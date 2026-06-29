"""
Pretrain VLAT_Enhanced encoders on medical image-caption data (ROCO, MedCAT, ImageCLEF).

Same objective stack as MUMC (https://github.com/pengfeiliHEU/MUMC):
  loss = loss_mlm + loss_ita + loss_itm

Example:
  python Pretrain_Enhanced.py \\
    --config configs/Pretrain.yaml \\
    --output_dir output/pretrain_enhanced
"""

import argparse
import datetime
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from transformers import BertTokenizer

import utils
from dataset import create_dataset, create_loader, create_sampler
from models.VLAT_Enhanced_Pretrain import VLAT_Enhanced_Pretrain, load_pretrain_checkpoint
from optim import create_optimizer
from scheduler import create_scheduler


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, distributed=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=50, fmt="{value:.6f}"))
    metric_logger.add_meter("loss_mlm", utils.SmoothedValue(window_size=50, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_ita", utils.SmoothedValue(window_size=50, fmt="{value:.4f}"))
    metric_logger.add_meter("loss_itm", utils.SmoothedValue(window_size=50, fmt="{value:.4f}"))

    header = f"Train Epoch: [{epoch}]"
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    if distributed and hasattr(data_loader, "sampler") and data_loader.sampler is not None:
        data_loader.sampler.set_epoch(epoch)

    for i, (image, text) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        optimizer.zero_grad()

        image = image.to(device, non_blocking=True)
        text_input = tokenizer(
            text, padding="longest", truncation=True, max_length=25, return_tensors="pt"
        ).to(device)

        if epoch > 0:
            alpha = config["alpha"]
        else:
            alpha = config["alpha"] * min(1, i / max(len(data_loader), 1))

        loss_mlm, loss_ita, loss_itm = model(image, text_input, alpha=alpha)
        loss = loss_mlm + loss_ita + loss_itm
        loss.backward()
        optimizer.step()

        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def epoch_total_loss(train_stats):
    return (
        float(train_stats["loss_mlm"])
        + float(train_stats["loss_ita"])
        + float(train_stats["loss_itm"])
    )


def build_checkpoint(model_without_ddp, optimizer, lr_scheduler, config, epoch, train_stats):
    total_loss = epoch_total_loss(train_stats)
    return {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "config": config,
        "epoch": epoch,
        "train_loss": total_loss,
        "train_stats": train_stats,
    }


def main(args, config):
    utils.init_distributed_mode(args)
    if args.distributed:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config["schedular"]["epochs"]
    warmup_steps = config["schedular"]["warmup_epochs"]

    print("Creating pretrain dataset")
    datasets = [create_dataset("pretrain", config)]
    print(f"Training samples: {len(datasets[0])}")

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True], num_tasks, global_rank)
    else:
        samplers = [None]

    data_loader = create_loader(
        datasets,
        samplers,
        batch_size=[config["batch_size"]],
        num_workers=[4],
        is_trains=[True],
        collate_fns=[None],
    )[0]

    print("Creating VLAT_Enhanced_Pretrain model")
    model = VLAT_Enhanced_Pretrain(
        config=config,
        text_encoder=args.text_encoder,
        bert_config=config["bert_config"],
        init_deit=not args.checkpoint,
    ).to(device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        best_loss = checkpoint.get("train_loss", float("inf"))
        start_epoch = load_pretrain_checkpoint(model, args.checkpoint, resume=args.resume) + 1
        if args.resume:
            print(f"  Resuming from epoch {start_epoch}, previous best loss: {best_loss:.4f}")
    else:
        best_loss = float("inf")

    arg_opt = utils.AttrDict(config["optimizer"])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config["schedular"])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    print("Start pretraining")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):
        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)

        train_stats = train(
            model, data_loader, optimizer, model_without_ddp.tokenizer,
            epoch, warmup_steps, device, lr_scheduler, config, args.distributed,
        )

        if utils.is_main_process():
            total_loss = epoch_total_loss(train_stats)
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "train_total_loss": f"{total_loss:.4f}",
                "epoch": epoch,
            }
            save_obj = build_checkpoint(
                model_without_ddp, optimizer, lr_scheduler, config, epoch, train_stats
            )

            last_path = os.path.join(args.output_dir, "enhanced_pretrain_last.pth")
            torch.save(save_obj, last_path)
            print(f"Saved last checkpoint: {last_path} (loss: {total_loss:.4f})")

            if total_loss < best_loss:
                best_loss = total_loss
                best_path = os.path.join(args.output_dir, "enhanced_pretrain_best.pth")
                torch.save(save_obj, best_path)
                print(f"Saved best checkpoint: {best_path} (loss: {best_loss:.4f})")

            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    if utils.is_main_process():
        print(f"\nPretraining finished. Best loss: {best_loss:.4f}")
        print(f"  Best: {os.path.join(args.output_dir, 'enhanced_pretrain_best.pth')}")
        print(f"  Last: {os.path.join(args.output_dir, 'enhanced_pretrain_last.pth')}")

    total_time = time.time() - start_time
    print(f"Pretraining time {str(datetime.timedelta(seconds=int(total_time)))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain VLAT_Enhanced on medical captions")
    parser.add_argument("--config", default="./configs/Pretrain.yaml")
    parser.add_argument("--checkpoint", default="", help="Resume or warm-start from checkpoint")
    parser.add_argument("--resume", action="store_true", help="Resume optimizer/scheduler state")
    parser.add_argument("--output_dir", default="output/pretrain_enhanced")
    parser.add_argument(
        "--text_encoder",
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--distributed", default=False, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(args.output_dir, "config.yaml"), "w"))

    main(args, config)
