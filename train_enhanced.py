"""
Training script for VLAT_Enhanced

This script trains the enhanced VLAT model with architectural improvements
"""

import os
import yaml
import wandb 
import utils
import random
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer

# Import enhanced model
from models.VLAT_Enhanced import VLAT_Enhanced
from models.VLAT_Enhanced_Pretrain import (
    transfer_pretrain_to_enhanced,
    transfer_clip_vision_to_enhanced,
)
try:
    from models.mumc_transfer import transfer_mumc_pretrain_to_enhanced
except ImportError:
    transfer_mumc_pretrain_to_enhanced = None
try:
    from models.VLAT_Enhanced_MVCM_Pretrain import transfer_mvcm_pretrain_to_enhanced
except ImportError:
    transfer_mvcm_pretrain_to_enhanced = None
from models.VLAT_Enhanced_FG_Pretrain import transfer_vlat_fg_pretrain_to_enhanced
from models.vit import interpolate_pos_embed

from dataset.utils import save_result
from dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn

from optim import create_optimizer
from scheduler import create_scheduler


def maybe_wandb_log(data):
    if wandb.run is not None:
        wandb.log(data)


def adapt_checkpoint_state_dict(state_dict, model):
    """Interpolate ViT pos_embed when loading a checkpoint trained at a different image size."""
    state_dict = dict(state_dict)
    key = 'image_encoder.vit_model.pos_embed'
    vit = getattr(getattr(model, 'image_encoder', None), 'vit_model', None)
    if key in state_dict and vit is not None:
        state_dict[key] = interpolate_pos_embed(state_dict[key], vit)
    return state_dict


def filter_compatible_state_dict(state_dict, model):
    """Drop checkpoint keys whose tensor shapes differ from the model (e.g. 414 vs 514 answer head)."""
    model_sd = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in model_sd:
            continue
        if model_sd[key].shape != value.shape:
            skipped.append(key)
            continue
        filtered[key] = value
    if skipped:
        print(f"  Skipped {len(skipped)} shape-mismatched keys: {skipped[:4]}{'...' if len(skipped) > 4 else ''}")
    return filtered


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    model.train()  
    metric_logger = utils.MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50    
    step_size = 100
    warmup_iterations = warmup_steps * step_size  

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if len(batch) == 8:
            image, question, answer, weights, n, answer_indices, question_answer_indices, answer_types = batch
        else:
            image, question, answer, weights, n, answer_indices = batch
            question_answer_indices, answer_types = None, None
        image, weights = image.to(device, non_blocking=True), weights.to(device, non_blocking=True)      
        question_input = tokenizer(question, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device) 
        answer_input = tokenizer(answer, padding='longest', return_tensors="pt").to(device) 

        if answer_indices is not None:
            answer_indices = answer_indices.to(device, non_blocking=True)
        if question_answer_indices is not None:
            question_answer_indices = question_answer_indices.to(device, non_blocking=True)

        if epoch > 0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        loss = model(
            image, question_input, answer_input, train=True, alpha=alpha, k=n, weights=weights,
            answer_indices=answer_indices,
            question_answer_indices=question_answer_indices,
            answer_types=answer_types,
            questions_text=question,
        )        
        
        optimizer.zero_grad()
        loss.backward()
        # Stability guard: clip gradients and skip the step on non-finite grads.
        # Under DDP the all-reduced grad_norm is identical on every rank, so the
        # skip decision stays in sync and cannot deadlock. Backward always runs.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 1.0))
        if torch.isfinite(grad_norm):
            optimizer.step()
            metric_logger.update(loss=loss.item())
        else:
            train.skipped = getattr(train, 'skipped', 0) + 1
            if utils.is_main_process():
                print(f"WARNING: non-finite grad_norm (={grad_norm}) at epoch {epoch} step {i}; "
                      f"skipping optimizer.step (skipped so far: {train.skipped})")

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations: 
            scheduler.step(i // step_size) 

        if i % print_freq == 0:
            maybe_wandb_log({
                "train/loss": float(loss.item()),
                "train/lr": float(optimizer.param_groups[0]["lr"]),
                "epoch": int(epoch),
                "step": int(i)
            })

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} 


def is_correct_answer(prediction, ground_truth_list):
    """
    Check if prediction matches any ground truth with partial matching support.
    
    Handles:
    - Exact match
    - Substring match
    - Multi-part answers (comma-separated)
    """
    for gt in ground_truth_list:
        # Exact match
        if prediction == gt:
            return True
        
        # Check if prediction is part of GT (e.g., "lung" in "lung cancer")
        if prediction in gt:
            return True
        
        # Check if GT is part of prediction
        if gt in prediction:
            return True
        
        # Handle comma-separated multi-part answers
        if ',' in gt:
            gt_parts = [part.strip() for part in gt.split(',')]
            if prediction in gt_parts:
                return True
            if any(prediction in part or part in prediction for part in gt_parts):
                return True
        
        if ',' in prediction:
            pred_parts = [part.strip() for part in prediction.split(',')]
            if gt in pred_parts:
                return True
            if any(gt in part or part in gt for part in pred_parts):
                return True
    
    return False


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config, infer_mode_override=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50
    
    result = []
    answer_list = [answer + config['eos'] for answer in data_loader.dataset.answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)    
    
    # Build mapping with answer_type metadata
    id_to_gt = {}
    id_to_type = {}
    for ann in data_loader.dataset.ann:
        id_to_gt[ann['question_id']] = ann.get('answer', [])
        id_to_type[ann['question_id']] = ann.get('answer_type', 'unknown').lower()
    
    # Track open/closed separately
    stats = {'open': {'correct': 0, 'total': 0}, 'closed': {'correct': 0, 'total': 0}}
    
    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device, non_blocking=True)             
        question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)        

        topk_ids, topk_probs = model(
            image, question_input, answer_input, None, train=False, k=config['k_test'],
            questions_text=question,
            answer_types=[id_to_type.get(int(q.item()), 'unknown') for q in question_id],
            infer_mode_override=infer_mode_override,
        )
        
        for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
            ques_id = int(ques_id.item())
            # topk() returns candidates sorted best-first; avoid argmax on padded slots
            predicted_answer = data_loader.dataset.answer_list[topk_id[0]]
            
            gt_answers = id_to_gt.get(ques_id, [])
            if not isinstance(gt_answers, list):
                gt_answers = [gt_answers]
            
            answer_type = id_to_type.get(ques_id, 'unknown')
            
            pred_normalized = str(predicted_answer).lower().strip()
            gt_normalized = [str(ans).lower().strip() for ans in gt_answers]
            
            # Use partial matching instead of exact match
            score = is_correct_answer(pred_normalized, gt_normalized)
            
            # Update stats by answer_type
            if answer_type in ['open', 'closed']:
                stats[answer_type]['correct'] += score
                stats[answer_type]['total'] += 1
            
            result.append({
                "question_id": ques_id, 
                "answer": predicted_answer,
                "correct": score,
                "answer_type": answer_type,
                "ground_truth": gt_answers
            })   
    
    # Print breakdown
    print(f"\n📊 Validation Breakdown:")
    for qtype in ['open', 'closed']:
        if stats[qtype]['total'] > 0:
            acc = stats[qtype]['correct'] / stats[qtype]['total'] * 100
            print(f"  {qtype.upper():7s}: {stats[qtype]['correct']:.1f}/{stats[qtype]['total']} = {acc:.2f}%")

    return result


def compute_split_accuracy(vqa_result):
    """Return overall, open, closed accuracy (percent) and counts."""
    open_c = open_t = closed_c = closed_t = 0
    for r in vqa_result:
        score = r.get('correct', 0.0)
        atype = str(r.get('answer_type', '')).lower()
        if atype == 'open':
            open_t += 1
            open_c += score
        elif atype == 'closed':
            closed_t += 1
            closed_c += score
    total = open_t + closed_t
    overall = (open_c + closed_c) / total * 100 if total else 0.0
    open_acc = open_c / open_t * 100 if open_t else 0.0
    closed_acc = closed_c / closed_t * 100 if closed_t else 0.0
    return overall, open_acc, closed_acc, open_c, open_t, closed_c, closed_t


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args, config):
    # Determine dataset name from config file path
    config_file = args.config
    if 'RAD' in config_file.upper():
        dataset_name = 'RAD'
    elif 'SLAKE' in config_file.upper():
        dataset_name = 'SLAKE'
    else:
        dataset_name = 'VQA'  # fallback
    
    # Initialize wandb (skip if evaluate-only mode)
    if not args.evaluate:
        experiment_name = f"{dataset_name}_Enhanced_{args.attention_mode}"
        if args.bidirectional:
            experiment_name += "_Bidirectional"
        if args.adaptive_gating:
            experiment_name += "_AdaptiveGate"
    
    utils.init_distributed_mode(args)
    if args.distributed:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device(args.device)

    if not args.evaluate and utils.is_main_process():
        wandb.init(project="VLAT_Enhanced_Study", name=experiment_name, config=config)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    start_epoch = 1
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    
    #### Dataset #### 
    print("Creating vqa datasets")
    train_dataset, test_dataset = create_dataset('vqa', config)
    num_answer_classes = len(train_dataset.answer_list)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        samplers = [train_sampler, None]  # evaluate on full 6719 test set
    else:
        samplers = [None, None]

    train_loader, test_loader = create_loader(
        [train_dataset, test_dataset], samplers,
        batch_size=[config['batch_size_train'], config['batch_size_test']],
        num_workers=[4, 4], is_trains=[True, False],
        collate_fns=[vqa_collate_fn, None]
    )
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Enhanced Model #### 
    print("="*60)
    print("Creating VLAT_Enhanced Model")
    
    # Parse head schedule
    head_schedule = None
    if args.head_schedule:
        head_schedule = [int(x) for x in args.head_schedule.split(',')]
    
    model = VLAT_Enhanced(
        embed_dim=args.embed_dim,
        fg_layers=args.fg_layers,
        head_schedule=head_schedule,
        attention_mode=args.attention_mode,
        use_bidirectional=args.bidirectional,
        use_adaptive_gating=args.adaptive_gating,
        use_answer_query=args.use_answer_query,
        num_answer_classes=num_answer_classes if (
            args.use_answer_query or args.use_open_answer_head or args.use_cmm_head
            or args.use_answer_boost
        ) else 0,
        answer_query_layers=args.answer_query_layers,
        answer_query_weight=args.answer_query_weight,
        answer_query_ffn_dim=args.answer_query_ffn_dim,
        use_asymmetric_query_loss=args.use_asymmetric_query_loss,
        answer_query_at_inference=args.answer_query_at_inference,
        answer_infer_mode=args.answer_infer_mode,
        answer_query_ensemble_weight=args.answer_query_ensemble_weight,
        answer_query_rerank_k=args.answer_query_rerank_k,
        answer_init_prefix=args.answer_init_prefix,
        answer_query_weight_open=args.answer_query_weight_open,
        answer_query_weight_closed=args.answer_query_weight_closed,
        answer_query_confidence_threshold=args.answer_query_confidence_threshold,
        use_fg_query_fusion=args.use_fg_query_fusion,
        use_fg_concat_query=args.use_fg_concat_query,
        query_cosine_weight=args.query_cosine_weight,
        use_bert_fusion_stream=args.use_bert_fusion_stream,
        hybrid_open_only=args.hybrid_open_only,
        hybrid_open_query_only=args.hybrid_open_query_only,
        open_decoder_loss_weight=args.open_decoder_loss_weight,
        use_answer_type_routing=args.use_answer_type_routing,
        query_only_train=args.query_only_train,
        query_train_open_only=args.query_train_open_only,
        use_closed_yn_head=args.use_closed_yn_head,
        closed_yn_loss_weight=args.closed_yn_loss_weight,
        closed_yn_infer_weight=args.closed_yn_infer_weight,
        closed_yn_train_only=args.closed_yn_train_only,
        closed_yn_hidden_dim=args.closed_yn_hidden_dim,
        use_open_answer_head=args.use_open_answer_head,
        open_answer_loss_weight=args.open_answer_loss_weight,
        open_answer_infer_weight=args.open_answer_infer_weight,
        open_answer_train_only=args.open_answer_train_only,
        open_answer_hidden_dim=args.open_answer_hidden_dim,
        use_dsap=args.use_dsap,
        dsap_sap_image_prompts=args.dsap_sap_image_prompts,
        dsap_sap_text_prompts=args.dsap_sap_text_prompts,
        dsap_dqap_prompts=args.dsap_dqap_prompts,
        dsap_train_prompts_only=args.dsap_train_prompts_only,
        use_itm_yn_head=args.use_itm_yn_head,
        itm_yn_loss_weight=args.itm_yn_loss_weight,
        itm_yn_infer_weight=args.itm_yn_infer_weight,
        use_laterality_expert=args.use_laterality_expert,
        laterality_loss_weight=args.laterality_loss_weight,
        laterality_infer_weight=args.laterality_infer_weight,
        laterality_confidence=args.laterality_confidence,
        use_modality_expert=args.use_modality_expert,
        modality_loss_weight=args.modality_loss_weight,
        modality_infer_weight=args.modality_infer_weight,
        modality_confidence=args.modality_confidence,
        e3_experts_train_only=args.e3_experts_train_only,
        e3_infer_enabled=args.e3_infer_enabled,
        e3_new_experts_only=args.e3_new_experts_only,
        closed_yn_compact=args.closed_yn_compact,
        closed_yn_focal_gamma=args.closed_yn_focal_gamma,
        closed_yn_exclusive_infer=args.closed_yn_exclusive_infer,
        closed_expert_train_only=args.closed_expert_train_only,
        use_cmm_head=args.use_cmm_head,
        cmm_loss_weight=args.cmm_loss_weight,
        cmm_infer_weight=args.cmm_infer_weight,
        cmm_train_only=args.cmm_train_only,
        cmm_num_layers=args.cmm_num_layers,
        cmm_use_bert_fusion=args.cmm_use_bert_fusion,
        use_answer_boost=args.use_answer_boost,
        answer_boost_train_only=args.answer_boost_train_only,
        answer_boost_loss_weight=args.answer_boost_loss_weight,
        answer_boost_closed_only=args.answer_boost_closed_only,
        use_mvcm_itlc=args.use_mvcm_itlc,
        mvcm_itlc_loss_weight=args.mvcm_itlc_loss_weight,
        mvcm_itc_loss_weight=args.mvcm_itc_loss_weight,
        mvcm_boost_train_only=args.mvcm_boost_train_only,
        dropout=args.dropout,
        prefusion_mlp=args.prefusion_mlp,
        prefusion_mlp_activation=args.prefusion_mlp_activation,
        image_size=config['image_res'],
        vision_encoder=args.vision_encoder,
        image_base=args.image_base,
        config_file=args.config_bert,
        encoder_base=args.text_encoder,
        decoder_base=args.text_decoder
    )
    print("="*60)
    
    model = model.to(device)   
    total_params = count_parameters(model)
    
    if not args.evaluate and utils.is_main_process():
        maybe_wandb_log({
            "Total Parameters": total_params,
            "Attention Mode": args.attention_mode,
            "Bidirectional": args.bidirectional,
            "Adaptive Gating": args.adaptive_gating
        })   
    print(f"Total Trainable Parameters: {total_params:,}")

    if args.pretrain_checkpoint and not args.resume and not config.get('skip_pretrain', False):
        print(f"\nLoading pretrain weights: {args.pretrain_checkpoint}")
        print(f"  pretrain_source: {args.pretrain_source}")
        if args.pretrain_source == 'mumc':
            transfer_mumc_pretrain_to_enhanced(model, args.pretrain_checkpoint)
        elif args.pretrain_source == 'mvcm_enhanced':
            transfer_mvcm_pretrain_to_enhanced(model, args.pretrain_checkpoint)
        elif args.pretrain_source == 'vlat_fg':
            transfer_vlat_fg_pretrain_to_enhanced(model, args.pretrain_checkpoint)
        elif args.vision_encoder == 'clip':
            transfer_clip_vision_to_enhanced(model, args.pretrain_checkpoint)
        else:
            transfer_pretrain_to_enhanced(model, args.pretrain_checkpoint)

    if args.use_answer_query and config.get('init_answer_embeddings', False) and not args.resume:
        print("\nInitializing answer-query embeddings from answer text...")
        model.register_answer_vocab(train_dataset.answer_list)
        model.init_answer_query_from_text(train_dataset.answer_list, device)
    elif args.use_answer_query:
        model.register_answer_vocab(train_dataset.answer_list)
    elif args.use_closed_yn_head or args.use_open_answer_head or args.use_itm_yn_head or args.use_cmm_head or args.use_answer_boost:
        model.register_answer_vocab(train_dataset.answer_list)

    if args.use_cmm_head and config.get('init_cmm_embeddings', False) and not args.resume:
        print("\nInitializing CMM answer embeddings from answer text...")
        model.init_cmm_from_text(train_dataset.answer_list, device)

    if args.e3_experts_train_only:
        for param in model.parameters():
            param.requires_grad = False
        train_heads = []
        if args.e3_new_experts_only:
            train_heads = [
                model.itm_yn_head, model.laterality_expert, model.modality_expert,
            ]
        else:
            train_heads = [
                model.itm_yn_head, model.laterality_expert, model.modality_expert,
                model.closed_yn_head, model.open_answer_head,
            ]
        for head in train_heads:
            if head is not None:
                for param in head.parameters():
                    param.requires_grad = True
        if args.e3_unfreeze_fg and not args.e3_new_experts_only:
            for param in model.fg_gf.parameters():
                param.requires_grad = True
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mode = 'new-experts-only' if args.e3_new_experts_only else (
            'experts+FG' if args.e3_unfreeze_fg else 'all-experts'
        )
        print(f"  E3 {mode} train-only: {n_trainable:,} trainable params")

    if args.dsap_train_prompts_only:
        for param in model.parameters():
            param.requires_grad = False
        for module in (model.sap_module, model.dqap_module):
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = True
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  DSAP prompt-only train: {n_trainable:,} trainable params")

    if args.open_answer_train_only and args.closed_yn_train_only:
        for param in model.parameters():
            param.requires_grad = False
        for head in (model.open_answer_head, model.closed_yn_head):
            if head is not None:
                for param in head.parameters():
                    param.requires_grad = True
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Open+Closed aux MLP train-only: {n_trainable:,} trainable params")

    elif args.open_answer_train_only:
        for param in model.parameters():
            param.requires_grad = False
        if model.open_answer_head is not None:
            for param in model.open_answer_head.parameters():
                param.requires_grad = True
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Open answer train-only: {n_trainable:,} trainable params")

    elif args.cmm_train_only:
        for param in model.parameters():
            param.requires_grad = False
        if model.cmm_head is not None:
            for param in model.cmm_head.parameters():
                param.requires_grad = True
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  CMM train-only: {n_trainable:,} trainable params")

    elif args.mvcm_boost_train_only or args.answer_boost_train_only:
        for param in model.parameters():
            param.requires_grad = False
        if model.answer_boost is not None:
            for mod in model.answer_boost.trainable_modules():
                for param in mod.parameters():
                    param.requires_grad = True
        if model.mvcm_align is not None:
            for param in model.mvcm_align.parameters():
                param.requires_grad = True
        tag = 'MVCM+boost' if args.mvcm_boost_train_only else 'answer boost'
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {tag} train-only: {n_trainable:,} trainable params")

    elif args.closed_expert_train_only:
        for param in model.parameters():
            param.requires_grad = False
        if model.closed_yn_head is not None:
            for param in model.closed_yn_head.parameters():
                param.requires_grad = True
        if model.answer_boost is not None:
            for mod in model.answer_boost.trainable_modules():
                for param in mod.parameters():
                    param.requires_grad = True
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Closed expert train-only (yn+ITM): {n_trainable:,} trainable params")

    elif args.closed_yn_train_only:
        for param in model.parameters():
            param.requires_grad = False
        if model.closed_yn_head is not None:
            for param in model.closed_yn_head.parameters():
                param.requires_grad = True
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Closed yes/no train-only: {n_trainable:,} trainable params")

    if args.query_only_train:
        if not args.closed_yn_train_only:
            freeze_modules = [model.image_encoder, model.text_encoder, model.decoder]
            if not args.query_train_unfreeze_fg:
                freeze_modules.append(model.fg_gf)
            for module in freeze_modules:
                for param in module.parameters():
                    param.requires_grad = False
            if getattr(model, 'image_prefusion_mlp', None) is not None:
                for param in model.image_prefusion_mlp.parameters():
                    param.requires_grad = False
            if getattr(model, 'text_prefusion_mlp', None) is not None:
                for param in model.text_prefusion_mlp.parameters():
                    param.requires_grad = False
        if model.answer_query_head is not None:
            for param in model.answer_query_head.parameters():
                param.requires_grad = True
        if not args.closed_yn_train_only:
            if model.open_answer_head is not None:
                for param in model.open_answer_head.parameters():
                    param.requires_grad = False
            if model.closed_yn_head is not None:
                for param in model.closed_yn_head.parameters():
                    param.requires_grad = False
            for module in (getattr(model, 'sap_module', None), getattr(model, 'dqap_module', None)):
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad = False
        if args.closed_yn_train_only:
            # Combined expert phase: keep closed head trainable too
            if model.closed_yn_head is not None:
                for param in model.closed_yn_head.parameters():
                    param.requires_grad = True
            for module in [model.image_encoder, model.text_encoder, model.decoder, model.fg_gf]:
                for param in module.parameters():
                    param.requires_grad = False
            if getattr(model, 'sap_module', None) is not None:
                for param in model.sap_module.parameters():
                    param.requires_grad = False
            if getattr(model, 'dqap_module', None) is not None:
                for param in model.dqap_module.parameters():
                    param.requires_grad = False
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mode = 'query+closed experts' if args.closed_yn_train_only else 'query head only'
        print(f"  Q2A query-only train ({mode}): {n_trainable:,} trainable params")

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
    
    # Evaluate-only mode
    if args.evaluate:
        if not args.checkpoint:
            print("Error: --checkpoint required for evaluation mode")
            return
        
        print(f"\n{'='*70}")
        print(f"EVALUATION MODE")
        print(f"{'='*70}")
        print(f"Loading checkpoint: {args.checkpoint}")
        
        checkpoint = torch.load(args.checkpoint, map_location=device)
        ckpt_sd = filter_compatible_state_dict(
            adapt_checkpoint_state_dict(checkpoint['model'], model_without_ddp),
            model_without_ddp,
        )
        missing, unexpected = model_without_ddp.load_state_dict(ckpt_sd, strict=False)
        if missing:
            print(f"  Checkpoint missing {len(missing)} keys (e.g. new modules): {missing[:4]}...")
        if unexpected:
            print(f"  Checkpoint unexpected keys: {len(unexpected)}")
        
        if 'accuracy' in checkpoint:
            print(f"Checkpoint accuracy: {checkpoint['accuracy']*100:.2f}%")
        if 'epoch' in checkpoint:
            print(f"Checkpoint epoch: {checkpoint['epoch']}")
        
        print(f"\nRunning evaluation on test set...")
        vqa_result = evaluation(model_without_ddp, test_loader, tokenizer, device, config)

        overall, open_acc, closed_acc, open_c, open_t, closed_c, closed_t = (
            compute_split_accuracy(vqa_result)
        )

        print(f"\n{'='*70}")
        print(f"FINAL RESULTS")
        print(f"{'='*70}")
        print(f"Overall Accuracy: {overall:.2f}% ({open_c + closed_c:.1f}/{open_t + closed_t})")
        open_tag = "  ← Q2A target" if args.use_answer_query else ""
        if open_t > 0:
            print(f"OPEN Questions:   {open_acc:.2f}% ({open_c:.1f}/{open_t}){open_tag}")
        if closed_t > 0:
            print(f"CLOSED Questions: {closed_acc:.2f}% ({closed_c:.1f}/{closed_t})")
        print(f"{'='*70}\n")
        
        return
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    best_acc = 0.0
    best_open_acc = 0.0
    save_best_by_open = config.get('save_best_by_open', False)
    save_best_by_closed = config.get('save_best_by_closed', False)
    best_closed_acc = 0.0
    if args.checkpoint and not args.resume and not args.evaluate:
        print(f"\nLoading base checkpoint (strict=False): {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        ckpt_sd = filter_compatible_state_dict(
            adapt_checkpoint_state_dict(checkpoint['model'], model_without_ddp),
            model_without_ddp,
        )
        missing, unexpected = model_without_ddp.load_state_dict(ckpt_sd, strict=False)
        print(f"  Loaded base weights: missing={len(missing)}, unexpected={len(unexpected)}")
        if 'accuracy' in checkpoint:
            acc = checkpoint['accuracy']
            acc_val = acc / 100.0 if acc > 1.0 else acc
            print(f"  Base checkpoint accuracy: {acc_val*100:.2f}%")

    if args.resume and args.checkpoint:
        print(f"\nResuming from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        ckpt_sd = filter_compatible_state_dict(
            adapt_checkpoint_state_dict(checkpoint['model'], model_without_ddp),
            model_without_ddp,
        )
        missing, unexpected = model_without_ddp.load_state_dict(ckpt_sd, strict=False)
        if missing:
            print(f"  Resume missing {len(missing)} keys: {missing[:4]}...")
        start_epoch = checkpoint.get('epoch', 0) + 1
        acc = checkpoint.get('accuracy', 0.0)
        best_acc = acc / 100.0 if acc > 1.0 else acc
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                if 'step' in state and isinstance(state['step'], torch.Tensor):
                    state['step'] = state['step'].cpu()
        # Re-sync LR schedule (old runs never stepped cosine scheduler correctly)
        resume_scheduler_epoch = start_epoch + warmup_steps - 1
        if resume_scheduler_epoch > 0:
            lr_scheduler.step(resume_scheduler_epoch)
        print(f"  Resuming epoch {start_epoch}, previous best acc: {best_acc*100:.2f}%")
        print(f"  LR after schedule sync: {optimizer.param_groups[0]['lr']:.2e}")

    # Training loop
    for epoch in range(start_epoch, max_epoch + 1):
        if args.distributed and samplers[0] is not None:
            samplers[0].set_epoch(epoch)

        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)

        train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config)
        
        if epoch % args.eval_freq == 0 and utils.is_main_process():
            print(f"\nRunning validation at epoch {epoch}...")
            infer_override = None
            warmup_epochs = int(config.get('hybrid_warmup_epochs', 0))
            if warmup_epochs > 0 and epoch < warmup_epochs:
                infer_override = 'decoder'
            e3_infer_warmup = int(config.get('e3_infer_warmup_epochs', 0))
            if hasattr(model_without_ddp, 'e3_infer_enabled'):
                model_without_ddp.e3_infer_enabled = (
                    epoch >= e3_infer_warmup and config.get('e3_infer_enabled', True)
                )
            query_warmup = int(config.get('query_infer_warmup_epochs', 0))
            if hasattr(model_without_ddp, 'answer_query_ensemble_weight'):
                base_w = float(config.get(
                    'query_infer_ensemble_weight',
                    config.get('answer_query_ensemble_weight', 0.35),
                ))
                warm_w = float(config.get('query_infer_ensemble_weight_warmup', 0.10))
                model_without_ddp.answer_query_ensemble_weight = (
                    warm_w if query_warmup > 0 and epoch < query_warmup else base_w
                )
            if hasattr(model_without_ddp, 'answer_boost') and model_without_ddp.answer_boost is not None:
                boost_warmup = int(config.get('answer_boost_infer_warmup_epochs', 0))
                model_without_ddp.answer_boost.infer_enabled = (
                    epoch >= boost_warmup and config.get('answer_boost_infer_enabled', True)
                )
            vqa_result = evaluation(
                model_without_ddp, test_loader, tokenizer, device, config,
                infer_mode_override=infer_override,
            )

            overall, open_acc, closed_acc, open_c, open_t, closed_c, closed_t = (
                compute_split_accuracy(vqa_result)
            )
            accuracy = overall / 100.0

            print(f"\nValidation Results:")
            print(f"  Overall: {overall:.2f}% ({open_c + closed_c:.1f}/{open_t + closed_t})")
            open_tag = "  ← Q2A target" if args.use_answer_query else ""
            print(f"  OPEN:    {open_acc:.2f}% ({open_c:.1f}/{open_t}){open_tag}")
            print(f"  CLOSED:  {closed_acc:.2f}% ({closed_c:.1f}/{closed_t})")
            
            # For display: consider score >= 1.0 as fully correct
            correct_examples = [r for r in vqa_result if r.get('correct', 0.0) >= 1.0][:3]
            incorrect_examples = [r for r in vqa_result if r.get('correct', 0.0) < 1.0][:3]
            
            print(f"\nSample Predictions:")
            for i, ex in enumerate(correct_examples, 1):
                print(f"  ✓ Correct {i}: Pred='{ex['answer']}' | GT={ex['ground_truth']}")
            for i, ex in enumerate(incorrect_examples, 1):
                print(f"  ✗ Wrong {i}: Pred='{ex['answer']}' | GT={ex['ground_truth']}")
            
            maybe_wandb_log({
                "val/accuracy": accuracy,
                "val/open_accuracy": open_acc / 100.0,
                "val/closed_accuracy": closed_acc / 100.0,
                "epoch": epoch
            })
            
            if accuracy > best_acc:
                prev_best = best_acc
                best_acc = accuracy
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'accuracy': accuracy,
                    'open_accuracy': open_acc / 100.0,
                    'closed_accuracy': closed_acc / 100.0,
                }
                best_path = os.path.join(args.output_dir, f"{dataset_name}_Enhanced_best.pth")
                torch.save(save_obj, best_path)
                print(f"\n🎉 New best overall! {overall:.2f}% (prev {prev_best*100:.2f}%)")
                print(f"  Saved: {best_path}")

            if save_best_by_open and open_acc / 100.0 > best_open_acc:
                prev_open = best_open_acc
                best_open_acc = open_acc / 100.0
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'accuracy': accuracy,
                    'open_accuracy': open_acc / 100.0,
                    'closed_accuracy': closed_acc / 100.0,
                }
                open_path = os.path.join(args.output_dir, f"{dataset_name}_Enhanced_best_open.pth")
                torch.save(save_obj, open_path)
                print(f"🎯 New best OPEN! {open_acc:.2f}% (prev {prev_open*100:.2f}%) → {open_path}")

            if save_best_by_closed and closed_acc / 100.0 > best_closed_acc:
                prev_closed = best_closed_acc
                best_closed_acc = closed_acc / 100.0
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'accuracy': accuracy,
                    'open_accuracy': open_acc / 100.0,
                    'closed_accuracy': closed_acc / 100.0,
                }
                closed_path = os.path.join(args.output_dir, f"{dataset_name}_Enhanced_best_closed.pth")
                torch.save(save_obj, closed_path)
                print(f"🎯 New best CLOSED! {closed_acc:.2f}% (prev {prev_closed*100:.2f}%) → {closed_path}")
                best_path = os.path.join(args.output_dir, f"{dataset_name}_Enhanced_best.pth")
                torch.save(save_obj, best_path)
                print(f"  Also saved as primary best: {best_path}")
        
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            'epoch': epoch,
        }    
        if utils.is_main_process():
            maybe_wandb_log(log_stats)
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            last_path = os.path.join(args.output_dir, f"{dataset_name}_Enhanced_last.pth")
            torch.save(save_obj, last_path)
            print(f"Saved last checkpoint: {last_path} (epoch {epoch})")

    if utils.is_main_process():
        print("\nFinal Evaluation...")
        vqa_result = evaluation(model_without_ddp, test_loader, tokenizer, device, config)
        correct = sum(r.get('correct', 0.0) for r in vqa_result)
        total = len(vqa_result)
        final_accuracy = correct / total if total > 0 else 0.0

        print(f"\n{'='*60}")
        print("TRAINING COMPLETED!")
        print(f"{'='*60}")
        print(f"Best Validation Accuracy: {best_acc*100:.2f}%")
        print(f"Final Validation Accuracy: {final_accuracy*100:.2f}%")
        print(f"\nCheckpoints saved:")
        print(f"  - Best: {os.path.join(args.output_dir, f'{dataset_name}_Enhanced_best.pth')}")
        print(f"  - Last: {os.path.join(args.output_dir, f'{dataset_name}_Enhanced_last.pth')}")
        print(f"{'='*60}")
        
        if not args.evaluate:
            maybe_wandb_log({
                "final/accuracy": final_accuracy,
                "final/best_accuracy": best_acc
            })
            if wandb.run is not None:
                wandb.finish()

    if args.distributed:
        torch.distributed.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VQA_Slake.yaml') 
    parser.add_argument('--config_bert', default='./configs/config_bert.json')
    parser.add_argument('--output_dir', default='output/vqa_enhanced')
    parser.add_argument('--text_encoder', default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
    parser.add_argument('--text_decoder', default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int)    
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--distributed', default=False, type=bool)
    
    # Enhanced model specific arguments
    parser.add_argument('--embed_dim', default=768, type=int, help='Embedding dimension')
    parser.add_argument('--fg_layers', default=6, type=int, help='Number of FineGrained layers')
    parser.add_argument('--head_schedule', default=None, type=str, 
                       help='Comma-separated head counts (e.g., "12,10,8,6,4,4")')
    parser.add_argument('--attention_mode', default='standard', choices=['standard', 'linear', 'sparse', 'flash'],
                       help='Attention mechanism: standard (O(N²)), linear (O(N)), sparse (O(N×k)), flash (O(N²) IO-efficient)')
    parser.add_argument('--bidirectional', action='store_true',
                       help='Enable bidirectional refinement (text also attends to image)')
    parser.add_argument('--adaptive_gating', action='store_true',
                       help='Enable adaptive gating (learn layer importance)')
    parser.add_argument('--use_answer_query', action='store_true',
                       help='Enable Q2A-style answer query head for direct class scoring')
    parser.add_argument('--answer_query_layers', default=2, type=int,
                       help='Number of answer-query transformer layers')
    parser.add_argument('--answer_query_weight', default=0.5, type=float,
                       help='Weight for answer-query auxiliary loss')
    parser.add_argument('--answer_query_ffn_dim', default=0, type=int,
                       help='FFN hidden dim in answer-query layers (0 = 4x embed_dim)')
    parser.add_argument('--use_asymmetric_query_loss', action='store_true',
                       help='Use Q2AT-style asymmetric loss for query head')
    parser.add_argument('--answer_init_prefix', default='answer: ',
                       help='Prefix for tokenizing answer strings at init')
    parser.add_argument('--answer_query_at_inference', action='store_true',
                       help='Use answer-query head at inference (default: decoder ranking)')
    parser.add_argument('--answer_infer_mode', default='decoder',
                       choices=['decoder', 'query', 'ensemble', 'hybrid'],
                       help='Inference strategy when answer-query head is enabled')
    parser.add_argument('--answer_query_ensemble_weight', default=0.3, type=float,
                       help='Weight for query logits in ensemble / open hybrid rerank')
    parser.add_argument('--answer_query_rerank_k', default=32, type=int,
                       help='Top-k candidates for ensemble / open hybrid reranking')
    parser.add_argument('--answer_query_weight_open', default=1.0, type=float,
                       help='Query aux loss multiplier for OPEN questions')
    parser.add_argument('--answer_query_weight_closed', default=0.25, type=float,
                       help='Query aux loss multiplier for CLOSED questions')
    parser.add_argument('--answer_query_confidence_threshold', default=0.55, type=float,
                       help='Query confidence threshold for closed routing at inference')
    parser.add_argument('--use_fg_query_fusion', action='store_true',
                       help='Feed VLAT FineGrained image/text streams into Q2A head')
    parser.add_argument('--use_fg_concat_query', action='store_true',
                       help='Q2AT strict: concat FG text+image tokens → AnswerQueryHead (skip BERT fusion)')
    parser.add_argument('--query_cosine_weight', default=0.5, type=float,
                       help='Cosine term weight in AnswerQueryHead logits (0 = linear only, Q2AT strict)')
    parser.add_argument('--use_bert_fusion_stream', action='store_true',
                       help='Add BERT fusion stream to FG Q2A head (triple-stream)')
    parser.add_argument('--hybrid_open_only', action='store_true', default=True,
                       help='Hybrid: Q2A on OPEN only, VLAT decoder on CLOSED')
    parser.add_argument('--hybrid_open_query_only', action='store_true', default=False,
                       help='OPEN uses query-only top-k (pure Q2AT). Default: query+decoder rerank')
    parser.add_argument('--open_decoder_loss_weight', default=0.05, type=float,
                       help='Decoder loss weight for OPEN samples when hybrid Q2A is enabled')
    parser.add_argument('--use_answer_type_routing', action='store_true', default=True,
                       help='Route OPEN/CLOSED via dataset answer_type at eval (fallback: question heuristics)')
    parser.add_argument('--no_answer_type_routing', action='store_false', dest='use_answer_type_routing',
                       help='Always route via question-text heuristics instead of answer_type metadata')
    parser.add_argument('--query_only_train', action='store_true',
                       help='Q2A pipeline: ASL on query head only, skip decoder loss')
    parser.add_argument('--query_train_unfreeze_fg', action='store_true',
                       help='With query_only_train: also train FineGrained (Q2AT primary finetune)')
    parser.add_argument('--query_train_open_only', action='store_true', default=True,
                       help='With query_only_train: train ASL on OPEN samples only (default True)')
    parser.add_argument('--query_train_all_types', action='store_false', dest='query_train_open_only',
                       help='With query_only_train: train ASL on OPEN+CLOSED (all 414 classes)')
    parser.add_argument('--use_closed_yn_head', action='store_true',
                       help='Binary yes/no head for CLOSED questions (Q2AT split)')
    parser.add_argument('--closed_yn_loss_weight', default=0.3, type=float,
                       help='Auxiliary CE weight for closed yes/no head during decoder training')
    parser.add_argument('--closed_yn_infer_weight', default=0.5, type=float,
                       help='Blend weight for closed yes/no head at inference (0=decoder only)')
    parser.add_argument('--closed_yn_train_only', action='store_true',
                       help='Freeze backbone; train closed yes/no head only')
    parser.add_argument('--closed_yn_exclusive_infer', action='store_true',
                       help='CLOSED infer: yes/no head + ITM only (no decoder blend)')
    parser.add_argument('--closed_expert_train_only', action='store_true',
                       help='Train closed yes/no + ITM experts on frozen backbone')
    parser.add_argument('--answer_boost_closed_only', action='store_true',
                       help='Answer boost: ITM yes/no only (skip lat/mod/synonyms)')
    parser.add_argument('--closed_yn_hidden_dim', default=0, type=int,
                       help='Hidden dim for closed yes/no MLP (0 = embed_dim)')
    parser.add_argument('--use_open_answer_head', action='store_true',
                       help='Lightweight 414-way head for OPEN questions')
    parser.add_argument('--open_answer_loss_weight', default=0.3, type=float,
                       help='Auxiliary CE weight for open answer head during decoder training')
    parser.add_argument('--open_answer_infer_weight', default=0.3, type=float,
                       help='Blend weight for open answer head at inference (0=decoder only)')
    parser.add_argument('--open_answer_train_only', action='store_true',
                       help='Freeze backbone; train open answer head only')
    parser.add_argument('--open_answer_hidden_dim', default=0, type=int,
                       help='Hidden dim for open answer MLP (0 = embed_dim)')
    parser.add_argument('--use_dsap', action='store_true',
                       help='DSAP: Semantic Alignment + Dynamic Question-Aware Prompts')
    parser.add_argument('--dsap_sap_image_prompts', default=8, type=int,
                       help='Number of SAP image soft prompts')
    parser.add_argument('--dsap_sap_text_prompts', default=8, type=int,
                       help='Number of SAP text soft prompts')
    parser.add_argument('--dsap_dqap_prompts', default=8, type=int,
                       help='Number of DQAP dynamic prompts')
    parser.add_argument('--dsap_train_prompts_only', action='store_true',
                       help='Freeze backbone; train DSAP prompt parameters only')
    parser.add_argument('--use_itm_yn_head', action='store_true',
                       help='ITM-style yes/no expert for CLOSED questions')
    parser.add_argument('--itm_yn_loss_weight', default=1.0, type=float)
    parser.add_argument('--itm_yn_infer_weight', default=0.55, type=float)
    parser.add_argument('--use_laterality_expert', action='store_true',
                       help='Laterality specialist for OPEN left/right questions')
    parser.add_argument('--laterality_loss_weight', default=1.0, type=float)
    parser.add_argument('--laterality_infer_weight', default=0.6, type=float)
    parser.add_argument('--laterality_confidence', default=0.45, type=float)
    parser.add_argument('--use_modality_expert', action='store_true',
                       help='Modality specialist for OPEN MRI/CT/x-ray questions')
    parser.add_argument('--modality_loss_weight', default=1.0, type=float)
    parser.add_argument('--modality_infer_weight', default=0.55, type=float)
    parser.add_argument('--modality_confidence', default=0.40, type=float)
    parser.add_argument('--e3_experts_train_only', action='store_true',
                       help='Freeze backbone; train E3 expert stack only')
    parser.add_argument('--e3_infer_enabled', default=True,
                       type=lambda x: str(x).lower() in ('1', 'true', 'yes'))
    parser.add_argument('--e3_new_experts_only', action='store_true',
                       help='Train only ITM/laterality/modality; keep loaded aux heads frozen')
    parser.add_argument('--e3_unfreeze_fg', action='store_true',
                       help='With e3_experts_train_only: also fine-tune FineGrained layers')
    parser.add_argument('--closed_yn_compact', action='store_true',
                       help='2-layer closed yn head (matches open_aux checkpoint)')
    parser.add_argument('--closed_yn_focal_gamma', default=0.0, type=float,
                       help='Focal loss gamma for closed yn (0=disabled)')
    parser.add_argument('--use_cmm_head', action='store_true',
                       help='Cross-Modal Matching head (CMI-MTL inspired)')
    parser.add_argument('--cmm_loss_weight', default=1.0, type=float,
                       help='Loss weight for CMM head')
    parser.add_argument('--cmm_infer_weight', default=0.35, type=float,
                       help='Inference blend weight for CMM scores')
    parser.add_argument('--cmm_train_only', action='store_true',
                       help='Train only CMM head; freeze backbone and aux heads')
    parser.add_argument('--cmm_num_layers', default=1, type=int,
                       help='Number of CMM cross-attention layers')
    parser.add_argument('--cmm_use_bert_fusion', action='store_true', default=True,
                       help='Include BERT fusion stream in CMM context')
    parser.add_argument('--use_answer_boost', action='store_true',
                       help='Unified boost stack: ITM + laterality + modality + synonyms')
    parser.add_argument('--answer_boost_train_only', action='store_true',
                       help='Train only answer boost modules; freeze backbone and aux heads')
    parser.add_argument('--answer_boost_loss_weight', default=1.0, type=float,
                       help='Global loss scale for answer boost training')
    parser.add_argument('--use_mvcm_itlc', action='store_true',
                       help='MVCM-lite local+global contrastive alignment loss')
    parser.add_argument('--mvcm_itlc_loss_weight', default=0.5, type=float,
                       help='Weight for ITLC (token-level contrastive)')
    parser.add_argument('--mvcm_itc_loss_weight', default=0.25, type=float,
                       help='Weight for global ITC contrastive (0=disable)')
    parser.add_argument('--mvcm_boost_train_only', action='store_true',
                       help='Train MVCM align + answer boost only on frozen open_aux base')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout rate')
    parser.add_argument('--prefusion_mlp', action='store_true',
                       help='Apply MLP to image/text embeddings before FineGrained fusion')
    parser.add_argument('--prefusion_mlp_activation', default='gelu',
                       choices=['relu', 'leaky_relu', 'gelu', 'none'],
                       help='Activation in pre-fusion MLP (none = linear only)')
    parser.add_argument('--vision_encoder', default='deit', choices=['deit', 'clip', 'biomedclip'],
                       help='Image encoder: deit, clip, or biomedclip (PMC-15M medical ViT)')
    parser.add_argument('--image_base', default=None, type=str,
                       help='HuggingFace model id for CLIP vision (default: BiomedCLIP)')
    
    parser.add_argument('--eval_freq', default=10, type=int, help='Evaluate every N epochs')
    parser.add_argument('--save_freq', default=50, type=int, help='Save checkpoint every N epochs')
    parser.add_argument('--checkpoint', default=None, type=str, help='Path to checkpoint for evaluation or resume')
    parser.add_argument('--resume', action='store_true', help='Resume training from --checkpoint')
    parser.add_argument('--pretrain_checkpoint', default=None, type=str,
                       help='Path to pretrain checkpoint (VLAT enhanced or MUMC official)')
    parser.add_argument('--pretrain_source', default='vlat', choices=['vlat', 'mumc', 'mvcm_enhanced', 'vlat_fg'],
                       help='Checkpoint format: vlat, mumc (official), or mvcm_enhanced (FG+MVCM pretrain)')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation only')
    
    args = parser.parse_args()
    
    # Load config
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    if config.get('use_answer_query', False):
        args.use_answer_query = True
    if 'answer_query_weight' in config:
        args.answer_query_weight = float(config['answer_query_weight'])
    if config.get('answer_query_at_inference', False):
        args.answer_query_at_inference = True
    if 'answer_infer_mode' in config:
        args.answer_infer_mode = str(config['answer_infer_mode'])
    if 'answer_query_ensemble_weight' in config:
        args.answer_query_ensemble_weight = float(config['answer_query_ensemble_weight'])
    if 'answer_query_rerank_k' in config:
        args.answer_query_rerank_k = int(config['answer_query_rerank_k'])
    if 'answer_query_layers' in config:
        args.answer_query_layers = int(config['answer_query_layers'])
    if 'answer_query_ffn_dim' in config:
        args.answer_query_ffn_dim = int(config['answer_query_ffn_dim'])
    if config.get('use_asymmetric_query_loss', False):
        args.use_asymmetric_query_loss = True
    if 'answer_init_prefix' in config:
        args.answer_init_prefix = str(config['answer_init_prefix'])
    if 'answer_query_weight_open' in config:
        args.answer_query_weight_open = float(config['answer_query_weight_open'])
    if 'answer_query_weight_closed' in config:
        args.answer_query_weight_closed = float(config['answer_query_weight_closed'])
    if 'answer_query_confidence_threshold' in config:
        args.answer_query_confidence_threshold = float(config['answer_query_confidence_threshold'])
    if config.get('use_fg_query_fusion', False):
        args.use_fg_query_fusion = True
    if config.get('use_fg_concat_query', False):
        args.use_fg_concat_query = True
    if 'query_cosine_weight' in config:
        args.query_cosine_weight = float(config['query_cosine_weight'])
    if config.get('use_bert_fusion_stream', False):
        args.use_bert_fusion_stream = True
    if config.get('hybrid_open_only', True):
        args.hybrid_open_only = True
    if config.get('hybrid_open_query_only', False):
        args.hybrid_open_query_only = True
    if 'open_decoder_loss_weight' in config:
        args.open_decoder_loss_weight = float(config['open_decoder_loss_weight'])
    if config.get('use_answer_type_routing', True):
        args.use_answer_type_routing = True
    if config.get('use_answer_type_routing') is False:
        args.use_answer_type_routing = False
    if config.get('prefusion_mlp', False):
        args.prefusion_mlp = True
    if 'prefusion_mlp_activation' in config:
        args.prefusion_mlp_activation = str(config['prefusion_mlp_activation'])
    if config.get('query_only_train', False):
        args.query_only_train = True
    if config.get('query_train_unfreeze_fg', False):
        args.query_train_unfreeze_fg = True
    if config.get('query_train_open_only') is False:
        args.query_train_open_only = False
    if config.get('query_train_all_types', False):
        args.query_train_open_only = False
    if config.get('use_closed_yn_head', False):
        args.use_closed_yn_head = True
    if 'closed_yn_loss_weight' in config:
        args.closed_yn_loss_weight = float(config['closed_yn_loss_weight'])
    if 'closed_yn_infer_weight' in config:
        args.closed_yn_infer_weight = float(config['closed_yn_infer_weight'])
    if config.get('closed_yn_train_only', False):
        args.closed_yn_train_only = True
    if 'closed_yn_hidden_dim' in config:
        args.closed_yn_hidden_dim = int(config['closed_yn_hidden_dim'])
    if config.get('use_open_answer_head', False):
        args.use_open_answer_head = True
    if 'open_answer_loss_weight' in config:
        args.open_answer_loss_weight = float(config['open_answer_loss_weight'])
    if 'open_answer_infer_weight' in config:
        args.open_answer_infer_weight = float(config['open_answer_infer_weight'])
    if config.get('open_answer_train_only', False):
        args.open_answer_train_only = True
    if 'open_answer_hidden_dim' in config:
        args.open_answer_hidden_dim = int(config['open_answer_hidden_dim'])
    if config.get('use_dsap', False):
        args.use_dsap = True
    if 'dsap_sap_image_prompts' in config:
        args.dsap_sap_image_prompts = int(config['dsap_sap_image_prompts'])
    if 'dsap_sap_text_prompts' in config:
        args.dsap_sap_text_prompts = int(config['dsap_sap_text_prompts'])
    if 'dsap_dqap_prompts' in config:
        args.dsap_dqap_prompts = int(config['dsap_dqap_prompts'])
    if config.get('dsap_train_prompts_only', False):
        args.dsap_train_prompts_only = True
    if config.get('use_itm_yn_head', False):
        args.use_itm_yn_head = True
    if 'itm_yn_loss_weight' in config:
        args.itm_yn_loss_weight = float(config['itm_yn_loss_weight'])
    if 'itm_yn_infer_weight' in config:
        args.itm_yn_infer_weight = float(config['itm_yn_infer_weight'])
    if config.get('use_laterality_expert', False):
        args.use_laterality_expert = True
    if 'laterality_loss_weight' in config:
        args.laterality_loss_weight = float(config['laterality_loss_weight'])
    if 'laterality_infer_weight' in config:
        args.laterality_infer_weight = float(config['laterality_infer_weight'])
    if 'laterality_confidence' in config:
        args.laterality_confidence = float(config['laterality_confidence'])
    if config.get('use_modality_expert', False):
        args.use_modality_expert = True
    if 'modality_loss_weight' in config:
        args.modality_loss_weight = float(config['modality_loss_weight'])
    if 'modality_infer_weight' in config:
        args.modality_infer_weight = float(config['modality_infer_weight'])
    if 'modality_confidence' in config:
        args.modality_confidence = float(config['modality_confidence'])
    if config.get('e3_experts_train_only', False):
        args.e3_experts_train_only = True
    if 'e3_infer_enabled' in config:
        args.e3_infer_enabled = bool(config['e3_infer_enabled'])
    if config.get('e3_new_experts_only', False):
        args.e3_new_experts_only = True
    if config.get('e3_unfreeze_fg', False):
        args.e3_unfreeze_fg = True
    if config.get('closed_yn_compact', False):
        args.closed_yn_compact = True
    if 'closed_yn_focal_gamma' in config:
        args.closed_yn_focal_gamma = float(config['closed_yn_focal_gamma'])
    if config.get('closed_yn_exclusive_infer', False):
        args.closed_yn_exclusive_infer = True
    if config.get('closed_expert_train_only', False):
        args.closed_expert_train_only = True
    if config.get('answer_boost_closed_only', False):
        args.answer_boost_closed_only = True
    if config.get('use_cmm_head', False):
        args.use_cmm_head = True
    if 'cmm_loss_weight' in config:
        args.cmm_loss_weight = float(config['cmm_loss_weight'])
    if 'cmm_infer_weight' in config:
        args.cmm_infer_weight = float(config['cmm_infer_weight'])
    if config.get('cmm_train_only', False):
        args.cmm_train_only = True
    if 'cmm_num_layers' in config:
        args.cmm_num_layers = int(config['cmm_num_layers'])
    if config.get('cmm_use_bert_fusion', True):
        args.cmm_use_bert_fusion = True
    if config.get('init_cmm_embeddings', False):
        pass  # handled after model build
    if config.get('use_answer_boost', False):
        args.use_answer_boost = True
    if config.get('answer_boost_train_only', False):
        args.answer_boost_train_only = True
    if 'answer_boost_loss_weight' in config:
        args.answer_boost_loss_weight = float(config['answer_boost_loss_weight'])
    if config.get('use_mvcm_itlc', False):
        args.use_mvcm_itlc = True
    if 'mvcm_itlc_loss_weight' in config:
        args.mvcm_itlc_loss_weight = float(config['mvcm_itlc_loss_weight'])
    if 'mvcm_itc_loss_weight' in config:
        args.mvcm_itc_loss_weight = float(config['mvcm_itc_loss_weight'])
    if config.get('mvcm_boost_train_only', False):
        args.mvcm_boost_train_only = True
        args.use_answer_boost = True
        args.use_mvcm_itlc = True
        args.answer_boost_train_only = True
    if 'text_decoder' in config:
        args.text_decoder = str(config['text_decoder'])
    if 'text_encoder' in config:
        args.text_encoder = str(config['text_encoder'])
    if 'pretrain_source' in config:
        args.pretrain_source = str(config['pretrain_source'])
    if 'vision_encoder' in config:
        args.vision_encoder = str(config['vision_encoder'])
    if config.get('image_base'):
        args.image_base = str(config['image_base'])
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)

