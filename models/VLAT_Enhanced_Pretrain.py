"""
Medical multimodal pretraining for VLAT_Enhanced (MUMC / ALBEF style).

Pretrains vision + text encoders on image-caption pairs (ROCO, MedCAT, ImageCLEF)
using three losses:
  - loss_mlm: masked language modeling with image context
  - loss_ita: image-text + unimodal contrastive (momentum queue)
  - loss_itm: image-text matching

FineGrained module and answer decoder are NOT used during pretraining;
weights are transferred to VLAT_Enhanced at finetune time.
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import BertTokenizer, BertConfig

from models.xbert import BertForMaskedLM
from models.vit import VisionTransformer, interpolate_pos_embed


def build_itm_head(input_dim, output_dim=2):
    return nn.Sequential(
        nn.Linear(input_dim, input_dim * 2),
        nn.LayerNorm(input_dim * 2),
        nn.GELU(),
        nn.Linear(input_dim * 2, output_dim),
    )


@torch.no_grad()
def concat_all_gather(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)


def itm_hard_neg_weights(image_feat, text_feat, temp, bs):
    """Stable hard-negative weights for ITM (guards against NaN/zero rows)."""
    t = temp.clamp(min=0.001)
    sim = torch.matmul(image_feat, text_feat.t()) / t
    sim_i2t = torch.nan_to_num(sim[:, :bs], nan=0.0, posinf=50.0, neginf=-50.0)
    sim_t2i = torch.nan_to_num(sim.t()[:, :bs], nan=0.0, posinf=50.0, neginf=-50.0)

    def _weights(sim_block):
        w = F.softmax(sim_block, dim=1) + 1e-4
        w.fill_diagonal_(0)
        row_sum = w.sum(dim=1, keepdim=True)
        uniform = torch.ones_like(w)
        uniform.fill_diagonal_(0)
        uniform = uniform / uniform.sum(dim=1, keepdim=True).clamp(min=1e-8)
        w = torch.where(row_sum > 1e-8, w / row_sum.clamp(min=1e-8), uniform)
        return torch.nan_to_num(w, nan=1.0 / max(bs, 1))

    return _weights(sim_i2t), _weights(sim_t2i)


class VLAT_Enhanced_Pretrain(nn.Module):
    def __init__(
        self,
        config,
        text_encoder="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        bert_config="./configs/config_bert.json",
        init_deit=True,
    ):
        super().__init__()

        self.image_res = config["image_res"]
        self.vision_width = config["vision_width"]
        self.embed_dim = config["embed_dim"]
        self.queue_size = config["queue_size"]
        self.momentum = config["momentum"]
        self.mlm_probability = config["mlm_probability"]

        self.visual_encoder = VisionTransformer(
            img_size=self.image_res,
            patch_size=16,
            embed_dim=self.vision_width,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        if init_deit:
            self._load_deit_weights()

        bert_cfg = BertConfig.from_json_file(bert_config)
        self.tokenizer = BertTokenizer.from_pretrained(text_encoder)
        self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_cfg)
        self.text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(self.vision_width, self.embed_dim)
        self.text_proj = nn.Linear(self.text_width, self.embed_dim)
        self.temp = nn.Parameter(torch.ones([]) * config["temp"])
        self.itm_head = build_itm_head(self.text_width)

        self.visual_encoder_m = VisionTransformer(
            img_size=self.image_res,
            patch_size=16,
            embed_dim=self.vision_width,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        self.text_encoder_m = BertForMaskedLM.from_pretrained(text_encoder, config=bert_cfg)
        self.vision_proj_m = nn.Linear(self.vision_width, self.embed_dim)
        self.text_proj_m = nn.Linear(self.text_width, self.embed_dim)

        self.model_pairs = [
            [self.visual_encoder, self.visual_encoder_m],
            [self.vision_proj, self.vision_proj_m],
            [self.text_encoder, self.text_encoder_m],
            [self.text_proj, self.text_proj_m],
        ]
        self.copy_params()

        self.register_buffer("image_queue", torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.image_queue = F.normalize(self.image_queue, dim=0)
        self.text_queue = F.normalize(self.text_queue, dim=0)

    def _load_deit_weights(self):
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu",
            check_hash=True,
        )
        state_dict = checkpoint["model"]
        pos_embed_reshaped = interpolate_pos_embed(state_dict["pos_embed"], self.visual_encoder)
        state_dict["pos_embed"] = pos_embed_reshaped
        msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
        print(f"Loaded DeiT weights: {msg}")

    def get_vision_embeds(self, image):
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image.device)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        return image_embeds, image_atts, image_feat

    @torch.no_grad()
    def get_momentum_vision_embeds(self, image):
        image_embeds_m = self.visual_encoder_m(image)
        image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
        return image_embeds_m, image_feat_m

    def get_text_embeds(self, text):
        text_output = self.text_encoder.bert(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode="text",
        )
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        return text_embeds, text_feat

    @torch.no_grad()
    def get_momentum_text_embeds(self, text):
        text_output_m = self.text_encoder_m.bert(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode="text",
        )
        text_feat_m = F.normalize(
            self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1
        )
        return text_feat_m

    def get_contrastive_loss(self, image_feat, text_feat, image_feat_m, text_feat_m, alpha=0.0):
        with torch.no_grad():
            self._momentum_update()
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)
            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp

            sim_targets = torch.zeros(sim_i2t_m.size(), device=image_feat.device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2i = image_feat @ image_feat_all / self.temp
        sim_t2t = text_feat @ text_feat_all / self.temp
        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1) * sim_targets, dim=1).mean()
        loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1) * sim_targets, dim=1).mean()
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
        loss_ita = (loss_i2t + loss_t2i + 0.5 * (loss_i2i + loss_t2t)) / 2

        self._dequeue_and_enqueue(image_feat_m, text_feat_m)
        return loss_ita

    def get_matching_loss(self, image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat):
        bs = image_embeds.size(0)
        output_pos = self.text_encoder.bert(
            encoder_embeds=text_embeds,
            attention_mask=text_atts,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            mode="fusion",
        )

        with torch.no_grad():
            weights_i2t, weights_t2i = itm_hard_neg_weights(
                image_feat, text_feat, self.temp, bs
            )

            image_embeds_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

            text_embeds_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_embeds_neg.append(text_embeds[neg_idx])
                text_atts_neg.append(text_atts[neg_idx])
            text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text_atts, text_atts_neg], dim=0)
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        output_neg = self.text_encoder.bert(
            encoder_embeds=text_embeds_all,
            attention_mask=text_atts_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
            mode="fusion",
        )

        vl_embeddings = torch.cat(
            [output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]], dim=0
        )
        vl_output = self.itm_head(vl_embeddings)
        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)], dim=0
        ).to(image_embeds.device)
        return F.cross_entropy(vl_output, itm_labels)

    def mask(self, input_ids, vocab_size, device, targets=None, probability_matrix=None):
        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        targets[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        indices_random = (
            torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        )
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long, device=device)
        input_ids[indices_random] = random_words[indices_random]
        return input_ids, targets

    def get_mlm_loss(self, text, image_embeds, image_atts, image_embeds_m, alpha=0.0):
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        input_ids, labels = self.mask(
            input_ids,
            self.text_encoder.config.vocab_size,
            image_embeds.device,
            targets=labels,
            probability_matrix=probability_matrix,
        )

        with torch.no_grad():
            logits_m = self.text_encoder_m(
                input_ids,
                attention_mask=text.attention_mask,
                encoder_hidden_states=image_embeds_m,
                encoder_attention_mask=image_atts,
                return_dict=True,
                return_logits=True,
            )

        mlm_output = self.text_encoder(
            input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            labels=labels,
            soft_labels=F.softmax(logits_m, dim=-1),
            alpha=alpha,
        )
        return mlm_output.loss.mean()

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)
                param_m.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        batch_size = image_feats.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0, (
            f"queue_size ({self.queue_size}) must be divisible by batch_size ({batch_size})"
        )
        self.image_queue[:, ptr : ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr : ptr + batch_size] = text_feats.T
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def forward(self, image, text, alpha=0.0):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        image_embeds, image_atts, image_feat = self.get_vision_embeds(image)
        text_embeds, text_feat = self.get_text_embeds(text)
        image_embeds_m, image_feat_m = self.get_momentum_vision_embeds(image)
        text_feat_m = self.get_momentum_text_embeds(text)

        loss_ita = self.get_contrastive_loss(image_feat, text_feat, image_feat_m, text_feat_m, alpha)
        loss_itm = self.get_matching_loss(
            image_embeds, image_atts, image_feat, text_embeds, text.attention_mask, text_feat
        )
        loss_mlm = self.get_mlm_loss(text, image_embeds, image_atts, image_embeds_m, alpha)
        return loss_mlm, loss_ita, loss_itm


def load_pretrain_checkpoint(model, checkpoint_path, resume=False):
    """Load a pretrain checkpoint into VLAT_Enhanced_Pretrain."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    if "visual_encoder.pos_embed" in state_dict:
        state_dict["visual_encoder.pos_embed"] = interpolate_pos_embed(
            state_dict["visual_encoder.pos_embed"], model.visual_encoder
        )
    if "visual_encoder_m.pos_embed" in state_dict:
        state_dict["visual_encoder_m.pos_embed"] = interpolate_pos_embed(
            state_dict["visual_encoder_m.pos_embed"], model.visual_encoder_m
        )

    for key in list(state_dict.keys()):
        if key in model.state_dict() and state_dict[key].shape != model.state_dict()[key].shape:
            del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrain checkpoint from {checkpoint_path}")
    print(f"  missing_keys: {msg.missing_keys}")
    print(f"  unexpected_keys: {msg.unexpected_keys}")
    return checkpoint.get("epoch", -1) if resume else -1


def transfer_pretrain_to_enhanced(enhanced_model, checkpoint_path):
    """
    Transfer encoder weights from a pretrain checkpoint into VLAT_Enhanced.
    Maps visual_encoder.* -> image_encoder.vit_model.* and text_encoder.* -> text_encoder.*
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    mapped = {}

    for key, value in state_dict.items():
        if key.startswith("visual_encoder."):
            mapped[key.replace("visual_encoder.", "image_encoder.vit_model.")] = value
        elif key.startswith("text_encoder."):
            mapped[key] = value

    if "image_encoder.vit_model.pos_embed" in mapped:
        mapped["image_encoder.vit_model.pos_embed"] = interpolate_pos_embed(
            mapped["image_encoder.vit_model.pos_embed"],
            enhanced_model.image_encoder.vit_model,
        )

    # Copy fusion-layer text encoder weights into the answer decoder (same as VLAT.py)
    for key, value in list(mapped.items()):
        if not key.startswith("text_encoder."):
            continue
        if "layer" in key:
            parts = key.split(".")
            layer_num = int(parts[4])
            if layer_num < 6:
                continue
            decoder_layer_num = layer_num - 6
            parts[4] = str(decoder_layer_num)
            decoder_key = ".".join(parts).replace("text_encoder", "decoder")
            mapped[decoder_key] = value
        else:
            mapped[key.replace("text_encoder", "decoder")] = value

    msg = enhanced_model.load_state_dict(mapped, strict=False)
    print(f"Transferred pretrain weights from {checkpoint_path}")
    print(f"  missing_keys: {len(msg.missing_keys)}")
    print(f"  unexpected_keys: {len(msg.unexpected_keys)}")
    return msg


def transfer_clip_vision_to_enhanced(enhanced_model, checkpoint_path):
    """Transfer CLIP vision weights from CLIP-RoBERTa medical pretrain into CLIPImageEncoder."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    mapped = {}
    for key, value in state_dict.items():
        if key.startswith("vision_encoder."):
            mapped[key.replace("vision_encoder.", "image_encoder.vision_encoder.")] = value
    msg = enhanced_model.load_state_dict(mapped, strict=False)
    print(f"Transferred CLIP vision pretrain from {checkpoint_path}")
    print(f"  missing_keys: {len(msg.missing_keys)}")
    print(f"  unexpected_keys: {len(msg.unexpected_keys)}")
    return msg
