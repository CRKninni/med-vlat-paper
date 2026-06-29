"""
VLAT_Enhanced_FG_Pretrain — clean MUMC-style pretraining with FG fully in the loop.

Design rationale
----------------
The old MVCM pretrain had a critical flaw: a separate ``albef`` sub-module ran
MLM/ITA/ITM on *raw* encoder outputs, bypassing FineGrained entirely. FG only
received gradient from the tiny MVCM contrastive losses (which also caused NaN
instability). The result: MLM converged to 5.66 (weak) and the transferred
decoder collapsed during finetune.

This module fixes the design:
  - ONE encoding path:  image_encoder → FineGrained → text_encoder (fusion)
  - ALL four losses flow gradient through FG:
      loss_mlm : masked text predicts via image_fg as cross-attn context → FG ✓
      loss_ita : contrastive on FG-refined CLS features               → FG ✓
      loss_itm : hard-neg matching with FG image/text                 → FG ✓
      loss_dec : decoder teacher-forced on FG-fusion states           → FG + decoder ✓
  - Momentum teacher (no FG in momentum — keeps memory reasonable)
  - ``core.*`` checkpoint keys map 1-to-1 onto VLAT_Enhanced at finetune time
  - ``mvcm_align`` module kept in core (can be activated during finetune)
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import BertConfig, BertTokenizer

from models.VLAT_Enhanced import VLAT_Enhanced
from models.vit import VisionTransformer, interpolate_pos_embed
from models.xbert import BertForMaskedLM

_SIM_CLAMP = 50.0


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers (same as old pretrain)
# ──────────────────────────────────────────────────────────────────────────────

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
    t = temp.clamp(min=0.001)
    sim = torch.matmul(image_feat, text_feat.t()) / t
    sim_i2t = torch.nan_to_num(sim[:, :bs], nan=0.0, posinf=50.0, neginf=-50.0)
    sim_t2i = torch.nan_to_num(sim.t()[:, :bs], nan=0.0, posinf=50.0, neginf=-50.0)

    def _weights(s):
        w = F.softmax(s, dim=1) + 1e-4
        w.fill_diagonal_(0)
        row_sum = w.sum(dim=1, keepdim=True)
        unif = torch.ones_like(w)
        unif.fill_diagonal_(0)
        unif = unif / unif.sum(dim=1, keepdim=True).clamp(min=1e-8)
        w = torch.where(row_sum > 1e-8, w / row_sum.clamp(min=1e-8), unif)
        return torch.nan_to_num(w, nan=1.0 / max(bs, 1))

    return _weights(sim_i2t), _weights(sim_t2i)


# ──────────────────────────────────────────────────────────────────────────────
# Main pretrain model
# ──────────────────────────────────────────────────────────────────────────────

class VLAT_Enhanced_FG_Pretrain(nn.Module):
    """
    Full VLAT_Enhanced (image_encoder + FineGrained + text_encoder + decoder)
    pretrained end-to-end with MLM + ITA + ITM + decoder-LM losses — all
    routed through FineGrained so every loss trains FG.

    Checkpoint layout (saved under ``self.core.*``):
        core.image_encoder.*   ← DeiT ViT
        core.text_encoder.*    ← PubMedBERT (12 layers, with MLM head)
        core.fine_grained.*    ← 6-layer FG (bidirectional, adaptive gating)
        core.mvcm_align.*      ← MVCM alignment projections (for finetune use)
        core.decoder.*         ← BertLMHeadModel (explicitly trained here!)

    At finetune: strip ``core.`` prefix → loads directly into VLAT_Enhanced.
    """

    def __init__(
        self,
        config,
        attention_mode="flash",
        fg_layers=6,
        head_schedule=None,
        text_encoder="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        bert_config="./configs/config_bert.json",
    ):
        super().__init__()

        self.embed_dim    = config["embed_dim"]
        self.vision_width = config["vision_width"]
        self.queue_size   = config["queue_size"]
        self.momentum     = config["momentum"]
        self.mlm_prob     = config["mlm_probability"]

        # ── core VLAT_Enhanced (everything that transfers to finetune) ────────
        self.core = VLAT_Enhanced(
            embed_dim=self.vision_width,
            fg_layers=fg_layers,
            head_schedule=head_schedule or [12, 12, 8, 6, 4, 4],
            attention_mode=attention_mode,
            use_bidirectional=True,
            use_adaptive_gating=True,
            use_mvcm_itlc=True,          # module kept in arch; not used in loss here
            mvcm_itlc_loss_weight=0.0,
            mvcm_itc_loss_weight=0.0,
            mvcm_boost_train_only=False,
            num_answer_classes=0,
            image_size=config["image_res"],
            config_file=bert_config,
            encoder_base=text_encoder,
            decoder_base=text_encoder,
            dropout=config.get("dropout", 0.1),
        )
        self.tokenizer = self.core.tokenizer

        # ── ITA projection heads (embed_dim = 256) ────────────────────────────
        self.vision_proj   = nn.Linear(self.vision_width, self.embed_dim)
        self.text_proj     = nn.Linear(self.vision_width, self.embed_dim)
        self.temp          = nn.Parameter(torch.ones([]) * config["temp"])
        self.itm_head      = build_itm_head(self.vision_width)

        # ── Momentum encoders (raw ViT + raw BERT, no FG) ─────────────────────
        bert_cfg = BertConfig.from_json_file(bert_config)
        self.visual_encoder_m = VisionTransformer(
            img_size=config["image_res"], patch_size=16,
            embed_dim=self.vision_width, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        self.text_encoder_m = BertForMaskedLM.from_pretrained(text_encoder, config=bert_cfg)
        self.vision_proj_m  = nn.Linear(self.vision_width, self.embed_dim)
        self.text_proj_m    = nn.Linear(self.vision_width, self.embed_dim)

        self.model_pairs = [
            [self.core.image_encoder.vit_model, self.visual_encoder_m],
            [self.vision_proj,                  self.vision_proj_m],
            [self.core.text_encoder,            self.text_encoder_m],
            [self.text_proj,                    self.text_proj_m],
        ]
        self.copy_params()

        # ── Contrastive queue ─────────────────────────────────────────────────
        self.register_buffer("image_queue", torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer("text_queue",  torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer("queue_ptr",   torch.zeros(1, dtype=torch.long))
        self.image_queue = F.normalize(self.image_queue, dim=0)
        self.text_queue  = F.normalize(self.text_queue,  dim=0)

    # ── Momentum helpers ──────────────────────────────────────────────────────

    @torch.no_grad()
    def copy_params(self):
        for src, tgt in self.model_pairs:
            for p, pm in zip(src.parameters(), tgt.parameters()):
                pm.data.copy_(p.data)
                pm.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        for src, tgt in self.model_pairs:
            for p, pm in zip(src.parameters(), tgt.parameters()):
                pm.data = pm.data * self.momentum + p.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat_m, text_feat_m):
        img_feats  = concat_all_gather(image_feat_m)
        txt_feats  = concat_all_gather(text_feat_m)
        batch_size = img_feats.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0, (
            f"queue_size ({self.queue_size}) must be divisible by batch_size ({batch_size})"
        )
        self.image_queue[:, ptr:ptr + batch_size] = img_feats.T
        self.text_queue[:, ptr:ptr + batch_size]  = txt_feats.T
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    # ── MLM token masking ─────────────────────────────────────────────────────

    def _mask_tokens(self, input_ids, vocab_size, device, targets, p_mask):
        masked_idx = torch.bernoulli(p_mask).bool()
        masked_idx[input_ids == self.tokenizer.pad_token_id] = False
        masked_idx[input_ids == self.tokenizer.cls_token_id] = False
        targets[~masked_idx] = -100

        replace_idx = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_idx
        input_ids[replace_idx] = self.tokenizer.mask_token_id

        random_idx = (
            torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool()
            & masked_idx & ~replace_idx
        )
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long, device=device)
        input_ids[random_idx] = random_words[random_idx]
        return input_ids, targets

    # ── Individual losses ─────────────────────────────────────────────────────

    def get_contrastive_loss(self, image_feat, text_feat, image_feat_m, text_feat_m, alpha):
        """ITA + unimodal contrastive with momentum queue (MUMC-style)."""
        with torch.no_grad():
            self._momentum_update()
            img_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)
            txt_feat_all = torch.cat([text_feat_m.t(),  self.text_queue.clone().detach()],  dim=1)

            sim_i2t_m = (image_feat_m @ txt_feat_all).clamp(-_SIM_CLAMP, _SIM_CLAMP) / self.temp
            sim_t2i_m = (text_feat_m  @ img_feat_all).clamp(-_SIM_CLAMP, _SIM_CLAMP) / self.temp

            targets = torch.zeros(sim_i2t_m.size(), device=image_feat.device)
            targets.fill_diagonal_(1)
            sim_i2t_tgt = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * targets
            sim_t2i_tgt = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * targets

        sim_i2i = (image_feat @ img_feat_all).clamp(-_SIM_CLAMP, _SIM_CLAMP) / self.temp
        sim_t2t = (text_feat  @ txt_feat_all).clamp(-_SIM_CLAMP, _SIM_CLAMP) / self.temp
        sim_i2t = (image_feat @ txt_feat_all).clamp(-_SIM_CLAMP, _SIM_CLAMP) / self.temp
        sim_t2i = (text_feat  @ img_feat_all).clamp(-_SIM_CLAMP, _SIM_CLAMP) / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_tgt, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_tgt, dim=1).mean()
        loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1) * targets,     dim=1).mean()
        loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1) * targets,     dim=1).mean()
        loss_ita = (loss_i2t + loss_t2i + 0.5 * (loss_i2i + loss_t2t)) / 2

        self._dequeue_and_enqueue(image_feat_m, text_feat_m)
        return loss_ita

    def get_matching_loss(self, image_fg, image_fg_atts, image_feat,
                          text_fg, text_atts, text_feat):
        """ITM with FG-refined image and text features."""
        bs = image_fg.size(0)

        output_pos = self.core.text_encoder.bert(
            encoder_embeds=text_fg,
            attention_mask=text_atts,
            encoder_hidden_states=image_fg,
            encoder_attention_mask=image_fg_atts,
            return_dict=True,
            mode="fusion",
        )

        with torch.no_grad():
            w_i2t, w_t2i = itm_hard_neg_weights(image_feat, text_feat, self.temp, bs)

            img_neg = torch.stack([image_fg[torch.multinomial(w_t2i[b], 1).item()]
                                   for b in range(bs)])
            txt_neg  = torch.stack([text_fg[torch.multinomial(w_i2t[b], 1).item()]
                                    for b in range(bs)])
            att_neg  = torch.stack([text_atts[torch.multinomial(w_i2t[b], 1).item()]
                                    for b in range(bs)])

        txt_all = torch.cat([text_fg, txt_neg], dim=0)
        att_all = torch.cat([text_atts, att_neg], dim=0)
        img_all = torch.cat([img_neg, image_fg], dim=0)
        iatt_all = torch.cat([image_fg_atts, image_fg_atts], dim=0)

        output_neg = self.core.text_encoder.bert(
            encoder_embeds=txt_all,
            attention_mask=att_all,
            encoder_hidden_states=img_all,
            encoder_attention_mask=iatt_all,
            return_dict=True,
            mode="fusion",
        )

        vl_emb = torch.cat([
            output_pos.last_hidden_state[:, 0, :],
            output_neg.last_hidden_state[:, 0, :],
        ], dim=0)
        vl_out = self.itm_head(vl_emb)
        labels = torch.cat([
            torch.ones(bs, dtype=torch.long),
            torch.zeros(2 * bs, dtype=torch.long),
        ]).to(image_fg.device)
        return F.cross_entropy(vl_out, labels)

    def get_mlm_loss(self, text, image_fg, image_fg_atts, image_embeds_m, alpha):
        """MLM with FG-refined image as cross-attention context."""
        input_ids = text.input_ids.clone()
        labels    = input_ids.clone()
        p_mask = torch.full(labels.shape, self.mlm_prob, device=input_ids.device)
        input_ids, labels = self._mask_tokens(
            input_ids, self.core.text_encoder.config.vocab_size,
            image_fg.device, labels, p_mask,
        )

        # Momentum teacher soft labels (raw image, no FG — acceptable slight mismatch)
        image_m_atts = torch.ones(image_embeds_m.size()[:-1],
                                   dtype=torch.long, device=image_embeds_m.device)
        with torch.no_grad():
            logits_m = self.text_encoder_m(
                input_ids,
                attention_mask=text.attention_mask,
                encoder_hidden_states=image_embeds_m,
                encoder_attention_mask=image_m_atts,
                return_dict=True,
                return_logits=True,
            )

        mlm_out = self.core.text_encoder(
            input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_fg,       # ← FG-refined image
            encoder_attention_mask=image_fg_atts,
            return_dict=True,
            labels=labels,
            soft_labels=F.softmax(logits_m, dim=-1),
            alpha=alpha,
        )
        return mlm_out.loss.mean()

    def get_decoder_lm_loss(self, question_states, question_atts, text):
        """Teacher-forced decoder LM on FG-fusion states. Trains core.decoder."""
        decoder_targets = text.input_ids.masked_fill(text.attention_mask == 0, -100)
        dec_out = self.core.decoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
            labels=decoder_targets,
            return_dict=True,
            reduction="mean",
        )
        return dec_out.loss

    # ── Full forward ──────────────────────────────────────────────────────────

    def forward(self, image, text, alpha=0.0):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        # ── Step 1: Raw encode both modalities ────────────────────────────────
        image_embeds = self.core.image_encoder(image)                         # [B,577,768]
        image_atts   = torch.ones(image_embeds.size()[:-1],
                                   dtype=torch.long, device=image.device)
        text_out = self.core.text_encoder.bert(
            text.input_ids, attention_mask=text.attention_mask,
            return_dict=True, mode="text",
        )
        text_embeds = text_out.last_hidden_state                              # [B,L,768]

        # ── Step 2: FineGrained refinement — FG gets gradient from ALL losses ─
        image_fg, text_fg = self.core.fine_grained(image_embeds, text_embeds)
        image_fg_atts = image_atts                                            # same shape

        # ── Step 3: FG CLS projections for ITA ───────────────────────────────
        image_feat = F.normalize(self.vision_proj(image_fg[:, 0, :]), dim=-1)
        text_feat  = F.normalize(self.text_proj(text_fg[:, 0, :]),  dim=-1)

        # ── Step 4: Momentum features (raw, no FG) ───────────────────────────
        with torch.no_grad():
            image_embeds_m = self.visual_encoder_m(image)
            image_feat_m   = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
            text_out_m = self.text_encoder_m.bert(
                text.input_ids, attention_mask=text.attention_mask,
                return_dict=True, mode="text",
            )
            text_feat_m = F.normalize(
                self.text_proj_m(text_out_m.last_hidden_state[:, 0, :]), dim=-1
            )

        # ── Step 5: ITA (contrastive on FG features) ─────────────────────────
        loss_ita = self.get_contrastive_loss(
            image_feat, text_feat, image_feat_m, text_feat_m, alpha
        )

        # ── Step 6: ITM (FG image as cross-attn context) ─────────────────────
        loss_itm = self.get_matching_loss(
            image_fg, image_fg_atts, image_feat,
            text_fg, text.attention_mask, text_feat,
        )

        # ── Step 7: MLM (FG image as cross-attn context) ─────────────────────
        loss_mlm = self.get_mlm_loss(text, image_fg, image_fg_atts, image_embeds_m, alpha)

        # ── Step 8: Decoder LM (FG-fusion states → caption generation) ───────
        fusion_out = self.core.text_encoder.bert(
            encoder_embeds=text_fg,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_fg,
            encoder_attention_mask=image_fg_atts,
            return_dict=True,
            mode="fusion",
        )
        loss_dec = self.get_decoder_lm_loss(
            fusion_out.last_hidden_state, text.attention_mask, text,
        )

        return loss_mlm, loss_ita, loss_itm, loss_dec


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint utilities
# ──────────────────────────────────────────────────────────────────────────────

def load_vlat_fg_pretrain_checkpoint(model, checkpoint_path):
    """Resume a VLAT_Enhanced_FG_Pretrain checkpoint (full state_dict)."""
    ck = torch.load(checkpoint_path, map_location="cpu")
    sd = ck["model"] if "model" in ck else ck

    vit = model.core.image_encoder.vit_model
    for pe_key in ("core.image_encoder.vit_model.pos_embed",
                   "visual_encoder_m.pos_embed"):
        if pe_key in sd:
            sd = dict(sd)
            sd[pe_key] = interpolate_pos_embed(sd[pe_key], vit)

    model_sd = model.state_dict()
    filtered = {k: v for k, v in sd.items()
                if k in model_sd and model_sd[k].shape == v.shape}
    msg = model.load_state_dict(filtered, strict=False)
    print(f"Resumed VLAT_Enhanced_FG_Pretrain from {checkpoint_path}")
    print(f"  missing_keys:    {len(msg.missing_keys)}")
    print(f"  unexpected_keys: {len(msg.unexpected_keys)}")
    return ck.get("epoch", -1)


def warmstart_from_enhanced_pretrain(model, checkpoint_path):
    """
    Warm-start core encoders + decoder from old encoder-only pretrain (clef_29).
    Uses transfer_pretrain_to_enhanced which maps text_encoder layers 6-11
    → decoder layers 0-5, giving the decoder a strong PubMedBERT initialization.
    Also seeds the momentum encoder copies.
    """
    from models.VLAT_Enhanced_Pretrain import transfer_pretrain_to_enhanced
    transfer_pretrain_to_enhanced(model.core, checkpoint_path)
    model.copy_params()   # sync momentum copies from newly loaded core params
    print(f"Warm-started from {checkpoint_path} (encoder + decoder seeded, momentum synced)")


def transfer_vlat_fg_pretrain_to_enhanced(enhanced_model, checkpoint_path):
    """
    Transfer core.* weights into VLAT_Enhanced at finetune time.

    Since core.decoder was explicitly trained here, we load it directly
    (no layer-remapping needed). All core.* keys strip the prefix and
    map 1-to-1 onto VLAT_Enhanced's state dict.
    """
    ck = torch.load(checkpoint_path, map_location="cpu")
    sd = ck["model"] if "model" in ck else ck

    mapped = {}
    for k, v in sd.items():
        if k.startswith("core."):
            mapped[k[5:]] = v     # strip "core." prefix

    pe_key = "image_encoder.vit_model.pos_embed"
    if pe_key in mapped:
        mapped[pe_key] = interpolate_pos_embed(
            mapped[pe_key], enhanced_model.image_encoder.vit_model
        )

    msg = enhanced_model.load_state_dict(mapped, strict=False)
    print(f"Transferred VLAT_Enhanced_FG_Pretrain weights from {checkpoint_path}")
    print(f"  mapped keys:     {len(mapped)}")
    print(f"  missing_keys:    {len(msg.missing_keys)}")
    print(f"  unexpected_keys: {len(msg.unexpected_keys)}")
    return msg
