"""
Enhanced VLAT Architecture with Multiple Improvements

Improvements over MAIN_VLAT.py:
1. ✅ Skip connections (residual connections) in FineGrained
2. ✅ Layer normalization between SA and GA
3. ✅ Hierarchical attention (variable heads per layer)
4. ✅ Adaptive gating (learnable layer importance)
5. ✅ Bidirectional refinement (optional)
6. ✅ Efficient attention options (linear/sparse/flash)

Usage:
    # Basic enhanced model
    model = VLAT_Enhanced()
    
    # With all features enabled
    model = VLAT_Enhanced(
        use_bidirectional=True,
        use_adaptive_gating=True,
        attention_mode='linear'
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from transformers import BertConfig, BertTokenizer, CLIPVisionModel
import timm

# Import custom BERT models that support 'mode' parameter
from models.xbert import BertForMaskedLM, BertLMHeadModel
from models.dsap_prompts import SemanticAlignmentPrompt, DynamicQuestionAwarePrompt
from models.rad_expert_heads import (
    ITMYesNoHead, LateralityExpert, ModalityExpert,
    LATERALITY_LABELS, MODALITY_LABELS,
    question_has_laterality, question_has_modality, question_negation_bias,
    build_laterality_answer_map, build_modality_answer_map, pick_best_from_candidates,
)
from models.vit import VisionTransformer, interpolate_pos_embed
from models.answer_query_head import (
    AnswerQueryHead, FGFusionAnswerQueryHead,
    asymmetric_loss, focal_cross_entropy,
)
from models.cross_modal_matching import CrossModalMatchingHead
from models.rad_open_synonyms import build_synonym_index, apply_synonym_boost
from models.answer_boost import AnswerBoostStack
from models.mvcm_itlc import MVCMAlignmentModule

torch.cuda.empty_cache()

try:
    from flash_attn import flash_attn_func
    _HAS_FLASH_ATTN = True
except ImportError:
    flash_attn_func = None
    _HAS_FLASH_ATTN = False


def _run_flash_attention(q, k, v, dropout_p=0.0, causal=False):
    """Flash attention: q/k/v shaped (B, num_heads, seq_len, head_dim)."""
    if not _HAS_FLASH_ATTN:
        raise RuntimeError("flash_attn is not installed")
    orig_dtype = q.dtype
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    if orig_dtype not in (torch.float16, torch.bfloat16):
        q, k, v = q.half(), k.half(), v.half()
    out = flash_attn_func(
        q, k, v, dropout_p=dropout_p, softmax_scale=None, causal=causal,
    )
    if out.dtype != orig_dtype:
        out = out.to(orig_dtype)
    return out.transpose(1, 2)


def tile(x, dim, n_tile):
    """Tile tensor along dimension (same as original VLAT)"""
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))


class ImprovedSelfAttention(nn.Module):
    """
    Self-Attention with configurable heads and attention mode
    """
    def __init__(self, embed_dim, num_heads, attention_mode='standard', dropout=0.1):
        super(ImprovedSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attention_mode = attention_mode
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # For linear attention
        if attention_mode == 'linear':
            self.feature_map = nn.ELU()
    
    def standard_attention(self, q, k, v, mask=None):
        """Standard scaled dot-product attention O(N²)"""
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        return out
    
    def linear_attention(self, q, k, v, mask=None):
        """Linear attention O(N) using kernel trick"""
        # Apply feature map: elu(x) + 1
        q = self.feature_map(q) + 1
        k = self.feature_map(k) + 1
        
        # Linear attention: Q(K^T V) instead of (QK^T)V
        kv = k.transpose(-2, -1) @ v  # (B, H, D, D)
        normalizer = q @ k.sum(dim=-2, keepdim=True).transpose(-2, -1)  # (B, H, N, 1)
        normalizer = normalizer.clamp(min=1e-6)
        out = (q @ kv) / normalizer
        return out
    
    def sparse_attention(self, q, k, v, mask=None, top_k=32):
        """Sparse attention: only attend to top-k tokens"""
        # Compute similarity scores
        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)  # (B, H, N, N)
        
        # Keep only top-k per query
        topk_vals, topk_indices = torch.topk(scores, k=min(top_k, scores.size(-1)), dim=-1)
        
        # Create sparse attention matrix
        attn = torch.zeros_like(scores).scatter_(-1, topk_indices, F.softmax(topk_vals, dim=-1))
        attn = self.dropout(attn)
        out = attn @ v
        return out

    def flash_attention(self, q, k, v):
        """Flash attention O(N²) with IO-efficient CUDA kernel (same math as standard)."""
        dropout_p = self.dropout.p if self.training else 0.0
        return _run_flash_attention(q, k, v, dropout_p=dropout_p, causal=False)
    
    def forward(self, x, mask=None):
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, N, head_dim)
        
        # Apply attention based on mode
        if self.attention_mode == 'linear':
            out = self.linear_attention(q, k, v, mask)
        elif self.attention_mode == 'sparse':
            out = self.sparse_attention(q, k, v, mask, top_k=32)
        elif self.attention_mode == 'flash':
            if mask is not None or not _HAS_FLASH_ATTN:
                out = self.standard_attention(q, k, v, mask)
            else:
                out = self.flash_attention(q, k, v)
        else:  # standard
            out = self.standard_attention(q, k, v, mask)
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.dropout(out)
        
        return out


class ImprovedGuidedAttention(nn.Module):
    """
    Cross-Attention with configurable heads and attention mode
    """
    def __init__(self, embed_dim, num_heads, attention_mode='standard', dropout=0.1):
        super(ImprovedGuidedAttention, self).__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attention_mode = attention_mode
        
        self.q = nn.Linear(embed_dim, embed_dim)
        self.kv = nn.Linear(embed_dim, embed_dim * 2)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        if attention_mode == 'linear':
            self.feature_map = nn.ELU()
    
    def forward(self, query_emb, key_value_emb):
        B, N_q, C = query_emb.shape
        N_kv = key_value_emb.shape[1]
        
        # Q from query, K,V from key_value
        q = self.q(query_emb).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(key_value_emb).reshape(B, N_kv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        # Compute attention
        if self.attention_mode == 'linear':
            q = self.feature_map(q) + 1
            k = self.feature_map(k) + 1
            kv = k.transpose(-2, -1) @ v
            normalizer = q @ k.sum(dim=-2, keepdim=True).transpose(-2, -1)
            normalizer = normalizer.clamp(min=1e-6)
            out = (q @ kv) / normalizer
        elif self.attention_mode == 'sparse':
            scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
            topk_vals, topk_indices = torch.topk(scores, k=min(32, scores.size(-1)), dim=-1)
            attn = torch.zeros_like(scores).scatter_(-1, topk_indices, F.softmax(topk_vals, dim=-1))
            attn = self.dropout(attn)
            out = attn @ v
        elif self.attention_mode == 'flash' and _HAS_FLASH_ATTN:
            dropout_p = self.dropout.p if self.training else 0.0
            out = _run_flash_attention(q, k, v, dropout_p=dropout_p, causal=False)
        else:  # standard (or flash fallback)
            attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = attn @ v
        
        out = out.transpose(1, 2).reshape(B, N_q, C)
        out = self.proj(out)
        out = self.dropout(out)
        
        return out


class AdaptiveGate(nn.Module):
    """
    Learnable gate to determine layer importance
    Gate value ∈ [0, 1]: 0 = skip layer, 1 = use fully
    """
    def __init__(self, embed_dim):
        super(AdaptiveGate, self).__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Gate is computed from mean pooling of tokens
        pooled = x.mean(dim=1, keepdim=True)  # (B, 1, C)
        gate = self.gate_network(pooled)  # (B, 1, 1)
        return gate


def _make_prefusion_activation(name):
    if name in (None, 'none'):
        return None
    if name == 'relu':
        return nn.ReLU(inplace=True)
    if name == 'leaky_relu':
        return nn.LeakyReLU(0.1, inplace=True)
    if name == 'gelu':
        return nn.GELU()
    raise ValueError(f"Unknown prefusion MLP activation: {name}")


class ClosedYesNoHead(nn.Module):
    """Binary yes/no classifier on fused question+image+text-FG features."""

    def __init__(self, embed_dim, hidden_dim=0, dropout=0.1, use_text_fg=True, compact=False):
        super().__init__()
        self.use_text_fg = use_text_fg
        in_dim = embed_dim * (3 if use_text_fg else 2)
        hidden_dim = hidden_dim or embed_dim
        if compact:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 2),
            )

    @staticmethod
    def pool_features(question_states, question_atts, image_fg, text_fg=None):
        q_mask = question_atts.unsqueeze(-1).float()
        q = (question_states * q_mask).sum(dim=1) / q_mask.sum(dim=1).clamp(min=1.0)
        i = image_fg.mean(dim=1)
        if text_fg is not None:
            t = text_fg.mean(dim=1)
            return torch.cat([q, t, i], dim=-1)
        return torch.cat([q, i], dim=-1)

    def forward(self, question_states, question_atts, image_fg, text_fg=None):
        feats = self.pool_features(
            question_states, question_atts, image_fg,
            text_fg if self.use_text_fg else None,
        )
        return self.net(feats)


class OpenAnswerHead(nn.Module):
    """Lightweight 414-way classifier for OPEN questions on fused features."""

    def __init__(self, num_classes, embed_dim, hidden_dim=0, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, question_states, question_atts, image_fg):
        fused = ClosedYesNoHead.pool_features(question_states, question_atts, image_fg)
        return self.net(fused)


class PreFusionMLP(nn.Module):
    """BERT-style FFN on token embeddings before FineGrained / cross-modal fusion."""

    def __init__(self, embed_dim, ffn_dim=None, activation='gelu', dropout=0.1):
        super().__init__()
        ffn_dim = ffn_dim or embed_dim * 4
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.act = _make_prefusion_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        if self.act is not None:
            x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.norm(residual + x)


class FineGrainedGlobalFeature(nn.Module):
    """
    FineGrained-GlobalFeature (FGGF) module — the core cross-modal interaction block of MED-VLAT.

    Architecture (two integrated components):
    ─────────────────────────────────────────
    FG  (Fine-Grained):  Unidirectional text-guided image refinement.
                         Image patches attend to question/caption tokens to produce
                         stable, locally-grounded anatomical representations.
                         Operates via ImprovedSelfAttention + ImprovedGuidedAttention
                         (image ← text) with adaptive gating per layer.

    GF  (Global Feature): Bidirectional multimodal reasoning (use_bidirectional=True).
                           After local grounding, text tokens attend back to the
                           refined image — enabling global semantic and clinical
                           understanding across both modalities.

    Design details:
    - Residual connections after every attention operation
    - LayerNorm after every residual block
    - Hierarchical attention scheduling: head_schedule progressively shifts from
      many fine-grained heads (early layers, broad coverage) to fewer wide heads
      (later layers, abstract semantics): default [12, 12, 8, 6, 4, 4]
    - Adaptive gating: learned per-sample sigmoid gate controls how much
      cross-modal signal each layer injects
    - Phase 1: text self-refinement over all layers (intra-modal)
    - Phase 2: interleaved image SA → image←text → (if bidir) text SA → text←image
    """
    def __init__(self, embed_dim=768, num_layers=6, head_schedule=None, 
                 attention_mode='standard', use_bidirectional=False, 
                 use_adaptive_gating=False, dropout=0.1):
        super(FineGrainedGlobalFeature, self).__init__()
        
        self.num_layers = num_layers
        self.use_bidirectional = use_bidirectional
        self.use_adaptive_gating = use_adaptive_gating
        
        # Default hierarchical head schedule: 12 → 12 → 8 → 6 → 4 → 4
        # Note: All head counts must be divisors of embed_dim (768)
        # Valid divisors: 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 768
        if head_schedule is None:
            head_schedule = [12, 12, 8, 6, 4, 4][:num_layers]
        else:
            assert len(head_schedule) == num_layers, f"head_schedule length {len(head_schedule)} != num_layers {num_layers}"
            # Validate all heads are divisors of embed_dim
            for i, heads in enumerate(head_schedule):
                assert embed_dim % heads == 0, f"Layer {i}: embed_dim {embed_dim} must be divisible by num_heads {heads}"
        
        self.head_schedule = head_schedule
        
        # Text self-attention layers (for initial text refinement)
        self.text_sa_initial = nn.ModuleList([
            ImprovedSelfAttention(embed_dim, head_schedule[i], attention_mode, dropout)
            for i in range(num_layers)
        ])
        self.text_ln_initial = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        
        # Image processing layers
        self.image_sa_layers = nn.ModuleList([
            ImprovedSelfAttention(embed_dim, head_schedule[i], attention_mode, dropout)
            for i in range(num_layers)
        ])
        self.image_ga_layers = nn.ModuleList([
            ImprovedGuidedAttention(embed_dim, head_schedule[i], attention_mode, dropout)
            for i in range(num_layers)
        ])
        
        # Layer norms (after each operation)
        self.image_ln_sa = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.image_ln_ga = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        
        # Bidirectional refinement: Text attends to Image
        if use_bidirectional:
            self.text_sa_layers = nn.ModuleList([
                ImprovedSelfAttention(embed_dim, head_schedule[i], attention_mode, dropout)
                for i in range(num_layers)
            ])
            self.text_ga_layers = nn.ModuleList([
                ImprovedGuidedAttention(embed_dim, head_schedule[i], attention_mode, dropout)
                for i in range(num_layers)
            ])
            self.text_ln_sa = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
            self.text_ln_ga = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        
        # Adaptive gating
        if use_adaptive_gating:
            self.image_gates = nn.ModuleList([AdaptiveGate(embed_dim) for _ in range(num_layers)])
            if use_bidirectional:
                self.text_gates = nn.ModuleList([AdaptiveGate(embed_dim) for _ in range(num_layers)])
    
    def forward(self, image_embeddings, text_embeddings, image_mask=None):
        # Initial text refinement with residual connections
        text_embeds = text_embeddings
        for i in range(self.num_layers):
            # Text self-attention with residual
            text_embeds = text_embeds + self.text_sa_initial[i](text_embeds)
            text_embeds = self.text_ln_initial[i](text_embeds)
        
        # Store initial refined text
        text_embeds_refined = text_embeds
        
        # Iterative image-text refinement
        image_embeds = image_embeddings
        
        for i in range(self.num_layers):
            # === Image Processing ===
            # Self-attention with residual
            image_sa_out = self.image_sa_layers[i](image_embeds, image_mask)
            image_embeds = image_embeds + image_sa_out
            image_embeds = self.image_ln_sa[i](image_embeds)
            
            # Guided attention (image attends to text) with residual
            image_ga_out = self.image_ga_layers[i](image_embeds, text_embeds_refined)
            
            # Apply adaptive gating if enabled
            if self.use_adaptive_gating:
                gate = self.image_gates[i](image_embeds)
                image_ga_out = gate * image_ga_out
            
            image_embeds = image_embeds + image_ga_out
            image_embeds = self.image_ln_ga[i](image_embeds)
            
            # === Bidirectional: Text Processing (optional) ===
            if self.use_bidirectional:
                # Text self-attention with residual
                text_sa_out = self.text_sa_layers[i](text_embeds_refined)
                text_embeds_refined = text_embeds_refined + text_sa_out
                text_embeds_refined = self.text_ln_sa[i](text_embeds_refined)
                    
                # Text attends to refined image
                text_ga_out = self.text_ga_layers[i](text_embeds_refined, image_embeds)
                
                if self.use_adaptive_gating:
                    gate = self.text_gates[i](text_embeds_refined)
                    text_ga_out = gate * text_ga_out
                
                text_embeds_refined = text_embeds_refined + text_ga_out
                text_embeds_refined = self.text_ln_ga[i](text_embeds_refined)
        
        return image_embeds, text_embeds_refined


class ImageEncoder(nn.Module):
    """DeiT ViT image encoder (ImageNet init)."""
    def __init__(self, image_size=224, output_dim=768, init_deit=True):
        super(ImageEncoder, self).__init__()
        self.vit_model = VisionTransformer(
            img_size=image_size, patch_size=16, embed_dim=output_dim, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        if init_deit:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.vit_model)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.vit_model.load_state_dict(state_dict, strict=False) 
            print(msg)

    def forward(self, images):
        image_embeds = self.vit_model(images) 
        return image_embeds


class BiomedCLIPImageEncoder(nn.Module):
    """BiomedCLIP ViT (PMC-15M medical pretrain) via timm."""
    def __init__(self, image_base="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"):
        super().__init__()
        self.image_base = image_base
        self.vit_model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            pretrained_cfg_overlay=dict(
                hf_hub_id=image_base,
                hf_hub_filename='open_clip_pytorch_model.bin',
            ),
            num_classes=0,
        )

    def forward(self, images):
        return self.vit_model.forward_features(images)


class CLIPImageEncoder(nn.Module):
    """CLIP-ViT vision backbone (e.g. openai/clip-vit-base-patch16)."""
    def __init__(self, image_base="openai/clip-vit-base-patch16"):
        super().__init__()
        self.image_base = image_base
        self.vision_encoder = CLIPVisionModel.from_pretrained(image_base)

    def forward(self, images):
        return self.vision_encoder(pixel_values=images).last_hidden_state


class VLAT_Enhanced(nn.Module):
    """
    Enhanced VLAT with multiple architectural improvements
    
    Args:
        embed_dim: Embedding dimension (default: 768)
        fg_layers: Number of FineGrained layers (default: 6)
        head_schedule: List of head counts per layer (default: [12,10,8,6,4,4])
        attention_mode: 'standard', 'linear', 'sparse', or 'flash' (default: 'standard')
        use_bidirectional: Enable bidirectional refinement (default: False)
        use_adaptive_gating: Enable adaptive layer gating (default: False)
        config_file: Path to BERT config
        decoder_base: Pretrained decoder model name
    """
    def __init__(self, embed_dim=768, fg_layers=6, head_schedule=None,
                 attention_mode='standard', use_bidirectional=False, 
                 use_adaptive_gating=False, use_answer_query=False,
                 num_answer_classes=0, answer_query_layers=2,
                 answer_query_weight=0.5, answer_query_ffn_dim=0,
                 use_asymmetric_query_loss=False,
                 answer_query_at_inference=False,
                 answer_infer_mode='decoder', answer_query_ensemble_weight=0.3,
                 answer_query_rerank_k=32, answer_init_prefix='answer: ',
                 answer_query_weight_open=1.0, answer_query_weight_closed=0.25,
                 answer_query_confidence_threshold=0.55,
                 use_fg_query_fusion=False,
                 use_fg_concat_query=False,
                 query_cosine_weight=0.5,
                 use_bert_fusion_stream=True,
                 hybrid_open_only=True,
                 hybrid_open_query_only=False,
                 open_decoder_loss_weight=0.05,
                 use_answer_type_routing=True,
                 query_only_train=False,
                 query_train_open_only=True,
                 use_closed_yn_head=False,
                 closed_yn_loss_weight=0.3,
                 closed_yn_infer_weight=0.5,
                 closed_yn_train_only=False,
                 closed_yn_hidden_dim=0,
                 use_open_answer_head=False,
                 open_answer_loss_weight=0.3,
                 open_answer_infer_weight=0.3,
                 open_answer_train_only=False,
                 open_answer_hidden_dim=0,
                 use_dsap=False,
                 dsap_sap_image_prompts=8,
                 dsap_sap_text_prompts=8,
                 dsap_dqap_prompts=8,
                 dsap_train_prompts_only=False,
                 use_itm_yn_head=False,
                 itm_yn_loss_weight=1.0,
                 itm_yn_infer_weight=0.55,
                 use_laterality_expert=False,
                 laterality_loss_weight=1.0,
                 laterality_infer_weight=0.6,
                 laterality_confidence=0.45,
                 use_modality_expert=False,
                 modality_loss_weight=1.0,
                 modality_infer_weight=0.55,
                 modality_confidence=0.40,
                 e3_experts_train_only=False,
                 e3_infer_enabled=True,
                 e3_new_experts_only=False,
                 closed_yn_compact=False,
                 closed_yn_focal_gamma=2.0,
                 closed_yn_exclusive_infer=False,
                 closed_expert_train_only=False,
                 use_cmm_head=False,
                 cmm_loss_weight=1.0,
                 cmm_infer_weight=0.35,
                 cmm_train_only=False,
                 cmm_num_layers=1,
                 cmm_use_bert_fusion=True,
                 cmm_open_only=True,
                 cmm_confidence_threshold=0.12,
                 use_open_synonym_boost=False,
                 rad_open_stack_train_only=False,
                 use_answer_boost=False,
                 answer_boost_train_only=False,
                 answer_boost_loss_weight=1.0,
                 answer_boost_closed_only=False,
                 use_mvcm_itlc=False,
                 mvcm_itlc_loss_weight=0.5,
                 mvcm_itc_loss_weight=0.25,
                 mvcm_boost_train_only=False,
                 dropout=0.1,
                 prefusion_mlp=False,
                 prefusion_mlp_activation='gelu',
                 image_size=224,
                 vision_encoder='deit',
                 image_base=None,
                 config_file="./configs/config_bert.json",
                 encoder_base=None,
                 decoder_base="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"):
        super().__init__()
        
        encoder_base = encoder_base or decoder_base
        self.encoder_base = encoder_base
        self.decoder_base = decoder_base
        self.use_prefusion_mlp = prefusion_mlp
        self.prefusion_mlp_activation = prefusion_mlp_activation
        self.use_bidirectional = use_bidirectional
        self.use_adaptive_gating = use_adaptive_gating
        self.use_answer_query = use_answer_query
        self.answer_query_weight = answer_query_weight
        self.use_asymmetric_query_loss = use_asymmetric_query_loss
        self.answer_init_prefix = answer_init_prefix
        self.answer_query_at_inference = answer_query_at_inference
        self.answer_infer_mode = answer_infer_mode
        self.answer_query_ensemble_weight = answer_query_ensemble_weight
        self.answer_query_rerank_k = answer_query_rerank_k
        self.answer_query_weight_open = answer_query_weight_open
        self.answer_query_weight_closed = answer_query_weight_closed
        self.answer_query_confidence_threshold = answer_query_confidence_threshold
        self.use_fg_query_fusion = use_fg_query_fusion
        self.use_fg_concat_query = use_fg_concat_query
        self.query_cosine_weight = query_cosine_weight
        self.use_bert_fusion_stream = use_bert_fusion_stream
        self.hybrid_open_only = hybrid_open_only
        self.hybrid_open_query_only = hybrid_open_query_only
        self.open_decoder_loss_weight = open_decoder_loss_weight
        self.use_answer_type_routing = use_answer_type_routing
        self.query_only_train = query_only_train
        self.query_train_open_only = query_train_open_only
        self.use_closed_yn_head = use_closed_yn_head
        self.closed_yn_loss_weight = closed_yn_loss_weight
        self.closed_yn_infer_weight = closed_yn_infer_weight
        self.closed_yn_train_only = closed_yn_train_only
        self.use_open_answer_head = use_open_answer_head
        self.open_answer_loss_weight = open_answer_loss_weight
        self.open_answer_infer_weight = open_answer_infer_weight
        self.open_answer_train_only = open_answer_train_only
        self.use_dsap = use_dsap
        self.dsap_train_prompts_only = dsap_train_prompts_only
        self.use_itm_yn_head = use_itm_yn_head
        self.itm_yn_loss_weight = itm_yn_loss_weight
        self.itm_yn_infer_weight = itm_yn_infer_weight
        self.use_laterality_expert = use_laterality_expert
        self.laterality_loss_weight = laterality_loss_weight
        self.laterality_infer_weight = laterality_infer_weight
        self.laterality_confidence = laterality_confidence
        self.use_modality_expert = use_modality_expert
        self.modality_loss_weight = modality_loss_weight
        self.modality_infer_weight = modality_infer_weight
        self.modality_confidence = modality_confidence
        self.e3_experts_train_only = e3_experts_train_only
        self.e3_infer_enabled = e3_infer_enabled
        self.e3_new_experts_only = e3_new_experts_only
        self.closed_yn_focal_gamma = closed_yn_focal_gamma
        self.closed_yn_exclusive_infer = closed_yn_exclusive_infer
        self.closed_expert_train_only = closed_expert_train_only
        self.answer_boost_closed_only = answer_boost_closed_only
        self.use_cmm_head = use_cmm_head
        self.cmm_loss_weight = cmm_loss_weight
        self.cmm_infer_weight = cmm_infer_weight
        self.cmm_train_only = cmm_train_only
        self.cmm_open_only = cmm_open_only
        self.cmm_confidence_threshold = cmm_confidence_threshold
        self.cmm_infer_enabled = True
        self.use_open_synonym_boost = use_open_synonym_boost
        self.rad_open_stack_train_only = rad_open_stack_train_only
        self.use_answer_boost = use_answer_boost
        self.answer_boost_train_only = answer_boost_train_only
        self.answer_boost_loss_weight = answer_boost_loss_weight
        self.use_mvcm_itlc = use_mvcm_itlc
        self.mvcm_itlc_loss_weight = mvcm_itlc_loss_weight
        self.mvcm_itc_loss_weight = mvcm_itc_loss_weight
        self.mvcm_boost_train_only = mvcm_boost_train_only
        self._synonym_index = None
        self._lat_answer_map = None
        self._mod_answer_map = None
        self.attention_mode = attention_mode
        self._answer_list_lower = []
        self._closed_answer_indices = set()
        self._yes_idx = None
        self._no_idx = None
        
        # Decoder config
        config_decoder = BertConfig.from_json_file(config_file)
        config_decoder.fusion_layer = 0
        config_decoder.num_hidden_layers = 6
        
        # Encoders
        self.vision_type = vision_encoder
        if vision_encoder == 'biomedclip':
            biomed_base = image_base or "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            self.image_encoder = BiomedCLIPImageEncoder(biomed_base)
        elif vision_encoder == 'clip':
            clip_base = image_base or "openai/clip-vit-base-patch16"
            self.image_encoder = CLIPImageEncoder(clip_base)
        else:
            self.image_encoder = ImageEncoder(image_size=image_size)
        bert_config = BertConfig.from_json_file(config_file)
        self.text_encoder = BertForMaskedLM.from_pretrained(encoder_base, config=bert_config)
        
        # Enhanced FineGrained module
        self.fg_gf = FineGrainedGlobalFeature(
            embed_dim=embed_dim,
            num_layers=fg_layers,
            head_schedule=head_schedule,
            attention_mode=attention_mode,
            use_bidirectional=use_bidirectional,
            use_adaptive_gating=use_adaptive_gating,
            dropout=dropout
        )

        self.image_prefusion_mlp = None
        self.text_prefusion_mlp = None
        if self.use_prefusion_mlp:
            self.image_prefusion_mlp = PreFusionMLP(
                embed_dim, activation=prefusion_mlp_activation, dropout=dropout
            )
            self.text_prefusion_mlp = PreFusionMLP(
                embed_dim, activation=prefusion_mlp_activation, dropout=dropout
        )
        
        # Decoder
        self.tokenizer = BertTokenizer.from_pretrained(decoder_base) 
        self.decoder = BertLMHeadModel.from_pretrained(decoder_base, config=config_decoder)

        self.answer_query_head = None
        if use_answer_query and num_answer_classes > 0:
            ffn_dim = answer_query_ffn_dim if answer_query_ffn_dim > 0 else embed_dim * 4
            if use_fg_concat_query or (not use_fg_query_fusion):
                self.answer_query_head = AnswerQueryHead(
                    num_classes=num_answer_classes,
                    embed_dim=embed_dim,
                    num_layers=answer_query_layers,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    cosine_weight=query_cosine_weight,
                )
            elif use_fg_query_fusion:
                self.answer_query_head = FGFusionAnswerQueryHead(
                    num_classes=num_answer_classes,
                    embed_dim=embed_dim,
                    num_layers=answer_query_layers,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    use_bert_fusion_stream=use_bert_fusion_stream,
                )

        self.closed_yn_head = None
        if use_closed_yn_head:
            self.closed_yn_head = ClosedYesNoHead(
                embed_dim, hidden_dim=closed_yn_hidden_dim, dropout=dropout,
                use_text_fg=not closed_yn_compact, compact=closed_yn_compact,
            )

        self.itm_yn_head = None
        if use_itm_yn_head:
            self.itm_yn_head = ITMYesNoHead(
                embed_dim, hidden_dim=closed_yn_hidden_dim or embed_dim, dropout=dropout,
            )
        self.laterality_expert = None
        if use_laterality_expert:
            self.laterality_expert = LateralityExpert(
                embed_dim, hidden_dim=open_answer_hidden_dim or embed_dim, dropout=dropout,
                num_classes=len(LATERALITY_LABELS),
            )
        self.modality_expert = None
        if use_modality_expert:
            self.modality_expert = ModalityExpert(
                embed_dim, hidden_dim=open_answer_hidden_dim or embed_dim, dropout=dropout,
                num_classes=len(MODALITY_LABELS),
            )

        self.open_answer_head = None
        if use_open_answer_head and num_answer_classes > 0:
            self.open_answer_head = OpenAnswerHead(
                num_answer_classes,
                embed_dim,
                hidden_dim=open_answer_hidden_dim or embed_dim,
                dropout=dropout,
            )

        self.cmm_head = None
        if use_cmm_head and num_answer_classes > 0:
            self.cmm_head = CrossModalMatchingHead(
                num_answer_classes,
                embed_dim=embed_dim,
                num_layers=cmm_num_layers,
                dropout=dropout,
                use_bert_fusion=cmm_use_bert_fusion,
            )

        self.answer_boost = None
        if use_answer_boost and num_answer_classes > 0:
            self.answer_boost = AnswerBoostStack(
                embed_dim=embed_dim,
                num_answer_classes=num_answer_classes,
                hidden_dim=closed_yn_hidden_dim or open_answer_hidden_dim or embed_dim,
                dropout=dropout,
                use_itm=True,
                use_laterality=not answer_boost_closed_only,
                use_modality=not answer_boost_closed_only,
                use_uncertainty_gate=True,
                use_synonym_boost=not answer_boost_closed_only,
            )

        self.mvcm_align = None
        if use_mvcm_itlc:
            self.mvcm_align = MVCMAlignmentModule(embed_dim=embed_dim)

        self.sap_module = None
        self.dqap_module = None
        if use_dsap:
            self.sap_module = SemanticAlignmentPrompt(
                embed_dim,
                n_image=dsap_sap_image_prompts,
                n_text=dsap_sap_text_prompts,
            )
            self.dqap_module = DynamicQuestionAwarePrompt(
                embed_dim, n_prompts=dsap_dqap_prompts,
            )
        
        print(f"\n{'='*60}")
        print("VLAT_Enhanced Configuration:")
        if vision_encoder == 'biomedclip':
            print(f"  Vision: BiomedCLIP ({image_base or 'PMC-15M ViT'})")
        elif vision_encoder == 'clip':
            print(f"  Vision: CLIP ({image_base or 'openai/clip-vit-base-patch16'})")
        else:
            print(f"  Vision: DeiT ViT")
        print(f"  Text encoder: {encoder_base}")
        print(f"  Text decoder: {decoder_base}")
        print(f"  FineGrained layers: {fg_layers}")
        print(f"  Head schedule: {head_schedule or [12,10,8,6,4,4][:fg_layers]}")
        print(f"  Attention mode: {attention_mode}")
        print(f"  Pre-fusion MLP: {prefusion_mlp} (activation={prefusion_mlp_activation})")
        print(f"  Bidirectional: {use_bidirectional}")
        print(f"  Adaptive gating: {use_adaptive_gating}")
        if use_answer_query:
            infer_mode = 'query' if query_only_train else answer_infer_mode
            if answer_query_at_inference and answer_infer_mode == 'decoder' and not query_only_train:
                infer_mode = 'query'
            fg_mode = "Q2AT-FG→Query" if use_fg_concat_query else (
                "FG-fusion" if use_fg_query_fusion else "BERT-fusion"
            )
            train_mode = 'Q2A-query-only (ASL)' if query_only_train else 'decoder+query'
            print(
                f"  Answer query head: True ({num_answer_classes} classes, "
                f"mode={fg_mode}, infer={infer_mode}, train={train_mode})"
            )
        else:
            print(f"  Answer query head: False")
        if use_closed_yn_head:
            print(
                f"  Closed yes/no head: True "
                f"(loss_w={closed_yn_loss_weight}, infer_w={closed_yn_infer_weight}, "
                f"train_only={closed_yn_train_only})"
            )
        if use_open_answer_head:
            print(
                f"  Open answer head: True "
                f"(classes={num_answer_classes}, loss_w={open_answer_loss_weight}, "
                f"infer_w={open_answer_infer_weight}, train_only={open_answer_train_only})"
            )
        if use_cmm_head:
            print(
                f"  CMM head: True "
                f"(classes={num_answer_classes}, layers={cmm_num_layers}, "
                f"loss_w={cmm_loss_weight}, infer_w={cmm_infer_weight}, "
                f"train_only={cmm_train_only}, bert_fusion={cmm_use_bert_fusion})"
            )
        if use_dsap:
            print(
                f"  DSAP prompts: True "
                f"(SAP img={dsap_sap_image_prompts}/txt={dsap_sap_text_prompts}, "
                f"DQAP={dsap_dqap_prompts}, train_only={dsap_train_prompts_only})"
            )
        if use_itm_yn_head or use_laterality_expert or use_modality_expert:
            print(
                f"  E3 experts: ITM-YN={use_itm_yn_head} (w={itm_yn_infer_weight}), "
                f"Laterality={use_laterality_expert}, Modality={use_modality_expert}, "
                f"train_only={e3_experts_train_only}"
            )
        if use_answer_boost:
            print(
                f"  Answer boost stack: True "
                f"(ITM+Laterality+Modality+synonyms, train_only={answer_boost_train_only})"
            )
        if use_mvcm_itlc:
            print(
                f"  MVCM-lite ITLC: True "
                f"(itlc_w={mvcm_itlc_loss_weight}, itc_w={mvcm_itc_loss_weight}, "
                f"boost_train_only={mvcm_boost_train_only})"
            )
        print(f"{'='*60}\n")

    _CLOSED_ANSWER_STRINGS = frozenset({
        'yes', 'no', 'left', 'right', 'anterior', 'posterior',
        'superior', 'inferior', 'normal', 'abnormal', 'male', 'female',
    })

    def register_answer_vocab(self, answer_list):
        """Cache answer strings and closed-class indices for hybrid routing."""
        self._answer_list_lower = [str(a).strip().lower() for a in answer_list]
        self._closed_answer_indices = {
            i for i, ans in enumerate(self._answer_list_lower)
            if ans in self._CLOSED_ANSWER_STRINGS
        }
        self._yes_idx = None
        self._no_idx = None
        for i, ans in enumerate(self._answer_list_lower):
            if ans == 'yes':
                self._yes_idx = i
            elif ans == 'no':
                self._no_idx = i
        print(
            f"  Registered {len(self._answer_list_lower)} answers for hybrid routing "
            f"(closed idx: {sorted(self._closed_answer_indices)}, "
            f"yes={self._yes_idx}, no={self._no_idx})"
        )
        if self.use_laterality_expert:
            self._lat_answer_map = build_laterality_answer_map(self._answer_list_lower)
        if self.use_modality_expert:
            self._mod_answer_map = build_modality_answer_map(self._answer_list_lower)
        if self.use_open_synonym_boost:
            self._synonym_index = build_synonym_index(answer_list)
        if self.answer_boost is not None:
            self.answer_boost.register_vocab(answer_list)

    @staticmethod
    def _is_likely_closed_question(question_text):
        q = str(question_text).lower().strip()
        closed_starts = (
            'is ', 'are ', 'was ', 'were ', 'does ', 'do ', 'did ',
            'can ', 'has ', 'have ', 'could ', 'would ', 'should ',
            'is there', 'are there', 'is this', 'is the', 'is it',
            'was there', 'were there', 'does the', 'does this',
        )
        return q.startswith(closed_starts)

    @staticmethod
    def _is_likely_open_question(question_text):
        q = str(question_text).lower().strip()
        open_starts = (
            'what ', 'where ', 'which ', 'how ', 'who ', 'when ', 'why ',
            'name ', 'describe ', 'list ', 'type ', 'in what', 'what kind',
            'what type', 'how many', 'how much', 'what is the', 'what are the',
        )
        return q.startswith(open_starts)

    def _open_route_mask(self, questions_text, batch_size, device, answer_types=None):
        """Route OPEN questions to Q2A head; CLOSED stay on VLAT decoder."""
        route = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if answer_types is not None and self.use_answer_type_routing:
            for b, atype in enumerate(answer_types):
                route[b] = str(atype).upper() == 'OPEN'
            return route
        if not questions_text:
            return route
        for b, q in enumerate(questions_text):
            if self.hybrid_open_only:
                if self._is_likely_open_question(q):
                    route[b] = True
                elif not self._is_likely_closed_question(q):
                    route[b] = True
            else:
                route[b] = not self._is_likely_closed_question(q)
        return route

    def _query_type_weights(self, answer_types, device):
        weights = []
        for answer_type in answer_types:
            if str(answer_type).upper() == 'OPEN':
                weights.append(self.answer_query_weight_open)
            else:
                weights.append(self.answer_query_weight_closed)
        return torch.tensor(weights, dtype=torch.float32, device=device)

    def _closed_route_mask(self, query_probs, questions_text):
        batch_size = query_probs.size(0)
        route = torch.zeros(batch_size, dtype=torch.bool, device=query_probs.device)
        for b in range(batch_size):
            top_prob, top_id = query_probs[b].max(dim=0)
            if questions_text and self._is_likely_closed_question(questions_text[b]):
                route[b] = True
            elif top_id.item() in self._closed_answer_indices:
                if top_prob.item() >= self.answer_query_confidence_threshold:
                    route[b] = True
        return route

    def init_answer_query_from_text(self, answer_list, device):
        if self.answer_query_head is None:
            return
        self.answer_query_head.init_from_text(
            self.text_encoder,
            self.tokenizer,
            answer_list,
            device,
            prefix=self.answer_init_prefix,
        )

    def init_cmm_from_text(self, answer_list, device):
        if self.cmm_head is None:
            return
        self.cmm_head.init_from_text(
            self.text_encoder,
            self.tokenizer,
            answer_list,
            device,
            prefix=self.answer_init_prefix,
        )

    def _effective_infer_mode(self):
        if self.answer_query_head is None:
            return 'decoder'
        if self.answer_query_at_inference and self.answer_infer_mode == 'decoder':
            return 'query'
        return self.answer_infer_mode

    def _encode(self, images, questions):
        """Shared VLAT encoding: FG-refined streams + BERT fusion for decoder."""
        image_embeds = self.image_encoder(images)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(images.device)
        text_atts = questions.attention_mask
        
        text_output = self.text_encoder.bert(
            questions.input_ids, 
            attention_mask=text_atts,
            return_dict=True, 
            mode='text'
        ) 
        text_embeds = text_output.last_hidden_state

        if self.sap_module is not None:
            image_embeds, text_embeds, image_atts, text_atts = self.sap_module(
                image_embeds, text_embeds, text_atts,
            )

        if self.dqap_module is not None:
            text_embeds, text_atts = self.dqap_module(text_embeds, text_atts)

        if self.use_prefusion_mlp:
            image_embeds = self.image_prefusion_mlp(image_embeds)
            text_embeds = self.text_prefusion_mlp(text_embeds)

        image_fg, text_fg = self.fg_gf(image_embeds, text_embeds)

        skip_bert_fusion = (
            self.query_only_train
            and self.use_answer_query
            and (self.use_fg_concat_query or not self.use_fg_query_fusion)
        )
        if skip_bert_fusion:
            return {
                'question_states': None,
                'question_atts': text_atts,
                'image_fg': image_fg,
                'text_fg': text_fg,
                'image_atts': image_atts,
            }

        fusion_outputs = self.text_encoder.bert(
            attention_mask=text_atts,
            inputs_embeds=text_fg,
            encoder_hidden_states=image_fg,
            encoder_attention_mask=image_atts,
            return_dict=True,
            mode='fusion',
        )
        return {
            'question_states': fusion_outputs.last_hidden_state,
            'question_atts': text_atts,
            'image_fg': image_fg,
            'text_fg': text_fg,
            'image_atts': image_atts,
        }

    def _fuse(self, images, questions):
        encoded = self._encode(images, questions)
        return encoded['question_states'], encoded['question_atts']

    def _query_logits(self, encoded):
        if self.use_fg_concat_query:
            text_fg = encoded['text_fg']
            image_fg = encoded['image_fg']
            text_atts = encoded['question_atts']
            image_atts = encoded['image_atts']
            image_len = image_fg.size(1)
            fused_embeds = torch.cat([text_fg, image_fg], dim=1)
            if text_atts is not None:
                image_mask = torch.ones(
                    text_atts.size(0), image_len,
                    dtype=text_atts.dtype, device=text_atts.device,
                )
                fused_mask = torch.cat([text_atts, image_mask], dim=1)
            else:
                fused_mask = None
            return self.answer_query_head(fused_embeds, fused_mask)

        if self.use_fg_query_fusion:
            return self.answer_query_head(
                encoded['text_fg'],
                encoded['image_fg'],
                text_attention_mask=encoded['question_atts'],
                bert_fusion=encoded['question_states'] if self.use_bert_fusion_stream else None,
                bert_attention_mask=encoded['question_atts'] if self.use_bert_fusion_stream else None,
            )
        if encoded['question_states'] is None:
            raise RuntimeError("BERT fusion states required for base answer-query head")
        return self.answer_query_head(
            encoded['question_states'], encoded['question_atts']
        )

    def _closed_yn_logits(self, encoded):
        if self.closed_yn_head is None:
            raise RuntimeError("Closed yes/no head is not enabled")
        if encoded['question_states'] is None:
            raise RuntimeError("Closed yes/no head requires BERT fusion states")
        return self.closed_yn_head(
            encoded['question_states'],
            encoded['question_atts'],
            encoded['image_fg'],
            encoded.get('text_fg'),
        )

    def _closed_yn_targets(self, answer_types, question_answer_indices, device):
        """Return (batch_indices, 0=no/1=yes) for CLOSED yes/no samples."""
        batch_idx, targets = [], []
        if answer_types is None or question_answer_indices is None:
            return batch_idx, targets
        if self._yes_idx is None or self._no_idx is None:
            return batch_idx, targets
        for b, atype in enumerate(answer_types):
            if str(atype).upper() != 'CLOSED':
                continue
            qidx = int(question_answer_indices[b].item())
            if qidx == self._yes_idx:
                batch_idx.append(b)
                targets.append(1)
            elif qidx == self._no_idx:
                batch_idx.append(b)
                targets.append(0)
        if not batch_idx:
            return batch_idx, targets
        return (
            torch.tensor(batch_idx, dtype=torch.long, device=device),
            torch.tensor(targets, dtype=torch.long, device=device),
        )

    def _closed_yn_loss(self, encoded, answer_types, question_answer_indices):
        batch_idx, targets = self._closed_yn_targets(
            answer_types, question_answer_indices, encoded['image_fg'].device
        )
        if not len(batch_idx):
            return None
        logits = self._closed_yn_logits(encoded)[batch_idx]
        if self.closed_yn_focal_gamma > 0:
            ce = F.cross_entropy(logits, targets, reduction='none')
            pt = torch.exp(-ce)
            gamma = self.closed_yn_focal_gamma
            return (((1 - pt) ** gamma) * ce).mean()
        return F.cross_entropy(logits, targets)

    def _itm_yn_logits(self, encoded):
        return self.itm_yn_head(
            encoded['question_states'], encoded['question_atts'], encoded['image_fg'],
        )

    def _itm_yn_loss(self, encoded, answer_types, question_answer_indices):
        batch_idx, targets = self._closed_yn_targets(
            answer_types, question_answer_indices, encoded['image_fg'].device
        )
        if not len(batch_idx) or self.itm_yn_head is None:
            return None
        logits = self._itm_yn_logits(encoded)[batch_idx]
        return F.cross_entropy(logits, targets)

    def _laterality_loss(self, encoded, answer_types, question_answer_indices, questions_text):
        if self.laterality_expert is None or questions_text is None:
            return None
        batch_idx, targets = [], []
        for b, (atype, q) in enumerate(zip(answer_types, questions_text)):
            if str(atype).upper() != 'OPEN' or not question_has_laterality(q):
                continue
            qidx = int(question_answer_indices[b].item())
            ans = self._answer_list_lower[qidx] if qidx >= 0 else ''
            if ans == 'left' or ans.startswith('left '):
                targets.append(0)
            elif ans == 'right' or ans.startswith('right '):
                targets.append(1)
            elif 'bilateral' in ans:
                targets.append(2)
            elif 'midline' in ans:
                targets.append(3)
            else:
                targets.append(4)
            batch_idx.append(b)
        if not batch_idx:
            return None
        device = encoded['image_fg'].device
        batch_idx = torch.tensor(batch_idx, dtype=torch.long, device=device)
        targets = torch.tensor(targets, dtype=torch.long, device=device)
        logits = self.laterality_expert(
            encoded['question_states'], encoded['question_atts'], encoded['image_fg'],
        )[batch_idx]
        return F.cross_entropy(logits, targets)

    def _modality_loss(self, encoded, answer_types, question_answer_indices, questions_text):
        if self.modality_expert is None or questions_text is None:
            return None
        batch_idx, targets = [], []
        mri_kw = ('mri', 'flair', 't1', 't2', 'dwi', 'diffusion')
        ct_kw = ('ct', 'computed tomography')
        xray_kw = ('xray', 'x-ray', 'x ray', 'plain film', 'radiograph')
        us_kw = ('ultrasound', 'sonograph')
        for b, (atype, q) in enumerate(zip(answer_types, questions_text)):
            if str(atype).upper() != 'OPEN' or not question_has_modality(q):
                continue
            qidx = int(question_answer_indices[b].item())
            ans = self._answer_list_lower[qidx] if qidx >= 0 else ''
            if any(k in ans for k in mri_kw):
                targets.append(0)
            elif any(k in ans for k in ct_kw):
                targets.append(1)
            elif any(k in ans for k in xray_kw):
                targets.append(2)
            elif any(k in ans for k in us_kw):
                targets.append(3)
            else:
                targets.append(4)
            batch_idx.append(b)
        if not batch_idx:
            return None
        device = encoded['image_fg'].device
        batch_idx = torch.tensor(batch_idx, dtype=torch.long, device=device)
        targets = torch.tensor(targets, dtype=torch.long, device=device)
        logits = self.modality_expert(
            encoded['question_states'], encoded['question_atts'], encoded['image_fg'],
        )[batch_idx]
        return F.cross_entropy(logits, targets)

    def _mvcm_align_loss(self, encoded):
        if self.mvcm_align is None:
            return None, None
        itlc, itc = self.mvcm_align(
            encoded['image_fg'],
            encoded['text_fg'],
            encoded['question_atts'],
            encoded.get('image_atts'),
            use_global=self.mvcm_itc_loss_weight > 0,
        )
        return itlc, itc

    def _apply_closed_yn_override(
        self, encoded, question_states, question_atts, answer_ids,
        topk_ids, topk_probs, answer_types, k,
    ):
        if self.closed_yn_head is None or answer_types is None:
            return topk_ids, topk_probs
        if self._yes_idx is None or self._no_idx is None:
            return topk_ids, topk_probs

        yn_logits = self._closed_yn_logits(encoded)
        yn_probs = F.softmax(yn_logits, dim=-1)

        dec_scores = self._decoder_first_token_scores(
            question_states, question_atts, answer_ids
        )
        dec_no = dec_scores[:, self._no_idx].clamp(min=1e-8)
        dec_yes = dec_scores[:, self._yes_idx].clamp(min=1e-8)

        if self.closed_yn_exclusive_infer:
            fused_no = yn_probs[:, 0]
            fused_yes = yn_probs[:, 1]
        else:
            w = self.closed_yn_infer_weight
            fused_no = (1.0 - w) * dec_no + w * yn_probs[:, 0]
            fused_yes = (1.0 - w) * dec_yes + w * yn_probs[:, 1]

        if self.itm_yn_head is not None and self.e3_infer_enabled and self.answer_boost is None:
            itm_probs = F.softmax(self._itm_yn_logits(encoded), dim=-1)
            wi = self.itm_yn_infer_weight
            fused_no = (1.0 - wi) * fused_no + wi * itm_probs[:, 0]
            fused_yes = (1.0 - wi) * fused_yes + wi * itm_probs[:, 1]

        if self.cmm_head is not None and self.cmm_infer_weight > 0 and not self.cmm_open_only:
            cmm_probs = F.softmax(self._cmm_logits(encoded), dim=-1)
            wc = self.cmm_infer_weight
            fused_no = (1.0 - wc) * fused_no + wc * cmm_probs[:, self._no_idx]
            fused_yes = (1.0 - wc) * fused_yes + wc * cmm_probs[:, self._yes_idx]

        questions_text = getattr(self, '_infer_questions_text', None)
        if self.answer_boost is not None:
            fused_no, fused_yes = self.answer_boost.apply_closed(
                encoded, fused_no, fused_yes, dec_no, dec_yes,
                self.itm_yn_infer_weight, questions_text,
                self._yes_idx, self._no_idx,
            )
        elif questions_text is not None:
            for b, q in enumerate(questions_text):
                bias = question_negation_bias(q)
                if bias != 0.0:
                    fused_yes[b] = fused_yes[b] * (1.0 + bias)
                    fused_no[b] = fused_no[b] * (1.0 - bias)

        for b, atype in enumerate(answer_types):
            if str(atype).upper() != 'CLOSED':
                continue
            pick = self._yes_idx if fused_yes[b] > fused_no[b] else self._no_idx
            prob = fused_yes[b] if pick == self._yes_idx else fused_no[b]
            topk_ids[b, 0] = pick
            topk_probs[b, 0] = prob
            if k > 1:
                other = self._no_idx if pick == self._yes_idx else self._yes_idx
                other_prob = fused_no[b] if other == self._no_idx else fused_yes[b]
                topk_ids[b, 1] = other
                topk_probs[b, 1] = other_prob
        return topk_ids, topk_probs

    def _open_answer_logits(self, encoded):
        if self.open_answer_head is None:
            raise RuntimeError("Open answer head is not enabled")
        if encoded['question_states'] is None:
            raise RuntimeError("Open answer head requires BERT fusion states")
        return self.open_answer_head(
            encoded['question_states'],
            encoded['question_atts'],
            encoded['image_fg'],
        )

    def _open_answer_loss(self, encoded, answer_types, question_answer_indices):
        if answer_types is None or question_answer_indices is None:
            return None
        batch_idx, targets = [], []
        for b, atype in enumerate(answer_types):
            if str(atype).upper() != 'OPEN':
                continue
            qidx = int(question_answer_indices[b].item())
            if qidx >= 0:
                batch_idx.append(b)
                targets.append(qidx)
        if not batch_idx:
            return None
        batch_idx = torch.tensor(batch_idx, dtype=torch.long, device=encoded['image_fg'].device)
        targets = torch.tensor(targets, dtype=torch.long, device=encoded['image_fg'].device)
        logits = self._open_answer_logits(encoded)[batch_idx]
        return F.cross_entropy(logits, targets)

    def _cmm_logits(self, encoded):
        if self.cmm_head is None:
            raise RuntimeError("CMM head is not enabled")
        return self.cmm_head(
            encoded['text_fg'],
            encoded['image_fg'],
            encoded['question_atts'],
            bert_fusion=encoded.get('question_states'),
            bert_attention_mask=encoded['question_atts'],
        )

    def _cmm_loss(self, question_answer_indices, encoded):
        if question_answer_indices is None:
            return None
        valid = question_answer_indices >= 0
        if not valid.any():
            return None
        logits = self._cmm_logits(encoded)[valid]
        targets = question_answer_indices[valid]
        return F.cross_entropy(logits, targets)

    def _blend_with_cmm(self, encoded, base_scores, open_mask=None):
        """Gated CMM blend — OPEN-only by default; skips when disabled or low confidence."""
        if (
            self.cmm_head is None
            or self.cmm_infer_weight <= 0
            or not getattr(self, 'cmm_infer_enabled', True)
        ):
            return base_scores
        cmm_probs = F.softmax(self._cmm_logits(encoded), dim=-1)
        w = self.cmm_infer_weight
        out = base_scores.clone()
        batch_size = base_scores.size(0)
        for b in range(batch_size):
            if open_mask is not None and not open_mask[b]:
                continue
            peak = cmm_probs[b].max().item()
            if peak < self.cmm_confidence_threshold:
                continue
            out[b] = (1.0 - w) * base_scores[b] + w * cmm_probs[b]
        return out

    def _apply_open_answer_override(
        self, encoded, question_states, question_atts, answer_ids,
        topk_ids, topk_probs, answer_types, k,
    ):
        if self.open_answer_head is None or answer_types is None:
            return topk_ids, topk_probs

        open_logits = self._open_answer_logits(encoded)
        open_probs = F.softmax(open_logits, dim=-1)
        dec_scores = self._decoder_first_token_scores(
            question_states, question_atts, answer_ids
        )
        w = self.open_answer_infer_weight
        fused = (1.0 - w) * dec_scores + w * open_probs
        open_mask = torch.tensor(
            [str(a).upper() == 'OPEN' for a in answer_types],
            device=fused.device,
        )
        fused = self._blend_with_cmm(encoded, fused, open_mask=open_mask)
        questions_text = getattr(self, '_infer_questions_text', None)

        for b, atype in enumerate(answer_types):
            if str(atype).upper() != 'OPEN':
                continue
            row = fused[b]
            if self.answer_boost is not None and questions_text is not None:
                row = self.answer_boost.apply_open(
                    row, encoded,
                    question_states[b], question_atts[b], encoded['image_fg'],
                    questions_text[b],
                    self.laterality_infer_weight, self.laterality_confidence,
                    self.modality_infer_weight, self.modality_confidence,
                )
            elif self.use_open_synonym_boost and self._synonym_index is not None:
                row = apply_synonym_boost(row, self._synonym_index)
            pick = int(row.argmax().item())
            topk_ids[b, 0] = pick
            topk_probs[b, 0] = row[pick]
            if k > 1:
                topk_vals, topk_cands = row.topk(min(k, row.size(0)))
                for slot in range(1, min(k, topk_cands.size(0))):
                    topk_ids[b, slot] = topk_cands[slot]
                    topk_probs[b, slot] = topk_vals[slot]
        return topk_ids, topk_probs

    def _apply_e3_open_expert_overrides(
        self, encoded, question_states, question_atts, answer_ids,
        topk_ids, topk_probs, answer_types, k, questions_text,
    ):
        if questions_text is None or answer_types is None:
            return topk_ids, topk_probs
        dec_scores = self._decoder_first_token_scores(
            question_states, question_atts, answer_ids
        )
        open_scores = None
        if self.open_answer_head is not None:
            open_scores = F.softmax(self._open_answer_logits(encoded), dim=-1)

        for b, (atype, q) in enumerate(zip(answer_types, questions_text)):
            if str(atype).upper() != 'OPEN':
                continue
            scores = dec_scores[b].clone()
            if open_scores is not None:
                wo = self.open_answer_infer_weight
                scores = (1.0 - wo) * scores + wo * open_scores[b]

            if self.laterality_expert is not None and question_has_laterality(q):
                lat_logits = self.laterality_expert(
                    question_states[b:b + 1], question_atts[b:b + 1], encoded['image_fg'][b:b + 1],
                )
                lat_probs = F.softmax(lat_logits, dim=-1)[0]
                conf, cls_id = lat_probs.max(dim=0)
                if conf.item() >= self.laterality_confidence and self._lat_answer_map:
                    pick = pick_best_from_candidates(scores, self._lat_answer_map[int(cls_id)])
                    topk_ids[b, 0] = pick
                    topk_probs[b, 0] = scores[pick]
                    continue

            if self.modality_expert is not None and question_has_modality(q):
                mod_logits = self.modality_expert(
                    question_states[b:b + 1], question_atts[b:b + 1], encoded['image_fg'][b:b + 1],
                )
                mod_probs = F.softmax(mod_logits, dim=-1)[0]
                conf, cls_id = mod_probs.max(dim=0)
                if conf.item() >= self.modality_confidence and self._mod_answer_map:
                    wm = self.modality_infer_weight
                    mod_pick = pick_best_from_candidates(scores, self._mod_answer_map[int(cls_id)])
                    mod_scores = scores.clone()
                    mod_scores[mod_pick] = mod_scores[mod_pick] * (1.0 + wm)
                    pick = int(mod_scores.argmax().item())
                    topk_ids[b, 0] = pick
                    topk_probs[b, 0] = mod_scores[pick]
                    continue

            pick = int(scores.argmax().item())
            topk_ids[b, 0] = pick
            topk_probs[b, 0] = scores[pick]
        return topk_ids, topk_probs

    def forward(self, images, questions, answers, weights, k, train=True, alpha=None,
                answer_indices=None, question_answer_indices=None, answer_types=None,
                questions_text=None, infer_mode_override=None):
        """Forward with optional answer-query auxiliary head."""
        
        encoded = self._encode(images, questions)
        question_states = encoded['question_states']
        question_atts = encoded['question_atts']
        if not train:
            self._infer_questions_text = questions_text

        if train:
            aux_expert_train = (
                (self.open_answer_train_only and self.open_answer_head is not None)
                or (self.closed_yn_train_only and self.closed_yn_head is not None)
                or (self.cmm_train_only and self.cmm_head is not None)
                or (self.query_only_train and self.answer_query_head is not None)
                or (self.answer_boost_train_only and self.answer_boost is not None)
                or (self.mvcm_boost_train_only and (
                    self.answer_boost is not None or self.mvcm_align is not None
                ))
                or (self.e3_experts_train_only and (
                    self.itm_yn_head is not None
                    or self.laterality_expert is not None
                    or self.modality_expert is not None
                ))
            )
            if (self.mvcm_boost_train_only or self.answer_boost_train_only) and (
                self.answer_boost is not None or self.mvcm_align is not None
            ) and not self.closed_expert_train_only:
                losses = []
                itlc, itc = self._mvcm_align_loss(encoded)
                if itlc is not None and self.mvcm_itlc_loss_weight > 0:
                    losses.append(self.mvcm_itlc_loss_weight * itlc)
                if itc is not None and self.mvcm_itc_loss_weight > 0:
                    losses.append(self.mvcm_itc_loss_weight * itc)
                if self.answer_boost is not None and question_answer_indices is not None and answer_types is not None:
                    itm_loss = self.answer_boost.closed_itm_loss(
                        encoded, answer_types, question_answer_indices,
                        self._yes_idx, self._no_idx,
                    )
                    if itm_loss is not None:
                        losses.append(self.itm_yn_loss_weight * itm_loss)
                    lat_loss = self.answer_boost.laterality_loss(
                        encoded, answer_types, question_answer_indices, questions_text,
                    )
                    if lat_loss is not None:
                        losses.append(self.laterality_loss_weight * lat_loss)
                    mod_loss = self.answer_boost.modality_loss(
                        encoded, answer_types, question_answer_indices, questions_text,
                    )
                    if mod_loss is not None:
                        losses.append(self.modality_loss_weight * mod_loss)
                if losses:
                    return sum(losses)
                return torch.tensor(0.0, device=encoded['image_fg'].device, requires_grad=True)
            if self.closed_expert_train_only and self.closed_yn_head is not None:
                losses = []
                yn_loss = self._closed_yn_loss(
                    encoded, answer_types, question_answer_indices
                )
                if yn_loss is not None:
                    losses.append(self.closed_yn_loss_weight * yn_loss)
                if self.answer_boost is not None and question_answer_indices is not None:
                    itm_loss = self.answer_boost.closed_itm_loss(
                        encoded, answer_types, question_answer_indices,
                        self._yes_idx, self._no_idx,
                    )
                    if itm_loss is not None:
                        losses.append(self.itm_yn_loss_weight * itm_loss)
                if losses:
                    return sum(losses)
                return torch.tensor(0.0, device=encoded['image_fg'].device, requires_grad=True)
            if self.e3_experts_train_only and (
                self.itm_yn_head is not None
                or self.laterality_expert is not None
                or self.modality_expert is not None
            ):
                losses = []
                if self.closed_yn_head is not None and not self.e3_new_experts_only:
                    yn_loss = self._closed_yn_loss(
                        encoded, answer_types, question_answer_indices
                    )
                    if yn_loss is not None:
                        losses.append(self.closed_yn_loss_weight * yn_loss)
                if self.itm_yn_head is not None:
                    itm_loss = self._itm_yn_loss(
                        encoded, answer_types, question_answer_indices
                    )
                    if itm_loss is not None:
                        losses.append(self.itm_yn_loss_weight * itm_loss)
                if self.laterality_expert is not None:
                    lat_loss = self._laterality_loss(
                        encoded, answer_types, question_answer_indices, questions_text
                    )
                    if lat_loss is not None:
                        losses.append(self.laterality_loss_weight * lat_loss)
                if self.modality_expert is not None:
                    mod_loss = self._modality_loss(
                        encoded, answer_types, question_answer_indices, questions_text
                    )
                    if mod_loss is not None:
                        losses.append(self.modality_loss_weight * mod_loss)
                if self.open_answer_head is not None and not self.e3_new_experts_only:
                    open_loss = self._open_answer_loss(
                        encoded, answer_types, question_answer_indices
                    )
                    if open_loss is not None:
                        losses.append(self.open_answer_loss_weight * open_loss)
                if losses:
                    return sum(losses)
                return torch.tensor(0.0, device=encoded['image_fg'].device, requires_grad=True)
            if aux_expert_train and (
                self.open_answer_train_only
                or self.closed_yn_train_only
                or self.cmm_train_only
                or self.query_only_train
            ):
                losses = []
                if self.closed_yn_train_only and self.closed_yn_head is not None:
                    yn_loss = self._closed_yn_loss(
                        encoded, answer_types, question_answer_indices
                    )
                    if yn_loss is not None:
                        losses.append(self.closed_yn_loss_weight * yn_loss)
                if self.open_answer_train_only and self.open_answer_head is not None:
                    open_loss = self._open_answer_loss(
                        encoded, answer_types, question_answer_indices
                    )
                    if open_loss is not None:
                        losses.append(self.open_answer_loss_weight * open_loss)
                if self.cmm_train_only and self.cmm_head is not None:
                    cmm_loss = self._cmm_loss(question_answer_indices, encoded)
                    if cmm_loss is not None:
                        losses.append(self.cmm_loss_weight * cmm_loss)
                if (
                    self.query_only_train
                    and self.answer_query_head is not None
                    and question_answer_indices is not None
                ):
                    query_logits = self._query_logits(encoded)
                    valid = question_answer_indices >= 0
                    if self.query_train_open_only and answer_types is not None:
                        for b, atype in enumerate(answer_types):
                            if str(atype).upper() != 'OPEN':
                                valid[b] = False
                    if valid.any():
                        targets = question_answer_indices[valid]
                        logits = query_logits[valid]
                        sample_weights = None
                        if answer_types is not None:
                            type_weights = self._query_type_weights(
                                answer_types, query_logits.device
                            )
                            sample_weights = type_weights[valid]
                        if self.use_asymmetric_query_loss:
                            q_loss = asymmetric_loss(
                                logits, targets, sample_weights=sample_weights
                            )
                        else:
                            per_sample = F.cross_entropy(logits, targets, reduction='none')
                            if sample_weights is not None:
                                q_loss = (
                                    per_sample * sample_weights
                                ).sum() / sample_weights.sum().clamp(min=1.0)
                            else:
                                q_loss = per_sample.mean()
                        losses.append(q_loss)
                if losses:
                    return sum(losses) / len(losses)
                return torch.zeros((), device=images.device, requires_grad=True)

            if (
                self.query_only_train
                and self.answer_query_head is not None
                and question_answer_indices is not None
            ):
                query_logits = self._query_logits(encoded)
                valid = question_answer_indices >= 0
                if (
                    self.query_train_open_only
                    and answer_types is not None
                ):
                    # Optional: train query head on OPEN only (v3). Default: all types.
                    for b, atype in enumerate(answer_types):
                        if str(atype).upper() != 'OPEN':
                            valid[b] = False
                if not valid.any():
                    return torch.zeros((), device=images.device, requires_grad=True)
                targets = question_answer_indices[valid]
                logits = query_logits[valid]
                sample_weights = None
                if answer_types is not None:
                    type_weights = self._query_type_weights(answer_types, query_logits.device)
                    sample_weights = type_weights[valid]
                if self.use_asymmetric_query_loss:
                    return asymmetric_loss(
                        logits, targets, sample_weights=sample_weights
                    )
                per_sample = F.cross_entropy(logits, targets, reduction='none')
                if sample_weights is not None:
                    return (
                        per_sample * sample_weights
                    ).sum() / sample_weights.sum().clamp(min=1.0)
                return per_sample.mean()

            if question_states is None:
                raise RuntimeError("Decoder training requires BERT fusion states")

            answer_targets = answers.input_ids.masked_fill(
                answers.input_ids == self.tokenizer.pad_token_id, -100
            )

            question_states_exp = []
            question_atts_exp = []
            for b, n in enumerate(k):
                question_states_exp += [question_states[b]] * n
                question_atts_exp += [question_atts[b]] * n
            question_states_exp = torch.stack(question_states_exp, 0)
            question_atts_exp = torch.stack(question_atts_exp, 0)
            
            answer_output = self.decoder(
                answers.input_ids, 
                attention_mask=answers.attention_mask, 
                encoder_hidden_states=question_states_exp,
                encoder_attention_mask=question_atts_exp,
                labels=answer_targets,
                return_dict=True,   
                reduction='none',
            )   
            
            per_answer_loss = weights * answer_output.loss
            if (
                self.hybrid_open_only
                and self.use_answer_query
                and answer_types is not None
                and self.open_decoder_loss_weight < 1.0
            ):
                type_weights = []
                for b, n_answers in enumerate(k):
                    is_open = str(answer_types[b]).upper() == 'OPEN'
                    w = self.open_decoder_loss_weight if is_open else 1.0
                    type_weights.extend([w] * n_answers)
                type_weights = torch.tensor(
                    type_weights, dtype=per_answer_loss.dtype, device=per_answer_loss.device
                )
                per_answer_loss = per_answer_loss * type_weights
            loss = per_answer_loss.sum() / images.size(0)

            if self.answer_query_head is not None and question_answer_indices is not None:
                query_logits = self._query_logits(encoded)
                valid = question_answer_indices >= 0
                if valid.any():
                    targets = question_answer_indices[valid]
                    logits = query_logits[valid]
                    sample_weights = None
                    if answer_types is not None:
                        type_weights = self._query_type_weights(answer_types, query_logits.device)
                        sample_weights = type_weights[valid]
                    if self.use_asymmetric_query_loss:
                        query_loss = asymmetric_loss(
                            logits, targets, sample_weights=sample_weights
                        )
                    else:
                        per_sample = F.cross_entropy(logits, targets, reduction='none')
                        if sample_weights is not None:
                            query_loss = (
                                per_sample * sample_weights
                            ).sum() / sample_weights.sum().clamp(min=1.0)
                        else:
                            query_loss = per_sample.mean()
                    loss = loss + self.answer_query_weight * query_loss

            if self.closed_yn_head is not None:
                yn_loss = self._closed_yn_loss(encoded, answer_types, question_answer_indices)
                if yn_loss is not None:
                    loss = loss + self.closed_yn_loss_weight * yn_loss

            if self.open_answer_head is not None:
                open_loss = self._open_answer_loss(
                    encoded, answer_types, question_answer_indices
                )
                if open_loss is not None:
                    loss = loss + self.open_answer_loss_weight * open_loss

            if self.cmm_head is not None and not self.cmm_train_only:
                cmm_loss = self._cmm_loss(question_answer_indices, encoded)
                if cmm_loss is not None:
                    loss = loss + self.cmm_loss_weight * cmm_loss

            if self.mvcm_align is not None and not self.mvcm_boost_train_only:
                itlc, itc = self._mvcm_align_loss(encoded)
                if itlc is not None and self.mvcm_itlc_loss_weight > 0:
                    loss = loss + self.mvcm_itlc_loss_weight * itlc
                if itc is not None and self.mvcm_itc_loss_weight > 0:
                    loss = loss + self.mvcm_itc_loss_weight * itc

            if (
                self.answer_boost is not None
                and not self.mvcm_boost_train_only
                and not self.answer_boost_train_only
                and question_answer_indices is not None
                and answer_types is not None
            ):
                itm_loss = self.answer_boost.closed_itm_loss(
                    encoded, answer_types, question_answer_indices,
                    self._yes_idx, self._no_idx,
                )
                if itm_loss is not None:
                    loss = loss + self.itm_yn_loss_weight * itm_loss
                lat_loss = self.answer_boost.laterality_loss(
                    encoded, answer_types, question_answer_indices, questions_text,
                )
                if lat_loss is not None:
                    loss = loss + self.laterality_loss_weight * lat_loss
                mod_loss = self.answer_boost.modality_loss(
                    encoded, answer_types, question_answer_indices, questions_text,
                )
                if mod_loss is not None:
                    loss = loss + self.modality_loss_weight * mod_loss

            return loss
            
        else:
            infer_mode = infer_mode_override or self._effective_infer_mode()
            if self.answer_query_head is not None and infer_mode == 'query':
                query_logits = self._query_logits(encoded)
                topk_probs, topk_ids = query_logits.topk(k, dim=1)
                topk_probs = F.softmax(topk_probs, dim=-1)
                return topk_ids, topk_probs

            if self.answer_query_head is not None and infer_mode == 'hybrid':
                if question_states is None:
                    raise RuntimeError("Hybrid inference requires BERT fusion states")
                topk_ids, topk_probs = self.rank_answer_hybrid(
                    encoded, question_states, question_atts,
                    answers.input_ids, answers.attention_mask, k,
                    questions_text=questions_text,
                    answer_types=answer_types,
                )
                if self.closed_yn_head is not None:
                    topk_ids, topk_probs = self._apply_closed_yn_override(
                        encoded, question_states, question_atts, answers.input_ids,
                        topk_ids, topk_probs, answer_types, k,
                    )
                if self.open_answer_head is not None:
                    topk_ids, topk_probs = self._apply_open_answer_override(
                        encoded, question_states, question_atts, answers.input_ids,
                        topk_ids, topk_probs, answer_types, k,
                    )
                return topk_ids, topk_probs

            if self.answer_query_head is not None and infer_mode == 'ensemble':
                if question_states is None:
                    raise RuntimeError("Ensemble inference requires BERT fusion states")
                return self.rank_answer_ensemble(
                    encoded, question_states, question_atts,
                    answers.input_ids, answers.attention_mask, k
                )

            topk_ids, topk_probs = self.rank_answer(
                question_states, question_atts, 
                answers.input_ids, answers.attention_mask, k
            )                    
            if self.closed_yn_head is not None:
                topk_ids, topk_probs = self._apply_closed_yn_override(
                    encoded, question_states, question_atts, answers.input_ids,
                    topk_ids, topk_probs, answer_types, k,
                )
            if self.open_answer_head is not None:
                skip_open = (
                    self.answer_query_head is not None
                    and infer_mode in ('hybrid', 'ensemble')
                )
                if not skip_open:
                    topk_ids, topk_probs = self._apply_open_answer_override(
                        encoded, question_states, question_atts, answers.input_ids,
                        topk_ids, topk_probs, answer_types, k,
                    )
            if (self.use_laterality_expert or self.use_modality_expert) and self.answer_boost is None:
                if self.e3_infer_enabled:
                    topk_ids, topk_probs = self._apply_e3_open_expert_overrides(
                        encoded, question_states, question_atts, answers.input_ids,
                        topk_ids, topk_probs, answer_types, k, questions_text,
                    )
            return topk_ids, topk_probs

    def _decoder_first_token_scores(self, question_states, question_atts, answer_ids):
        num_ques = question_states.size(0)
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)

        start_output = self.decoder(
            start_ids,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
            return_dict=True,
            reduction="none",
        )
        logits = start_output.logits[:, 0, :]
        answer_first_token = answer_ids[:, 1]
        return F.softmax(logits, dim=1).index_select(dim=1, index=answer_first_token)

    def rank_answer_ensemble(self, encoded, question_states, question_atts, answer_ids, answer_atts, k):
        """Fuse decoder first-token scores with answer-query logits, then stage-2 rerank."""
        num_ques = question_states.size(0)
        num_answers = answer_ids.size(0)
        rerank_k = min(self.answer_query_rerank_k, num_answers, max(k, 1))

        decoder_scores = self._decoder_first_token_scores(
            question_states, question_atts, answer_ids
        )
        query_logits = self._query_logits(encoded)
        query_scores = F.softmax(query_logits, dim=-1)

        combined = (
            decoder_scores.log().clamp(min=-1e4)
            + self.answer_query_ensemble_weight * query_scores.log().clamp(min=-1e4)
        )
        topk_probs, topk_ids = combined.topk(rerank_k, dim=1)

        input_ids = []
        input_atts = []
        for b, candidate_ids in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=candidate_ids))
            input_atts.append(answer_atts.index_select(dim=0, index=candidate_ids))
        input_ids = torch.cat(input_ids, dim=0)
        input_atts = torch.cat(input_atts, dim=0)

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)
        question_states = tile(question_states, 0, rerank_k)
        question_atts = tile(question_atts, 0, rerank_k)

        output = self.decoder(
            input_ids,
            attention_mask=input_atts,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
            labels=targets_ids,
            return_dict=True,
            reduction="none",
        )

        logits = output.logits
        log_probs_tokens = F.log_softmax(logits, dim=-1)
        targets_for_gather = targets_ids.clone()
        targets_for_gather[targets_ids == -100] = 0
        targets_ids_expanded = targets_for_gather.unsqueeze(-1)
        log_probs_targets = log_probs_tokens.gather(
            dim=-1, index=targets_ids_expanded
        ).squeeze(-1)
        log_probs_targets = log_probs_targets.masked_fill(targets_ids == -100, 0)
        answer_loss = -log_probs_targets

        topk_probs = topk_probs.view(-1, 1)
        log_probs = torch.cat([topk_probs.clamp(min=1e-8).log(), -answer_loss], dim=1)
        log_probs_sum = log_probs.sum(1).view(num_ques, rerank_k)

        topk_probs = F.softmax(log_probs_sum, dim=-1)
        final_k = min(k, rerank_k)
        topk_probs, rerank_id = topk_probs.topk(final_k, dim=1)
        topk_ids = torch.gather(topk_ids, 1, rerank_id)
        if final_k < k:
            topk_ids = F.pad(topk_ids, (0, k - final_k), value=0)
            topk_probs = F.pad(topk_probs, (0, k - final_k), value=0.0)
            return topk_ids, topk_probs

    def rank_answer_q2a(self, encoded, question_states, question_atts, answer_ids, answer_atts, k):
        """
        Q2AT-style two-stage open answering:
        1) Score all answer prototypes with the query head
        2) Decoder rerank on top-k candidates (query-primary, like Q2AT + refinement)
        """
        return self.rank_answer_ensemble(
            encoded, question_states, question_atts, answer_ids, answer_atts, k
        )

    def rank_answer_hybrid(self, encoded, question_states, question_atts, answer_ids, answer_atts, k,
                           questions_text=None, answer_types=None):
        """
        CLOSED → VLAT decoder ranking.
        OPEN → Q2AT query head (optionally with decoder rerank on top-k candidates).
        """
        batch_size = question_states.size(0)
        open_mask = self._open_route_mask(
            questions_text, batch_size, question_states.device, answer_types=answer_types
        )

        dec_ids, dec_probs = self.rank_answer(
            question_states, question_atts, answer_ids, answer_atts, k
        )
        final_ids = dec_ids.clone()
        final_probs = dec_probs.clone()

        if open_mask.any() and self.open_answer_head is not None:
            open_encoded = {
                key: val[open_mask] if isinstance(val, torch.Tensor) else val
                for key, val in encoded.items()
            }
            open_states = question_states[open_mask]
            open_atts = question_atts[open_mask]
            open_logits = self._open_answer_logits(open_encoded)
            open_probs = F.softmax(open_logits, dim=-1)
            open_top_probs, open_top_ids = open_probs.topk(k, dim=1)
            final_ids[open_mask] = open_top_ids
            final_probs[open_mask] = open_top_probs
        elif open_mask.any() and self.answer_query_head is not None:
            open_encoded = {
                key: val[open_mask] if isinstance(val, torch.Tensor) else val
                for key, val in encoded.items()
            }
            open_states = question_states[open_mask]
            open_atts = question_atts[open_mask]
            if self.hybrid_open_query_only:
                query_logits = self._query_logits(open_encoded)
                query_probs = F.softmax(query_logits, dim=-1)
                query_top_probs, query_top_ids = query_probs.topk(k, dim=1)
                # Decoder-first: keep decoder unless query is more confident on its pick
                dec_open_scores = self._decoder_first_token_scores(
                    open_states, open_atts, answer_ids
                )
                open_ids = dec_ids[open_mask].clone()
                open_probs = dec_probs[open_mask].clone()
                open_rows = open_mask.nonzero(as_tuple=False).squeeze(-1)
                for b in range(query_top_ids.size(0)):
                    dec_row = int(open_rows[b].item())
                    q_top = int(query_top_ids[b, 0].item())
                    d_top = int(dec_ids[dec_row, 0].item())
                    q_score = query_probs[b, q_top]
                    d_score = dec_open_scores[b, d_top]
                    if q_score > d_score:
                        open_ids[b] = query_top_ids[b]
                        open_probs[b] = query_top_probs[b]
            else:
                open_ids, open_probs = self.rank_answer_q2a(
                    open_encoded, open_states, open_atts,
                    answer_ids, answer_atts, k
                )
            final_ids[open_mask] = open_ids
            final_probs[open_mask] = open_probs

        return final_ids, final_probs
    
    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):
        """Same two-stage ranking as original"""
        if question_states is None:
            raise RuntimeError("Decoder ranking requires BERT fusion states")
        
        num_ques = question_states.size(0)
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)
        
        start_output = self.decoder(
            start_ids, 
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,                                      
            return_dict=True, 
            reduction="none"
        )              
        logits = start_output.logits[:, 0, :]
        
        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(dim=1, index=answer_first_token) 
        topk_probs, topk_ids = prob_first_token.topk(k, dim=1) 
        
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids, dim=0)  
        input_atts = torch.cat(input_atts, dim=0)  

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)
        
        output = self.decoder(
            input_ids, 
            attention_mask=input_atts, 
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,     
            labels=targets_ids,
            return_dict=True, 
            reduction="none"
        )
        
        # Compute log-likelihood for each token (BUG FIX: was using raw logits!)
        logits = output.logits  # [batch*k, seq_len, vocab_size]
        log_probs_tokens = F.log_softmax(logits, dim=-1)
        
        # Gather log probabilities of target tokens
        # First, replace -100 (padding) with 0 to avoid index errors
        targets_for_gather = targets_ids.clone()
        targets_for_gather[targets_ids == -100] = 0
        
        targets_ids_expanded = targets_for_gather.unsqueeze(-1)
        log_probs_targets = log_probs_tokens.gather(dim=-1, index=targets_ids_expanded).squeeze(-1)
        
        # Mask padding tokens (set their log prob to 0)
        log_probs_targets = log_probs_targets.masked_fill(targets_ids == -100, 0)
        
        # Negative log-likelihood
        answer_loss = -log_probs_targets  # [batch*k, seq_len]
        
        topk_probs = topk_probs.view(-1, 1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss], dim=1)

        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques, k)

        topk_probs = F.softmax(log_probs_sum, dim=-1)
        topk_probs, rerank_id = topk_probs.topk(k, dim=1) 
        topk_ids = torch.gather(topk_ids, 1, rerank_id)    

        return topk_ids, topk_probs

