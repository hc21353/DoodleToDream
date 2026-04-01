from __future__ import annotations



from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        dead_code_threshold: float = 0.5,
        reset_interval: int = 200,
        commitment_warmup_steps: int = 500,
        init_batch_samples: int = 4096,
    ) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)
        self.ema_decay = float(ema_decay)
        self.ema_eps = float(ema_eps)
        self.dead_code_threshold = float(dead_code_threshold)
        self.reset_interval = int(reset_interval)
        self.commitment_warmup_steps = int(commitment_warmup_steps)
        self.init_batch_samples = int(init_batch_samples)

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        nn.init.uniform_(self.embedding.weight, -0.02, 0.02)

        self.register_buffer("ema_cluster_size", torch.zeros(self.num_embeddings))
        self.register_buffer("ema_embed_avg", torch.zeros(self.num_embeddings, self.embedding_dim))
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.long))
        self.register_buffer("is_initialized", torch.tensor(False))

    def _maybe_init_from_batch(self, flat32: torch.Tensor) -> None:
        if bool(self.is_initialized.item()):
            return
        if flat32.numel() == 0:
            return
        with torch.no_grad():
            n = flat32.size(0)
            sample_count = min(max(self.num_embeddings, 1), n, self.init_batch_samples)
            if n >= self.num_embeddings:
                perm = torch.randperm(n, device=flat32.device)[:self.num_embeddings]
                chosen = flat32[perm]
            else:
                repeats = math.ceil(self.num_embeddings / max(n, 1))
                chosen = flat32.repeat(repeats, 1)[:self.num_embeddings]
            self.embedding.weight.data.copy_(chosen.to(self.embedding.weight.dtype))
            self.ema_embed_avg.data.copy_(chosen.to(self.ema_embed_avg.dtype))
            self.ema_cluster_size.data.fill_(1.0)
            self.is_initialized.fill_(True)

    def _effective_commitment(self) -> float:
        step = int(self.num_updates.item())
        if self.commitment_warmup_steps <= 0:
            return self.commitment_cost
        alpha = min(1.0, float(step + 1) / float(self.commitment_warmup_steps))
        return float(self.commitment_cost) * alpha

    def _refresh_dead_codes(self, flat32: torch.Tensor) -> None:
        if flat32.numel() == 0:
            return
        dead = self.ema_cluster_size < float(self.dead_code_threshold)
        dead_count = int(dead.sum().item())
        if dead_count <= 0:
            return
        with torch.no_grad():
            n = int(flat32.size(0))
            if n <= 0:
                return




            if n >= dead_count:
                perm = torch.randperm(n, device=flat32.device)[:dead_count]
            else:
                perm = torch.randint(0, n, (dead_count,), device=flat32.device)

            new_codes = flat32[perm]
            if new_codes.size(0) != dead_count:
                raise RuntimeError(
                    f"dead-code refresh mismatch: dead_count={dead_count}, sampled={new_codes.size(0)}"
                )

            dead_idx = dead.nonzero(as_tuple=False).squeeze(1)
            self.embedding.weight.data[dead_idx] = new_codes.to(self.embedding.weight.dtype)
            self.ema_embed_avg.data[dead_idx] = new_codes.to(self.ema_embed_avg.dtype)
            self.ema_cluster_size.data[dead_idx] = 1.0

    def _quantize_valid(self, flat_valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        flat32 = torch.nan_to_num(flat_valid.float(), nan=0.0, posinf=0.0, neginf=0.0)
        self._maybe_init_from_batch(flat32)

        emb32 = torch.nan_to_num(self.embedding.weight.float(), nan=0.0, posinf=0.0, neginf=0.0)
        distances = (
            flat32.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat32 @ emb32.t()
            + emb32.pow(2).sum(dim=1, keepdim=True).t()
        )
        distances = torch.nan_to_num(distances, nan=1e6, posinf=1e6, neginf=1e6)
        indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(indices)

        one_hot = F.one_hot(indices, num_classes=self.num_embeddings).float()
        avg_probs = one_hot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        usage = float((avg_probs > 0).float().sum().item()) / float(self.num_embeddings)

        if self.training:
            with torch.no_grad():
                self.num_updates.add_(1)
                cluster_size = one_hot.sum(dim=0)
                embed_sum = one_hot.t() @ flat32

                self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=(1.0 - self.ema_decay))
                self.ema_embed_avg.mul_(self.ema_decay).add_(embed_sum, alpha=(1.0 - self.ema_decay))

                n = self.ema_cluster_size.sum()
                cluster_size_norm = (
                    (self.ema_cluster_size + self.ema_eps)
                    / (n + self.num_embeddings * self.ema_eps)
                    * n
                ).clamp(min=self.ema_eps)
                embed_normalized = self.ema_embed_avg / cluster_size_norm.unsqueeze(1)
                embed_normalized = torch.nan_to_num(embed_normalized, nan=0.0, posinf=0.0, neginf=0.0)
                self.embedding.weight.data.copy_(embed_normalized.to(self.embedding.weight.dtype))

                if self.reset_interval > 0 and (int(self.num_updates.item()) % self.reset_interval == 0):
                    self._refresh_dead_codes(flat32)

        commitment = self._effective_commitment()
        loss = commitment * F.smooth_l1_loss(flat32, quantized.detach().float())

        info = {
            "perplexity": float(perplexity.item()),
            "codebook_usage": usage,
        }
        return quantized, indices, loss, info

    def forward(self, z_e: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        if mask is None:
            flat = z_e.reshape(-1, self.embedding_dim)
            quantized, indices, loss, info = self._quantize_valid(flat)
            quantized_st = flat + (quantized - flat).detach()
            return quantized_st.view_as(z_e), indices.view(z_e.shape[:-1]), loss, info

        flat = z_e.reshape(-1, self.embedding_dim)
        valid = mask.reshape(-1).bool()
        if int(valid.sum().item()) == 0:
            q = z_e.clone()
            idx = torch.zeros(z_e.shape[:-1], dtype=torch.long, device=z_e.device)
            zero = z_e.new_tensor(0.0)
            return q, idx, zero, {"perplexity": 0.0, "codebook_usage": 0.0}

        flat_valid = flat[valid]
        quantized_valid, indices_valid, loss, info = self._quantize_valid(flat_valid)
        flat_out = flat.clone()
        flat_out[valid] = flat_valid + (quantized_valid - flat_valid).detach()
        idx_full = torch.zeros((flat.shape[0],), dtype=torch.long, device=z_e.device)
        idx_full[valid] = indices_valid
        return flat_out.view_as(z_e), idx_full.view(z_e.shape[:-1]), loss, info

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(indices.clamp(min=0, max=self.num_embeddings - 1))

class CoordConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        yy = torch.linspace(-1.0, 1.0, steps=h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, -1, -1, w)
        xx = torch.linspace(-1.0, 1.0, steps=w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, -1, h, -1)
        return self.conv(torch.cat([x, xx, yy], dim=1))

class ConvEncoder(nn.Module):
    def __init__(self, hidden_dims: List[int], latent_dim: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = 1
        for idx, out_ch in enumerate(hidden_dims):
            conv = CoordConv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1) if idx == 0 else nn.Conv2d(
                in_ch, out_ch, kernel_size=4, stride=2, padding=1
            )
            layers.extend([conv, nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)])
            in_ch = out_ch
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_ch, latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.body(x)
        return self.head(feat)

class DeconvTrunk(nn.Module):
    def __init__(self, hidden_dims: List[int], latent_dim: int, output_size: int = 64) -> None:
        super().__init__()
        start_ch = hidden_dims[-1]
        self.fc = nn.Sequential(nn.Linear(latent_dim, start_ch * 4 * 4), nn.ReLU(inplace=True))
        channels = list(reversed(hidden_dims))
        layers: List[nn.Module] = []
        in_ch = channels[0]
        spatial = 4
        for out_ch in channels[1:]:
            layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch
            spatial *= 2
        while spatial < output_size:
            next_ch = max(in_ch // 2, 16)
            layers.extend([
                nn.ConvTranspose2d(in_ch, next_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(next_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = next_ch
            spatial *= 2
        self.body = nn.Sequential(*layers)
        self.out_channels = in_ch

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(z.size(0), -1, 4, 4)
        return self.body(x)

class ShapeStrokeEmbeddingAE(nn.Module):
    def __init__(self, hidden_dims: List[int], embedding_dim: int, image_size: int = 64) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.image_size = int(image_size)
        self.encoder = ConvEncoder(hidden_dims=hidden_dims, latent_dim=embedding_dim)
        self.decoder_trunk = DeconvTrunk(hidden_dims=hidden_dims, latent_dim=embedding_dim, output_size=image_size)
        trunk_ch = self.decoder_trunk.out_channels
        self.bitmap_head = nn.Conv2d(trunk_ch, 1, kernel_size=3, padding=1)
        self.dist_head = nn.Conv2d(trunk_ch, 1, kernel_size=3, padding=1)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def decode_logits_from_embedding(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.decoder_trunk(embeddings)
        bitmap_logits = self.bitmap_head(feat)
        dist_raw = self.dist_head(feat)
        return bitmap_logits, dist_raw

    def decode_from_embedding(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        bitmap_logits, dist_raw = self.decode_logits_from_embedding(embeddings)
        return {
            "bitmap_logits": bitmap_logits,
            "dist_raw": dist_raw,
            "bitmap": torch.sigmoid(bitmap_logits),
            "dist_map": torch.sigmoid(dist_raw),
        }

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        emb = self.encode(images)
        out = self.decode_from_embedding(emb)
        out["embedding"] = emb
        return out

class LocationBBoxEmbeddingAE(nn.Module):
    def __init__(self, hidden_dims: List[int], embedding_dim: int) -> None:
        super().__init__()
        dims = [4] + list(hidden_dims) + [embedding_dim]
        enc: List[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            enc.extend([nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True)])
        enc.append(nn.Linear(dims[-2], dims[-1]))
        self.encoder = nn.Sequential(*enc)

        dec_dims = [embedding_dim] + list(reversed(hidden_dims)) + [4]
        dec: List[nn.Module] = []
        for in_dim, out_dim in zip(dec_dims[:-2], dec_dims[1:-1]):
            dec.extend([nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True)])
        dec.append(nn.Linear(dec_dims[-2], dec_dims[-1]))
        self.decoder = nn.Sequential(*dec)

    def encode(self, vectors: torch.Tensor) -> torch.Tensor:
        return self.encoder(vectors)

    def decode_from_embedding(self, embeddings: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decoder(embeddings))

    def forward(self, vectors: torch.Tensor) -> Dict[str, torch.Tensor]:
        emb = self.encode(vectors)
        recon = self.decode_from_embedding(emb)
        return {"embedding": emb, "recon": recon}


class _ConvNormAct1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.norm2(out)
        x = self.act(x + out)
        if mask is not None:
            x = x * mask
        return x

class SequenceTokenizerVQ(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_embeddings: int,
        max_len: int,
        commitment_cost: float = 0.25,
        num_heads: int = 0,
        num_layers: int = 4,
        decoder_num_layers: int | None = None,
        ff_dim: int = 0,
        dropout: float = 0.1,
        kernel_size: int = 5,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        dead_code_threshold: float = 0.5,
        reset_interval: int = 200,
        commitment_warmup_steps: int = 500,
        init_batch_samples: int = 4096,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)
        self.max_len = int(max_len)

        self.input_proj = nn.Conv1d(self.input_dim, self.model_dim, kernel_size=1)
        self.input_norm = nn.GroupNorm(1, self.model_dim)
        self.position_embedding = nn.Embedding(max(1, self.max_len), self.model_dim)
        self.encoder = nn.ModuleList([
            _ConvNormAct1D(self.model_dim, kernel_size=kernel_size, dropout=dropout)
            for _ in range(int(num_layers))
        ])
        self.pre_vq = nn.Sequential(
            nn.Conv1d(self.model_dim, self.model_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(self.model_dim, self.model_dim, kernel_size=1),
        )
        self.vq = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=self.model_dim,
            commitment_cost=commitment_cost,
            ema_decay=ema_decay,
            ema_eps=ema_eps,
            dead_code_threshold=dead_code_threshold,
            reset_interval=reset_interval,
            commitment_warmup_steps=commitment_warmup_steps,
            init_batch_samples=init_batch_samples,
        )

        dec_layers = int(num_layers if decoder_num_layers is None else decoder_num_layers)
        self.post_vq = nn.Conv1d(self.model_dim, self.model_dim, kernel_size=1)
        self.decoder = nn.ModuleList([
            _ConvNormAct1D(self.model_dim, kernel_size=kernel_size, dropout=dropout)
            for _ in range(dec_layers)
        ])
        self.local_head = nn.Conv1d(self.model_dim, self.input_dim, kernel_size=1)
        self.output_head = nn.Conv1d(self.model_dim, self.input_dim, kernel_size=1)

    @staticmethod
    def _mask_1d(valid_mask: torch.Tensor, channels_first: bool = True) -> torch.Tensor:
        if channels_first:
            return valid_mask.float().unsqueeze(1)
        return valid_mask.float().unsqueeze(-1)

    def _position_cf(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        pos = torch.arange(seq_len, device=device)
        if self.max_len > 0:
            pos = pos.clamp(max=self.max_len - 1)
        pos = self.position_embedding(pos).transpose(0, 1).unsqueeze(0)
        return pos.to(dtype=dtype)

    def encode_hidden(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        x = features.transpose(1, 2)
        mask_cf = self._mask_1d(valid_mask, channels_first=True)
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = x + self._position_cf(features.size(1), features.device, x.dtype)
        x = torch.clamp(x, -8.0, 8.0)
        x = x * mask_cf
        for block in self.encoder:
            x = block(x, mask_cf)
            x = torch.clamp(x, -8.0, 8.0)
        x = torch.tanh(self.pre_vq(x)) * mask_cf
        return x.transpose(1, 2)

    def decode_quantized(self, z_q: torch.Tensor, valid_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = z_q.transpose(1, 2)
        mask_cf = self._mask_1d(valid_mask, channels_first=True)
        x = (x + self._position_cf(z_q.size(1), z_q.device, x.dtype)) * mask_cf
        local_recon = self.local_head(x).transpose(1, 2) * self._mask_1d(valid_mask, channels_first=False)
        x = self.post_vq(x) * mask_cf
        x = torch.clamp(x, -8.0, 8.0)
        for block in self.decoder:
            x = block(x, mask_cf)
            x = torch.clamp(x, -8.0, 8.0)
        recon_seq = self.output_head(x).transpose(1, 2) * self._mask_1d(valid_mask, channels_first=False)
        return {"recon_seq": recon_seq, "recon_local": local_recon}

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> Dict[str, torch.Tensor | float]:
        hidden = self.encode_hidden(features, valid_mask)
        z_q, indices, vq_loss, info = self.vq(hidden, mask=valid_mask.bool())
        dec = self.decode_quantized(z_q, valid_mask)
        return {
            "recon": dec["recon_seq"],
            "recon_seq": dec["recon_seq"],
            "recon_local": dec["recon_local"],
            "indices": indices,
            "vq_loss": vq_loss,
            **info,
        }

    def decode_sequence_indices(self, indices: torch.Tensor, valid_mask: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        squeeze = False
        if indices.ndim == 1:
            indices = indices.unsqueeze(0)
            squeeze = True
        if valid_mask is None:
            valid_mask = torch.ones(indices.shape, device=indices.device, dtype=torch.float32)
        elif valid_mask.ndim == 1:
            valid_mask = valid_mask.unsqueeze(0)
        safe_indices = indices.clamp(min=0, max=self.vq.num_embeddings - 1)
        z_q = self.vq.decode_indices(safe_indices)
        dec = self.decode_quantized(z_q, valid_mask.float())
        if squeeze:
            return {k: v.squeeze(0) for k, v in dec.items()}
        return dec

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        valid_mask = torch.ones(indices.shape, device=indices.device, dtype=torch.float32)
        return self.decode_sequence_indices(indices, valid_mask)["recon_seq"]


class LocationMLPTokenizerVQ(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_embeddings: int,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        dead_code_threshold: float = 0.5,
        reset_interval: int = 200,
        commitment_warmup_steps: int = 500,
        init_batch_samples: int = 4096,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
        )
        self.pre_vq = nn.Linear(self.model_dim, self.model_dim)
        self.vq = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=self.model_dim,
            commitment_cost=commitment_cost,
            ema_decay=ema_decay,
            ema_eps=ema_eps,
            dead_code_threshold=dead_code_threshold,
            reset_interval=reset_interval,
            commitment_warmup_steps=commitment_warmup_steps,
            init_batch_samples=init_batch_samples,
        )
        self.local_head = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.input_dim),
        )
        self.output_head = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.input_dim),
        )

    def encode_hidden(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(features)
        x = torch.tanh(self.pre_vq(x))
        x = x * valid_mask.unsqueeze(-1)
        return x

    def decode_quantized(self, z_q: torch.Tensor, valid_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        local_recon = self.local_head(z_q) * valid_mask.unsqueeze(-1)
        recon_seq = self.output_head(z_q) * valid_mask.unsqueeze(-1)
        return {"recon_seq": recon_seq, "recon_local": local_recon}

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> Dict[str, torch.Tensor | float]:
        hidden = self.encode_hidden(features, valid_mask)
        z_q, indices, vq_loss, info = self.vq(hidden, mask=valid_mask.bool())
        dec = self.decode_quantized(z_q, valid_mask)
        return {
            "recon": dec["recon_seq"],
            "recon_seq": dec["recon_seq"],
            "recon_local": dec["recon_local"],
            "indices": indices,
            "vq_loss": vq_loss,
            **info,
        }

    def decode_sequence_indices(self, indices: torch.Tensor, valid_mask: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        squeeze = False
        if indices.ndim == 1:
            indices = indices.unsqueeze(0)
            squeeze = True
        if valid_mask is None:
            valid_mask = torch.ones(indices.shape, device=indices.device, dtype=torch.float32)
        elif valid_mask.ndim == 1:
            valid_mask = valid_mask.unsqueeze(0)
        safe_indices = indices.clamp(min=0, max=self.vq.num_embeddings - 1)
        z_q = self.vq.decode_indices(safe_indices)
        dec = self.decode_quantized(z_q, valid_mask.float())
        if squeeze:
            return {k: v.squeeze(0) for k, v in dec.items()}
        return dec

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        valid_mask = torch.ones(indices.shape, device=indices.device, dtype=torch.float32)
        return self.decode_sequence_indices(indices, valid_mask)["recon_seq"]




class DirectBBoxQuantizer(nn.Module):
    def __init__(self, bins_w: int = 6, bins_h: int = 6, bins_cx: int = 8, bins_cy: int = 8) -> None:
        super().__init__()
        self.bins_w = int(bins_w)
        self.bins_h = int(bins_h)
        self.bins_cx = int(bins_cx)
        self.bins_cy = int(bins_cy)

        w_centers = (torch.arange(self.bins_w, dtype=torch.float32) + 0.5) / float(self.bins_w)
        h_centers = (torch.arange(self.bins_h, dtype=torch.float32) + 0.5) / float(self.bins_h)
        cx_centers = (torch.arange(self.bins_cx, dtype=torch.float32) + 0.5) / float(self.bins_cx)
        cy_centers = (torch.arange(self.bins_cy, dtype=torch.float32) + 0.5) / float(self.bins_cy)

        grid = torch.stack(
            torch.meshgrid(w_centers, h_centers, cx_centers, cy_centers, indexing="ij"),
            dim=-1,
        ).reshape(-1, 4)
        self.register_buffer("codebook", grid)
        self.is_direct_bbox_quantizer = True

    @property
    def num_embeddings(self) -> int:
        return int(self.codebook.size(0))

    def encode_bboxes(self, bboxes: torch.Tensor) -> torch.Tensor:
        b = torch.clamp(bboxes, 0.0, 1.0)
        w_idx = torch.clamp((b[..., 0] * self.bins_w).floor().long(), 0, self.bins_w - 1)
        h_idx = torch.clamp((b[..., 1] * self.bins_h).floor().long(), 0, self.bins_h - 1)
        cx_idx = torch.clamp((b[..., 2] * self.bins_cx).floor().long(), 0, self.bins_cx - 1)
        cy_idx = torch.clamp((b[..., 3] * self.bins_cy).floor().long(), 0, self.bins_cy - 1)
        idx = (((w_idx * self.bins_h) + h_idx) * self.bins_cx + cx_idx) * self.bins_cy + cy_idx
        return idx.long()

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        safe = torch.clamp(indices.long(), 0, self.num_embeddings - 1)
        flat = safe.reshape(-1)
        decoded = self.codebook.index_select(0, flat)
        return decoded.view(*safe.shape, 4)

    def decode_sequence_indices(self, indices: torch.Tensor, valid_mask: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        decoded = self.decode_indices(indices)
        return {"recon_seq": decoded, "recon_local": decoded}

class StrokeAutoregressiveGenerator(nn.Module):

    def __init__(
        self,
        num_classes: int,
        shape_vocab_size: int,
        loc_vocab_size: int,
        max_strokes: int,
        shape_codebook: torch.Tensor,
        loc_codebook: torch.Tensor,
        loc_bins_w: int,
        loc_bins_h: int,
        loc_bins_cx: int,
        loc_bins_cy: int,
        model_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        token_residual_scale: float = 0.10,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.shape_vocab_size = int(shape_vocab_size)
        self.loc_vocab_size = int(loc_vocab_size)
        self.max_strokes = int(max_strokes)
        self.model_dim = int(model_dim)
        self.token_residual_scale = float(token_residual_scale)






        self.shape_end = self.shape_vocab_size
        self.shape_start = self.shape_vocab_size + 1
        self.shape_pad = self.shape_vocab_size + 2
        self.loc_end = self.loc_vocab_size
        self.loc_start = self.loc_vocab_size + 1
        self.loc_pad = self.loc_vocab_size + 2
        self.loc_bins_w = int(loc_bins_w)
        self.loc_bins_h = int(loc_bins_h)
        self.loc_bins_cx = int(loc_bins_cx)
        self.loc_bins_cy = int(loc_bins_cy)






        self.loc_w_end = self.loc_bins_w
        self.loc_w_start = self.loc_bins_w + 1
        self.loc_w_pad = self.loc_bins_w + 2

        self.loc_h_end = self.loc_bins_h
        self.loc_h_start = self.loc_bins_h + 1
        self.loc_h_pad = self.loc_bins_h + 2

        self.loc_cx_end = self.loc_bins_cx
        self.loc_cx_start = self.loc_bins_cx + 1
        self.loc_cx_pad = self.loc_bins_cx + 2

        self.loc_cy_end = self.loc_bins_cy
        self.loc_cy_start = self.loc_bins_cy + 1
        self.loc_cy_pad = self.loc_bins_cy + 2

        shape_codebook = torch.as_tensor(shape_codebook, dtype=torch.float32)
        loc_codebook = torch.as_tensor(loc_codebook, dtype=torch.float32)
        if shape_codebook.ndim != 2 or int(shape_codebook.size(0)) != self.shape_vocab_size:
            raise ValueError(
                f"shape_codebook must have shape [{self.shape_vocab_size}, D], got {tuple(shape_codebook.shape)}"
            )
        if loc_codebook.ndim != 2 or int(loc_codebook.size(0)) != self.loc_vocab_size:
            raise ValueError(
                f"loc_codebook must have shape [{self.loc_vocab_size}, D], got {tuple(loc_codebook.shape)}"
            )

        self.shape_code_dim = int(shape_codebook.size(1))
        self.loc_code_dim = int(loc_codebook.size(1))
        self.register_buffer("shape_codebook", shape_codebook.clone())
        self.register_buffer("loc_codebook", loc_codebook.clone())

        self.shape_codebook_proj = nn.Sequential(
            nn.LayerNorm(self.shape_code_dim),
            nn.Linear(self.shape_code_dim, model_dim),
        )
        self.loc_codebook_proj = nn.Sequential(
            nn.LayerNorm(self.loc_code_dim),
            nn.Linear(self.loc_code_dim, model_dim),
        )
        self.shape_special_embedding = nn.Embedding(3, model_dim)
        self.loc_special_embedding = nn.Embedding(3, model_dim)
        self.shape_residual_embedding = nn.Embedding(self.shape_vocab_size + 3, model_dim)
        self.loc_residual_embedding = nn.Embedding(self.loc_vocab_size + 3, model_dim)
        nn.init.normal_(self.shape_residual_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.loc_residual_embedding.weight, mean=0.0, std=0.02)

        self.class_embedding = nn.Embedding(self.num_classes, model_dim)
        self.class_global_bias_proj = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
        )
        self.class_film_proj = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim * 2),
        )
        self.position_embedding = nn.Embedding(self.max_strokes + 1, model_dim)
        self.input_norm = nn.LayerNorm(model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(model_dim)
        self.class_output_norm = nn.LayerNorm(model_dim)


        self.shape_head = nn.Linear(model_dim, self.shape_vocab_size + 1)


        self.loc_condition_proj = nn.Sequential(
            nn.LayerNorm(model_dim * 2),
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )


        self.loc_w_head = nn.Linear(model_dim, self.loc_bins_w + 1)
        self.loc_h_head = nn.Linear(model_dim, self.loc_bins_h + 1)
        self.loc_cx_head = nn.Linear(model_dim, self.loc_bins_cx + 1)
        self.loc_cy_head = nn.Linear(model_dim, self.loc_bins_cy + 1)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)

    def _embed_tokens(
        self,
        tokens: torch.Tensor,
        vocab_size: int,
        codebook: torch.Tensor,
        projector: nn.Module,
        special_embedding: nn.Embedding,
        residual_embedding: nn.Embedding,
    ) -> torch.Tensor:
        bsz, seq_len = tokens.shape
        base_dtype = self.position_embedding.weight.dtype
        out = torch.zeros((bsz, seq_len, self.model_dim), device=tokens.device, dtype=base_dtype)

        normal_mask = tokens < vocab_size
        if normal_mask.any():
            normal_ids = tokens[normal_mask].long()
            code_feats = codebook.index_select(0, normal_ids)
            projected = projector(code_feats).to(dtype=base_dtype)
            out[normal_mask] = projected

        special_mask = ~normal_mask
        if special_mask.any():
            special_ids = (tokens[special_mask] - vocab_size).clamp(min=0, max=2).long()
            special = special_embedding(special_ids).to(dtype=base_dtype)
            out[special_mask] = special

        residual_ids = tokens.clamp(min=0, max=residual_embedding.num_embeddings - 1).long()
        residual = residual_embedding(residual_ids).to(dtype=base_dtype)
        return out + self.token_residual_scale * residual

    def _encode_prefix(self, class_ids: torch.Tensor, input_shape: torch.Tensor, input_loc: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_shape.shape
        positions = torch.arange(seq_len, device=input_shape.device).unsqueeze(0).expand(bsz, -1)
        base_dtype = self.position_embedding.weight.dtype
        class_embed = self.class_embedding(class_ids).to(dtype=base_dtype)
        class_bias = self.class_global_bias_proj(class_embed).unsqueeze(1)
        x = (
            self._embed_tokens(
                input_shape,
                vocab_size=self.shape_vocab_size,
                codebook=self.shape_codebook,
                projector=self.shape_codebook_proj,
                special_embedding=self.shape_special_embedding,
                residual_embedding=self.shape_residual_embedding,
            )
            + self._embed_tokens(
                input_loc,
                vocab_size=self.loc_vocab_size,
                codebook=self.loc_codebook,
                projector=self.loc_codebook_proj,
                special_embedding=self.loc_special_embedding,
                residual_embedding=self.loc_residual_embedding,
            )
            + self.position_embedding(positions)
            + class_bias
        )
        x = self.input_norm(x)
        causal_mask = self._causal_mask(seq_len, input_shape.device)
        pad_mask = (input_shape == self.shape_pad) & (input_loc == self.loc_pad)
        hidden = self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        hidden = self.output_norm(hidden)
        class_scale, class_shift = self.class_film_proj(class_embed).chunk(2, dim=-1)
        class_scale = torch.tanh(class_scale).unsqueeze(1)
        class_shift = class_shift.unsqueeze(1)
        hidden = hidden * (1.0 + class_scale) + class_shift
        return self.class_output_norm(hidden)

    def _loc_logits_from_hidden(self, hidden: torch.Tensor, current_shape_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        shape_condition = self._embed_tokens(
            current_shape_tokens,
            vocab_size=self.shape_vocab_size,
            codebook=self.shape_codebook,
            projector=self.shape_codebook_proj,
            special_embedding=self.shape_special_embedding,
            residual_embedding=self.shape_residual_embedding,
        )
        loc_hidden = self.loc_condition_proj(torch.cat([hidden, shape_condition], dim=-1))
        return {
            "w": self.loc_w_head(loc_hidden),
            "h": self.loc_h_head(loc_hidden),
            "cx": self.loc_cx_head(loc_hidden),
            "cy": self.loc_cy_head(loc_hidden),
        }

    def _decompose_joint_loc_tokens(self, loc_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        loc_tokens = loc_tokens.long()

        w_tok = torch.full_like(loc_tokens, fill_value=self.loc_w_pad)
        h_tok = torch.full_like(loc_tokens, fill_value=self.loc_h_pad)
        cx_tok = torch.full_like(loc_tokens, fill_value=self.loc_cx_pad)
        cy_tok = torch.full_like(loc_tokens, fill_value=self.loc_cy_pad)

        normal_mask = loc_tokens < self.loc_vocab_size
        if normal_mask.any():
            flat = loc_tokens[normal_mask]

            cy_ids = flat % self.loc_bins_cy
            flat = flat // self.loc_bins_cy

            cx_ids = flat % self.loc_bins_cx
            flat = flat // self.loc_bins_cx

            h_ids = flat % self.loc_bins_h
            w_ids = flat // self.loc_bins_h

            w_tok[normal_mask] = w_ids
            h_tok[normal_mask] = h_ids
            cx_tok[normal_mask] = cx_ids
            cy_tok[normal_mask] = cy_ids

        end_mask = loc_tokens == self.loc_end
        start_mask = loc_tokens == self.loc_start
        pad_mask = loc_tokens == self.loc_pad

        w_tok[end_mask] = self.loc_w_end
        h_tok[end_mask] = self.loc_h_end
        cx_tok[end_mask] = self.loc_cx_end
        cy_tok[end_mask] = self.loc_cy_end

        w_tok[start_mask] = self.loc_w_start
        h_tok[start_mask] = self.loc_h_start
        cx_tok[start_mask] = self.loc_cx_start
        cy_tok[start_mask] = self.loc_cy_start

        w_tok[pad_mask] = self.loc_w_pad
        h_tok[pad_mask] = self.loc_h_pad
        cx_tok[pad_mask] = self.loc_cx_pad
        cy_tok[pad_mask] = self.loc_cy_pad

        return {
            "w": w_tok.long(),
            "h": h_tok.long(),
            "cx": cx_tok.long(),
            "cy": cy_tok.long(),
        }


    def _compose_joint_loc_tokens(
        self,
        w_tokens: torch.Tensor,
        h_tokens: torch.Tensor,
        cx_tokens: torch.Tensor,
        cy_tokens: torch.Tensor,
    ) -> torch.Tensor:
        w_tokens = w_tokens.long()
        h_tokens = h_tokens.long()
        cx_tokens = cx_tokens.long()
        cy_tokens = cy_tokens.long()

        out = torch.full_like(w_tokens, fill_value=self.loc_pad)

        normal_mask = (
            (w_tokens >= 0) & (w_tokens < self.loc_bins_w) &
            (h_tokens >= 0) & (h_tokens < self.loc_bins_h) &
            (cx_tokens >= 0) & (cx_tokens < self.loc_bins_cx) &
            (cy_tokens >= 0) & (cy_tokens < self.loc_bins_cy)
        )
        if normal_mask.any():
            out[normal_mask] = (
                (
                    (
                        w_tokens[normal_mask] * self.loc_bins_h
                        + h_tokens[normal_mask]
                    ) * self.loc_bins_cx
                    + cx_tokens[normal_mask]
                ) * self.loc_bins_cy
                + cy_tokens[normal_mask]
            )

        end_mask = (
            (w_tokens == self.loc_w_end) &
            (h_tokens == self.loc_h_end) &
            (cx_tokens == self.loc_cx_end) &
            (cy_tokens == self.loc_cy_end)
        )
        start_mask = (
            (w_tokens == self.loc_w_start) &
            (h_tokens == self.loc_h_start) &
            (cx_tokens == self.loc_cx_start) &
            (cy_tokens == self.loc_cy_start)
        )
        pad_mask = (
            (w_tokens == self.loc_w_pad) &
            (h_tokens == self.loc_h_pad) &
            (cx_tokens == self.loc_cx_pad) &
            (cy_tokens == self.loc_cy_pad)
        )

        out[end_mask] = self.loc_end
        out[start_mask] = self.loc_start
        out[pad_mask] = self.loc_pad
        return out.long()


    def _predict_joint_loc_tokens_from_hidden(
        self,
        hidden: torch.Tensor,
        current_shape_tokens: torch.Tensor,
    ) -> torch.Tensor:
        loc_logits = self._loc_logits_from_hidden(hidden, current_shape_tokens)
        pred_w = loc_logits["w"].argmax(dim=-1)
        pred_h = loc_logits["h"].argmax(dim=-1)
        pred_cx = loc_logits["cx"].argmax(dim=-1)
        pred_cy = loc_logits["cy"].argmax(dim=-1)
        return self._compose_joint_loc_tokens(pred_w, pred_h, pred_cx, pred_cy)

    def forward(
        self,
        class_ids: torch.Tensor,
        input_shape: torch.Tensor,
        input_loc: torch.Tensor,
        loc_condition_shape_tokens: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        hidden = self._encode_prefix(class_ids=class_ids, input_shape=input_shape, input_loc=input_loc)
        shape_logits = self.shape_head(hidden)

        if loc_condition_shape_tokens is None:
            loc_condition_shape_tokens = torch.argmax(shape_logits.detach(), dim=-1)

        loc_logits = self._loc_logits_from_hidden(hidden, loc_condition_shape_tokens.long())
        loc_joint_pred = self._predict_joint_loc_tokens_from_hidden(hidden, loc_condition_shape_tokens.long())

        return {
            "shape_logits": shape_logits,
            "loc_logits": loc_logits,
            "loc_joint_pred": loc_joint_pred,
            "hidden": hidden,
        }

    @torch.no_grad()
    def sample(
        self,
        class_ids: torch.Tensor,
        max_steps: int | None = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = class_ids.device
        batch_size = class_ids.shape[0]
        max_steps = int(max_steps or self.max_strokes)

        input_shape = torch.full((batch_size, 1), fill_value=self.shape_start, device=device, dtype=torch.long)
        input_loc = torch.full((batch_size, 1), fill_value=self.loc_start, device=device, dtype=torch.long)

        finished = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        generated_shape: List[torch.Tensor] = []
        generated_loc: List[torch.Tensor] = []
        lengths = torch.zeros((batch_size,), dtype=torch.long, device=device)

        for step in range(max_steps + 1):
            out = self.forward(class_ids=class_ids, input_shape=input_shape, input_loc=input_loc)
            hidden_last = out["hidden"][:, -1:, :]
            shape_logits = out["shape_logits"][:, -1, :]
            next_shape = nucleus_sample(shape_logits, temperature=temperature, top_p=top_p)

            loc_logits = self._loc_logits_from_hidden(hidden_last, next_shape.unsqueeze(1))
            next_w = nucleus_sample(loc_logits["w"][:, -1, :], temperature=temperature, top_p=top_p)
            next_h = nucleus_sample(loc_logits["h"][:, -1, :], temperature=temperature, top_p=top_p)
            next_cx = nucleus_sample(loc_logits["cx"][:, -1, :], temperature=temperature, top_p=top_p)
            next_cy = nucleus_sample(loc_logits["cy"][:, -1, :], temperature=temperature, top_p=top_p)

            loc_all_end = (
                (next_w == self.loc_w_end) &
                (next_h == self.loc_h_end) &
                (next_cx == self.loc_cx_end) &
                (next_cy == self.loc_cy_end)
            )
            loc_any_end = (
                (next_w == self.loc_w_end) |
                (next_h == self.loc_h_end) |
                (next_cx == self.loc_cx_end) |
                (next_cy == self.loc_cy_end)
            )
            loc_inconsistent_end = loc_any_end & (~loc_all_end)

            next_finished = (next_shape == self.shape_end) | loc_all_end | loc_inconsistent_end

            next_loc = self._compose_joint_loc_tokens(next_w, next_h, next_cx, next_cy)
            safe_shape = next_shape.clone()
            safe_loc = next_loc.clone()

            safe_shape[next_finished] = self.shape_end
            safe_loc[next_finished] = self.loc_end

            just_finished = (~finished) & next_finished
            lengths = torch.where(just_finished, torch.full_like(lengths, step), lengths)
            finished = finished | next_finished

            generated_shape.append(safe_shape)
            generated_loc.append(safe_loc)

            if finished.all():
                break

            append_shape = safe_shape.clone()
            append_loc = safe_loc.clone()
            append_shape[finished] = self.shape_pad
            append_loc[finished] = self.loc_pad

            input_shape = torch.cat([input_shape, append_shape.unsqueeze(1)], dim=1)
            input_loc = torch.cat([input_loc, append_loc.unsqueeze(1)], dim=1)

        if generated_shape:
            shape_seq = torch.stack(generated_shape, dim=1)
            loc_seq = torch.stack(generated_loc, dim=1)
        else:
            shape_seq = torch.empty((batch_size, 0), device=device, dtype=torch.long)
            loc_seq = torch.empty((batch_size, 0), device=device, dtype=torch.long)

        if not finished.all():
            lengths[~finished] = int(shape_seq.size(1))

        return shape_seq, loc_seq, lengths

def nucleus_sample(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 0.9) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    keep = cumulative <= top_p
    keep[..., 0] = True
    filtered = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
    filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    sampled = torch.multinomial(filtered, num_samples=1).squeeze(-1)
    return sorted_idx.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()

def shape_ae_recon_loss(
    bitmap_logits: torch.Tensor,
    dist_raw: torch.Tensor,
    target_bitmap: torch.Tensor,
    target_dist: torch.Tensor,
    bce_weight: float = 0.30,
    l1_weight: float = 0.50,
    dist_weight: float = 2.0,
    dice_weight: float = 0.0,
) -> torch.Tensor:
    loss = bitmap_logits.new_tensor(0.0)
    if bce_weight > 0:
        loss = loss + float(bce_weight) * F.binary_cross_entropy_with_logits(bitmap_logits, target_bitmap)
    if l1_weight > 0:
        loss = loss + float(l1_weight) * F.l1_loss(torch.sigmoid(bitmap_logits), target_bitmap)
    if dist_weight > 0:
        loss = loss + float(dist_weight) * F.l1_loss(torch.sigmoid(dist_raw), target_dist)
    if dice_weight > 0:
        loss = loss + float(dice_weight) * dice_loss_from_logits(bitmap_logits, target_bitmap)
    return loss

from pathlib import Path

from typing import Dict, List, Sequence

import imageio.v2 as imageio

import numpy as np

import torch

from torch import nn

from torch.amp import GradScaler, autocast

from tqdm import tqdm

def make_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def maybe_clip_grad(model: nn.Module, max_norm: float | None) -> None:
    if max_norm is not None and max_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

def _loader(dataset, batch_size: int, shuffle: bool, cfg: Dict) -> DataLoader:
    num_workers = int(cfg["project"].get("num_workers", 2))
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)

def build_shape_ae(cfg: Dict) -> ShapeStrokeEmbeddingAE:
    p = cfg["shape_ae"]
    return ShapeStrokeEmbeddingAE(
        hidden_dims=list(p["hidden_dims"]),
        embedding_dim=int(p["embedding_dim"]),
        image_size=int(cfg["dataset"]["image_size"]),
    )

def build_location_ae(cfg: Dict) -> LocationBBoxEmbeddingAE:
    p = cfg["location_ae"]
    return LocationBBoxEmbeddingAE(
        hidden_dims=list(p["hidden_dims"]),
        embedding_dim=int(p["embedding_dim"]),
    )

def build_shape_tokenizer(cfg: Dict) -> SequenceTokenizerVQ:
    p = cfg["shape_tokenizer"]
    return SequenceTokenizerVQ(
        input_dim=int(cfg["shape_ae"]["embedding_dim"]),
        model_dim=int(p["model_dim"]),
        num_embeddings=int(p["num_embeddings"]),
        max_len=int(cfg["dataset"]["max_strokes"]),
        commitment_cost=float(p["commitment_cost"]),
        num_layers=int(p["num_layers"]),
        decoder_num_layers=int(p["num_layers"]),
        dropout=float(p["dropout"]),
        kernel_size=int(p["kernel_size"]),
        ema_decay=float(p["ema_decay"]),
        ema_eps=float(p["ema_eps"]),
        dead_code_threshold=float(p["dead_code_threshold"]),
        reset_interval=int(p["reset_interval"]),
        commitment_warmup_steps=int(p["commitment_warmup_steps"]),
        init_batch_samples=int(p["init_batch_samples"]),
    )

def _extract_tokenizer_codebook(tokenizer, fallback_dim: int, fallback_size: int) -> torch.Tensor:
    if tokenizer is not None and hasattr(tokenizer, "codebook"):
        weight = torch.as_tensor(tokenizer.codebook, dtype=torch.float32).detach().cpu()
        if weight.ndim == 2 and int(weight.size(0)) == int(fallback_size):
            return weight
    if tokenizer is not None and hasattr(tokenizer, "vq") and hasattr(tokenizer.vq, "embedding"):
        weight = tokenizer.vq.embedding.weight.detach().float().cpu()
        if weight.ndim == 2 and int(weight.size(0)) == int(fallback_size):
            return weight
    return torch.randn(int(fallback_size), int(fallback_dim), dtype=torch.float32) * 0.02

def load_checkpoint_weights(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    payload = CheckpointIO.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state"])
    return model

def resolve_optional_ckpt(path_value: Any, model_name: str) -> Optional[Path]:
    if path_value is None:
        return None
    raw = str(path_value).strip()
    if not raw:
        return None
    path = Path(os.path.expanduser(raw))
    if not path.exists():
        raise FileNotFoundError(f"[{model_name}] checkpoint not found: {path}")
    return path.resolve()

def maybe_train_or_load_simple(model, train_loader, val_loader, cfg_root, cfg_section, device, kind, epoch_fn, override_ckpt=None):
    if override_ckpt is not None:
        print(f"[{kind}] loading pretrained checkpoint: {override_ckpt}")
        load_checkpoint_weights(model, override_ckpt, device)
        return Path(override_ckpt), []
    ckpt_path, history = train_simple_model(model, train_loader, val_loader, cfg_root, cfg_section, device, kind, epoch_fn)
    load_checkpoint_weights(model, ckpt_path, device)
    return Path(ckpt_path), history

def maybe_train_or_load_tokenizer(
    model,
    train_loader,
    val_loader,
    cfg_root,
    cfg_section,
    device,
    kind,
    override_ckpt=None,
    init_ckpt=None,
    aux_decoder_model=None,
    stats=None,
):
    if override_ckpt is not None:
        print(f"[{kind}] loading pretrained checkpoint (skip training): {override_ckpt}")
        load_checkpoint_weights(model, override_ckpt, device)
        return Path(override_ckpt), []

    if init_ckpt is not None:
        print(f"[{kind}] loading init checkpoint (continue training): {init_ckpt}")
        load_checkpoint_weights(model, init_ckpt, device)

    if train_loader is None or val_loader is None:
        raise ValueError(f"[{kind}] train_loader/val_loader must be provided when training is enabled")

    ckpt_path, history = train_tokenizer_model(
        model,
        train_loader,
        val_loader,
        cfg_root,
        cfg_section,
        device,
        kind,
        aux_decoder_model=aux_decoder_model,
        stats=stats,
    )
    load_checkpoint_weights(model, ckpt_path, device)
    return Path(ckpt_path), history

def maybe_train_or_load_generator(
    model,
    train_loader,
    val_loader,
    cfg_root,
    device,
    override_ckpt=None,
    init_ckpt=None,
):
    if override_ckpt is not None:
        print(f"[generator] loading pretrained checkpoint (skip training): {override_ckpt}")
        load_checkpoint_weights(model, override_ckpt, device)
        return Path(override_ckpt), []

    if init_ckpt is not None:
        print(f"[generator] loading init checkpoint (continue training): {init_ckpt}")
        load_checkpoint_weights(model, init_ckpt, device)

    if train_loader is None or val_loader is None:
        raise ValueError("[generator] train_loader/val_loader must be provided when training is enabled")

    ckpt_path = train_generator(model, train_loader, val_loader, cfg_root, device)
    load_checkpoint_weights(model, ckpt_path, device)
    return Path(ckpt_path), []

def build_location_tokenizer(cfg: Dict) -> SequenceTokenizerVQ:
    p = cfg["location_tokenizer"]
    return SequenceTokenizerVQ(
        input_dim=int(cfg["location_ae"]["embedding_dim"]),
        model_dim=int(p["model_dim"]),
        num_embeddings=int(p["num_embeddings"]),
        max_len=int(cfg["dataset"]["max_strokes"]),
        commitment_cost=float(p["commitment_cost"]),
        num_layers=int(p["num_layers"]),
        decoder_num_layers=int(p["num_layers"]),
        dropout=float(p["dropout"]),
        kernel_size=int(p["kernel_size"]),
        ema_decay=float(p["ema_decay"]),
        ema_eps=float(p["ema_eps"]),
        dead_code_threshold=float(p["dead_code_threshold"]),
        reset_interval=int(p["reset_interval"]),
        commitment_warmup_steps=int(p["commitment_warmup_steps"]),
        init_batch_samples=int(p["init_batch_samples"]),
    )

def build_generator(
    cfg: Dict,
    shape_tokenizer: torch.nn.Module | None = None,
    location_tokenizer: torch.nn.Module | None = None,
) -> StrokeAutoregressiveGenerator:
    p = cfg["generator"]
    shape_codebook = _extract_tokenizer_codebook(
        shape_tokenizer,
        fallback_dim=int(cfg["shape_tokenizer"]["model_dim"]),
        fallback_size=int(cfg["shape_tokenizer"]["num_embeddings"]),
    )
    loc_codebook = _extract_tokenizer_codebook(
        location_tokenizer,
        fallback_dim=int(cfg["location_tokenizer"]["model_dim"]),
        fallback_size=int(cfg["location_tokenizer"]["num_embeddings"]),
    )
    return StrokeAutoregressiveGenerator(
        num_classes=len(cfg["dataset"]["classes"]),
        shape_vocab_size=int(cfg["shape_tokenizer"]["num_embeddings"]),
        loc_vocab_size=int(cfg["location_tokenizer"]["num_embeddings"]),
        max_strokes=int(cfg["dataset"]["max_strokes"]),
        shape_codebook=shape_codebook,
        loc_codebook=loc_codebook,
        model_dim=int(p["model_dim"]),
        num_heads=int(p["num_heads"]),
        num_layers=int(p["num_layers"]),
        ff_dim=int(p["ff_dim"]),
        dropout=float(p["dropout"]),
        token_residual_scale=float(p["token_residual_scale"]),
    )

def scheduled_sampling_ratio(cfg: Dict, epoch: int) -> float:
    p = cfg["generator"]
    start = float(p.get("scheduled_sampling_start", 1.0))
    end = float(p.get("scheduled_sampling_end", start))
    warmup = int(p.get("scheduled_sampling_warmup_epochs", 0))
    decay = int(p.get("scheduled_sampling_decay_epochs", max(1, int(p.get("epochs", 1)))))
    if epoch <= warmup:
        return start
    progress = min(1.0, max(0.0, float(epoch - warmup) / float(max(1, decay))))
    return start + (end - start) * progress

def build_mixed_loc_condition_shapes(
    model: StrokeAutoregressiveGenerator,
    target_shape: torch.Tensor,
    pred_shape: torch.Tensor,
    teacher_force_ratio: float,
) -> torch.Tensor:
    teacher_force_ratio = float(np.clip(teacher_force_ratio, 0.0, 1.0))

    if teacher_force_ratio >= 1.0:
        return target_shape

    mixed = target_shape.clone()


    valid_mask = target_shape != model.shape_pad

    replace_mask = (
        torch.rand(target_shape.shape, device=target_shape.device) > teacher_force_ratio
    ) & valid_mask

    mixed[replace_mask] = pred_shape[replace_mask]
    return mixed


def build_generator_position_weights(
    target_tokens: torch.Tensor,
    pad_idx: int,
    min_weight: float,
    max_weight: float,
    power: float,
) -> torch.Tensor:
    if target_tokens.ndim != 2:
        raise ValueError(f"target_tokens must have shape [B, T], got {tuple(target_tokens.shape)}")

    batch_size, seq_len = target_tokens.shape
    device = target_tokens.device
    dtype = torch.float32

    min_weight = float(min_weight)
    max_weight = float(max_weight)
    power = float(power)

    if seq_len <= 1:
        step_weights = torch.full((seq_len,), fill_value=max_weight, device=device, dtype=dtype)
    else:
        progress = torch.linspace(0.0, 1.0, steps=seq_len, device=device, dtype=dtype)
        step_weights = max_weight - (max_weight - min_weight) * progress.pow(power)

    weights = step_weights.unsqueeze(0).expand(batch_size, seq_len)
    valid_mask = (target_tokens != pad_idx).to(dtype)
    return weights * valid_mask

def weighted_token_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int,
    position_weights: torch.Tensor,
) -> torch.Tensor:
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_idx,
        reduction="none",
    ).reshape_as(targets)

    valid_mask = (targets != pad_idx).to(token_loss.dtype)
    weights = position_weights.to(token_loss.dtype) * valid_mask
    denom = weights.sum().clamp_min(1.0)
    return (token_loss * weights).sum() / denom


def run_shape_ae_epoch(model, loader, device, cfg, optimizer=None, mixed_precision=True):
    is_train = optimizer is not None
    model.train(is_train)
    mixed_precision = False
    scaler = GradScaler("cuda", enabled=False) if is_train else None
    total_meter, recon_meter = AverageMeter(), AverageMeter()
    iterator = tqdm(loader, leave=False)
    for batch in iterator:
        images = batch["image"].to(device)
        dist_map = batch["dist_map"].to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=mixed_precision and device.type == "cuda"):
            out = model(images)
            recon_loss = shape_ae_recon_loss(
                out["bitmap_logits"], out["dist_raw"], images, dist_map,
                bce_weight=float(cfg["bitmap_bce_weight"]),
                l1_weight=float(cfg["bitmap_l1_weight"]),
                dist_weight=float(cfg["distance_weight"]),
                dice_weight=float(cfg["dice_weight"]),
            )
            loss = recon_loss
        if is_train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            maybe_clip_grad(model, float(cfg["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
        bsz = int(images.size(0))
        total_meter.update(float(loss.item()), bsz)
        recon_meter.update(float(recon_loss.item()), bsz)
        iterator.set_description(f"loss={total_meter.avg:.4f}")
    return {"loss": total_meter.avg, "recon": recon_meter.avg}

def run_location_ae_epoch(model, loader, device, cfg, optimizer=None, mixed_precision=True):
    is_train = optimizer is not None
    model.train(is_train)
    scaler = GradScaler("cuda", enabled=mixed_precision and device.type == "cuda") if is_train else None
    total_meter = AverageMeter()
    iterator = tqdm(loader, leave=False)
    for batch in iterator:
        vectors = batch["vector"].to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=mixed_precision and device.type == "cuda"):
            out = model(vectors)
            loss = F.smooth_l1_loss(out["recon"], vectors)
        if is_train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            maybe_clip_grad(model, float(cfg["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
        bsz = int(vectors.size(0))
        total_meter.update(float(loss.item()), bsz)
        iterator.set_description(f"loss={total_meter.avg:.4f}")
    return {"loss": total_meter.avg}

def _bbox_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    w, h, cx, cy = boxes.unbind(dim=-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

def masked_bbox_iou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    pred = _bbox_xyxy(pred_boxes)
    target = _bbox_xyxy(target_boxes)

    inter_x1 = torch.maximum(pred[..., 0], target[..., 0])
    inter_y1 = torch.maximum(pred[..., 1], target[..., 1])
    inter_x2 = torch.minimum(pred[..., 2], target[..., 2])
    inter_y2 = torch.minimum(pred[..., 3], target[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter = inter_w * inter_h

    area_pred = (pred[..., 2] - pred[..., 0]).clamp(min=0.0) * (pred[..., 3] - pred[..., 1]).clamp(min=0.0)
    area_target = (target[..., 2] - target[..., 0]).clamp(min=0.0) * (target[..., 3] - target[..., 1]).clamp(min=0.0)
    union = (area_pred + area_target - inter).clamp(min=1e-6)
    iou = inter / union

    mask = valid_mask.float()
    if mask.ndim == 2:
        masked = iou * mask
        denom = mask.sum().clamp(min=1.0)
        return 1.0 - masked.sum() / denom
    return 1.0 - iou.mean()

def _expand_mask_nd(mask: torch.Tensor, target_ndim: int) -> torch.Tensor:
    out = mask.float()
    while out.ndim < target_ndim:
        out = out.unsqueeze(-1)
    return out

def masked_bitmap_l1(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).abs()
    return (diff * _expand_mask_nd(valid_mask, diff.ndim)).sum() / _expand_mask_nd(valid_mask, diff.ndim).sum().clamp(min=1.0)

def masked_bitmap_iou_loss(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    pred_bin = (pred >= threshold).float()
    target_bin = (target >= threshold).float()
    mask = _expand_mask_nd(valid_mask, pred_bin.ndim)
    inter = (pred_bin * target_bin * mask).sum()
    union = (((pred_bin + target_bin) > 0).float() * mask).sum().clamp(min=1.0)
    return 1.0 - inter / union

def compute_small_stroke_weights(
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    gamma: float = 0.50,
    w_min: float = 0.75,
    w_max: float = 4.00,
) -> torch.Tensor:


    fg = target.flatten(2).sum(dim=2)
    mask = valid_mask.float()

    fg_mean = (fg * mask).sum() / mask.sum().clamp(min=1.0)
    w = ((fg_mean + 1.0) / (fg + 1.0)).pow(gamma)

    w = torch.clamp(w, min=w_min, max=w_max)
    w = w * mask
    return w

def masked_bitmap_l1_stroke_weighted(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    stroke_weight: torch.Tensor,
) -> torch.Tensor:


    per_stroke = (pred - target).abs().flatten(2).mean(dim=2)
    w = stroke_weight * valid_mask.float()
    return (per_stroke * w).sum() / w.sum().clamp(min=1.0)

def masked_bitmap_soft_dice_loss_stroke_weighted(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    stroke_weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:

    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)

    inter = (pred * target).flatten(2).sum(dim=2)
    denom = pred.flatten(2).sum(dim=2) + target.flatten(2).sum(dim=2)
    dice = (2.0 * inter + eps) / (denom + eps)

    w = stroke_weight * valid_mask.float()
    return 1.0 - (dice * w).sum() / w.sum().clamp(min=1.0)

def masked_bitmap_bce_stroke_weighted(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    stroke_weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred = pred.clamp(eps, 1.0 - eps)
    bce = -(target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))
    per_stroke = bce.flatten(2).mean(dim=2)
    w = stroke_weight * valid_mask.float()
    return (per_stroke * w).sum() / w.sum().clamp(min=1.0)

def run_tokenizer_epoch(
    model,
    loader,
    device,
    cfg_section: Dict,
    kind: str,
    optimizer=None,
    mixed_precision=True,
    grad_clip_norm=None,
    aux_decoder_model=None,
    stats=None,
):
    is_train = optimizer is not None
    model.train(is_train)
    if aux_decoder_model is not None:
        aux_decoder_model.eval()

    total_meter = AverageMeter()
    seq_meter = AverageMeter()
    local_meter = AverageMeter()
    vq_meter = AverageMeter()
    ppl_meter = AverageMeter()
    bbox_meter = AverageMeter()
    iou_meter = AverageMeter()

    iterator = tqdm(loader, leave=False)
    seq_weight = float(cfg_section.get("seq_recon_weight", 1.0))
    local_weight = float(cfg_section.get("local_recon_weight", 1.0))
    cosine_weight = float(cfg_section.get("cosine_weight", 0.0))
    vq_weight = float(cfg_section.get("vq_weight", 1.0))
    bbox_weight = float(cfg_section.get("bbox_weight", 0.0))
    iou_weight = float(cfg_section.get("iou_weight", 0.0))
    image_recon_weight = float(cfg_section.get("image_recon_weight", 0.0))
    image_local_weight = float(cfg_section.get("image_local_weight", 0.0))
    image_iou_weight = float(cfg_section.get("image_iou_weight", 0.0))
    image_bce_weight = float(cfg_section.get("image_bce_weight", 0.0))

    for batch in iterator:
        feats = batch["features"].to(device)
        mask = batch["valid_mask"].to(device)

        raw_bboxes = batch.get("raw_bboxes", None)
        if raw_bboxes is not None:
            raw_bboxes = raw_bboxes.to(device)

        raw_shape_images = batch.get("raw_shape_images", None)
        if raw_shape_images is not None:
            raw_shape_images = raw_shape_images.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        feats = torch.nan_to_num(feats.float(), nan=0.0, posinf=0.0, neginf=0.0)
        out = model(feats, mask)

        if "shape" in kind:
            seq_loss = masked_smooth_l1(out["recon_seq"], feats, mask, beta=0.5)
            local_loss = masked_smooth_l1(out["recon_local"], feats, mask, beta=0.5)
            cos_loss = (
                masked_cosine(out["recon_seq"], feats, mask)
                + 0.5 * masked_cosine(out["recon_local"], feats, mask)
            ) if cosine_weight > 0 else feats.new_tensor(0.0)

            bbox_loss = feats.new_tensor(0.0)
            iou_loss = feats.new_tensor(0.0)

            if (
                aux_decoder_model is not None
                and stats is not None
                and (
                    image_recon_weight > 0
                    or image_local_weight > 0
                    or image_iou_weight > 0
                    or image_bce_weight > 0
                )
            ):
                recon_emb = denormalize_feature_batch(out["recon_seq"], stats["shape_mean"], stats["shape_std"])
                local_emb = denormalize_feature_batch(out["recon_local"], stats["shape_mean"], stats["shape_std"])
                B, T, D = recon_emb.shape

                flat_recon = recon_emb.reshape(-1, D)
                flat_local = local_emb.reshape(-1, D)

                if raw_shape_images is not None:
                    target_bitmap = raw_shape_images.float()
                else:
                    orig_emb = denormalize_feature_batch(feats, stats["shape_mean"], stats["shape_std"])
                    flat_orig = orig_emb.reshape(-1, D)
                    with torch.no_grad():
                        target_bitmap = aux_decoder_model.decode_from_embedding(flat_orig)["bitmap"].view(
                            B, T, 1, aux_decoder_model.image_size, aux_decoder_model.image_size
                        )

                pred_seq_bitmap = aux_decoder_model.decode_from_embedding(flat_recon)["bitmap"].view(
                    B, T, 1, aux_decoder_model.image_size, aux_decoder_model.image_size
                )
                pred_local_bitmap = aux_decoder_model.decode_from_embedding(flat_local)["bitmap"].view(
                    B, T, 1, aux_decoder_model.image_size, aux_decoder_model.image_size
                )

                stroke_weight = compute_small_stroke_weights(
                    target_bitmap,
                    mask,
                    gamma=float(cfg_section.get("small_stroke_gamma", 0.50)),
                    w_min=float(cfg_section.get("small_stroke_weight_min", 0.75)),
                    w_max=float(cfg_section.get("small_stroke_weight_max", 4.00)),
                )

                img_seq_loss = masked_bitmap_l1_stroke_weighted(
                    pred_seq_bitmap, target_bitmap, mask, stroke_weight
                )
                img_local_loss = masked_bitmap_l1_stroke_weighted(
                    pred_local_bitmap, target_bitmap, mask, stroke_weight
                )
                img_iou_loss = 0.5 * (
                    masked_bitmap_soft_dice_loss_stroke_weighted(
                        pred_seq_bitmap, target_bitmap, mask, stroke_weight
                    )
                    + masked_bitmap_soft_dice_loss_stroke_weighted(
                        pred_local_bitmap, target_bitmap, mask, stroke_weight
                    )
                )
                img_seq_bce = masked_bitmap_bce_stroke_weighted(
                    pred_seq_bitmap, target_bitmap, mask, stroke_weight
                )
                img_local_bce = masked_bitmap_bce_stroke_weighted(
                    pred_local_bitmap, target_bitmap, mask, stroke_weight
                )

                bbox_loss = (
                    image_recon_weight * img_seq_loss
                    + image_local_weight * img_local_loss
                    + image_bce_weight * 0.5 * (img_seq_bce + img_local_bce)
                )
                iou_loss = image_iou_weight * img_iou_loss

        else:
            seq_loss = masked_smooth_l1(out["recon_seq"], feats, mask, beta=0.25)
            local_loss = masked_smooth_l1(out["recon_local"], feats, mask, beta=0.25)
            cos_loss = feats.new_tensor(0.0)

            bbox_loss = feats.new_tensor(0.0)
            iou_loss = feats.new_tensor(0.0)
            if (aux_decoder_model is not None) and (raw_bboxes is not None) and (stats is not None):
                recon_emb = denormalize_feature_batch(out["recon_seq"], stats["loc_mean"], stats["loc_std"])
                local_emb = denormalize_feature_batch(out["recon_local"], stats["loc_mean"], stats["loc_std"])

                pred_bbox_seq = aux_decoder_model.decode_from_embedding(recon_emb.reshape(-1, recon_emb.size(-1))).view_as(raw_bboxes)
                pred_bbox_local = aux_decoder_model.decode_from_embedding(local_emb.reshape(-1, local_emb.size(-1))).view_as(raw_bboxes)

                pred_bbox_seq = torch.nan_to_num(pred_bbox_seq, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                pred_bbox_local = torch.nan_to_num(pred_bbox_local, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

                bbox_loss = 0.5 * (
                    masked_smooth_l1(pred_bbox_seq, raw_bboxes, mask, beta=0.10)
                    + masked_smooth_l1(pred_bbox_local, raw_bboxes, mask, beta=0.10)
                )
                iou_loss = 0.5 * (
                    masked_bbox_iou_loss(pred_bbox_seq, raw_bboxes, mask)
                    + masked_bbox_iou_loss(pred_bbox_local, raw_bboxes, mask)
                )

        recon_loss = (
            seq_weight * seq_loss
            + local_weight * local_loss
            + cosine_weight * cos_loss
            + bbox_weight * bbox_loss
            + iou_weight * iou_loss
        )
        vq_loss = vq_weight * out["vq_loss"]
        loss = recon_loss + vq_loss
        loss = torch.nan_to_num(loss, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))

        if not torch.isfinite(loss):
            raise RuntimeError(f"[{kind}] non-finite loss detected")

        if is_train:
            loss.backward()
            maybe_clip_grad(model, grad_clip_norm)
            optimizer.step()

        bsz = int(feats.size(0))
        total_meter.update(float(loss.item()), bsz)
        seq_meter.update(float(seq_loss.item()), bsz)
        local_meter.update(float(local_loss.item()), bsz)
        vq_meter.update(float(vq_loss.item()), bsz)
        ppl_meter.update(float(out["perplexity"]), bsz)
        bbox_meter.update(float(bbox_loss.item()), bsz)
        iou_meter.update(float(iou_loss.item()), bsz)
        iterator.set_description(
            f"loss={total_meter.avg:.4f} seq={seq_meter.avg:.4f} local={local_meter.avg:.4f} vq={vq_meter.avg:.4f}"
        )

    return {
        "loss": total_meter.avg,
        "seq": seq_meter.avg,
        "local": local_meter.avg,
        "vq": vq_meter.avg,
        "perplexity": ppl_meter.avg,
        "bbox": bbox_meter.avg,
        "iou": iou_meter.avg,
    }

def train_tokenizer_model(model, train_loader, val_loader, cfg_root: Dict, cfg_section: Dict, device, kind: str, aux_decoder_model=None, stats=None):
    optimizer = make_optimizer(model, lr=float(cfg_section["lr"]), weight_decay=float(cfg_section["weight_decay"]))
    scheduler = None
    if str(cfg_section.get("scheduler", "")).lower() == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(cfg_section.get("scheduler_factor", 0.5)),
            patience=int(cfg_section.get("scheduler_patience", 5)),
            min_lr=float(cfg_section.get("min_lr", 0.0)),
        )
    best = float("inf")
    no_improve = 0
    early_stopping_patience = int(cfg_section.get("early_stopping_patience", 0))
    ckpt_path = checkpoint_root(cfg_root) / f"{kind}_best.pt"
    history = []
    for epoch in range(1, int(cfg_section["epochs"]) + 1):
        train_metrics = run_tokenizer_epoch(
            model,
            train_loader,
            device,
            cfg_section=cfg_section,
            kind=kind,
            optimizer=optimizer,
            mixed_precision=bool(cfg_root["project"]["mixed_precision"]),
            grad_clip_norm=float(cfg_section["grad_clip_norm"]),
            aux_decoder_model=aux_decoder_model,
            stats=stats,
        )
        val_metrics = run_tokenizer_epoch(
            model,
            val_loader,
            device,
            cfg_section=cfg_section,
            kind=kind,
            optimizer=None,
            mixed_precision=bool(cfg_root["project"]["mixed_precision"]),
            grad_clip_norm=float(cfg_section["grad_clip_norm"]),
            aux_decoder_model=aux_decoder_model,
            stats=stats,
        )
        score = float(val_metrics["loss"])
        if scheduler is not None and math.isfinite(score):
            scheduler.step(score)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append({"epoch": epoch, "lr": current_lr, "train": train_metrics, "val": val_metrics})
        print(
            f"[{kind}] epoch={epoch:03d} lr={current_lr:.2e} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_perplexity={val_metrics['perplexity']:.2f}"
        )
        if not math.isfinite(score):
            raise RuntimeError(f"[{kind}] non-finite validation loss detected: {score}")
        if score < best:
            best = score
            no_improve = 0
            CheckpointIO.save(ckpt_path, {"model_state": model.state_dict(), "history": history, "best_score": best})
        else:
            no_improve += 1
        if early_stopping_patience > 0 and no_improve >= early_stopping_patience:
            print(f"[{kind}] early stopping triggered after {epoch} epochs (best_val_loss={best:.4f})")
            break
    if not Path(ckpt_path).exists():
        raise RuntimeError(f"[{kind}] checkpoint was not saved")
    return ckpt_path, history

def train_simple_model(model, train_loader, val_loader, cfg_root: Dict, cfg_section: Dict, device, kind: str, epoch_fn):
    optimizer = make_optimizer(model, lr=float(cfg_section["lr"]), weight_decay=float(cfg_section["weight_decay"]))
    best = float("inf")
    ckpt_path = checkpoint_root(cfg_root) / f"{kind}_best.pt"
    history = []
    for epoch in range(1, int(cfg_section["epochs"]) + 1):
        train_metrics = epoch_fn(
            model,
            train_loader,
            device,
            cfg_section,
            optimizer=optimizer,
            mixed_precision=bool(cfg_root["project"]["mixed_precision"]),
        )
        val_metrics = epoch_fn(
            model,
            val_loader,
            device,
            cfg_section,
            optimizer=None,
            mixed_precision=bool(cfg_root["project"]["mixed_precision"]),
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        score = float(val_metrics["loss"])
        print(
            f"[{kind}] epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f}"
        )
        if score < best:
            best = score
            CheckpointIO.save(
                ckpt_path,
                {"model_state": model.state_dict(), "history": history, "best_score": best},
            )
    return ckpt_path, history

def build_mixed_teacher_inputs(
    model: StrokeAutoregressiveGenerator,
    class_ids: torch.Tensor,
    input_shape: torch.Tensor,
    input_loc: torch.Tensor,
    target_shape: torch.Tensor,
    target_loc: torch.Tensor,
    teacher_force_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    teacher_force_ratio = float(np.clip(teacher_force_ratio, 0.0, 1.0))
    if teacher_force_ratio >= 1.0:
        return input_shape, input_loc

    with torch.no_grad():
        warm_hidden = model._encode_prefix(class_ids=class_ids, input_shape=input_shape, input_loc=input_loc)
        warm_shape_logits = model.shape_head(warm_hidden)
        pred_shape_full = warm_shape_logits.argmax(dim=-1)
        pred_loc_full = model._predict_loc_tokens_from_hidden(warm_hidden, pred_shape_full.long())
        pred_prev_shape = pred_shape_full[:, :-1]
        pred_prev_loc = pred_loc_full[:, :-1]

    mixed_input_shape = input_shape.clone()
    mixed_input_loc = input_loc.clone()

    prev_target_shape = target_shape[:, :-1].clone()
    prev_target_loc = target_loc[:, :-1].clone()
    valid_prev = (prev_target_shape != model.shape_pad) & (prev_target_loc != model.loc_pad)

    replace_mask = (torch.rand(prev_target_shape.shape, device=prev_target_shape.device) > teacher_force_ratio) & valid_prev
    prev_target_shape[replace_mask] = pred_prev_shape[replace_mask]
    prev_target_loc[replace_mask] = pred_prev_loc[replace_mask]

    mixed_input_shape[:, 1:] = prev_target_shape
    mixed_input_loc[:, 1:] = prev_target_loc
    return mixed_input_shape, mixed_input_loc

def render_sketch_stroke_batch(point_strokes: Sequence[np.ndarray], cfg: Dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_size = int(cfg["dataset"]["image_size"])
    source_canvas_size = int(cfg["dataset"]["source_canvas_size"])
    line_width = int(cfg["dataset"]["shape_stroke_width"])
    distance_decay = float(cfg["dataset"]["distance_decay"])
    center_scale_margin = float(cfg["dataset"]["shape_center_scale_margin"])
    max_canvas_coverage = float(cfg["dataset"]["shape_max_canvas_coverage"])

    images, dist_maps, bboxes = [], [], []
    for pts in point_strokes:
        rendered = render_single_stroke_to_normalized_bbox(
            pts,
            image_size=image_size,
            source_canvas_size=source_canvas_size,
            line_width=line_width,
            distance_decay=distance_decay,
            center_scale_margin=center_scale_margin,
            max_canvas_coverage=max_canvas_coverage,
        )
        images.append(to_float_tensor_image(rendered.image))
        dist_maps.append(rendered.dist_map.astype(np.float32))
        bboxes.append(rendered.bbox.astype(np.float32))
    if not images:
        return (
            torch.zeros((0, 1, image_size, image_size), dtype=torch.float32),
            torch.zeros((0, 1, image_size, image_size), dtype=torch.float32),
            torch.zeros((0, 4), dtype=torch.float32),
        )
    images_t = torch.from_numpy(np.stack(images, axis=0)).unsqueeze(1)
    dist_t = torch.from_numpy(np.stack(dist_maps, axis=0)).unsqueeze(1)
    bboxes_t = torch.from_numpy(np.stack(bboxes, axis=0))
    return images_t, dist_t, bboxes_t

_STATS_CACHE: Dict[str, Dict[str, np.ndarray]] = {}

def embedding_stats_path(cfg: Dict) -> Path:
    return embedding_root(cfg) / "embedding_stats_ver12.npz"

def compute_embedding_stats(cfg: Dict, train_emb_npz_path: str | Path) -> Path:
    out_path = embedding_stats_path(cfg)
    if out_path.exists():
        return out_path
    npz = np.load(train_emb_npz_path)
    mask = npz["valid_mask"].astype(bool)
    shape_flat = npz["shape_embeddings"][mask]
    loc_flat = npz["loc_embeddings"][mask]
    shape_mean = shape_flat.mean(axis=0).astype(np.float32)
    shape_std = np.maximum(shape_flat.std(axis=0).astype(np.float32), 1e-4)
    loc_mean = loc_flat.mean(axis=0).astype(np.float32)
    loc_std = np.maximum(loc_flat.std(axis=0).astype(np.float32), 1e-4)
    np.savez_compressed(
        out_path,
        shape_mean=shape_mean,
        shape_std=shape_std,
        loc_mean=loc_mean,
        loc_std=loc_std,
    )
    return out_path

def load_embedding_stats(cfg: Dict) -> Dict[str, np.ndarray]:
    path = str(embedding_stats_path(cfg))
    if path not in _STATS_CACHE:
        npz = np.load(path)
        _STATS_CACHE[path] = {
            "shape_mean": npz["shape_mean"].astype(np.float32),
            "shape_std": npz["shape_std"].astype(np.float32),
            "loc_mean": npz["loc_mean"].astype(np.float32),
            "loc_std": npz["loc_std"].astype(np.float32),
        }
    return _STATS_CACHE[path]

def write_embedding_stats_npz(
    cfg: Dict[str, Any],
    shape_mean: np.ndarray,
    shape_std: np.ndarray,
    loc_mean: np.ndarray,
    loc_std: np.ndarray,
) -> Path:
    out_path = embedding_stats_path(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        shape_mean=np.asarray(shape_mean, dtype=np.float32),
        shape_std=np.asarray(shape_std, dtype=np.float32),
        loc_mean=np.asarray(loc_mean, dtype=np.float32),
        loc_std=np.asarray(loc_std, dtype=np.float32),
    )
    _STATS_CACHE.pop(str(out_path), None)
    return out_path

def write_mixed_embedding_stats(
    dst_cfg: Dict[str, Any],
    shape_stats: Dict[str, np.ndarray],
    loc_stats: Dict[str, np.ndarray],
) -> Path:
    return write_embedding_stats_npz(
        dst_cfg,
        shape_mean=shape_stats["shape_mean"],
        shape_std=shape_stats["shape_std"],
        loc_mean=loc_stats["loc_mean"],
        loc_std=loc_stats["loc_std"],
    )

def normalize_feature_batch(x: torch.Tensor, mean: np.ndarray, std: np.ndarray, clamp_value: float = 8.0) -> torch.Tensor:
    mean_t = torch.as_tensor(mean, device=x.device, dtype=x.dtype)
    std_t = torch.as_tensor(std, device=x.device, dtype=x.dtype)
    while mean_t.ndim < x.ndim:
        mean_t = mean_t.unsqueeze(0)
        std_t = std_t.unsqueeze(0)
    std_t = torch.clamp(std_t, min=1e-4)
    out = (x - mean_t) / std_t
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = torch.clamp(out, -clamp_value, clamp_value)
    return out

def denormalize_feature_batch(x: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    mean_t = torch.as_tensor(mean, device=x.device, dtype=x.dtype)
    std_t = torch.as_tensor(std, device=x.device, dtype=x.dtype)
    while mean_t.ndim < x.ndim:
        mean_t = mean_t.unsqueeze(0)
        std_t = std_t.unsqueeze(0)
    return x * std_t + mean_t

def encode_embedding_sequences(cfg, split: str, shape_ae, location_ae, device):
    ensure_workspace(cfg)
    out_path = embedding_root(cfg) / f"{split}_embeddings_ver12.npz"
    if out_path.exists():
        return out_path

    index = JsonlSketchIndex(cfg, split)
    max_strokes = int(cfg["dataset"]["max_strokes"])
    shape_dim = int(cfg["shape_ae"]["embedding_dim"])
    loc_dim = int(cfg["location_ae"]["embedding_dim"])
    image_size = int(cfg["dataset"]["image_size"])

    shape_embeddings = np.zeros((len(index), max_strokes, shape_dim), dtype=np.float32)
    loc_embeddings = np.zeros((len(index), max_strokes, loc_dim), dtype=np.float32)
    raw_bboxes = np.zeros((len(index), max_strokes, 4), dtype=np.float32)
    raw_shape_images = np.zeros((len(index), max_strokes, image_size, image_size), dtype=np.float32)
    valid_mask = np.zeros((len(index), max_strokes), dtype=np.float32)
    lengths = np.zeros((len(index),), dtype=np.int64)
    class_ids = np.zeros((len(index),), dtype=np.int64)

    shape_ae.eval()
    location_ae.eval()

    with torch.no_grad():
        for sketch_idx in tqdm(range(len(index)), desc=f"Encoding embeddings [{split}]"):
            record = index.get_drawing(sketch_idx, apply_augment=False)
            point_strokes = record["point_strokes"][:max_strokes]
            L = len(point_strokes)
            lengths[sketch_idx] = L
            class_ids[sketch_idx] = int(record["class_id"])
            if L == 0:
                continue

            images_t, _, bboxes_t = render_sketch_stroke_batch(point_strokes, cfg)

            shape_emb = shape_ae.encode(images_t.to(device)).cpu().numpy().astype(np.float32)
            loc_emb = location_ae.encode(bboxes_t.to(device)).cpu().numpy().astype(np.float32)

            shape_embeddings[sketch_idx, :L] = shape_emb
            loc_embeddings[sketch_idx, :L] = loc_emb
            raw_bboxes[sketch_idx, :L] = bboxes_t.numpy().astype(np.float32)
            raw_shape_images[sketch_idx, :L] = images_t.squeeze(1).numpy().astype(np.float32)
            valid_mask[sketch_idx, :L] = 1.0

    np.savez_compressed(
        out_path,
        class_ids=class_ids,
        lengths=lengths,
        valid_mask=valid_mask,
        shape_embeddings=shape_embeddings,
        loc_embeddings=loc_embeddings,
        raw_bboxes=raw_bboxes,
        raw_shape_images=raw_shape_images,
        source_indices=np.arange(len(index), dtype=np.int64),
    )
    return out_path

def _generator_split_limit(cfg: Dict, split: str) -> int:
    total = int(cfg["dataset"]["generator_max_drawings_per_class"])
    train_n = int(total * float(cfg["dataset"]["train_ratio"]))
    val_n = int(total * float(cfg["dataset"]["val_ratio"]))
    test_n = max(0, total - train_n - val_n)
    mapping = {"train": train_n, "val": val_n, "test": test_n}
    return int(mapping[split])

def make_generator_token_npz(cfg: Dict, split: str, token_npz_path: str | Path) -> Path:
    rep_total = int(cfg["dataset"]["representation_max_drawings_per_class"])
    gen_total = int(cfg["dataset"]["generator_max_drawings_per_class"])
    if gen_total >= rep_total:
        return Path(token_npz_path)

    out_path = token_root(cfg) / f"{split}_tokens_generator_ver13.npz"
    if out_path.exists():
        return out_path

    npz = np.load(token_npz_path)
    class_ids = npz["class_ids"].astype(np.int64)
    lengths = npz["lengths"].astype(np.int64)
    shape_tokens = npz["shape_tokens"].astype(np.int64)
    loc_tokens = npz["loc_tokens"].astype(np.int64)
    source_indices = npz["source_indices"].astype(np.int64) if "source_indices" in npz else np.arange(len(class_ids), dtype=np.int64)

    limit = _generator_split_limit(cfg, split)
    seed = int(cfg["project"]["seed"]) + {"train": 11, "val": 23, "test": 37}[split]
    rng = np.random.default_rng(seed)

    keep_indices: List[int] = []
    for class_id in sorted(np.unique(class_ids).tolist()):
        cls_idx = np.where(class_ids == class_id)[0]
        if cls_idx.size <= limit:
            keep_indices.extend(cls_idx.tolist())
        else:
            chosen = rng.choice(cls_idx, size=limit, replace=False)
            keep_indices.extend(chosen.tolist())

    keep_indices = np.array(sorted(keep_indices), dtype=np.int64)
    np.savez_compressed(
        out_path,
        class_ids=class_ids[keep_indices],
        lengths=lengths[keep_indices],
        shape_tokens=shape_tokens[keep_indices],
        loc_tokens=loc_tokens[keep_indices],
        source_indices=source_indices[keep_indices],
    )
    return out_path

def encode_split_to_token_npz(cfg, split: str, emb_npz_path, shape_tokenizer, location_tokenizer, device):
    out_path = token_root(cfg) / f"{split}_tokens_ver15.npz"
    if out_path.exists():
        return out_path

    npz = np.load(emb_npz_path)
    shape_embeddings = torch.from_numpy(npz["shape_embeddings"]).float()
    loc_embeddings = torch.from_numpy(npz["loc_embeddings"]).float()
    valid_mask = torch.from_numpy(npz["valid_mask"]).float()
    lengths = npz["lengths"].astype(np.int64)
    class_ids = npz["class_ids"].astype(np.int64)
    source_indices = npz["source_indices"].astype(np.int64) if "source_indices" in npz else np.arange(len(class_ids), dtype=np.int64)

    if shape_embeddings.size(0) == 0:
        empty = np.zeros((0, int(cfg["dataset"]["max_strokes"])), dtype=np.int64)
        np.savez_compressed(
            out_path,
            class_ids=class_ids,
            lengths=lengths,
            shape_tokens=empty,
            loc_tokens=empty.copy(),
            source_indices=source_indices,
        )
        return out_path

    stats = load_embedding_stats(cfg)
    shape_mean = stats["shape_mean"]
    shape_std = stats["shape_std"]
    loc_mean = stats["loc_mean"]
    loc_std = stats["loc_std"]

    shape_tokenizer.eval()
    location_tokenizer.eval()

    batch_size = 128
    all_shape_tokens = []
    all_loc_tokens = []

    with torch.no_grad():
        for start in tqdm(range(0, shape_embeddings.size(0), batch_size), desc=f"Encoding tokens [{split}]"):
            end = min(start + batch_size, shape_embeddings.size(0))
            shape_batch = normalize_feature_batch(shape_embeddings[start:end].to(device), shape_mean, shape_std)
            loc_batch = normalize_feature_batch(loc_embeddings[start:end].to(device), loc_mean, loc_std)
            mask_batch = valid_mask[start:end].to(device)

            shape_out = shape_tokenizer(shape_batch, mask_batch)
            loc_out = location_tokenizer(loc_batch, mask_batch)

            all_shape_tokens.append(shape_out["indices"].detach().cpu().numpy().astype(np.int64))
            all_loc_tokens.append(loc_out["indices"].detach().cpu().numpy().astype(np.int64))

    shape_tokens = np.concatenate(all_shape_tokens, axis=0)
    loc_tokens = np.concatenate(all_loc_tokens, axis=0)

    np.savez_compressed(
        out_path,
        class_ids=class_ids,
        lengths=lengths,
        shape_tokens=shape_tokens,
        loc_tokens=loc_tokens,
        source_indices=source_indices,
    )
    return out_path

def train_generator(model, train_loader, val_loader, cfg, device):
    optimizer = make_optimizer(model, lr=float(cfg["generator"]["lr"]), weight_decay=float(cfg["generator"]["weight_decay"]))
    best = float("inf")
    ckpt_path = checkpoint_root(cfg) / "generator_best.pt"
    history = []
    for epoch in range(1, int(cfg["generator"]["epochs"]) + 1):
        tf_ratio = scheduled_sampling_ratio(cfg, epoch)
        train_metrics = run_generator_epoch(
            model, train_loader, device, optimizer=optimizer,
            mixed_precision=bool(cfg["project"]["mixed_precision"]),
            grad_clip_norm=float(cfg["generator"]["grad_clip_norm"]),
            teacher_force_ratio=tf_ratio,
        )
        val_metrics = run_generator_epoch(
            model, val_loader, device, optimizer=None,
            mixed_precision=bool(cfg["project"]["mixed_precision"]),
            grad_clip_norm=float(cfg["generator"]["grad_clip_norm"]),
            teacher_force_ratio=1.0,
        )
        history.append({"epoch": epoch, "teacher_force_ratio": tf_ratio, "train": train_metrics, "val": val_metrics})
        score = float(val_metrics["loss"])
        print(
            f"[generator] epoch={epoch:03d} tf_ratio={tf_ratio:.3f} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"(shape={train_metrics['shape_loss']:.4f}, loc={train_metrics['loc_loss']:.4f}) "
            f"val_loss={val_metrics['loss']:.4f} "
            f"(shape={val_metrics['shape_loss']:.4f}, loc={val_metrics['loc_loss']:.4f})"
        )
        if not math.isfinite(score):
            raise RuntimeError(f"[generator] non-finite validation loss detected: {score}")
        if score < best:
            best = score
            CheckpointIO.save(ckpt_path, {"model_state": model.state_dict(), "history": history, "best_score": best})
    return ckpt_path

class StrokeAutoregressiveGenerator(nn.Module):
    def __init__(
        self,
        num_classes: int,
        shape_vocab_size: int,
        loc_vocab_size: int,
        max_strokes: int,
        shape_codebook: torch.Tensor,
        loc_codebook: torch.Tensor,
        model_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        token_residual_scale: float = 0.10,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.shape_vocab_size = int(shape_vocab_size)
        self.loc_vocab_size = int(loc_vocab_size)
        self.max_strokes = int(max_strokes)
        self.model_dim = int(model_dim)
        self.token_residual_scale = float(token_residual_scale)





        self.shape_end = self.shape_vocab_size
        self.shape_start = self.shape_vocab_size + 1
        self.shape_pad = self.shape_vocab_size + 2

        self.loc_end = self.loc_vocab_size
        self.loc_start = self.loc_vocab_size + 1
        self.loc_pad = self.loc_vocab_size + 2

        shape_codebook = torch.as_tensor(shape_codebook, dtype=torch.float32)
        loc_codebook = torch.as_tensor(loc_codebook, dtype=torch.float32)
        if shape_codebook.ndim != 2 or int(shape_codebook.size(0)) != self.shape_vocab_size:
            raise ValueError(f"shape_codebook must have shape [{self.shape_vocab_size}, D], got {tuple(shape_codebook.shape)}")
        if loc_codebook.ndim != 2 or int(loc_codebook.size(0)) != self.loc_vocab_size:
            raise ValueError(f"loc_codebook must have shape [{self.loc_vocab_size}, D], got {tuple(loc_codebook.shape)}")

        self.shape_code_dim = int(shape_codebook.size(1))
        self.loc_code_dim = int(loc_codebook.size(1))
        self.register_buffer("shape_codebook", shape_codebook.clone())
        self.register_buffer("loc_codebook", loc_codebook.clone())

        self.shape_codebook_proj = nn.Sequential(
            nn.LayerNorm(self.shape_code_dim),
            nn.Linear(self.shape_code_dim, model_dim),
        )
        self.loc_codebook_proj = nn.Sequential(
            nn.LayerNorm(self.loc_code_dim),
            nn.Linear(self.loc_code_dim, model_dim),
        )
        self.shape_special_embedding = nn.Embedding(3, model_dim)
        self.loc_special_embedding = nn.Embedding(3, model_dim)
        self.shape_residual_embedding = nn.Embedding(self.shape_vocab_size + 3, model_dim)
        self.loc_residual_embedding = nn.Embedding(self.loc_vocab_size + 3, model_dim)
        nn.init.normal_(self.shape_residual_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.loc_residual_embedding.weight, mean=0.0, std=0.02)



        self.class_token_embedding = nn.Embedding(self.num_classes, model_dim)
        self.class_global_bias_proj = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
        )
        self.class_film_proj = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim * 2),
        )
        self.position_embedding = nn.Embedding(self.max_strokes + 1, model_dim)
        self.input_norm = nn.LayerNorm(model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(model_dim)
        self.class_output_norm = nn.LayerNorm(model_dim)

        self.shape_head = nn.Linear(model_dim, self.shape_vocab_size + 1)
        self.loc_condition_proj = nn.Sequential(
            nn.LayerNorm(model_dim * 2),
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        self.loc_head = nn.Linear(model_dim, self.loc_vocab_size + 1)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)

    def _embed_tokens(
        self,
        tokens: torch.Tensor,
        vocab_size: int,
        codebook: torch.Tensor,
        projector: nn.Module,
        special_embedding: nn.Embedding,
        residual_embedding: nn.Embedding,
    ) -> torch.Tensor:
        bsz, seq_len = tokens.shape
        base_dtype = self.position_embedding.weight.dtype
        out = torch.zeros((bsz, seq_len, self.model_dim), device=tokens.device, dtype=base_dtype)

        normal_mask = tokens < vocab_size
        if normal_mask.any():
            normal_ids = tokens[normal_mask].long()
            code_feats = codebook.index_select(0, normal_ids)
            projected = projector(code_feats).to(dtype=base_dtype)
            out[normal_mask] = projected

        special_mask = ~normal_mask
        if special_mask.any():
            special_ids = (tokens[special_mask] - vocab_size).clamp(min=0, max=2).long()
            special = special_embedding(special_ids).to(dtype=base_dtype)
            out[special_mask] = special

        residual_ids = tokens.clamp(min=0, max=residual_embedding.num_embeddings - 1).long()
        residual = residual_embedding(residual_ids).to(dtype=base_dtype)
        return out + self.token_residual_scale * residual

    def _encode_prefix(self, class_ids: torch.Tensor, input_shape: torch.Tensor, input_loc: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_shape.shape
        positions = torch.arange(seq_len, device=input_shape.device).unsqueeze(0).expand(bsz, -1)
        base_dtype = self.position_embedding.weight.dtype
        class_token = self.class_token_embedding(class_ids).to(dtype=base_dtype)
        class_bias = self.class_global_bias_proj(class_token).unsqueeze(1)

        x = (
            self._embed_tokens(
                input_shape,
                self.shape_vocab_size,
                self.shape_codebook,
                self.shape_codebook_proj,
                self.shape_special_embedding,
                self.shape_residual_embedding,
            )
            + self._embed_tokens(
                input_loc,
                self.loc_vocab_size,
                self.loc_codebook,
                self.loc_codebook_proj,
                self.loc_special_embedding,
                self.loc_residual_embedding,
            )
            + self.position_embedding(positions)
            + class_bias
        )

        if seq_len > 0:
            x[:, 0, :] = class_token

        x = self.input_norm(x)
        causal_mask = self._causal_mask(seq_len, input_shape.device)
        pad_mask = (input_shape == self.shape_pad) & (input_loc == self.loc_pad)
        if seq_len > 0:
            pad_mask = pad_mask.clone()
            pad_mask[:, 0] = False

        hidden = self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        hidden = self.output_norm(hidden)

        class_scale, class_shift = self.class_film_proj(class_token).chunk(2, dim=-1)
        class_scale = torch.tanh(class_scale).unsqueeze(1)
        class_shift = class_shift.unsqueeze(1)
        hidden = hidden * (1.0 + class_scale) + class_shift
        return self.class_output_norm(hidden)

    def _loc_logits_from_hidden(self, hidden: torch.Tensor, current_shape_tokens: torch.Tensor) -> torch.Tensor:
        shape_condition = self._embed_tokens(
            current_shape_tokens,
            self.shape_vocab_size,
            self.shape_codebook,
            self.shape_codebook_proj,
            self.shape_special_embedding,
            self.shape_residual_embedding,
        )
        loc_hidden = self.loc_condition_proj(torch.cat([hidden, shape_condition], dim=-1))
        return self.loc_head(loc_hidden)

    def _predict_loc_tokens_from_hidden(self, hidden: torch.Tensor, current_shape_tokens: torch.Tensor) -> torch.Tensor:
        return self._loc_logits_from_hidden(hidden, current_shape_tokens).argmax(dim=-1)

    def forward(
        self,
        class_ids: torch.Tensor,
        input_shape: torch.Tensor,
        input_loc: torch.Tensor,
        loc_condition_shape: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        hidden = self._encode_prefix(class_ids=class_ids, input_shape=input_shape, input_loc=input_loc)
        shape_logits = self.shape_head(hidden)
        if loc_condition_shape is None:
            loc_condition_shape = shape_logits.argmax(dim=-1)
        loc_logits = self._loc_logits_from_hidden(hidden, loc_condition_shape.long())
        return {"hidden": hidden, "shape_logits": shape_logits, "loc_logits": loc_logits}

    @torch.no_grad()
    def sample(
        self,
        class_ids: torch.Tensor,
        max_steps: int | None = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = class_ids.device
        batch_size = int(class_ids.shape[0])
        max_steps = int(max_steps or self.max_strokes)

        input_shape = torch.full((batch_size, 1), fill_value=self.shape_start, device=device, dtype=torch.long)
        input_loc = torch.full((batch_size, 1), fill_value=self.loc_start, device=device, dtype=torch.long)

        finished = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        generated_shape: List[torch.Tensor] = []
        generated_loc: List[torch.Tensor] = []
        lengths = torch.zeros((batch_size,), dtype=torch.long, device=device)

        for step in range(max_steps + 1):
            hidden = self._encode_prefix(class_ids=class_ids, input_shape=input_shape, input_loc=input_loc)
            shape_logits = self.shape_head(hidden)[:, -1, :]
            next_shape = nucleus_sample(shape_logits, temperature=temperature, top_p=top_p)

            loc_logits = self._loc_logits_from_hidden(hidden[:, -1:, :], next_shape.unsqueeze(1))[:, -1, :]
            next_loc = nucleus_sample(loc_logits, temperature=temperature, top_p=top_p)

            next_finished = (next_shape == self.shape_end) | (next_loc == self.loc_end)

            safe_shape = next_shape.clone()
            safe_loc = next_loc.clone()
            safe_shape[next_finished] = self.shape_end
            safe_loc[next_finished] = self.loc_end

            just_finished = (~finished) & next_finished
            lengths = torch.where(just_finished, torch.full_like(lengths, step), lengths)
            finished = finished | next_finished

            generated_shape.append(safe_shape)
            generated_loc.append(safe_loc)

            if finished.all():
                break

            append_shape = safe_shape.clone()
            append_loc = safe_loc.clone()
            append_shape[finished] = self.shape_pad
            append_loc[finished] = self.loc_pad

            input_shape = torch.cat([input_shape, append_shape.unsqueeze(1)], dim=1)
            input_loc = torch.cat([input_loc, append_loc.unsqueeze(1)], dim=1)

        if generated_shape:
            shape_seq = torch.stack(generated_shape, dim=1)
            loc_seq = torch.stack(generated_loc, dim=1)
        else:
            shape_seq = torch.empty((batch_size, 0), device=device, dtype=torch.long)
            loc_seq = torch.empty((batch_size, 0), device=device, dtype=torch.long)

        if not finished.all():
            lengths[~finished] = int(shape_seq.size(1))

        return shape_seq, loc_seq, lengths

def run_generator_epoch(
    model,
    loader,
    device,
    optimizer=None,
    mixed_precision=True,
    grad_clip_norm=None,
    teacher_force_ratio: float = 1.0,
):
    is_train = optimizer is not None
    model.train(is_train)
    scaler = GradScaler("cuda", enabled=mixed_precision and device.type == "cuda") if is_train else None
    total_meter, shape_meter, loc_meter = AverageMeter(), AverageMeter(), AverageMeter()

    iterator = tqdm(loader, leave=False)
    for batch in iterator:
        class_ids = batch["class_id"].to(device)
        input_shape = batch["input_shape"].to(device)
        input_loc = batch["input_loc"].to(device)
        target_shape = batch["target_shape"].to(device)
        target_loc = batch["target_loc"].to(device)

        if is_train:
            input_shape, input_loc = build_mixed_teacher_inputs(
                model=model,
                class_ids=class_ids,
                input_shape=input_shape,
                input_loc=input_loc,
                target_shape=target_shape,
                target_loc=target_loc,
                teacher_force_ratio=teacher_force_ratio,
            )
            optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=mixed_precision and device.type == "cuda"):
            hidden = model._encode_prefix(class_ids=class_ids, input_shape=input_shape, input_loc=input_loc)
            shape_logits_full = model.shape_head(hidden)
            pred_shape = shape_logits_full.detach().argmax(dim=-1)

            if is_train:
                loc_condition_shape = build_mixed_loc_condition_shapes(
                    model=model,
                    target_shape=target_shape,
                    pred_shape=pred_shape,
                    teacher_force_ratio=teacher_force_ratio,
                )
            else:
                loc_condition_shape = pred_shape

            loc_logits_full = model._loc_logits_from_hidden(hidden, loc_condition_shape.long())

            position_weights = build_generator_position_weights(
                target_tokens=target_shape,
                pad_idx=model.shape_pad,
                min_weight=float(cfg["generator"].get("early_stroke_loss_min_weight", 1.0)),
                max_weight=float(cfg["generator"].get("early_stroke_loss_max_weight", 1.0)),
                power=float(cfg["generator"].get("early_stroke_loss_power", 1.0)),
            )

            shape_loss = weighted_token_cross_entropy(
                logits=shape_logits_full,
                targets=target_shape,
                pad_idx=model.shape_pad,
                position_weights=position_weights,
            )
            loc_loss = weighted_token_cross_entropy(
                logits=loc_logits_full,
                targets=target_loc,
                pad_idx=model.loc_pad,
                position_weights=position_weights,
            )
            loss = shape_loss + loc_loss

        if is_train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            maybe_clip_grad(model, grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

        bsz = int(class_ids.size(0))
        total_meter.update(float(loss.item()), bsz)
        shape_meter.update(float(shape_loss.item()), bsz)
        loc_meter.update(float(loc_loss.item()), bsz)
        iterator.set_description(f"loss={total_meter.avg:.4f} shape={shape_meter.avg:.4f} loc={loc_meter.avg:.4f}")

    return {"loss": total_meter.avg, "shape_loss": shape_meter.avg, "loc_loss": loc_meter.avg}

def evaluate_generator_teacher_forced(model, loader, device, mixed_precision=True):
    model.eval()
    shape_correct = 0
    shape_total = 0
    loc_correct = 0
    loc_total = 0

    with torch.no_grad():
        for batch in loader:
            class_ids = batch["class_id"].to(device)
            input_shape = batch["input_shape"].to(device)
            input_loc = batch["input_loc"].to(device)
            target_shape = batch["target_shape"].to(device)
            target_loc = batch["target_loc"].to(device)

            with autocast("cuda", enabled=mixed_precision and device.type == "cuda"):
                hidden = model._encode_prefix(class_ids=class_ids, input_shape=input_shape, input_loc=input_loc)
                shape_logits = model.shape_head(hidden)
                shape_pred = shape_logits.argmax(dim=-1)
                loc_pred = model._predict_loc_tokens_from_hidden(hidden, shape_pred.long())

            shape_mask = target_shape != model.shape_pad
            loc_mask = target_loc != model.loc_pad

            shape_correct += int(((shape_pred == target_shape) & shape_mask).sum().item())
            shape_total += int(shape_mask.sum().item())
            loc_correct += int(((loc_pred == target_loc) & loc_mask).sum().item())
            loc_total += int(loc_mask.sum().item())

    return {
        "shape_token_acc": shape_correct / max(1, shape_total),
        "location_token_acc": loc_correct / max(1, loc_total),
    }

def _to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    return img

def save_gif(frames: Sequence[np.ndarray], out_path: str | Path, fps: int = 4) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_frames = [_to_rgb(frame.astype(np.uint8)) for frame in frames]
    imageio.mimsave(out_path, rgb_frames, duration=1.0 / max(1, fps))
    return out_path

def save_triptych(prefix_canvas: np.ndarray, pred_canvas: np.ndarray, gt_canvas: np.ndarray, out_path: str | Path) -> Path:
    import matplotlib.pyplot as plt
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    titles = ["prefix", "pred full", "gt full"]
    for ax, img, title in zip(axes, [prefix_canvas, pred_canvas, gt_canvas], titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def evaluate_continuations(cfg, generator, shape_ae, location_ae, shape_tokenizer, location_tokenizer, test_npz_path, device):
    npz = np.load(test_npz_path)
    class_ids = npz["class_ids"].astype(np.int64)
    lengths = npz["lengths"].astype(np.int64)
    shape_tokens = npz["shape_tokens"].astype(np.int64)
    loc_tokens = npz["loc_tokens"].astype(np.int64)

    primer = int(cfg["continuation_eval"]["primer_strokes"])
    metric_samples = int(cfg["continuation_eval"]["metric_samples"])
    preview_samples = int(cfg["continuation_eval"]["preview_samples"])
    temperature = float(cfg["continuation_eval"]["temperature"])
    top_p = float(cfg["continuation_eval"]["top_p"])

    valid_indices = np.where(lengths >= max(2, primer + 1))[0]
    if valid_indices.size == 0:
        return {"exact_match_ratio": 0.0, "mean_completion_iou": 0.0, "mean_full_canvas_iou": 0.0}, []

    rng = np.random.default_rng(int(cfg["project"]["seed"]))
    metric_indices = rng.choice(valid_indices, size=min(metric_samples, valid_indices.size), replace=False)

    exact_matches = []
    completion_ious = []
    full_ious = []
    preview_paths = []

    for rank, idx in enumerate(metric_indices):
        class_id = int(class_ids[idx])
        length = int(lengths[idx])
        gt_shape = shape_tokens[idx, :length]
        gt_loc = loc_tokens[idx, :length]
        prefix_shape = gt_shape[:primer]
        prefix_loc = gt_loc[:primer]

        pred_shape, pred_loc = sample_continuation(
            generator=generator,
            class_id=class_id,
            prefix_shape_tokens=prefix_shape,
            prefix_loc_tokens=prefix_loc,
            device=device,
            max_steps=int(cfg["dataset"]["max_strokes"]),
            temperature=temperature,
            top_p=top_p,
        )

        gt_canvas, gt_frames = decode_token_sequence(cfg, shape_ae, location_ae, shape_tokenizer, location_tokenizer, gt_shape, gt_loc)
        prefix_canvas, _ = decode_token_sequence(cfg, shape_ae, location_ae, shape_tokenizer, location_tokenizer, prefix_shape, prefix_loc)
        pred_canvas, pred_frames = decode_token_sequence(cfg, shape_ae, location_ae, shape_tokenizer, location_tokenizer, pred_shape, pred_loc)

        gt_completion = np.maximum(gt_canvas.astype(np.int16) - prefix_canvas.astype(np.int16), 0).astype(np.uint8)
        pred_completion = np.maximum(pred_canvas.astype(np.int16) - prefix_canvas.astype(np.int16), 0).astype(np.uint8)

        exact_matches.append(float(np.array_equal(pred_shape, gt_shape) and np.array_equal(pred_loc, gt_loc)))
        completion_ious.append(binary_iou(pred_completion, gt_completion))
        full_ious.append(binary_iou(pred_canvas, gt_canvas))

        if rank < preview_samples:
            triptych_path = artifact_root(cfg) / f"continuation_preview_{rank:03d}.png"
            gif_path = artifact_root(cfg) / f"continuation_pred_{rank:03d}.gif"
            save_triptych(prefix_canvas, pred_canvas, gt_canvas, triptych_path)
            save_gif(pred_frames if len(pred_frames) > 0 else [pred_canvas], gif_path)
            preview_paths.append(str(triptych_path))
            preview_paths.append(str(gif_path))

    metrics = {
        "exact_match_ratio": float(np.mean(exact_matches)),
        "mean_completion_iou": float(np.mean(completion_ious)),
        "mean_full_canvas_iou": float(np.mean(full_ious)),
    }
    return metrics, preview_paths

def decode_token_sequence(cfg, shape_ae, location_ae, shape_tokenizer, location_tokenizer, shape_tokens, loc_tokens):
    device = next(shape_ae.parameters()).device
    stats = load_embedding_stats(cfg)
    shape_arr = np.asarray(shape_tokens, dtype=np.int64)
    loc_arr = np.asarray(loc_tokens, dtype=np.int64)
    max_shape = int(cfg["shape_tokenizer"]["num_embeddings"])
    max_loc = int(cfg["location_tokenizer"]["num_embeddings"])

    valid_len = 0
    for s, l in zip(shape_arr, loc_arr):
        if s < 0 or l < 0 or s >= max_shape or l >= max_loc:
            break
        valid_len += 1

    if valid_len == 0:
        return np.zeros((int(cfg["dataset"]["canvas_size"]), int(cfg["dataset"]["canvas_size"])), dtype=np.uint8), []

    with torch.no_grad():
        shape_idx = torch.from_numpy(shape_arr[:valid_len]).long().unsqueeze(0).to(device)
        loc_idx = torch.from_numpy(loc_arr[:valid_len]).long().unsqueeze(0).to(device)
        valid_mask = torch.ones((1, valid_len), device=device, dtype=torch.float32)

        shape_emb_norm = shape_tokenizer.decode_sequence_indices(shape_idx, valid_mask)["recon_seq"].squeeze(0)
        shape_emb = denormalize_feature_batch(shape_emb_norm, stats["shape_mean"], stats["shape_std"])
        decoded_shapes = shape_ae.decode_from_embedding(shape_emb)["bitmap"].squeeze(1).cpu().numpy()

        loc_emb_norm = location_tokenizer.decode_sequence_indices(loc_idx, valid_mask)["recon_seq"].squeeze(0)
        loc_emb = denormalize_feature_batch(loc_emb_norm, stats["loc_mean"], stats["loc_std"])
        decoded_bboxes = location_ae.decode_from_embedding(loc_emb).cpu().numpy()

    decoded_shapes = np.clip(decoded_shapes * 255.0, 0, 255).astype(np.uint8)
    decoded_bboxes = np.clip(decoded_bboxes, 0.0, 1.0)
    canvas, frames = compose_strokes_from_shape_and_location(
        decoded_shapes,
        decoded_bboxes,
        canvas_size=int(cfg["dataset"]["canvas_size"]),
        decode_threshold=float(cfg["dataset"]["shape_decode_threshold"]),
        use_bbox_size=bool(cfg["dataset"].get("decode_use_bbox_size", False)),
    )
    return canvas, frames

def sample_continuation(generator, class_id, prefix_shape_tokens, prefix_loc_tokens, device, max_steps, temperature=1.0, top_p=0.9):
    generator.eval()
    input_shape = [generator.shape_start] + [int(x) for x in np.asarray(prefix_shape_tokens, dtype=np.int64).tolist()]
    input_loc = [generator.loc_start] + [int(x) for x in np.asarray(prefix_loc_tokens, dtype=np.int64).tolist()]

    with torch.no_grad():
        for _ in range(max_steps):
            class_ids = torch.tensor([class_id], dtype=torch.long, device=device)
            shape_t = torch.tensor([input_shape], dtype=torch.long, device=device)
            loc_t = torch.tensor([input_loc], dtype=torch.long, device=device)

            hidden = generator._encode_prefix(class_ids=class_ids, input_shape=shape_t, input_loc=loc_t)
            shape_logits = generator.shape_head(hidden)
            next_shape = int(nucleus_sample(shape_logits[:, -1, :], temperature=temperature, top_p=top_p).item())
            if next_shape == generator.shape_end:
                break

            shape_cond = torch.tensor([[next_shape]], dtype=torch.long, device=device)
            loc_logits = generator._loc_logits_from_hidden(hidden[:, -1:, :], shape_cond)
            next_loc = int(nucleus_sample(loc_logits[:, -1, :], temperature=temperature, top_p=top_p).item())
            if next_loc == generator.loc_end:
                break

            input_shape.append(next_shape)
            input_loc.append(next_loc)

            if len(input_shape) - 1 >= int(generator.max_strokes):
                break

    shape_seq = np.array(input_shape[1:], dtype=np.int64)
    loc_seq = np.array(input_loc[1:], dtype=np.int64)
    return shape_seq, loc_seq

def sample_class_conditioned_sketches(
    cfg,
    generator,
    shape_ae,
    location_ae,
    shape_tokenizer,
    location_tokenizer,
    class_id: int,
    num_samples: int,
    device,
    max_steps: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
):
    generator.eval()
    num_samples = int(max(1, num_samples))
    max_steps = int(max_steps or cfg["dataset"]["max_strokes"])
    temperature = float(cfg["generator"]["temperature"] if temperature is None else temperature)
    top_p = float(cfg["generator"]["top_p"] if top_p is None else top_p)

    class_ids = torch.full((num_samples,), fill_value=int(class_id), dtype=torch.long, device=device)
    shape_seq, loc_seq, lengths = generator.sample(
        class_ids=class_ids,
        max_steps=max_steps,
        temperature=temperature,
        top_p=top_p,
    )

    shape_np = shape_seq.detach().cpu().numpy()
    loc_np = loc_seq.detach().cpu().numpy()
    lengths_np = lengths.detach().cpu().numpy().astype(np.int64)

    samples: List[Dict[str, Any]] = []
    for i in range(num_samples):
        seq_cap = shape_np.shape[1] if shape_np.ndim == 2 else 0
        L = int(np.clip(lengths_np[i], 0, seq_cap))
        shape_tokens_i = shape_np[i, :L] if L > 0 else np.zeros((0,), dtype=np.int64)
        loc_tokens_i = loc_np[i, :L] if L > 0 else np.zeros((0,), dtype=np.int64)
        canvas_i, frames_i = decode_token_sequence(
            cfg,
            shape_ae,
            location_ae,
            shape_tokenizer,
            location_tokenizer,
            shape_tokens_i,
            loc_tokens_i,
        )
        samples.append(
            {
                "canvas": canvas_i,
                "frames": frames_i,
                "shape_tokens": shape_tokens_i,
                "loc_tokens": loc_tokens_i,
                "length": L,
            }
        )
    return samples

