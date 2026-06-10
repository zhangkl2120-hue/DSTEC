import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ema_torch(x, alpha=0.3):
    out = torch.zeros_like(x)
    out[..., 0] = x[..., 0]
    for t in range(1, x.shape[-1]):
        out[..., t] = alpha * x[..., t] + (1 - alpha) * out[..., t - 1]
    return out


def _zscore_torch(x, eps=1e-6):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp_min(eps)
    return (x - mean) / std


def _l2_normalize_torch(x, eps=1e-6):
    norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(eps)
    return x / norm


def build_query_key_torch(x0, mask=None, alpha=0.3):
    if mask is not None:
        denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        obs_mean = (x0 * mask).sum(dim=-1, keepdim=True) / denom
        x_work = mask * x0 + (1.0 - mask) * obs_mean
    else:
        x_work = x0

    shape_key = _zscore_torch(x_work)

    trend = _ema_torch(x_work, alpha=alpha)
    residual_key = _zscore_torch(x_work - trend)

    diff = x_work[..., 1:] - x_work[..., :-1]
    diff = F.pad(diff, (1, 0), mode="constant", value=0.0)
    diff_key = _zscore_torch(diff)

    key = torch.cat([shape_key, residual_key, diff_key], dim=-1)
    key = _l2_normalize_torch(key)
    return key


class PatternBankImputer(nn.Module):
    def __init__(
        self,
        bank_path,
        top_m=5,
        tau=0.2,
        feature_idx=0,
        top_proto=3,
        alpha=0.3,
        lambda_key=0.1,
    ):
        super().__init__()

        bank = np.load(bank_path)

        self.top_m = top_m
        self.tau = tau
        self.feature_idx = feature_idx
        self.top_proto = top_proto
        self.alpha = alpha
        self.lambda_key = lambda_key

        self.use_proto = ("prototype_keys" in bank.files) and ("donor_values" in bank.files)

        if self.use_proto:
            prototype_keys = bank["prototype_keys"].astype(np.float32)
            donor_values = bank["donor_values"].astype(np.float32)
            donor_keys = bank["donor_keys"].astype(np.float32)

            self.register_buffer("prototype_keys", torch.from_numpy(prototype_keys))
            self.register_buffer("donor_values", torch.from_numpy(donor_values))
            self.register_buffer("donor_keys", torch.from_numpy(donor_keys))

            self.N = donor_values.shape[0]
            self.C = donor_values.shape[1]
            self.R = donor_values.shape[2]
            self.L = donor_values.shape[3]

            print(
                f"[PatternBankImputer] proto mode | "
                f"prototype_keys={prototype_keys.shape}, "
                f"donor_values={donor_values.shape}, "
                f"donor_keys={donor_keys.shape}"
            )

        else:
            bank_values = bank["bank_values"].astype(np.float32)
            self.register_buffer("bank_values", torch.from_numpy(bank_values))

            self.N = bank_values.shape[0]
            self.K = bank_values.shape[1]
            self.L = bank_values.shape[2]

            print(
                f"[PatternBankImputer] flat mode | "
                f"bank_values={bank_values.shape}"
            )

    def forward(self, x):
        if self.use_proto:
            return self._forward_proto(x)
        else:
            return self._forward_flat(x)

    def _forward_flat(self, x):
        B, N, Fdim, L = x.shape

        x0 = x[:, :, self.feature_idx, :]
        mask = (x0 != 0).float()

        bank = self.bank_values.to(x.device)

        if bank.shape[0] != N:
            raise RuntimeError(f"Pattern bank N={bank.shape[0]}, but input N={N}")
        if bank.shape[-1] != L:
            raise RuntimeError(f"Pattern bank L={bank.shape[-1]}, but input L={L}")

        x_expand = x0.unsqueeze(2)
        m_expand = mask.unsqueeze(2)
        bank_expand = bank.unsqueeze(0)

        dist = ((x_expand - bank_expand) ** 2 * m_expand).sum(dim=-1)
        dist = dist / (m_expand.sum(dim=-1) + 1e-6)

        top_m = min(self.top_m, bank.shape[1])
        top_dist, top_idx = torch.topk(dist, k=top_m, dim=-1, largest=False)

        weight = torch.softmax(-top_dist / self.tau, dim=-1)

        bank_b = bank.unsqueeze(0).expand(B, -1, -1, -1)
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, -1, L)
        top_cand = torch.gather(bank_b, dim=2, index=gather_idx)

        x_hat = (weight.unsqueeze(-1) * top_cand).sum(dim=2)
        x_filled0 = mask * x0 + (1.0 - mask) * x_hat

        x_filled = x.clone()
        x_filled[:, :, self.feature_idx, :] = x_filled0

        conf = self._confidence_from_dist(top_dist, B, N, x.device)

        return x_filled, conf

    def _forward_proto(self, x):
        B, N, Fdim, L = x.shape

        x0 = x[:, :, self.feature_idx, :]
        mask = (x0 != 0).float()

        donor_values = self.donor_values.to(x.device)
        donor_keys = self.donor_keys.to(x.device)
        prototype_keys = self.prototype_keys.to(x.device)

        if donor_values.shape[0] != N:
            raise RuntimeError(f"Pattern bank N={donor_values.shape[0]}, but input N={N}")
        if donor_values.shape[-1] != L:
            raise RuntimeError(f"Pattern bank L={donor_values.shape[-1]}, but input L={L}")

        C = donor_values.shape[1]
        R = donor_values.shape[2]
        D = prototype_keys.shape[-1]

        q_key = build_query_key_torch(x0, mask=mask, alpha=self.alpha)

        if q_key.shape[-1] != D:
            raise RuntimeError(f"Query key D={q_key.shape[-1]}, but prototype D={D}")

        dist_proto = ((q_key.unsqueeze(2) - prototype_keys.view(1, 1, C, D)) ** 2).mean(dim=-1)

        top_p = min(self.top_proto, C)
        _, proto_idx = torch.topk(dist_proto, k=top_p, dim=-1, largest=False)

        donor_values_expand = donor_values.unsqueeze(0).expand(B, -1, -1, -1, -1)
        gather_idx_v = proto_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, R, L)
        cand_values = torch.gather(donor_values_expand, dim=2, index=gather_idx_v)

        donor_keys_expand = donor_keys.unsqueeze(0).expand(B, -1, -1, -1, -1)
        gather_idx_k = proto_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, R, D)
        cand_keys = torch.gather(donor_keys_expand, dim=2, index=gather_idx_k)

        K = top_p * R
        cand_values = cand_values.reshape(B, N, K, L)
        cand_keys = cand_keys.reshape(B, N, K, D)

        x_expand = x0.unsqueeze(2)
        m_expand = mask.unsqueeze(2)

        dist_value = ((x_expand - cand_values) ** 2 * m_expand).sum(dim=-1)
        dist_value = dist_value / (m_expand.sum(dim=-1) + 1e-6)

        dist_key = ((q_key.unsqueeze(2) - cand_keys) ** 2).mean(dim=-1)

        dist = dist_value + self.lambda_key * dist_key

        top_m = min(self.top_m, K)
        top_dist, top_idx = torch.topk(dist, k=top_m, dim=-1, largest=False)

        weight = torch.softmax(-top_dist / self.tau, dim=-1)

        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, -1, L)
        top_cand = torch.gather(cand_values, dim=2, index=gather_idx)

        x_hat = (weight.unsqueeze(-1) * top_cand).sum(dim=2)

        x_filled0 = mask * x0 + (1.0 - mask) * x_hat

        x_filled = x.clone()
        x_filled[:, :, self.feature_idx, :] = x_filled0

        conf = self._confidence_from_dist(top_dist, B, N, x.device)

        return x_filled, conf

    @staticmethod
    def _confidence_from_dist(top_dist, B, N, device):
        if top_dist.shape[-1] >= 2:
            margin = top_dist[..., 1] - top_dist[..., 0]
            conf = torch.sigmoid(margin).unsqueeze(-1)
        else:
            conf = torch.ones(B, N, 1, device=device)
        return conf
