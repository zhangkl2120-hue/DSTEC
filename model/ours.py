import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
try:
    from .pattern_bank import PatternBankImputer
except ImportError:
    from pattern_bank import PatternBankImputer

class AdaptiveNoiseLayer(nn.Module):
    def __init__(self, input_dim, sigma=0.1):
        super().__init__()
        hidden = max(1, input_dim // 2)
        self.sigma = sigma
        self.noise_gate = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
            nn.Sigmoid()
        )
        self.enabled = True

    def set_enabled(self, flag: bool):
        self.enabled = flag

    def forward(self, x, obs_mask=None):
        x_flat = x.permute(0, 1, 3, 2)
        out = x_flat
        if self.training and self.enabled and self.sigma > 0:
            if obs_mask is None:
                obs_mask = (x != 0).float()
            m = obs_mask.permute(0, 1, 3, 2)

            alpha = self.noise_gate(x_flat) * m
            noise = torch.randn_like(x_flat) * m
            out = x_flat + (alpha * self.sigma) * noise

        return out.permute(0, 1, 3, 2)


def ema_trend(x, alpha=0.3):
    B,N,F,T = x.shape
    out = torch.zeros_like(x)
    out[..., 0] = x[..., 0]
    for t in range(1, T):
        out[..., t] = alpha * x[..., t] + (1 - alpha) * out[..., t-1]
    return out

def robust_view(x, missing_is_zero=True, alpha=0.3, clip_k=3.0, eps=1e-6):
    if missing_is_zero:
        mask = (x != 0).float()
    else:
        mask = torch.ones_like(x)

    trend = ema_trend(x, alpha=alpha)
    r = (x - trend) * mask

    mu = r.mean(dim=-1, keepdim=True)
    std = r.std(dim=-1, keepdim=True).clamp_min(eps)
    r = torch.clamp(r, mu - clip_k * std, mu + clip_k * std)

    return r

class TemporalGraphLearner(nn.Module):
    def __init__(self, num_nodes, in_channels, T, emb_dim=32, topk=20, tau=0.2,
                 ema_decay=0.9, use_ema=True):
        super().__init__()
        self.num_nodes = num_nodes
        self.topk = topk
        self.tau = tau
        self.ema_decay = ema_decay
        self.use_ema = use_ema
        self.freeze_graph = False

        self.encoder = nn.Sequential(
            nn.Linear(in_channels * T, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim)
        )
        if use_ema:
            self.register_buffer("A_ema", torch.zeros(num_nodes, num_nodes))
            self.register_buffer("_ema_inited", torch.tensor(False))

    def set_freeze(self, flag: bool = True):
        self.freeze_graph = flag
    def forward(self, x, reduce="mean"):
        if self.use_ema and self.freeze_graph and self._ema_inited.item():
            return self.A_ema
        B, N, Fin, T = x.shape
        assert N == self.num_nodes, f"N mismatch: got {N}, expected {self.num_nodes}"

        feat = x.permute(0, 1, 3, 2).reshape(B, N, T * Fin)

        E = self.encoder(feat)

        if reduce == "mean":
            E = E.mean(dim=0)
        elif reduce == "median":
            E = E.median(dim=0).values
        elif reduce is None:
            raise ValueError("reduce=None is slow,please use mean/median")
        else:
            raise ValueError(f"Unknown reduce={reduce}")


        logits = (E @ E.t()) / math.sqrt(E.size(-1))
        logits = F.relu(logits)

        logits.fill_diagonal_(-1e9)
        if self.topk is not None and self.topk < N:
            _, idx = torch.topk(logits, self.topk, dim=1)
            mask = torch.full_like(logits, -1e9)
            mask.scatter_(1, idx, 0.0)
            logits = logits + mask

        A = torch.softmax(logits / self.tau, dim=1)

        if not self.use_ema:
            return A

        if not self._ema_inited.item():
            self.A_ema.copy_(A.detach())
            self._ema_inited.fill_(True)

        if self.training and (not self.freeze_graph):
            self.A_ema.mul_(self.ema_decay).add_(A.detach(), alpha=1 - self.ema_decay)

        return self.A_ema






class AdaptiveGraphGenerator(nn.Module):
    def __init__(self, num_nodes, k=10, embedding_dim=40):
        super(AdaptiveGraphGenerator, self).__init__()
        self.node_vec1 = nn.Parameter(torch.randn(num_nodes, embedding_dim), requires_grad=True)
        self.node_vec2 = nn.Parameter(torch.randn(embedding_dim, num_nodes), requires_grad=True)
        self.k = k
        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.node_vec1)
        nn.init.xavier_uniform_(self.node_vec2)

    def forward(self):
        A = F.relu(self.node_vec1 @ self.node_vec2)
        A = A.masked_fill(torch.eye(A.size(0), device=A.device, dtype=torch.bool), -1e9)

        if self.k is not None and self.k < A.size(1):
            _, idx = torch.topk(A, self.k, dim=1)
            neg = -1e9 if A.dtype == torch.float32 else -1e4
            mask = torch.full_like(A, neg)
            mask.scatter_(1, idx, 0.0)
            A = A + mask

        A = F.softmax(A, dim=1)
        return A

class GATSpatialTopK(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.0, leaky=0.2):
        super().__init__()
        self.heads = heads
        self.d = out_dim // heads
        self.scale = math.sqrt(self.d)
        assert out_dim % heads == 0
        self.W = nn.Linear(in_dim, heads * self.d, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, self.d))
        self.a_dst = nn.Parameter(torch.randn(heads, self.d))
        self.leaky = nn.LeakyReLU(leaky)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_bt, neigh_idx, edge_bias=None):
        B, N, _ = x_bt.shape
        K = neigh_idx.shape[1]

        h = self.W(x_bt).view(B, N, self.heads, self.d)

        neighbors = neigh_idx.reshape(-1)
        h_neighbors = h[:, neighbors]
        h_j = h_neighbors.view(B, N, K, self.heads, self.d)
        h_i = h.unsqueeze(2)

        e = (h_i * self.a_src).sum(-1) + (h_j * self.a_dst).sum(-1)
        e = self.leaky(e)

        if edge_bias is not None:
            e = e + edge_bias.unsqueeze(0).unsqueeze(-1)
        e = e / self.scale
        alpha = torch.softmax(e, dim=2)
        alpha = self.drop(alpha)

        out = (alpha.unsqueeze(-1) * h_j).sum(dim=2)
        out = out.reshape(B, N, self.heads * self.d)
        return out


class DilatedTCNTimeConv(nn.Module):
    def __init__(self, C_in, C_out, stride_t=1, k=3, dilations=(1, 2,4), dropout=0.0, mid_act=True):
        super().__init__()
        assert k % 2 == 1, "k same padding"

        layers = []
        for i, d in enumerate(dilations):
            in_ch = C_in if i == 0 else C_out
            stride = (1, stride_t) if i == 0 else (1, 1)
            pad = d * (k - 1) // 2

            layers.append(
                nn.Conv2d(in_ch, C_out, kernel_size=(1, k),
                          stride=stride, padding=(0, pad), dilation=(1, d))
            )
            if i != len(dilations) - 1 and mid_act:
                layers += [nn.ReLU(), nn.Dropout(dropout)]

        self.net = nn.Sequential(*layers)
        if (C_in != C_out) or (stride_t != 1):
            self.shortcut = nn.Conv2d(C_in, C_out, kernel_size=(1, 1),
                                      stride=(1, stride_t), padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        y = self.net(x)
        return y + self.shortcut(x)


class STPositionalEncoding(nn.Module):
    def __init__(self, num_nodes, channels, max_len=64, dropout=0.1,
                 use_se=True, use_te=True, init_scale=0.1):
        super().__init__()
        self.use_se = use_se
        self.use_te = use_te
        self.dropout = nn.Dropout(dropout)

        if use_se:
            self.node_emb = nn.Parameter(torch.empty(num_nodes, channels))
            nn.init.xavier_uniform_(self.node_emb)
            self.se_scale = nn.Parameter(torch.tensor(float(init_scale)))
        else:
            self.register_parameter('node_emb', None)
            self.register_parameter('se_scale', None)

        if use_te:
            pe = torch.zeros(max_len, channels)
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, channels, 2, dtype=torch.float32) *
                                 (-(math.log(10000.0) / max(channels, 1))))
            pe[:, 0::2] = torch.sin(position * div_term)
            if channels > 1:
                pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
            self.register_buffer('temporal_pe', pe)
            self.te_scale = nn.Parameter(torch.tensor(float(init_scale)))
        else:
            self.register_buffer('temporal_pe', torch.zeros(1, channels))
            self.register_parameter('te_scale', None)

    def forward(self, x):
        B, N, C, T = x.shape
        out = x
        if self.use_se and self.node_emb is not None:
            if N != self.node_emb.size(0):
                raise ValueError(f"Node number mismatch in STPositionalEncoding: got {N}, expected {self.node_emb.size(0)}")
            out = out + self.se_scale * self.node_emb.unsqueeze(0).unsqueeze(-1)
        if self.use_te:
            if T > self.temporal_pe.size(0):
                pe = torch.zeros(T, C, device=x.device, dtype=x.dtype)
                position = torch.arange(0, T, dtype=x.dtype, device=x.device).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, C, 2, dtype=x.dtype, device=x.device) *
                                     (-(math.log(10000.0) / max(C, 1))))
                pe[:, 0::2] = torch.sin(position * div_term)
                if C > 1:
                    pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
            else:
                pe = self.temporal_pe[:T].to(device=x.device, dtype=x.dtype)
            out = out + self.te_scale * pe.transpose(0, 1).unsqueeze(0).unsqueeze(0)
        return self.dropout(out)




class STGAT_block(nn.Module):
    def __init__(self, in_channels, out_channels, time_strides=1, num_nodes=None, max_len=64, dropout=0.1, use_pe=True):
        super().__init__()
        self.time_conv = DilatedTCNTimeConv(
            in_channels, out_channels, stride_t=time_strides, k=3, dilations=(1, 2, 4), dropout=dropout
        )
        self.pe = STPositionalEncoding(num_nodes, out_channels, max_len=max_len, dropout=dropout) if (use_pe and num_nodes is not None) else nn.Identity()
        self.s_gat = GATSpatialTopK(out_channels, out_channels, heads=4, dropout=0.1)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(out_channels)

    def forward(self, x, neigh_idx, edge_bias=None):
        h = self.time_conv(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        h = self.pe(h)
        B, N, Fout, Tt = h.shape

        h_bt = h.permute(0, 3, 1, 2).reshape(B * Tt, N, Fout)
        out_bt = self.s_gat(h_bt, neigh_idx, edge_bias=edge_bias)
        h2 = out_bt.view(B, Tt, N, Fout).permute(0, 2, 3, 1)
        h2 = torch.relu(h2)

        res = self.residual(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        y = self.ln((h2 + res).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return y


class GAT_Submodule(nn.Module):

    def __init__(self, DEVICE, nb_block, in_channels, nb_time_filter, time_strides,
                 num_for_predict, len_input, topk=20, num_nodes=None, dropout=0.1, use_pe=True):
        super().__init__()
        self.topk = topk

        self.BlockList = nn.ModuleList()
        max_len = int(len_input / time_strides)
        self.BlockList.append(STGAT_block(in_channels, nb_time_filter, time_strides=time_strides,
                                          num_nodes=num_nodes, max_len=max_len, dropout=dropout, use_pe=use_pe))
        for _ in range(nb_block - 1):
            self.BlockList.append(STGAT_block(nb_time_filter, nb_time_filter, time_strides=1,
                                              num_nodes=num_nodes, max_len=max_len, dropout=dropout, use_pe=use_pe))

        self.final_conv = nn.Conv2d(int(len_input / time_strides), num_for_predict,
                                    kernel_size=(1, nb_time_filter))


    def forward(self, x, dyn_adj=None, return_stage1=False):
        if dyn_adj is None:
            raise ValueError("dyn_adj is required")
        N = dyn_adj.shape[0]
        k = min(self.topk, N - 1)

        _, neigh_idx = torch.topk(dyn_adj, k=k, dim=1)
        edge_w = dyn_adj.gather(1, neigh_idx)
        edge_bias = torch.log(edge_w + 1e-6)

        x_stage1 = None
        for i, block in enumerate(self.BlockList):
            x = block(x, neigh_idx=neigh_idx, edge_bias=edge_bias)
            if return_stage1 and i == 0:
                x_stage1 = x

        out = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)

        if not return_stage1:
            return out

        out1 = self.final_conv(x_stage1.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        return out, out1




def normalize_adj_with_self_loop(A, eps=1e-6):
    N = A.shape[0]
    I = torch.eye(N, device=A.device, dtype=A.dtype)
    A_hat = A.clamp_min(0.0) + I
    A_hat = A_hat / (A_hat.sum(dim=1, keepdim=True) + eps)
    return A_hat

class ASTSpatialAttentionGCN(nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim=None, dropout=0.1):
        super().__init__()
        if attn_dim is None:
            attn_dim = max(16, min(32, out_dim // 2))
        self.q_proj = nn.Linear(in_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(in_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.out_proj = nn.Linear(out_dim, out_dim, bias=False)
        self.scale = math.sqrt(attn_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_bt, adj):
        BT, N, _ = x_bt.shape
        q = self.q_proj(x_bt)
        k = self.k_proj(x_bt)
        v = self.v_proj(x_bt)

        logits = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        s_attn = torch.softmax(logits, dim=-1)

        A = adj.clamp_min(0.0)
        P = s_attn * A.unsqueeze(0)
        P = P / (P.sum(dim=-1, keepdim=True) + 1e-6)

        out = torch.matmul(P, v)
        out = self.out_proj(out)
        return self.drop(out)


class STASTGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_strides=1, dropout=0.1,
                 num_nodes=None, max_len=64, use_pe=True):
        super().__init__()
        self.time_conv = DilatedTCNTimeConv(
            in_channels, out_channels, stride_t=time_strides,
            k=3, dilations=(1, 2, 4), dropout=dropout
        )
        self.pe = STPositionalEncoding(num_nodes, out_channels, max_len=max_len, dropout=dropout) if (use_pe and num_nodes is not None) else nn.Identity()
        self.ast_gcn = ASTSpatialAttentionGCN(out_channels, out_channels, dropout=dropout)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(out_channels)

    def forward(self, x, adj):
        h = self.time_conv(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        h = self.pe(h)
        B, N, Fout, Tt = h.shape

        h_bt = h.permute(0, 3, 1, 2).reshape(B * Tt, N, Fout)
        out_bt = self.ast_gcn(h_bt, adj)
        h2 = out_bt.view(B, Tt, N, Fout).permute(0, 2, 3, 1)
        h2 = torch.relu(h2)

        res = self.residual(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        y = self.ln((h2 + res).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return y


class ASTGCN_Submodule(nn.Module):
    def __init__(self, DEVICE, nb_block, in_channels, nb_time_filter, time_strides,
                 num_for_predict, len_input, dropout=0.1, num_nodes=None, use_pe=True):
        super().__init__()
        self.BlockList = nn.ModuleList()
        max_len = int(len_input / time_strides)
        self.BlockList.append(STASTGCNBlock(in_channels, nb_time_filter, time_strides=time_strides, dropout=dropout,
                                            num_nodes=num_nodes, max_len=max_len, use_pe=use_pe))
        for _ in range(nb_block - 1):
            self.BlockList.append(STASTGCNBlock(nb_time_filter, nb_time_filter, time_strides=1, dropout=dropout,
                                                num_nodes=num_nodes, max_len=max_len, use_pe=use_pe))

        self.final_conv = nn.Conv2d(int(len_input / time_strides), num_for_predict,
                                    kernel_size=(1, nb_time_filter))

    def forward(self, x, dyn_adj=None, return_stage1=False, return_repr=False):
        if dyn_adj is None:
            raise ValueError("dyn_adj is required")

        adj = normalize_adj_with_self_loop(dyn_adj)

        x_stage1 = None
        for i, block in enumerate(self.BlockList):
            x = block(x, adj=adj)
            if return_stage1 and i == 0:
                x_stage1 = x

        H = x.mean(dim=-1)

        out = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)

        if return_repr:
            if not return_stage1:
                return out, H
            out1 = self.final_conv(x_stage1.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
            return out, out1, H

        if not return_stage1:
            return out

        out1 = self.final_conv(x_stage1.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        return out, out1


class DualBranchGAT(nn.Module):
    def __init__(self, DEVICE, nb_block, in_channels, nb_time_filter, time_strides,
                 num_for_predict, len_input, num_of_vertices, adj_mx_time, adj_mx_space,
                 pattern_bank_path=None, pattern_top_m=5, pattern_tau=0.2, **kwargs):
        super().__init__()

        self.num_for_predict = num_for_predict
        self.nb_time_filter = nb_time_filter

        self.adp_gen_space = AdaptiveGraphGenerator(num_of_vertices, k=20, embedding_dim=32)

        self.noise_layer = AdaptiveNoiseLayer(in_channels)

        if pattern_bank_path is not None and pattern_bank_path != "":
            self.pattern_bank = PatternBankImputer(
                bank_path=pattern_bank_path,
                top_m=pattern_top_m,
                tau=pattern_tau,
                feature_idx=0
            )
            print("[PatternBank] enabled:", pattern_bank_path)
        else:
            self.pattern_bank = None
            print("[PatternBank] disabled")

        self.t_graph = TemporalGraphLearner(
            num_nodes=num_of_vertices,
            in_channels=in_channels,
            T=len_input,
            emb_dim=32,
            topk=20,
            tau=0.2,
            use_ema=True,
            ema_decay=0.9
        )

        self.temporal_expert = ASTGCN_Submodule(
            DEVICE, nb_block, in_channels, nb_time_filter, time_strides,
            num_for_predict, len_input, dropout=0.1, num_nodes=num_of_vertices, use_pe=True
        )
        self.spatial_expert = ASTGCN_Submodule(
            DEVICE, nb_block, in_channels, nb_time_filter, time_strides,
            num_for_predict, len_input, dropout=0.1, num_nodes=num_of_vertices, use_pe=True
        )

        if adj_mx_space is not None:
            A_s = torch.tensor(adj_mx_space, dtype=torch.float32)
            A_s.fill_diagonal_(0)
            A_s = A_s / (A_s.max() + 1e-6)
            A_s = A_s / (A_s.sum(dim=1, keepdim=True) + 1e-6)
            self.register_buffer("A_s_base", A_s)
        else:
            self.register_buffer("A_s_base", torch.zeros(num_of_vertices, num_of_vertices))

        repr_dim = nb_time_filter
        concat_dim = repr_dim * 2
        self.pred_head = nn.Sequential(
            nn.Linear(concat_dim, repr_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(repr_dim, num_for_predict)
        )

    def forward(self, x, return_stage1=False):
        B, N, Fin, T_in = x.shape

        if self.pattern_bank is not None:
            x_filled, cache_conf = self.pattern_bank(x)
        else:
            x_filled, cache_conf = x, None

        adp_adj_s = 0.5 * self.A_s_base + 0.5 * self.adp_gen_space()

        use_cached_t = (
            self.t_graph.use_ema
            and self.t_graph._ema_inited.item()
            and ((not self.training) or self.t_graph.freeze_graph)
        )
        if use_cached_t:
            adp_adj_t = self.t_graph.A_ema
        else:
            x_graph = robust_view(x_filled)
            adp_adj_t = self.t_graph(x_graph)

        Yt, Ht = self.temporal_expert(x_filled, dyn_adj=adp_adj_t, return_repr=True)
        Ys, Hs = self.spatial_expert(x, dyn_adj=adp_adj_s, return_repr=True)

        H_cat = torch.cat([Ht, Hs], dim=-1)
        Y = self.pred_head(H_cat)

        if return_stage1:
            return Yt, Ys, Y, Ht, Hs
        return Yt, Ys, Y, Ht, Hs


def make_model(DEVICE, nb_block, in_channels,
               nb_time_filter, time_strides,
               num_for_predict, len_input, num_of_vertices,
               adj_mx_space=None, adj_mx_time=None,
               pattern_bank_path=None,
               pattern_top_m=5,
               pattern_tau=0.2):

    model = DualBranchGAT(
        DEVICE=DEVICE,
        nb_block=nb_block,
        in_channels=in_channels,
        nb_time_filter=nb_time_filter,
        time_strides=time_strides,
        num_for_predict=num_for_predict,
        len_input=len_input,
        num_of_vertices=num_of_vertices,
        adj_mx_space=adj_mx_space,
        adj_mx_time=adj_mx_time,
        pattern_bank_path=pattern_bank_path,
        pattern_top_m=pattern_top_m,
        pattern_tau=pattern_tau,
    ).to(DEVICE)

    return model
