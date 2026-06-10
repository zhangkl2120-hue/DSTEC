
import os
import argparse
import configparser
from typing import Dict, Tuple, List

import numpy as np
from sklearn.cluster import MiniBatchKMeans



def zscore_last(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def ema_np(x: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    out[:, 0] = x[:, 0]
    for t in range(1, x.shape[1]):
        out[:, t] = alpha * x[:, t] + (1.0 - alpha) * out[:, t - 1]
    return out


def l2_normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norm + eps)


def build_pattern_key(windows: np.ndarray, alpha: float = 0.3, use_l2_norm: bool = True) -> np.ndarray:
    windows = windows.astype(np.float32)

    shape_key = zscore_last(windows)

    trend = ema_np(windows, alpha=alpha)
    residual_key = zscore_last(windows - trend)

    diff = np.diff(windows, axis=-1)
    diff = np.pad(diff, ((0, 0), (1, 0)), mode="constant")
    diff_key = zscore_last(diff)

    key = np.concatenate([shape_key, residual_key, diff_key], axis=-1).astype(np.float32)

    if use_l2_norm:
        key = l2_normalize(key).astype(np.float32)

    return key


def pairwise_dist_to_centers(keys: np.ndarray, centers: np.ndarray) -> np.ndarray:
    diff = keys[:, None, :] - centers[None, :, :]
    return np.mean(diff * diff, axis=-1).astype(np.float32)


def select_with_padding(indices: np.ndarray, scores: np.ndarray, keep: int, rng: np.random.Generator) -> np.ndarray:
    if len(indices) == 0:
        raise ValueError("select_with_padding received empty indices.")

    order = np.argsort(scores)
    selected = indices[order[:min(keep, len(order))]]

    if len(selected) < keep:
        extra = rng.choice(selected, size=keep - len(selected), replace=True)
        selected = np.concatenate([selected, extra], axis=0)

    return selected.astype(np.int64)



def read_config(config_path: str) -> Dict[str, str]:
    cfg = configparser.ConfigParser()
    cfg.read(config_path)

    data_config = cfg["Data"]
    training_config = cfg["Training"]

    graph_signal_matrix_filename = data_config["graph_signal_matrix_filename"]
    dataset_name = data_config["dataset_name"]

    num_of_hours = int(training_config["num_of_hours"])
    num_of_days = int(training_config["num_of_days"])
    num_of_weeks = int(training_config["num_of_weeks"])

    raw_file = os.path.basename(graph_signal_matrix_filename).split(".")[0]
    dirpath = os.path.dirname(graph_signal_matrix_filename)

    prepared_npz_path = os.path.join(
        dirpath,
        f"{raw_file}_r{num_of_hours}_d{num_of_days}_w{num_of_weeks}_astcgn.npz"
    )

    return {
        "dataset_name": dataset_name,
        "dirpath": dirpath,
        "raw_file": raw_file,
        "prepared_npz_path": prepared_npz_path,
        "num_of_hours": num_of_hours,
        "num_of_days": num_of_days,
        "num_of_weeks": num_of_weeks,
        "rdw_tag": f"r{num_of_hours}d{num_of_days}w{num_of_weeks}",
    }


def load_train_windows(prepared_npz_path: str, feature_idx: int = 0) -> np.ndarray:
    if not os.path.exists(prepared_npz_path):
        raise FileNotFoundError(
            f"Prepared file not found: {prepared_npz_path}\n"
            f"Please run prepareData.py with the matching config first."
        )

    data = np.load(prepared_npz_path)
    train_x = data["train_x"]
    if train_x.ndim != 4:
        raise RuntimeError(f"Expected train_x to be 4-D (B,N,F,L), got {train_x.shape}")

    train_x = train_x[:, :, feature_idx, :].astype(np.float32)
    return train_x



def build_uniform_bank(
    train_x: np.ndarray,
    keep_per_node: int = 256,
    alpha: float = 0.3,
    seed: int = 2026,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    B, N, L = train_x.shape
    values_list, keys_list, times_list = [], [], []

    for n in range(N):
        node_windows = train_x[:, n, :]

        if B >= keep_per_node:
            idx = np.linspace(0, B - 1, keep_per_node).round().astype(np.int64)
        else:
            extra = rng.choice(B, size=keep_per_node - B, replace=True)
            idx = np.concatenate([np.arange(B), extra], axis=0).astype(np.int64)

        selected_windows = node_windows[idx]
        selected_keys = build_pattern_key(selected_windows, alpha=alpha)
        selected_times = idx.astype(np.int64)

        values_list.append(selected_windows)
        keys_list.append(selected_keys)
        times_list.append(selected_times)

    bank_values = np.stack(values_list, axis=0).astype(np.float32)
    bank_keys = np.stack(keys_list, axis=0).astype(np.float32)
    bank_times = np.stack(times_list, axis=0).astype(np.int64)

    return {
        "bank_values": bank_values,
        "bank_keys": bank_keys,
        "bank_times": bank_times,
    }



def build_proto_bank(
    train_x: np.ndarray,
    n_prototypes: int = 32,
    donors_per_proto: int = 8,
    max_cluster_samples: int = 200000,
    alpha: float = 0.3,
    seed: int = 2026,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    B, N, L = train_x.shape
    total_windows = B * N
    sample_size = min(max_cluster_samples, total_windows)

    flat_indices = rng.choice(total_windows, size=sample_size, replace=False)
    sample_b = flat_indices // N
    sample_n = flat_indices % N
    sample_windows = train_x[sample_b, sample_n, :]
    sample_keys = build_pattern_key(sample_windows, alpha=alpha)

    print("[ProtoBank] cluster samples:", sample_windows.shape)
    print("[ProtoBank] sample keys:", sample_keys.shape)

    kmeans = MiniBatchKMeans(
        n_clusters=n_prototypes,
        random_state=seed,
        batch_size=4096,
        n_init=3,
        max_iter=100,
        reassignment_ratio=0.01,
        verbose=0,
    )
    kmeans.fit(sample_keys)

    prototype_keys = kmeans.cluster_centers_.astype(np.float32)
    prototype_keys = l2_normalize(prototype_keys).astype(np.float32)

    C = n_prototypes
    R = donors_per_proto
    D = prototype_keys.shape[-1]

    donor_values = np.zeros((N, C, R, L), dtype=np.float32)
    donor_keys = np.zeros((N, C, R, D), dtype=np.float32)
    donor_times = np.zeros((N, C, R), dtype=np.int64)
    donor_dists = np.zeros((N, C, R), dtype=np.float32)
    group_counts = np.zeros((N, C), dtype=np.int64)

    for n in range(N):
        node_windows = train_x[:, n, :]
        node_keys = build_pattern_key(node_windows, alpha=alpha)

        dist = pairwise_dist_to_centers(node_keys, prototype_keys)
        assign = dist.argmin(axis=1)

        for c in range(C):
            idx_c = np.where(assign == c)[0]
            group_counts[n, c] = len(idx_c)

            if len(idx_c) > 0:
                selected = select_with_padding(idx_c, dist[idx_c, c], R, rng)
            else:
                all_idx = np.arange(B)
                selected = select_with_padding(all_idx, dist[:, c], R, rng)

            donor_values[n, c] = node_windows[selected]
            donor_keys[n, c] = node_keys[selected]
            donor_times[n, c] = selected
            donor_dists[n, c] = dist[selected, c]

        if (n + 1) % 100 == 0 or n == N - 1:
            print(f"[ProtoBank] processed node {n + 1}/{N}")

    bank_values = donor_values.reshape(N, C * R, L)
    bank_keys = donor_keys.reshape(N, C * R, D)
    bank_times = donor_times.reshape(N, C * R)

    return {
        "prototype_keys": prototype_keys,
        "donor_values": donor_values,
        "donor_keys": donor_keys,
        "donor_times": donor_times,
        "donor_dists": donor_dists,
        "group_counts": group_counts,
        "bank_values": bank_values.astype(np.float32),
        "bank_keys": bank_keys.astype(np.float32),
        "bank_times": bank_times.astype(np.int64),
    }



def _slice_bank(bank: Dict[str, np.ndarray], k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = bank["bank_values"]
    keys = bank["bank_keys"]
    times = bank["bank_times"]

    N, K, L = values.shape
    if k >= K:
        return values, keys, times

    idx = np.linspace(0, K - 1, k).round().astype(np.int64)
    return values[:, idx, :], keys[:, idx, :], times[:, idx]


def build_hybrid_bank(
    uniform_bank: Dict[str, np.ndarray],
    proto_bank: Dict[str, np.ndarray],
    k_uniform: int,
    k_proto: int,
) -> Dict[str, np.ndarray]:
    u_values, u_keys, u_times = _slice_bank(uniform_bank, k_uniform)
    p_values, p_keys, p_times = _slice_bank(proto_bank, k_proto)

    bank_values = np.concatenate([u_values, p_values], axis=1).astype(np.float32)
    bank_times = np.concatenate([u_times, p_times], axis=1).astype(np.int64)

    if u_keys.shape[-1] == p_keys.shape[-1]:
        bank_keys = np.concatenate([u_keys, p_keys], axis=1).astype(np.float32)
    else:
        bank_keys = np.zeros((bank_values.shape[0], bank_values.shape[1], p_keys.shape[-1]), dtype=np.float32)

    source_type = np.concatenate([
        np.zeros((u_values.shape[0], u_values.shape[1]), dtype=np.int64),
        np.ones((p_values.shape[0], p_values.shape[1]), dtype=np.int64),
    ], axis=1)

    return {
        "bank_values": bank_values,
        "bank_keys": bank_keys,
        "bank_times": bank_times,
        "source_type": source_type,
        "k_uniform": np.array(k_uniform),
        "k_proto": np.array(k_proto),
    }


def parse_variant(variant: str) -> Tuple[int, int]:
    try:
        u_part, p_part = variant.split("_")
        k_uniform = int(u_part.replace("U", ""))
        k_proto = int(p_part.replace("P", ""))
        return k_uniform, k_proto
    except Exception as e:
        raise ValueError(f"Invalid variant format: {variant}. Expected e.g. U256_P256") from e



def save_npz(path: str, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)
    print("[Saved]", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--prepared_npz_path", type=str, default=None, help="Optional override.")
    parser.add_argument("--out_dir", type=str, default=None, help="Optional output directory override.")

    parser.add_argument("--feature_idx", type=int, default=0)
    parser.add_argument("--keep_per_node", type=int, default=256)

    parser.add_argument("--n_prototypes", type=int, default=32)
    parser.add_argument("--donors_per_proto", type=int, default=8)
    parser.add_argument("--max_cluster_samples", type=int, default=200000)

    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument(
        "--variants",
        nargs="+",
        default=["U256_P256", "U192_P64"],
        help="Hybrid variants, e.g. U256_P256 U192_P64 U256_P128."
    )
    parser.add_argument("--no_intermediate", action="store_true", help="Only save hybrid banks.")

    args = parser.parse_args()

    info = read_config(args.config)
    dataset = info["dataset_name"]
    rdw_tag = info["rdw_tag"]

    prepared_npz_path = args.prepared_npz_path or info["prepared_npz_path"]
    out_dir = args.out_dir or info["dirpath"]

    print("========== Build All Pattern Banks ==========")
    print("dataset:", dataset)
    print("rdw_tag:", rdw_tag)
    print("prepared:", prepared_npz_path)
    print("out_dir:", out_dir)
    print("variants:", args.variants)

    train_x = load_train_windows(prepared_npz_path, feature_idx=args.feature_idx)
    B, N, L = train_x.shape

    print("[Data] train_x:", train_x.shape)
    print("[Data] mean/std/min/max:", train_x.mean(), train_x.std(), train_x.min(), train_x.max())

    print("\n========== [1/3] Uniform Bank ==========")
    uniform_bank = build_uniform_bank(
        train_x=train_x,
        keep_per_node=args.keep_per_node,
        alpha=args.alpha,
        seed=args.seed,
    )
    uniform_path = os.path.join(out_dir, f"{dataset}_pattern_bank_norm_{rdw_tag}.npz")
    if not args.no_intermediate:
        save_npz(
            uniform_path,
            **uniform_bank,
            source_prepared_npz=np.array(prepared_npz_path),
            keep_per_node=np.array(args.keep_per_node),
            input_len=np.array(L),
            alpha=np.array(args.alpha),
        )
    print("[Uniform] bank_values:", uniform_bank["bank_values"].shape)

    print("\n========== [2/3] Prototype Bank ==========")
    proto_bank = build_proto_bank(
        train_x=train_x,
        n_prototypes=args.n_prototypes,
        donors_per_proto=args.donors_per_proto,
        max_cluster_samples=args.max_cluster_samples,
        alpha=args.alpha,
        seed=args.seed,
    )
    proto_path = os.path.join(
        out_dir,
        f"{dataset}_pattern_bank_proto_C{args.n_prototypes}_R{args.donors_per_proto}_{rdw_tag}.npz"
    )
    if not args.no_intermediate:
        save_npz(
            proto_path,
            **proto_bank,
            source_prepared_npz=np.array(prepared_npz_path),
            n_prototypes=np.array(args.n_prototypes),
            donors_per_proto=np.array(args.donors_per_proto),
            input_len=np.array(L),
            alpha=np.array(args.alpha),
        )
    print("[Proto] prototype_keys:", proto_bank["prototype_keys"].shape)
    print("[Proto] donor_values:", proto_bank["donor_values"].shape)
    print("[Proto] flat bank_values:", proto_bank["bank_values"].shape)

    print("\n========== [3/3] Hybrid Banks ==========")
    for variant in args.variants:
        k_uniform, k_proto = parse_variant(variant)
        hybrid_bank = build_hybrid_bank(
            uniform_bank=uniform_bank,
            proto_bank=proto_bank,
            k_uniform=k_uniform,
            k_proto=k_proto,
        )
        hybrid_path = os.path.join(
            out_dir,
            f"{dataset}_pattern_bank_hybrid_{variant}_{rdw_tag}.npz"
        )
        save_npz(
            hybrid_path,
            **hybrid_bank,
            uniform_source=np.array(uniform_path),
            proto_source=np.array(proto_path),
            source_prepared_npz=np.array(prepared_npz_path),
            input_len=np.array(L),
            alpha=np.array(args.alpha),
        )
        print(f"[Hybrid {variant}] bank_values:", hybrid_bank["bank_values"].shape)
        print(f"[Hybrid {variant}] mean/std/min/max:",
              hybrid_bank["bank_values"].mean(),
              hybrid_bank["bank_values"].std(),
              hybrid_bank["bank_values"].min(),
              hybrid_bank["bank_values"].max())

    print("\nDone.")


if __name__ == "__main__":
    main()
