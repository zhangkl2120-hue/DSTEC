

import os
import re
import json
import time
import random
import argparse
import configparser
import itertools
from typing import Dict, Tuple, List, Any, Optional
import numpy as np
import networkx as nx
import sys


def normalize_nonzero_to_01(A):
    A = A.astype(np.float32).copy()
    nz = A[A > 0]
    if nz.size:
        mn, mx = float(nz.min()), float(nz.max())
        if mx > mn:
            A[A > 0] = (A[A > 0] - mn) / (mx - mn)
        else:
            A[A > 0] = 1.0
    return A

def should_overwrite(path: str, policy: str) -> bool:
    if not os.path.exists(path):
        return True
    if policy == "force":
        return True
    if policy == "skip":
        return False
    if policy == "ask":
        if not sys.stdin.isatty():
            print(f"[INFO] {path} exists and non-interactive. skip overwrite.")
            return False
        ans = input(f"{os.path.basename(path)} already exists. Re-run LLM to overwrite? [y/N]: ").strip().lower()
        return ans in ("y", "yes")
    raise ValueError(policy)

try:
    import requests
except Exception:
    requests = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from lib.utils import get_adjacency_matrix


def get_proj_root() -> str:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.isdir(os.path.join(this_dir, "data")):
        return this_dir
    return this_dir


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def read_config(config_path: str) -> Tuple[np.ndarray, str]:
    print("Read config:", config_path)
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    data_cfg = cfg["Data"]

    adj_filename = data_cfg["adj_filename"]
    num_nodes = int(data_cfg["num_of_vertices"])
    dataset_name = data_cfg["dataset_name"]
    id_filename = data_cfg["id_filename"] if cfg.has_option("Data", "id_filename") else None

    adj_mx, _ = get_adjacency_matrix(adj_filename, num_nodes, id_filename)
    adj_mx = adj_mx.astype(np.float32)
    np.fill_diagonal(adj_mx, 0.0)
    print("dataset_name:", dataset_name)
    print("adj_mx shape:", adj_mx.shape, "nonzero:", int((adj_mx > 0).sum()))
    return adj_mx, dataset_name


def dump_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(items: List[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def compute_struct_stats(adj_mx: np.ndarray, struct_mode="fast", bet_k=64, seed=0) -> Dict[int, dict]:
    N = adj_mx.shape[0]
    A_bin = (adj_mx > 0).astype(np.int32)
    G = nx.from_numpy_array(A_bin)

    deg_dict = dict(G.degree())
    deg_arr = np.array([deg_dict.get(i, 0) for i in range(N)], dtype=np.float32)
    order = np.argsort(deg_arr)
    rank = np.empty_like(order)
    rank[order] = np.arange(N)
    deg_rank_norm = rank.astype(np.float32) / max(N - 1, 1)

    tri_dict = nx.triangles(G)
    clust_dict = nx.clustering(G)

    if struct_mode == "full":
        print("Compute betweenness/closeness (full/approx)...")
        bet_dict = nx.betweenness_centrality(G, k=min(bet_k, N), normalized=True, seed=seed)
        clo_dict = nx.closeness_centrality(G)
    else:
        bet_dict = {i: 0.0 for i in range(N)}
        clo_dict = {i: 0.0 for i in range(N)}

    stats = {}
    it_nodes = tqdm(range(N), desc="struct_stats") if tqdm is not None else range(N)

    for i in it_nodes:
        lengths = nx.single_source_shortest_path_length(G, i, cutoff=3)
        bfs_1hop = sum(1 for d in lengths.values() if d == 1)
        bfs_2hop = sum(1 for d in lengths.values() if d == 2)
        bfs_leq3 = sum(1 for d in lengths.values() if 1 <= d <= 3)

        dfs_T = nx.dfs_tree(G, source=i)
        dlen = nx.single_source_shortest_path_length(dfs_T, i)
        dfs_tree_size = len(dlen)
        dfs_depth = max(dlen.values()) if dlen else 0

        deg = int(deg_dict.get(i, 0))
        tri_center = int(tri_dict.get(i, 0))
        wedge_count = deg * (deg - 1) // 2
        open_wedge = int(wedge_count - tri_center)

        stats[i] = {
            "degree": float(deg),
            "degree_rank_norm": float(deg_rank_norm[i]),
            "betweenness": float(bet_dict.get(i, 0.0)),
            "closeness": float(clo_dict.get(i, 0.0)),
            "triangles": float(tri_dict.get(i, 0.0)),
            "clustering": float(clust_dict.get(i, 0.0)),
            "bfs_1hop": int(bfs_1hop),
            "bfs_2hop": int(bfs_2hop),
            "bfs_leq3": int(bfs_leq3),
            "dfs_tree_size": int(dfs_tree_size),
            "dfs_tree_depth": int(dfs_depth),
            "motif_tri_center": int(tri_center),
            "motif_open_wedge": int(open_wedge),
        }
    return stats



def make_level_mapping(values, low_q=0.33, high_q=0.66) -> Tuple[float, float]:
    arr = np.array(values, dtype=np.float32)
    low = float(np.quantile(arr, low_q))
    high = float(np.quantile(arr, high_q))
    return low, high


def to_level(x: float, low: float, high: float) -> str:
    if x <= low:
        return "low"
    elif x <= high:
        return "medium"
    return "high"


def build_node_descriptions(stats_dict: Dict[int, dict]) -> Dict[str, str]:
    N = len(stats_dict)
    deg_rank = [stats_dict[i]["degree_rank_norm"] for i in range(N)]
    bet = [stats_dict[i]["betweenness"] for i in range(N)]
    clo = [stats_dict[i]["closeness"] for i in range(N)]
    bfs3 = [stats_dict[i]["bfs_leq3"] for i in range(N)]
    dfsd = [stats_dict[i]["dfs_tree_depth"] for i in range(N)]
    tri = [stats_dict[i]["motif_tri_center"] for i in range(N)]
    wedge = [stats_dict[i]["motif_open_wedge"] for i in range(N)]

    deg_low, deg_high = make_level_mapping(deg_rank)
    bet_low, bet_high = make_level_mapping(bet)
    clo_low, clo_high = make_level_mapping(clo)
    bfs_low, bfs_high = make_level_mapping(bfs3)
    dfs_low, dfs_high = make_level_mapping(dfsd)
    tri_low, tri_high = make_level_mapping(tri)
    wed_low, wed_high = make_level_mapping(wedge)

    desc = {}
    for i in range(N):
        s = stats_dict[i]
        desc[str(i)] = (
            f"Node {i} structural profile: "
            f"degree={int(s['degree'])} (degree-rank={s['degree_rank_norm']:.3f}, level={to_level(s['degree_rank_norm'], deg_low, deg_high)}); "
            f"betweenness={s['betweenness']:.6f} (level={to_level(s['betweenness'], bet_low, bet_high)}); "
            f"closeness={s['closeness']:.6f} (level={to_level(s['closeness'], clo_low, clo_high)}); "
            f"clustering={s['clustering']:.6f}, triangles={int(s['triangles'])}. "
            f"BFS within 3 hops: {int(s['bfs_leq3'])} (level={to_level(float(s['bfs_leq3']), bfs_low, bfs_high)}). "
            f"DFS tree depth={int(s['dfs_tree_depth'])} (level={to_level(float(s['dfs_tree_depth']), dfs_low, dfs_high)}), "
            f"DFS tree size={int(s['dfs_tree_size'])}. "
            f"Motifs: tri_center={int(s['motif_tri_center'])} (level={to_level(float(s['motif_tri_center']), tri_low, tri_high)}), "
            f"open_wedge={int(s['motif_open_wedge'])} (level={to_level(float(s['motif_open_wedge']), wed_low, wed_high)})."
        )
    return desc


def build_candidate_edges(adj_mx, stats_dict, max_hop=3, min_common=2, min_jaccard=0.05,
                          max_candidates=None, seed=1234):
    N = adj_mx.shape[0]
    A_bin = (adj_mx > 0).astype(np.int32)

    neigh = [set(np.nonzero(A_bin[i] > 0)[0].tolist()) - {i} for i in range(N)]

    G = nx.from_numpy_array(A_bin)
    candidates = []

    rng = np.random.default_rng(seed)

    for i in range(N):
        lengths = nx.single_source_shortest_path_length(G, i, cutoff=max_hop)
        for j, dist in lengths.items():
            if j <= i:
                continue
            if dist <= 1:
                continue
            if A_bin[i, j] > 0:
                continue

            ni = neigh[i]
            nj = neigh[j]
            common = ni & nj
            union = ni | nj
            num_common = len(common)
            jaccard = num_common / len(union) if union else 0.0

            if num_common >= min_common or jaccard >= min_jaccard:
                candidates.append((i, j, dist, num_common, jaccard))

    rng.shuffle(candidates)
    if max_candidates is not None and max_candidates < len(candidates):
        candidates = candidates[:max_candidates]

    print(f"[Candidate Edges] {len(candidates)} candidates found "
          f"(max_hop={max_hop}, min_common={min_common}, min_jaccard={min_jaccard})")
    return candidates


def build_candidate_verification_prompts(adj_mx, stats_dict, candidates,
                                         max_candidates=None, seed=1234):
    N = adj_mx.shape[0]

    def node_profile(idx: int, s: dict) -> str:
        return (
            f"Node {idx}: "
            f"degree-rank={float(s.get('degree_rank_norm', 0.0)):.3f} (deg={int(s.get('degree', 0))}), "
            f"betweenness={float(s.get('betweenness', 0.0)):.6f}, "
            f"closeness={float(s.get('closeness', 0.0)):.6f}, "
            f"clustering={float(s.get('clustering', 0.0)):.6f}, "
            f"BFS-≤3hop={int(s.get('bfs_leq3', 0))}, "
            f"DFS-depth={int(s.get('dfs_tree_depth', 0))}, "
            f"triangles={int(s.get('motif_tri_center', 0))}, "
            f"open-wedges={int(s.get('motif_open_wedge', 0))}"
        )

    neigh = [set(np.nonzero(adj_mx[i] > 0)[0].tolist()) - {i} for i in range(N)]

    rng = np.random.default_rng(seed)
    cand_list = list(candidates)
    if max_candidates is not None and max_candidates < len(cand_list):
        rng.shuffle(cand_list)
        cand_list = cand_list[:max_candidates]

    prompts = []
    for item in cand_list:
        i, j, dist, num_common, jaccard = int(item[0]), int(item[1]), item[2], item[3], item[4]

        ni = neigh[i]
        nj = neigh[j]
        common = ni & nj
        union = ni | nj
        num_common_actual = len(common)
        jaccard_actual = num_common_actual / len(union) if union else 0.0

        s_i = stats_dict[i]
        s_j = stats_dict[j]

        prompt = (
            "Act as a traffic graph relation verifier. "
            "Evaluate whether a candidate structural relation between "
            f"node {i} and node {j} should be added.\n\n"
            f"[Structural Evidence — Node {i}]\n{node_profile(i, s_i)}\n\n"
            f"[Structural Evidence — Node {j}]\n{node_profile(j, s_j)}\n\n"
            f"[Edge-local Evidence]\n"
            f"shortest_path_distance={dist}\n"
            f"common_neighbors={num_common_actual}\n"
            f"jaccard_similarity={jaccard_actual:.6f}\n\n"
            "Plausibility Rubric:\n"
            "- High: multiple structural cues consistently support the relation.\n"
            "- Medium: evidence is partially supportive or mixed.\n"
            "- Low: evidence is weak, contradictory, or structurally implausible.\n\n"
            "Risk Rubric:\n"
            "- Low: unlikely to introduce spurious propagation.\n"
            "- Medium: possible shortcut effect or insufficient support.\n"
            "- High: likely to create unreliable shortcuts or amplify abnormal signals.\n\n"
            "Decision Rule: Add only when plausibility is High and risk is Low; otherwise reject.\n\n"
            "Output Format (strict JSON):\n"
            '{"plausibility": "<High|Medium|Low>", "risk": "<Low|Medium|High>", '
            '"decision": "<add|reject>", "reason": "<brief justification>"}'
        )

        prompts.append({
            "edge_id": f"{i}_{j}",
            "src": i,
            "dst": j,
            "dist": dist,
            "num_common": num_common_actual,
            "jaccard": jaccard_actual,
            "prompt": prompt,
        })

    return prompts

def _safe_level_id(x: float, low: float, high: float) -> int:
    if x <= low:
        return 0
    elif x <= high:
        return 1
    return 2


def _level_sim(a: int, b: int) -> float:
    return 1.0 - abs(int(a) - int(b)) / 2.0


def score_edges_heuristic(adj_mx: np.ndarray,
                          stats_dict: Dict[int, dict],
                          max_edges: Optional[int] = None,
                          seed: int = 1234,
                          lambda_node: float = 0.6,
                          lambda_jaccard: float = 0.2,
                          lambda_cn: float = 0.2) -> List[dict]:

    N = adj_mx.shape[0]
    A_bin = (adj_mx > 0).astype(np.int32)

    edges = np.transpose(np.nonzero(np.triu(A_bin > 0, k=1)))
    edges = edges.tolist()

    if max_edges is not None and max_edges < len(edges):
        rng = np.random.default_rng(seed)
        rng.shuffle(edges)
        edges = edges[:max_edges]
    neigh = [set(np.nonzero(A_bin[i] > 0)[0].tolist()) - {i} for i in range(N)]

    feat_names = [
        "degree_rank_norm",
        "betweenness",
        "closeness",
        "bfs_leq3",
        "dfs_tree_depth",
        "motif_tri_center",
        "motif_open_wedge",
    ]

    level_map = {}
    for name in feat_names:
        vals = [float(stats_dict[i].get(name, 0.0)) for i in range(N)]
        low, high = make_level_mapping(vals)
        level_map[name] = (low, high)

    cn_cache = {}
    max_cn = 1
    for i, j in edges:
        ni = neigh[i] - {j}
        nj = neigh[j] - {i}
        common = ni & nj
        cn = len(common)
        cn_cache[(int(i), int(j))] = cn
        if cn > max_cn:
            max_cn = cn

    results = []
    for i, j in edges:
        i = int(i)
        j = int(j)

        s_i = stats_dict[i]
        s_j = stats_dict[j]

        sims = []
        for name in feat_names:
            low, high = level_map[name]
            li = _safe_level_id(float(s_i.get(name, 0.0)), low, high)
            lj = _safe_level_id(float(s_j.get(name, 0.0)), low, high)
            sims.append(_level_sim(li, lj))
        s_node = float(np.mean(sims)) if sims else 0.0

        ni = neigh[i] - {j}
        nj = neigh[j] - {i}
        common = ni & nj
        union = ni | nj

        jaccard = len(common) / len(union) if len(union) > 0 else 0.0
        cn_norm = cn_cache[(i, j)] / max_cn

        score = (
            lambda_node * s_node
            + lambda_jaccard * float(jaccard)
            + lambda_cn * float(cn_norm)
        )
        score = max(0.0, min(1.0, float(score)))

        results.append({
            "edge_id": f"{i}_{j}",
            "src": i,
            "dst": j,
            "score": score
        })

    return results




def parse_score(text: str) -> Optional[float]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "score" in obj:
            s = float(obj["score"])
            return max(0.0, min(1.0, s))
    except Exception:
        pass
    m = re.search(r"score\s*[:=]\s*([01](?:\.\d+)?)", text, flags=re.IGNORECASE)
    if m:
        s = float(m.group(1))
        return max(0.0, min(1.0, s))
    m = re.search(r"([01](?:\.\d+)?)", text)
    if m:
        s = float(m.group(1))
        return max(0.0, min(1.0, s))
    return None


def parse_verification(text: str) -> Optional[dict]:
    text = text.strip()

    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text.split("\n", 1)[1].strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "plausibility" in obj and "risk" in obj:
            return {
                "plausibility": str(obj.get("plausibility", "Medium")),
                "risk": str(obj.get("risk", "Medium")),
                "decision": str(obj.get("decision", "reject")),
                "reason": str(obj.get("reason", "")),
            }
    except Exception:
        pass

    pl = re.search(r'plausibility["\s:]+(High|Medium|Low)', text, re.IGNORECASE)
    ri = re.search(r'risk["\s:]+(Low|Medium|High)', text, re.IGNORECASE)
    de = re.search(r'decision["\s:]+(add|reject)', text, re.IGNORECASE)

    if pl and ri:
        return {
            "plausibility": pl.group(1).capitalize(),
            "risk": ri.group(1).capitalize(),
            "decision": de.group(1).lower() if de else "reject",
            "reason": "parsed from fallback",
        }

    return None


def deepseek_verify(prompt: str, api_key: str, timeout: int = 60, retries: int = 3) -> dict:
    if requests is None:
        raise RuntimeError("requests not installed. pip install requests")
    API_URL = "https://api.deepseek.com/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}", "Connection": "close"}
    data = {
        "model": "deepseek-chat", "messages": [
            {"role": "system", "content": (
                "You are a traffic graph relation verifier. Evaluate candidate structural relations. "
                "Reply with strict JSON: {\"plausibility\":\"High|Medium|Low\",\"risk\":\"Low|Medium|High\","
                "\"decision\":\"add|reject\",\"reason\":\"...\"}. "
                "Decision: add ONLY if plausibility=High AND risk=Low.")},
            {"role": "user", "content": prompt},
        ], "temperature": 0.0, "max_tokens": 150, "stream": False,
    }
    for attempt in range(retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=data, timeout=timeout)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.strip("`").strip()
                if content.lower().startswith("json"):
                    content = content.split("\n", 1)[1].strip()
            result = parse_verification(content)
            if result is not None:
                return result
            print(f"[WARN] Cannot parse verification: {content}")
            return {"plausibility": "Medium", "risk": "Medium", "decision": "reject", "reason": "parse failure"}
        except Exception as e:
            print(f"[WARN] DeepSeek failed attempt {attempt+1}/{retries}: {e}")
            time.sleep(3 * (attempt + 1))
    return {"plausibility": "Medium", "risk": "Medium", "decision": "reject", "reason": "api failure fallback"}


def openrouter_verify(prompt: str, api_key: str, model: str = "openai/gpt-5.2-pro",
                      timeout: int = 60, retries: int = 3,
                      referer: str = "", title: str = "") -> dict:
    if requests is None:
        raise RuntimeError("requests not installed. pip install requests")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    if referer: headers["HTTP-Referer"] = referer
    if title: headers["X-Title"] = title
    data = {
        "model": model, "messages": [
            {"role": "system", "content": (
                "You are a traffic graph relation verifier. Evaluate candidate structural relations. "
                "Reply with strict JSON: {\"plausibility\":\"High|Medium|Low\",\"risk\":\"Low|Medium|High\","
                "\"decision\":\"add|reject\",\"reason\":\"...\"}. "
                "Decision: add ONLY if plausibility=High AND risk=Low.")},
            {"role": "user", "content": prompt},
        ], "temperature": 0.0, "max_tokens": 150, "stream": False,
    }
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(data), timeout=timeout)
            if resp.status_code == 403:
                raise RuntimeError(f"OpenRouter 403: {resp.text[:500]}")
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.strip("`").strip()
                if content.lower().startswith("json"):
                    content = content.split("\n", 1)[1].strip()
            result = parse_verification(content)
            if result is not None:
                return result
            print(f"[WARN] Cannot parse verification: {content}")
            return {"plausibility": "Medium", "risk": "Medium", "decision": "reject", "reason": "parse failure"}
        except Exception as e:
            print(f"[WARN] OpenRouter failed attempt {attempt+1}/{retries}: {e}")
            time.sleep(2 * (attempt + 1))
    return {"plausibility": "Medium", "risk": "Medium", "decision": "reject", "reason": "api failure fallback"}


def verify_candidate_edges(items: List[dict], llm: str, seed: int = 1234,
                           sleep_base: float = 0.2) -> List[dict]:
    results = []
    it = tqdm(items, desc=f"verifying({llm})") if tqdm is not None else items

    if llm == "stub":
        random.seed(seed)
        for item in it:
            pl = random.choices(["High","Medium","Low"], weights=[0.3,0.4,0.3])[0]
            ri = random.choices(["Low","Medium","High"], weights=[0.35,0.35,0.3])[0]
            decision = "add" if (pl == "High" and ri == "Low") else "reject"
            results.append({"edge_id": item["edge_id"], "src": item["src"], "dst": item["dst"],
                            "plausibility": pl, "risk": ri, "decision": decision,
                            "reason": f"stub: pl={pl} ri={ri}"})
        return results

    if llm == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set. export it.")
        for item in it:
            r = deepseek_verify(item["prompt"], api_key=api_key)
            results.append({"edge_id": item["edge_id"], "src": item["src"], "dst": item["dst"], **r})
            time.sleep(sleep_base + random.random() * 0.1)
        return results

    if llm == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set. export it.")
        model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-5.2-pro")
        referer = os.environ.get("OPENROUTER_REFERER", "")
        title = os.environ.get("OPENROUTER_TITLE", "")
        for item in it:
            r = openrouter_verify(item["prompt"], api_key=api_key, model=model, referer=referer, title=title)
            results.append({"edge_id": item["edge_id"], "src": item["src"], "dst": item["dst"], **r})
            time.sleep(sleep_base + random.random() * 0.1)
        return results

    raise ValueError(f"Unknown llm mode: {llm}")


def build_As_expanded(adj_mx: np.ndarray, verification_results: List[dict]) -> np.ndarray:
    N = adj_mx.shape[0]
    A = adj_mx.astype(np.float32).copy()

    A_plus = np.zeros((N, N), dtype=np.float32)
    added = 0
    for it in verification_results:
        if it.get("decision", "reject") == "add":
            i, j = int(it["src"]), int(it["dst"])
            if 0 <= i < N and 0 <= j < N:
                A_plus[i, j] = 1.0
                A_plus[j, i] = 1.0
                added += 1

    A_sum = A + A_plus
    eps = 1e-6
    row_sum = A_sum.sum(axis=1, keepdims=True)
    As = A_sum / (row_sum + eps)
    As = As.astype(np.float32)

    nz_orig = int((A > 0).sum())
    print(f"As expanded. original edges: {nz_orig}, added edges: {added}, "
          f"total nonzero: {int((As > 0).sum())}")
    nz = As[As > 0]
    if nz.size > 0:
        print(f"As stats: min={nz.min():.4f} max={nz.max():.4f} mean={nz.mean():.4f}")
    return As


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configurations/PEMS04_astgcn.conf")
    p.add_argument("--only", type=str, default="all",
                   choices=["all", "struct", "desc", "candidates", "verify", "as"])
    p.add_argument("--llm", type=str, default="stub",
                   choices=["stub", "deepseek", "openrouter", "heuristic"])
    p.add_argument("--max_edges", type=int, default=None)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--sleep", type=float, default=0.2)
    p.add_argument("--force", action="store_true")
    p.add_argument("--space_graph_policy", type=str, default="skip", choices=["skip", "ask", "force"])
    p.add_argument("--struct_mode", type=str, default="fast", choices=["fast", "full"],
                   help="fast: pass betweenness/closeness；full")
    p.add_argument("--bet_k", type=int, default=64, help="full moda betweenness sim k")

    return p.parse_args()


def main():
    args = parse_args()
    PROJ = get_proj_root()

    adj_mx, dataset = read_config(args.config)
    out_dir = os.path.join(PROJ, "data", dataset)
    ensure_dir(out_dir)

    struct_path = os.path.join(out_dir, f"{dataset}_struct_stats.json")
    desc_path = os.path.join(out_dir, f"{dataset}_node_desc.json")
    candidates_path = os.path.join(out_dir, f"{dataset}_candidates.jsonl")
    if args.llm == "heuristic":
        scores_path = os.path.join(out_dir, f"{dataset}_edge_scores_heur.jsonl")
        as_path = os.path.join(out_dir, f"{dataset}_As_heur.npy")
    else:
        verif_path = os.path.join(out_dir, f"{dataset}_verification_results.jsonl")
        as_path = os.path.join(out_dir, f"{dataset}_As_ours.npy")

    def need_run(path: str) -> bool:
        return args.force or (not os.path.exists(path))

    if args.only in ("all", "struct", "desc", "candidates", "verify", "as"):
        if need_run(struct_path):
            print("=== [1/5] struct stats ===")
            stats = compute_struct_stats(adj_mx, struct_mode=args.struct_mode,
                                         bet_k=args.bet_k, seed=args.seed)
            dump_json({str(k): v for k, v in stats.items()}, struct_path)
            print("Saved:", struct_path)
        else:
            print("Skip struct (exists):", struct_path)
    if args.only == "struct":
        return

    if args.only in ("all", "desc", "candidates", "verify", "as"):
        if need_run(desc_path):
            print("=== [2/5] node desc ===")
            stats_raw = load_json(struct_path)
            stats_dict = {int(k): v for k, v in stats_raw.items()}
            desc = build_node_descriptions(stats_dict)
            dump_json(desc, desc_path)
            print("Saved:", desc_path)
        else:
            print("Skip desc (exists):", desc_path)
    if args.only == "desc":
        return

    if args.llm != "heuristic":
        if args.only in ("all", "candidates", "verify", "as"):
            if need_run(candidates_path):
                print("=== [3/5] candidate edges (DSTEC C set) ===")
                stats_raw = load_json(struct_path)
                stats_dict = {int(k): v for k, v in stats_raw.items()}
                candidates = build_candidate_edges(
                    adj_mx, stats_dict, max_hop=3, min_common=2, min_jaccard=0.05,
                    max_candidates=args.max_edges, seed=args.seed
                )
                cand_items = [{"src": int(c[0]), "dst": int(c[1]),
                               "dist": c[2], "cn": c[3], "jaccard": float(c[4])}
                              for c in candidates]
                write_jsonl(cand_items, candidates_path)
                print("Saved:", candidates_path, "num_candidates:", len(cand_items))
            else:
                print("Skip candidates (exists):", candidates_path)
        if args.only == "candidates":
            return
    else:
        if args.only == "candidates":
            print("[INFO] heuristic mode uses existing edges, no candidates needed.")
            return

    verif_updated = False
    if args.only in ("all", "verify", "as"):
        if args.llm == "heuristic":
            if args.only != "as":
                do_scores = should_overwrite(scores_path, args.space_graph_policy) or args.force
                if do_scores:
                    print("=== [4/5] heuristic edge scoring ===")
                    stats_raw = load_json(struct_path)
                    stats_dict = {int(k): v for k, v in stats_raw.items()}
                    scores = score_edges_heuristic(
                        adj_mx=adj_mx, stats_dict=stats_dict,
                        max_edges=args.max_edges, seed=args.seed,
                        lambda_node=0.6, lambda_jaccard=0.2, lambda_cn=0.2,
                    )
                    if os.path.exists(scores_path):
                        os.rename(scores_path, scores_path + ".bak")
                    write_jsonl(scores, scores_path)
                    verif_updated = True
                    print("Saved:", scores_path, "num_edges:", len(scores))
                else:
                    print("Reuse existing scores:", scores_path)
        else:
            if args.only != "as":
                do_verify = should_overwrite(verif_path, args.space_graph_policy) or args.force
                if do_verify:
                    print("=== [4/5] LLM relation verification (DSTEC rubric) ===")
                    stats_raw = load_json(struct_path)
                    stats_dict = {int(k): v for k, v in stats_raw.items()}
                    cand_items = read_jsonl(candidates_path)
                    candidates = [(it["src"], it["dst"], it["dist"], it["cn"], it["jaccard"])
                                  for it in cand_items]
                    prompts = build_candidate_verification_prompts(
                        adj_mx, stats_dict, candidates,
                        max_candidates=args.max_edges, seed=args.seed
                    )
                    results = verify_candidate_edges(prompts, llm=args.llm,
                                                     seed=args.seed, sleep_base=args.sleep)
                    if os.path.exists(verif_path):
                        os.rename(verif_path, verif_path + ".bak")
                    write_jsonl(results, verif_path)
                    verif_updated = True
                    print("Saved:", verif_path, "num_verified:", len(results))
                    num_added = sum(1 for r in results if r.get("decision") == "add")
                    print(f"Added edges: {num_added}/{len(results)} "
                          f"({100.0*num_added/max(1,len(results)):.1f}%)")
                else:
                    print("Reuse existing verification:", verif_path)
            else:
                print("Skip verify because --only as (will reuse existing results).")
    if args.only == "verify":
        return

    if args.only in ("all", "as"):
        do_as = should_overwrite(as_path, args.space_graph_policy) or args.force
        if args.only != "as":
            do_as = do_as or (not os.path.exists(as_path)) or verif_updated

        if do_as:
            print("=== [5/5] build expanded spatial graph As (DSTEC) ===")
            if args.llm == "heuristic":
                scores = read_jsonl(scores_path)
                As = build_As_expanded(adj_mx, [])
                As = np.zeros_like(As)
                score_map = {}
                for it in scores:
                    score_map[(int(it["src"]), int(it["dst"]))] = float(it["score"])
                N = adj_mx.shape[0]
                for i in range(N):
                    for j in range(N):
                        if adj_mx[i, j] > 0:
                            s = score_map.get((i, j), score_map.get((j, i), 1.0))
                            As[i, j] = max(0.0, min(1.0, s))
                As = np.maximum(As, As.T)
                row_sum = As.sum(axis=1, keepdims=True)
                As = As / (row_sum + 1e-6)
            else:
                results = read_jsonl(verif_path)
                As = build_As_expanded(adj_mx, results)
            np.save(as_path, As.astype(np.float32))
            print("Saved:", as_path, "shape:", As.shape)
        else:
            print("Reuse existing As:", as_path)

    print("Done.")


if __name__ == "__main__":
    main()
