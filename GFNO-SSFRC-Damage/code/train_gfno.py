# -*- coding: utf-8 -*-
"""
GFNO v5 — PURE PYTORCH (NO PyG / NO torch_cluster) — NO-LEAKAGE — v6
-------------------------------------------------------------------
Fixes / Features:
  ✅ Group split by base RVE id (all rot/jit variants stay in the same split)
  ✅ Label normalization (mean/std) computed on TRAIN split only
  ✅ Pure PyTorch KNN (torch.cdist + topk), no torch-cluster needed
  ✅ Pure PyTorch local message passing: mean neighbor aggregation + Linear
  ✅ GFNO spectral conv (per-graph eigenbasis U) + local conv + residual + LayerNorm
  ✅ Two-head regression: diag(3) + shear(3)
  ✅ Uncertainty weighting + shear extra weight
  ✅ U cache on disk

NEW in v6:
  ✅ VAL metrics split: total/diag/shear
  ✅ Save TWO checkpoints: best_total and best_shear (to stabilize shear R2 selection)
  ✅ Clamp uncertainty weights s_* to avoid runaway negative values

Notes:
  - Your augmentation uses 24 cubic rotations (det=+1), so rot_id is categorical.
    => ROT_FEAT_MODE="onehot" is the safest reviewer-proof choice.
"""

from __future__ import annotations

# ============================================================
# ✅ CRITICAL FIX (MUST be before numpy/torch imports)
# ============================================================
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import re, json, math, time, random, argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset


# ============================================================
# ✅ PATHS (GitHub-friendly relative defaults)
# ============================================================
# Expected repository structure:
# GFNO-SSFRC-Damage/
# ├── data/dataset/*.npz
# ├── code/train_gfno.py
# └── results/
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = str(ROOT_DIR / "data" / "dataset")
OUTDIR   = str(ROOT_DIR / "results" / "gfno_run")


# ============================================================
# ✅ TRAINING HYPERPARAMS
# ============================================================
SEED = 20251112

EPOCHS = 220
BATCH_SIZE = 16

LR = 1e-3
WEIGHT_DECAY = 1e-5
WARMUP_EPOCHS = 10

HIDDEN = 128
NUM_LAYERS = 6

KNN_K = 10
MAX_SEGMENTS = 768
M_MODES = 64  # <= MAX_SEGMENTS

VAL_RATIO = 0.15
TEST_RATIO = 0.15

USE_AMP = False
PIN_MEMORY = False
NUM_WORKERS = 4


# ============================================================
# ✅ LOSS / REGULARIZATION
# ============================================================
USE_UNCERTAINTY_WEIGHTING = True
SHEAR_EXTRA_WEIGHT = 2.5

LAMBDA_TRACE = 0.00
LAMBDA_FROB  = 0.00

DIAG_NONNEG = False
GRAD_CLIP_NORM = 1.0

# ✅ v6: clamp for uncertainty weights to avoid runaway negative values
UW_CLAMP_MIN = -6.0
UW_CLAMP_MAX = +6.0


# ============================================================
# ✅ Rotation feature (Scheme A)
# ============================================================
ROT_N = 24
ROT_FEAT_MODE = "onehot"  # "onehot" (dim=24) or "cs2" (dim=2, NOT recommended here)

ROT_RE = re.compile(r"_rot(\d+)", re.IGNORECASE)

def rot_feat_from_id(rot_id: int) -> np.ndarray:
    if ROT_FEAT_MODE.lower() == "onehot":
        v = np.zeros((ROT_N,), dtype=np.float32)
        v[int(rot_id) % ROT_N] = 1.0
        return v
    th = 2.0 * math.pi * (rot_id % ROT_N) / float(ROT_N)
    return np.array([math.cos(th), math.sin(th)], dtype=np.float32)

def parse_rot_id_from_filename(fn: str) -> int:
    m = ROT_RE.search(fn)
    if not m:
        return 0
    return int(m.group(1))


# ============================================================
# ✅ NPZ KEYS (robust)
# ============================================================
SEG_KEYS = ["segments", "Segments", "segment", "fiber_segments"]
GF_KEYS  = ["global_features", "global_feat", "gf", "Global_Features"]
Y_KEYS   = ["label_D6", "Label_D6", "D6", "y", "label", "Y"]


# -------------------- utils --------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def first_key(z, keys: List[str]) -> str:
    for k in keys:
        if k in z.files:
            return k
    return ""

def is_sample_npz(filename: str) -> bool:
    fn = filename.lower()
    if not fn.endswith(".npz"):
        return False
    bad_tokens = ["_all", "audit", "report", "summary", "labels", "log", "verify"]
    if any(t in fn for t in bad_tokens):
        return False
    if ("_rot" in fn) and ("_jit" in fn):
        return True
    return fn.startswith("rve_")

def resolve_sample_npz(data_dir: str) -> List[str]:
    files = [f for f in os.listdir(data_dir) if is_sample_npz(f)]
    files.sort()
    if not files:
        raise RuntimeError(f"[DATA_DIR] no SAMPLE .npz found in: {data_dir}\n"
                           f"Tip: ensure 4320 RVE_..._rotXX_jitY.npz exist.")
    return files


# -------------------- GROUP SPLIT (NO LEAKAGE) --------------------
GROUP_RE = re.compile(r"^(.*)_rot\d+_jit\d+\.npz$", re.IGNORECASE)

def group_id_from_filename(fn: str) -> str:
    m = GROUP_RE.match(fn)
    if m:
        return m.group(1)
    base = os.path.splitext(fn)[0]
    if "_rot" in base:
        base = base.split("_rot")[0]
    if "_jit" in base:
        base = base.split("_jit")[0]
    return base

def make_group_splits(files: List[str], val_ratio: float, test_ratio: float, seed: int
                     ) -> Tuple[List[str], List[str], List[str], Dict[str, int]]:
    groups: Dict[str, List[str]] = {}
    for fn in files:
        gid = group_id_from_filename(fn)
        groups.setdefault(gid, []).append(fn)

    group_ids = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    nG = len(group_ids)
    n_testG = max(1, int(round(nG * float(test_ratio))))
    n_valG  = max(1, int(round(nG * float(val_ratio))))
    n_trainG = nG - n_testG - n_valG
    if n_trainG <= 0:
        raise RuntimeError(f"Bad group split: nG={nG}, val={n_valG}, test={n_testG} -> train={n_trainG}")

    train_g = group_ids[:n_trainG]
    val_g   = group_ids[n_trainG:n_trainG + n_valG]
    test_g  = group_ids[n_trainG + n_valG:]

    assert set(train_g).isdisjoint(set(val_g))
    assert set(train_g).isdisjoint(set(test_g))
    assert set(val_g).isdisjoint(set(test_g))
    assert len(train_g) + len(val_g) + len(test_g) == nG

    def expand(gids: List[str]) -> List[str]:
        out = []
        for g in gids:
            out += sorted(groups[g])
        return out

    train_files = expand(train_g)
    val_files   = expand(val_g)
    test_files  = expand(test_g)

    def groups_in(file_list: List[str]) -> set:
        return set(group_id_from_filename(f) for f in file_list)

    assert groups_in(train_files).isdisjoint(groups_in(val_files))
    assert groups_in(train_files).isdisjoint(groups_in(test_files))
    assert groups_in(val_files).isdisjoint(groups_in(test_files))

    meta = {"n_groups": nG, "train_groups": len(train_g), "val_groups": len(val_g), "test_groups": len(test_g)}
    return train_files, val_files, test_files, meta


# ============================================================
# ✅ Pure PyTorch KNN graph
# ============================================================
def compute_knn_edge_index_pure(pos: torch.Tensor, k: int) -> torch.Tensor:
    """
    pos: [N,3] float32 on CPU
    returns edge_index: [2, E] long, directed i -> nn_j (excluding self)
    """
    n = int(pos.size(0))
    if n < 2:
        return torch.empty((2, 0), dtype=torch.long)

    k_use = int(min(k, n - 1))

    dist = torch.cdist(pos, pos, p=2)  # [N,N]
    dist.fill_diagonal_(float("inf"))
    nn_idx = torch.topk(dist, k=k_use, largest=False).indices  # [N,k]

    src = torch.arange(n, dtype=torch.long).unsqueeze(1).repeat(1, k_use).reshape(-1)
    dst = nn_idx.reshape(-1).to(torch.long)
    return torch.stack([src, dst], dim=0)  # [2,E]


def build_laplacian_eigvecs(edge_index: torch.Tensor, n: int, m_modes: int) -> torch.Tensor:
    """
    Dense Laplacian eigendecomposition (OK for n<=768). Returns U: [n, m]
    """
    if n <= 1:
        return torch.ones((n, 1), dtype=torch.float32)

    A = torch.zeros((n, n), dtype=torch.float32)
    if edge_index.numel() > 0:
        src, dst = edge_index[0], edge_index[1]
        A[src, dst] = 1.0
        A[dst, src] = 1.0

    deg = torch.sum(A, dim=1)
    L = torch.diag(deg) - A
    _, evecs = torch.linalg.eigh(L)
    m = int(min(m_modes, n))
    return evecs[:, :m].contiguous()


def orientation_tensor_6(dirv: np.ndarray) -> np.ndarray:
    d = dirv.astype(np.float64)
    A = (d[:, :, None] * d[:, None, :]).mean(axis=0)
    return np.array([A[0,0], A[1,1], A[2,2], A[0,1], A[0,2], A[1,2]], dtype=np.float32)

def spatial_cov_6(mid_n: np.ndarray) -> np.ndarray:
    x = mid_n.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    C = (x[:, :, None] * x[:, None, :]).mean(axis=0)
    return np.array([C[0,0], C[1,1], C[2,2], C[0,1], C[0,2], C[1,2]], dtype=np.float32)

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12)
    return 1.0 - ss_res / ss_tot


# ============================================================
# ✅ Dataset (returns per-graph tensors)
# ============================================================
class GFNOReadyDataset(Dataset):
    def __init__(self, data_dir: str, outdir: str,
                 file_list: List[str],
                 stats_files: List[str],
                 max_segments: int = 768, k: int = 10, m_modes: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.outdir = outdir
        self.files = list(file_list)

        self.max_segments = int(max_segments)
        self.k = int(k)
        self.m_modes = int(m_modes)

        # infer gf_dim
        gf_dim = None
        probe_n = min(50, len(self.files))
        for fn in self.files[:probe_n]:
            with np.load(os.path.join(data_dir, fn), allow_pickle=True) as z:
                gfk = first_key(z, GF_KEYS)
                if not gfk:
                    raise KeyError(f"[{fn}] missing global_features keys={z.files}")
                gf = np.asarray(z[gfk]).reshape(-1).astype(np.float32)
                if gf_dim is None:
                    gf_dim = int(gf.size)
                if int(gf.size) != int(gf_dim):
                    raise ValueError(f"[{fn}] global_features dim mismatch ({gf.size} vs {gf_dim})")
        if gf_dim is None:
            raise RuntimeError("Cannot infer global_features dim.")

        self.gf_dim = int(gf_dim)
        self.rot_dim = int(ROT_N if ROT_FEAT_MODE.lower() == "onehot" else 2)
        self.gf_dim_aug = int(self.gf_dim + self.rot_dim)

        print(f"[INFO] Using {len(self.files)} sample npz. gf_dim={self.gf_dim}  rot_dim={self.rot_dim}  gf_dim_aug={self.gf_dim_aug}")

        # ---- TRAIN-ONLY label mean/std ----
        Y = []
        for fn in stats_files:
            with np.load(os.path.join(self.data_dir, fn), allow_pickle=True) as z:
                y = self._load_label6(z)
            Y.append(y.reshape(1, 6))
        if not Y:
            raise RuntimeError("No labels loaded for stats_files (train split).")
        Y = np.concatenate(Y, axis=0).astype(np.float32)
        self.y_mean = Y.mean(axis=0).astype(np.float32)
        self.y_std  = Y.std(axis=0).astype(np.float32)
        self.y_std[self.y_std < 1e-8] = 1.0
        print("[INFO] Label stats computed on TRAIN only.")

        self.cache_dir = os.path.join(outdir, "cache_U")
        os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self): return len(self.files)

    def _load_segments(self, z) -> np.ndarray:
        k = first_key(z, SEG_KEYS)
        if not k:
            raise KeyError(f"Missing segments keys={z.files}")
        seg = np.asarray(z[k], dtype=np.float32)
        if seg.ndim != 2:
            raise ValueError(f"segments must be 2D, got {seg.shape}")
        if seg.shape[1] == 6:
            return seg
        if seg.shape[0] == 6 and seg.shape[1] != 6:
            return seg.T
        raise ValueError(f"segments shape must be (N,6) got {seg.shape}")

    def _load_global(self, z) -> np.ndarray:
        k = first_key(z, GF_KEYS)
        if not k:
            raise KeyError(f"Missing global_features keys={z.files}")
        gf = np.asarray(z[k], dtype=np.float32).reshape(-1)
        if gf.size != self.gf_dim:
            raise ValueError(f"global_features dim mismatch expect {self.gf_dim}, got {gf.size}")
        return gf

    def _load_label6(self, z) -> np.ndarray:
        k = first_key(z, Y_KEYS)
        if not k:
            raise KeyError(f"Missing label_D6 keys={z.files}")
        y = np.asarray(z[k], dtype=np.float32).reshape(-1)
        if y.size != 6:
            raise ValueError(f"label must be length 6, got {y.shape}")
        return y.astype(np.float32)

    def __getitem__(self, idx: int):
        fn = self.files[idx]
        fpath = os.path.join(self.data_dir, fn)

        with np.load(fpath, allow_pickle=True) as z:
            seg = self._load_segments(z)
            gf0 = self._load_global(z)
            y6  = self._load_label6(z)

        rot_id = parse_rot_id_from_filename(fn)
        rot_feat = rot_feat_from_id(rot_id)
        gf = np.concatenate([gf0.astype(np.float32).reshape(-1), rot_feat.reshape(-1)], axis=0).astype(np.float32)

        N = int(seg.shape[0])
        if N <= 0:
            raise RuntimeError(f"[{fn}] empty segments")

        if N > self.max_segments:
            p1 = seg[:, 0:3]; p2 = seg[:, 3:6]
            length0 = np.linalg.norm(p2 - p1, axis=1)
            sel = np.argsort(-length0)[:self.max_segments]
            seg = seg[sel]
            N = int(seg.shape[0])

        p1 = seg[:, 0:3]
        p2 = seg[:, 3:6]
        mid = 0.5 * (p1 + p2)
        vec = p2 - p1
        length = np.linalg.norm(vec, axis=1, keepdims=True).astype(np.float32)
        dirv = vec / np.clip(length, 1e-8, None)

        mn = mid.min(axis=0, keepdims=True)
        mx = mid.max(axis=0, keepdims=True)
        scale = np.clip(mx - mn, 1e-6, None)
        mid_n = (mid - mn) / scale

        A6 = orientation_tensor_6(dirv)
        C6 = spatial_cov_6(mid_n)
        g2 = np.concatenate([A6, C6], axis=0).astype(np.float32)

        gf_tile = np.tile(gf.reshape(1, -1), (N, 1))
        g2_tile = np.tile(g2.reshape(1, -1), (N, 1))
        x = np.concatenate([mid_n, dirv, length, gf_tile, g2_tile], axis=1).astype(np.float32)
        pos = mid_n.astype(np.float32)

        x_t = torch.from_numpy(x)       # [N,F] CPU
        pos_t = torch.from_numpy(pos)   # [N,3] CPU

        edge_index = compute_knn_edge_index_pure(pos_t, self.k)

        u_cache_path = os.path.join(self.cache_dir, fn.replace(".npz", f"_N{N}_k{self.k}_m{self.m_modes}.pt"))
        if os.path.isfile(u_cache_path):
            try:
                U = torch.load(u_cache_path, map_location="cpu", weights_only=True)
            except TypeError:
                U = torch.load(u_cache_path, map_location="cpu")
        else:
            U = build_laplacian_eigvecs(edge_index, n=N, m_modes=self.m_modes)
            torch.save(U.cpu(), u_cache_path)

        y_norm = (y6 - self.y_mean) / self.y_std
        y_t = torch.from_numpy(y_norm.astype(np.float32))  # [6]

        return {"x": x_t, "pos": pos_t, "edge_index": edge_index, "U": U, "y": y_t, "fn": fn, "rot_id": int(rot_id)}


# ============================================================
# ✅ Collate to a "batch graph" without PyG
# ============================================================
def collate_graphs(batch_list: List[dict]):
    xs, poss, ys = [], [], []
    edge_indices = []
    Us = []
    fns = []
    rot_ids = []
    ptr = [0]

    for item in batch_list:
        x = item["x"]; pos = item["pos"]; ei = item["edge_index"]; U = item["U"]; y = item["y"]
        n = int(x.size(0)); base = ptr[-1]

        xs.append(x)
        poss.append(pos)
        ys.append(y.view(1, -1))
        fns.append(item["fn"])
        rot_ids.append(item["rot_id"])

        edge_indices.append((ei + base) if ei.numel() > 0 else ei)
        Us.append(U)
        ptr.append(base + n)

    x_cat = torch.cat(xs, dim=0)
    pos_cat = torch.cat(poss, dim=0)
    y_cat = torch.cat(ys, dim=0)  # [B,6]
    edge_index_cat = torch.cat(edge_indices, dim=1) if edge_indices else torch.empty((2, 0), dtype=torch.long)

    B = len(batch_list)
    batch_vec = torch.empty((x_cat.size(0),), dtype=torch.long)
    for bi in range(B):
        batch_vec[ptr[bi]:ptr[bi+1]] = bi

    return {
        "x": x_cat,
        "pos": pos_cat,
        "edge_index": edge_index_cat,
        "U_list": Us,  # list of [Ni,m]
        "ptr": torch.tensor(ptr, dtype=torch.long),  # [B+1]
        "batch": batch_vec,  # [N]
        "y": y_cat,          # [B,6]
        "fn": fns,
        "rot_id": rot_ids,
    }


# ============================================================
# ✅ Model: spectral + local + residual (pure torch)
# ============================================================
class GraphSpectralConv(nn.Module):
    def __init__(self, channels: int, m_modes: int):
        super().__init__()
        self.channels = int(channels)
        self.m_modes = int(m_modes)
        self.W = nn.Parameter(torch.randn(self.m_modes, self.channels, self.channels) * 0.02)

    def forward_one(self, h: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        m = min(int(U.size(1)), self.m_modes)
        U = U[:, :m]
        W = self.W[:m]
        h_hat = U.t() @ h
        out_hat = torch.einsum("mc,mco->mo", h_hat, W)
        return U @ out_hat

    def forward(self, h: torch.Tensor, U_list: List[torch.Tensor], ptr: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(h)
        B = int(ptr.numel()) - 1
        for bi in range(B):
            s = int(ptr[bi].item()); t = int(ptr[bi+1].item())
            if t > s:
                out[s:t] = self.forward_one(h[s:t], U_list[bi].to(h.device))
        return out


class LocalMeanAgg(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.lin = nn.Linear(channels, channels)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return self.lin(h)

        src = edge_index[0]
        dst = edge_index[1]
        n = int(h.size(0))

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, h[src])

        deg = torch.zeros((n,), dtype=h.dtype, device=h.device)
        deg.index_add_(0, dst, torch.ones((dst.numel(),), dtype=h.dtype, device=h.device))
        deg = deg.clamp_min(1.0).unsqueeze(1)

        return self.lin(agg / deg)


class GFNOBlock(nn.Module):
    def __init__(self, channels: int, m_modes: int):
        super().__init__()
        self.spec = GraphSpectralConv(channels, m_modes)
        self.local = LocalMeanAgg(channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, U_list: List[torch.Tensor], ptr: torch.Tensor) -> torch.Tensor:
        h = h + self.spec(h, U_list, ptr) + self.local(h, edge_index)
        h = self.norm(h)
        return F.gelu(h)


def global_mean_pool_pure(h: torch.Tensor, batch: torch.Tensor, B: int) -> torch.Tensor:
    out = torch.zeros((B, h.size(1)), dtype=h.dtype, device=h.device)
    out.index_add_(0, batch, h)
    deg = torch.zeros((B,), dtype=h.dtype, device=h.device)
    deg.index_add_(0, batch, torch.ones_like(batch, dtype=h.dtype))
    deg = deg.clamp_min(1.0).unsqueeze(1)
    return out / deg


class TwoHeadGFNO(nn.Module):
    def __init__(self, in_dim: int, hidden: int, m_modes: int, num_layers: int):
        super().__init__()
        self.lift = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([GFNOBlock(hidden, m_modes) for _ in range(int(num_layers))])
        self.trunk = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU())

        self.head_diag = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))
        self.head_shear = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, batch_dict: dict) -> torch.Tensor:
        x = batch_dict["x"]
        edge_index = batch_dict["edge_index"]
        U_list = batch_dict["U_list"]
        ptr = batch_dict["ptr"]
        batch = batch_dict["batch"]
        B = int(ptr.numel()) - 1

        h = F.gelu(self.lift(x))
        for blk in self.blocks:
            h = blk(h, edge_index, U_list, ptr)

        g = global_mean_pool_pure(h, batch, B)
        g = self.trunk(g)

        d = self.head_diag(g)
        s = self.head_shear(g)
        if DIAG_NONNEG:
            d = F.softplus(d)
        return torch.cat([d, s], dim=1)


# -------------------- loss weighting --------------------
class UncertaintyWeights(nn.Module):
    def __init__(self, clamp_min: float = -6.0, clamp_max: float = 6.0):
        super().__init__()
        self.s_diag = nn.Parameter(torch.zeros(1))
        self.s_shear = nn.Parameter(torch.zeros(1))
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

    def forward(self, L_diag: torch.Tensor, L_shear: torch.Tensor) -> torch.Tensor:
        # ✅ v6: clamp to avoid runaway negative s
        s_diag = torch.clamp(self.s_diag, self.clamp_min, self.clamp_max)
        s_shear = torch.clamp(self.s_shear, self.clamp_min, self.clamp_max)

        loss = torch.exp(-s_diag) * L_diag + s_diag
        loss = loss + torch.exp(-s_shear) * L_shear + s_shear
        return loss


def D6_to_tensor(D6: torch.Tensor) -> torch.Tensor:
    B = D6.size(0)
    D = torch.zeros((B, 3, 3), dtype=D6.dtype, device=D6.device)
    D[:, 0, 0] = D6[:, 0]
    D[:, 1, 1] = D6[:, 1]
    D[:, 2, 2] = D6[:, 2]
    D[:, 0, 1] = D[:, 1, 0] = D6[:, 3]
    D[:, 1, 2] = D[:, 2, 1] = D6[:, 4]
    D[:, 0, 2] = D[:, 2, 0] = D6[:, 5]
    return D


def physics_invariants_loss(pred_n: torch.Tensor, y_n: torch.Tensor,
                            y_mean: torch.Tensor, y_std: torch.Tensor) -> torch.Tensor:
    pred = pred_n * y_std + y_mean
    y    = y_n    * y_std + y_mean
    Dp = D6_to_tensor(pred)
    Dy = D6_to_tensor(y)

    trace_p = torch.diagonal(Dp, dim1=1, dim2=2).sum(dim=1)
    trace_y = torch.diagonal(Dy, dim1=1, dim2=2).sum(dim=1)
    L_trace = F.mse_loss(trace_p, trace_y)

    frob_p = torch.sqrt(torch.sum(Dp * Dp, dim=(1, 2)) + 1e-12)
    frob_y = torch.sqrt(torch.sum(Dy * Dy, dim=(1, 2)) + 1e-12)
    L_frob = F.mse_loss(frob_p, frob_y)

    return LAMBDA_TRACE * L_trace + LAMBDA_FROB * L_frob


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    batch["x"] = batch["x"].to(device, non_blocking=False)
    batch["pos"] = batch["pos"].to(device, non_blocking=False)
    batch["edge_index"] = batch["edge_index"].to(device, non_blocking=False)
    batch["ptr"] = batch["ptr"].to(device, non_blocking=False)
    batch["batch"] = batch["batch"].to(device, non_blocking=False)
    batch["y"] = batch["y"].to(device, non_blocking=False)
    return batch


@torch.no_grad()
def evaluate_split_mse(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    Returns:
      total_mse_sum, diag_mse_sum, shear_mse_sum, n_samples
    (all in normalized space)
    """
    model.eval()
    total_sum = 0.0
    diag_sum = 0.0
    shear_sum = 0.0
    n = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        y = batch["y"]              # [B,6]
        pred = model(batch)         # [B,6]
        B = int(y.size(0))

        diff = pred - y
        total_sum += float(torch.sum(diff * diff).item())
        diag_sum  += float(torch.sum(diff[:, :3] * diff[:, :3]).item())
        shear_sum += float(torch.sum(diff[:, 3:] * diff[:, 3:]).item())
        n += B

    return total_sum, diag_sum, shear_sum, n


@torch.no_grad()
def predict_denorm(model: nn.Module, loader: DataLoader, device: torch.device,
                   y_mean: np.ndarray, y_std: np.ndarray):
    model.eval()
    Ys, Ps, FNs = [], [], []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        y_n = batch["y"]
        pred_n = model(batch)
        Ys.append(y_n.detach().cpu().numpy())
        Ps.append(pred_n.detach().cpu().numpy())
        FNs += list(batch["fn"])

    y_n = np.concatenate(Ys, axis=0) if Ys else np.zeros((0, 6), dtype=np.float32)
    p_n = np.concatenate(Ps, axis=0) if Ps else np.zeros((0, 6), dtype=np.float32)

    y = y_n * y_std.reshape(1, 6) + y_mean.reshape(1, 6)
    p = p_n * y_std.reshape(1, 6) + y_mean.reshape(1, 6)
    return y_n, p_n, y, p, FNs


def cosine_with_warmup_lr(optimizer, epoch: int, total_epochs: int, base_lr: float, warmup_epochs: int):
    if epoch <= warmup_epochs:
        lr = base_lr * (epoch / max(1, warmup_epochs))
    else:
        t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * t))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the pure-PyTorch GFNO model for SSFRC damage tensor prediction."
    )
    parser.add_argument("--data_dir", type=str, default=DATA_DIR,
                        help="Path to the folder containing all .npz samples. Default: data/dataset")
    parser.add_argument("--outdir", type=str, default=OUTDIR,
                        help="Output folder for checkpoints, logs, cache, and predictions. Default: results/gfno_run")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--num_layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--knn_k", type=int, default=KNN_K)
    parser.add_argument("--max_segments", type=int, default=MAX_SEGMENTS)
    parser.add_argument("--m_modes", type=int, default=M_MODES)
    return parser.parse_args()


def main():
    global DATA_DIR, OUTDIR, EPOCHS, BATCH_SIZE, NUM_WORKERS, SEED, LR, HIDDEN, NUM_LAYERS, KNN_K, MAX_SEGMENTS, M_MODES

    args = parse_args()
    DATA_DIR = args.data_dir
    OUTDIR = args.outdir
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    SEED = args.seed
    LR = args.lr
    HIDDEN = args.hidden
    NUM_LAYERS = args.num_layers
    KNN_K = args.knn_k
    MAX_SEGMENTS = args.max_segments
    M_MODES = args.m_modes

    os.makedirs(OUTDIR, exist_ok=True)
    if not os.path.isdir(DATA_DIR):
        raise RuntimeError(f"DATA_DIR not found: {DATA_DIR}")

    set_seed(SEED)

    all_files = resolve_sample_npz(DATA_DIR)
    train_files, val_files, test_files, split_meta = make_group_splits(
        all_files, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, seed=SEED
    )
    print("[SPLIT][GROUP]", split_meta)
    print("[SPLIT][FILES] train/val/test =", len(train_files), len(val_files), len(test_files))

    ds = GFNOReadyDataset(
        DATA_DIR, OUTDIR,
        file_list=all_files,
        stats_files=train_files,
        max_segments=MAX_SEGMENTS, k=KNN_K, m_modes=M_MODES
    )

    idx_map = {fn: i for i, fn in enumerate(ds.files)}
    train_idx = [idx_map[f] for f in train_files]
    val_idx   = [idx_map[f] for f in val_files]
    test_idx  = [idx_map[f] for f in test_files]

    train_set = Subset(ds, train_idx)
    val_set   = Subset(ds, val_idx)
    test_set  = Subset(ds, test_idx)

    persistent = bool(NUM_WORKERS > 0)
    dl_kwargs = dict(
        batch_size=BATCH_SIZE,
        pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS,
        persistent_workers=persistent,
        collate_fn=collate_graphs,
    )
    train_loader = DataLoader(train_set, shuffle=True,  **dl_kwargs)
    val_loader   = DataLoader(val_set,   shuffle=False, **dl_kwargs)
    test_loader  = DataLoader(test_set,  shuffle=False, **dl_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(USE_AMP and device.type == "cuda")
    print(f"[DEVICE] {device}  AMP={amp_enabled}")

    in_dim = 7 + ds.gf_dim_aug + 12
    model = TwoHeadGFNO(in_dim=in_dim, hidden=HIDDEN, m_modes=M_MODES, num_layers=NUM_LAYERS).to(device)

    uw = UncertaintyWeights(UW_CLAMP_MIN, UW_CLAMP_MAX).to(device) if USE_UNCERTAINTY_WEIGHTING else None
    params = list(model.parameters()) + ([] if uw is None else list(uw.parameters()))
    optim = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    cfg = {
        "DATA_DIR": DATA_DIR, "OUTDIR": OUTDIR, "SEED": SEED,
        "EPOCHS": EPOCHS, "BATCH_SIZE": BATCH_SIZE,
        "LR": LR, "WEIGHT_DECAY": WEIGHT_DECAY, "WARMUP_EPOCHS": WARMUP_EPOCHS,
        "HIDDEN": HIDDEN, "NUM_LAYERS": NUM_LAYERS,
        "KNN_K": KNN_K, "MAX_SEGMENTS": MAX_SEGMENTS, "M_MODES": M_MODES,
        "VAL_RATIO": VAL_RATIO, "TEST_RATIO": TEST_RATIO,
        "DEVICE": str(device), "AMP": amp_enabled,
        "USE_UNCERTAINTY_WEIGHTING": USE_UNCERTAINTY_WEIGHTING,
        "UW_CLAMP": [UW_CLAMP_MIN, UW_CLAMP_MAX],
        "SHEAR_EXTRA_WEIGHT": SHEAR_EXTRA_WEIGHT,
        "LAMBDA_TRACE": LAMBDA_TRACE, "LAMBDA_FROB": LAMBDA_FROB,
        "DIAG_NONNEG": DIAG_NONNEG,
        "LABEL_MEAN_TRAIN": ds.y_mean.tolist(),
        "LABEL_STD_TRAIN": ds.y_std.tolist(),
        "IN_DIM": in_dim,
        "NOTE": "PURE TORCH v6: split val metrics + best_total/best_shear checkpoints + clamped uncertainty weights",
        "SPLIT_META": split_meta,
        "ROT": dict(ROT_N=ROT_N, ROT_FEAT_MODE=ROT_FEAT_MODE, ROT_FEAT_DIM=ds.rot_dim, ROT_REGEX=ROT_RE.pattern),
    }
    with open(os.path.join(OUTDIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    log_path = os.path.join(OUTDIR, "train_log.csv")
    with open(log_path, "w", encoding="utf-8") as fp:
        fp.write("epoch,lr,train_loss,val_total_mse,val_diag_mse,val_shear_mse\n")

    ckpt_total = os.path.join(OUTDIR, "gfno_best_total.pt")
    ckpt_shear = os.path.join(OUTDIR, "gfno_best_shear.pt")
    best_total = math.inf
    best_shear = math.inf

    y_mean_t = torch.tensor(ds.y_mean, dtype=torch.float32, device=device).view(1, 6)
    y_std_t  = torch.tensor(ds.y_std,  dtype=torch.float32, device=device).view(1, 6)

    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        lr_now = cosine_with_warmup_lr(optim, ep, EPOCHS, LR, WARMUP_EPOCHS)

        model.train()
        if uw is not None:
            uw.train()

        sum_loss = 0.0
        seen = 0

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            y_n = batch["y"]

            optim.zero_grad(set_to_none=True)
            pred_n = model(batch)

            pred_d, pred_s = pred_n[:, :3], pred_n[:, 3:]
            y_d,    y_s    = y_n[:, :3],   y_n[:, 3:]

            L_diag  = F.mse_loss(pred_d, y_d)
            L_shear = F.mse_loss(pred_s, y_s) * SHEAR_EXTRA_WEIGHT
            loss = uw(L_diag, L_shear) if (uw is not None) else (L_diag + L_shear)

            if (LAMBDA_TRACE > 0.0) or (LAMBDA_FROB > 0.0):
                loss = loss + physics_invariants_loss(pred_n, y_n, y_mean_t, y_std_t)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_NORM)
            optim.step()

            sum_loss += float(loss.item()) * int(y_n.size(0))
            seen += int(y_n.size(0))

        train_loss = sum_loss / max(1, seen)

        # ---- val: total/diag/shear ----
        tot_sum, d_sum, s_sum, n_val = evaluate_split_mse(model, val_loader, device)
        val_total = tot_sum / max(1, n_val)
        val_diag  = d_sum / max(1, n_val)
        val_shear = s_sum / max(1, n_val)

        with open(log_path, "a", encoding="utf-8") as fp:
            fp.write(f"{ep},{lr_now:.6g},{train_loss:.6f},{val_total:.6f},{val_diag:.6f},{val_shear:.6f}\n")

        extra = ""
        if uw is not None:
            extra = f"  (s_diag={float(uw.s_diag.item()):+.3f}, s_shear={float(uw.s_shear.item()):+.3f})"
        print(f"[Epoch {ep:03d}] lr={lr_now:.3g}  train_loss={train_loss:.6f}  "
              f"val_total={val_total:.6f}  val_diag={val_diag:.6f}  val_shear={val_shear:.6f}{extra}")

        # ---- save best by total ----
        if val_total + 1e-12 < best_total:
            best_total = val_total
            torch.save({"model": model.state_dict(),
                        "uw": (None if uw is None else uw.state_dict()),
                        "epoch": ep,
                        "val_total": val_total,
                        "val_diag": val_diag,
                        "val_shear": val_shear}, ckpt_total)

        # ---- save best by shear ----
        if val_shear + 1e-12 < best_shear:
            best_shear = val_shear
            torch.save({"model": model.state_dict(),
                        "uw": (None if uw is None else uw.state_dict()),
                        "epoch": ep,
                        "val_total": val_total,
                        "val_diag": val_diag,
                        "val_shear": val_shear}, ckpt_shear)

    print(f"\nTraining done. elapsed = {(time.time()-t0)/60.0:.1f} min")
    print("==> Evaluate best checkpoints on TEST …")

    # =========================
    # Evaluate BEST_TOTAL
    # =========================
    if os.path.isfile(ckpt_total):
        ckpt = torch.load(ckpt_total, map_location=device)
        model.load_state_dict(ckpt["model"])
        if uw is not None and ckpt.get("uw", None) is not None:
            uw.load_state_dict(ckpt["uw"])
        print(f"[LOAD] best_total @ epoch={ckpt.get('epoch')}  val_total={ckpt.get('val_total'):.6f}  val_shear={ckpt.get('val_shear'):.6f}")

    y_n_true, p_n_pred, y_true, y_pred, fns = predict_denorm(model, test_loader, device, ds.y_mean, ds.y_std)
    cols = ["D11", "D22", "D33", "D12", "D23", "D13"]

    pred_csv = os.path.join(OUTDIR, "pred_test_denorm_best_total.csv")
    out_df = pd.DataFrame(y_pred, columns=[c + "_pred" for c in cols])
    for i, c in enumerate(cols):
        out_df[c + "_true"] = y_true[:, i]
    out_df["file"] = fns[:len(out_df)]
    out_df.to_csv(pred_csv, index=False)

    r2s_total = {c: r2_score(y_true[:, i], y_pred[:, i]) for i, c in enumerate(cols)}
    r2_df = pd.DataFrame([r2s_total])
    r2_path = os.path.join(OUTDIR, "r2_test_best_total.csv")
    r2_df.to_csv(r2_path, index=False)

    print("[TEST][best_total] R2:", r2s_total)
    print(f"Saved -> {pred_csv}")
    print(f"Saved -> {r2_path}")

    # =========================
    # Evaluate BEST_SHEAR
    # =========================
    if os.path.isfile(ckpt_shear):
        ckpt = torch.load(ckpt_shear, map_location=device)
        model.load_state_dict(ckpt["model"])
        if uw is not None and ckpt.get("uw", None) is not None:
            uw.load_state_dict(ckpt["uw"])
        print(f"[LOAD] best_shear @ epoch={ckpt.get('epoch')}  val_total={ckpt.get('val_total'):.6f}  val_shear={ckpt.get('val_shear'):.6f}")

    y_n_true, p_n_pred, y_true, y_pred, fns = predict_denorm(model, test_loader, device, ds.y_mean, ds.y_std)

    pred_csv = os.path.join(OUTDIR, "pred_test_denorm_best_shear.csv")
    out_df = pd.DataFrame(y_pred, columns=[c + "_pred" for c in cols])
    for i, c in enumerate(cols):
        out_df[c + "_true"] = y_true[:, i]
    out_df["file"] = fns[:len(out_df)]
    out_df.to_csv(pred_csv, index=False)

    r2s_shear = {c: r2_score(y_true[:, i], y_pred[:, i]) for i, c in enumerate(cols)}
    r2_df = pd.DataFrame([r2s_shear])
    r2_path = os.path.join(OUTDIR, "r2_test_best_shear.csv")
    r2_df.to_csv(r2_path, index=False)

    print("[TEST][best_shear] R2:", r2s_shear)
    print(f"Saved -> {pred_csv}")
    print(f"Saved -> {r2_path}")


if __name__ == "__main__":
    main()