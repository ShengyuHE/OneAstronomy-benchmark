"""
AION模型DDP分布式训练脚本
v4: 没有逻辑错误的第一个版本
v4.12: 扩大数据集包含未匹配数据集，固定输出token数，修正输入输出策略及collate_fn，增加标量损失权重，输入锚点比例为[0.8,1.6]，输出beta参数为[1,4]
未匹配数据集不使用红移作为输入anchor, 参考4M的混合策略，修改了输入锚点token的比例，增加了image or spectrum --> all的比例，增加了primary_mod占输出总预算的比例
增加target中包含标量的概率
参考AION论文第5章训练方法论实现：
- 采用多模态掩码建模（Multimodal Masked Modeling）
- 使用Cross-Entropy损失函数
- 支持PyTorch DDP分布式训练
- 支持混合精度训练（AMP）
- 支持梯度累积
- 支持TensorBoard/WandB日志记录
- 支持模型检查点保存与恢复

使用方法：
    # 单机多卡训练
    torchrun --standalone --nnodes=1 --nproc_per_node=8 train_aion_ddp.py --config configs/aion_base.yaml

    # 快速测试
    torchrun --standalone --nnodes=1 --nproc_per_node=2 train_aion_ddp.py --batch_size 4 --max_samples 1000
"""

import os
import sys
import math
import argparse
import importlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from contextlib import nullcontext
import json
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, IterableDataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, LambdaLR
from torch.amp import GradScaler, autocast

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import hashlib

_TOK_IMAGE_SAMPLING_INFO = None


def _get_tok_image_sampling_info(anchor_mod: str):
    global _TOK_IMAGE_SAMPLING_INFO
    if _TOK_IMAGE_SAMPLING_INFO is not None:
        return _TOK_IMAGE_SAMPLING_INFO
    if anchor_mod == "tok_image_hsc":
        stats_path = Path(__file__).parent / "tools" / "token_distribution_hsc" / "token_stats.json"
    else:
        stats_path = Path(__file__).parent / "tools" / "token_distribution_legacy" / "token_stats.json"
    try:
        with open(stats_path, "r") as f:
            stats = json.load(f)
    except Exception:
        _TOK_IMAGE_SAMPLING_INFO = {"weights": {}, "default_weight": 1.0}
        return _TOK_IMAGE_SAMPLING_INFO
    most_common = stats.get("most_common", [])[:10]
    if not most_common:
        _TOK_IMAGE_SAMPLING_INFO = {"weights": {}, "default_weight": 1.0}
        return _TOK_IMAGE_SAMPLING_INFO
    default_ratio = most_common[min(len(most_common) - 1, 9)]["ratio"]
    default_ratio = float(default_ratio) if default_ratio > 0 else 1.0
    default_weight = 1.0 / default_ratio
    weight_map = {}
    for item in most_common:
        token_id = int(item["token_id"])
        ratio = float(item["ratio"])
        if ratio <= 0:
            continue
        weight_map[token_id] = 1.0 / ratio
    _TOK_IMAGE_SAMPLING_INFO = {"weights": weight_map, "default_weight": default_weight}
    return _TOK_IMAGE_SAMPLING_INFO


def _weighted_sample_tok_image_indices(token_tensor, batch_idx, candidate_indices, num_samples, anchor_mod):
    if num_samples <= 0 or len(candidate_indices) == 0:
        return []
    if num_samples >= len(candidate_indices):
        return list(candidate_indices)
    info = _get_tok_image_sampling_info(anchor_mod)
    weights_map = info["weights"]
    default_weight = info["default_weight"]
    token_ids = token_tensor[batch_idx, candidate_indices].tolist()
    weights = []
    for token_id in token_ids:
        weights.append(weights_map.get(int(token_id), default_weight))
    weights_array = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights_array).all():
        return random.sample(candidate_indices, num_samples)
    total = float(weights_array.sum())
    if total <= 0:
        return random.sample(candidate_indices, num_samples)
    probabilities = weights_array / total
    chosen_pos = np.random.choice(len(candidate_indices), size=num_samples, replace=False, p=probabilities)
    chosen_pos = chosen_pos.tolist()
    return [candidate_indices[i] for i in chosen_pos]


# AION imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from aion import AION
from aion.codecs import CodecManager
from aion.modalities import (
    LegacySurveyImage,
    HSCImage,
    DESISpectrum,
    SDSSSpectrum,
    LegacySurveyCatalog,
    LegacySurveySegmentationMap,
    LegacySurveyFluxG,
    LegacySurveyFluxR,
    LegacySurveyFluxI,
    LegacySurveyFluxZ,
    LegacySurveyFluxW1,
    LegacySurveyFluxW2,
    LegacySurveyFluxW3,
    LegacySurveyFluxW4,
    LegacySurveyShapeR,
    LegacySurveyShapeE1,
    LegacySurveyShapeE2,
    LegacySurveyEBV,
    Z,
    HSCAG,
    HSCAR,
    HSCAI,
    HSCAZ,
    HSCAY,
    HSCMagG,
    HSCMagR,
    HSCMagI,
    HSCMagZ,
    HSCMagY,
    HSCShape11,
    HSCShape22,
    HSCShape12,
    GaiaFluxG,
    GaiaFluxBp,
    GaiaFluxRp,
    GaiaParallax,
    Ra,
    Dec,
    GaiaXpBp,
    GaiaXpRp,
)


# ===================== 工具函数 =====================

def setup_distributed():
    """初始化分布式训练环境"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    is_distributed = world_size > 1
    
    if is_distributed:
        torch.cuda.set_device(local_rank)
        # 增加NCCL超时时间到120分钟，防止epoch末尾的同步超时
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
        torch.backends.cudnn.benchmark = True
        # 设置NCCL环境变量以提高稳定性
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    
    return {
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "is_distributed": is_distributed,
        "is_main_process": rank == 0,
    }


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def synchronize_device():
    """同步当前设备，确保计时包含完整CUDA执行时间"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _distributed_reduce_scalar(value: float, op=dist.ReduceOp.SUM) -> float:
    """对标量做分布式归约；未初始化分布式时直接返回"""
    if not (dist.is_available() and dist.is_initialized()):
        return value
    tensor = torch.tensor(
        value,
        dtype=torch.float64,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    dist.all_reduce(tensor, op=op)
    return float(tensor.item())


def _next_batch(data_iter, data_loader):
    """获取下一个batch；到达末尾时自动重建迭代器。"""
    try:
        return next(data_iter), data_iter
    except StopIteration:
        data_iter = iter(data_loader)
        return next(data_iter), data_iter


def set_seed(seed: int, rank: int = 0):
    """设置随机种子以确保可重复性"""
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_tensor(data_array, dtype="float32", device="cuda"):
    """将数据转换为tensor"""
    return torch.tensor(np.array(data_array).astype(dtype), device=device)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """统计模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# ===================== 数据加载 =====================

class AIONDataset(Dataset):
    """AION训练数据集（JSON token文件加载）
    
    直接加载已编码的token数据，无需codec_manager
    """
    
    def __init__(
        self,
        data_path: str,
        max_samples: Optional[int] = None,
        world_size: int = 1,
        rank: int = 0,
    ):
        super().__init__()
        self.data_path = data_path
        self.max_samples = max_samples
        self.world_size = world_size
        self.rank = rank
        
        # 加载JSON token数据（支持文件或文件夹）
        print(f"[Rank {rank}] Loading data from: {data_path}")
        path_obj = Path(data_path)
        
        self._file_paths: List[str] = []
        self._path_to_file_id: Dict[str, int] = {}
        self._file_group_ids: List[int] = []
        self.id_to_group: List[str] = []
        self._group_to_id: Dict[str, int] = {}
        self._file_formats: List[str] = []
        self._offsets: Optional[np.ndarray] = None
        self._file_ids: Optional[np.ndarray] = None
        self.group_ids: Optional[np.ndarray] = None

        self._fh_cache: Dict[int, Any] = {}
        self._fh_cache_order: List[int] = []
        self._fh_cache_max = 16
        if path_obj.is_file():
            # 单个JSON文件
            self._register_file(str(path_obj), "__single__")
        elif path_obj.is_dir():
            json_files = sorted(list(path_obj.rglob("*.json"))) + sorted(list(path_obj.rglob("*.jsonl")))
            if not json_files:
                raise ValueError(f"在文件夹 {data_path} 中未找到任何.json或.jsonl文件")

            print(f"[Rank {rank}] Discovered {len(json_files)} data files")
            for jf in json_files:
                rel = jf.relative_to(path_obj)
                group = rel.parts[0] if len(rel.parts) > 1 else "__root__"
                self._register_file(str(jf), group)
        else:
            raise ValueError(f"路径不存在或无效: {data_path}")
        
        offsets_list: List[int] = []
        file_ids_list: List[int] = []
        group_ids_list: List[int] = []

        for file_id, fp in enumerate(self._file_paths):
            if max_samples and len(offsets_list) >= max_samples:
                break

            fmt = self._file_formats[file_id]
            group_id = self._file_group_ids[file_id]
            if fmt == "jsonl":
                file_offsets = self._build_offsets_jsonl(fp)
            elif fmt == "json_array":
                file_offsets = self._build_offsets_json_array(fp)
            elif fmt == "json_object":
                file_offsets = [self._first_non_ws_offset(fp)]
            else:
                raise ValueError(f"未知数据格式: {fmt} ({fp})")

            if file_offsets:
                if max_samples:
                    remaining = max_samples - len(offsets_list)
                    if remaining <= 0:
                        break
                    if len(file_offsets) > remaining:
                        file_offsets = file_offsets[:remaining]
                offsets_list.extend(file_offsets)
                file_ids_list.extend([file_id] * len(file_offsets))
                group_ids_list.extend([group_id] * len(file_offsets))

        if len(offsets_list) == 0:
            raise ValueError(f"在路径 {data_path} 中未找到任何有效数据")

        if max_samples and len(offsets_list) > max_samples:
            offsets_list = offsets_list[:max_samples]
            file_ids_list = file_ids_list[:max_samples]
            group_ids_list = group_ids_list[:max_samples]

        self._offsets = np.asarray(offsets_list, dtype=np.int64)
        self._file_ids = np.asarray(file_ids_list, dtype=np.int32)
        self.group_ids = np.asarray(group_ids_list, dtype=np.int32)

        print(f"[Rank {rank}] Data index ready: {len(self._offsets)} records")

    def __del__(self):
        for fh in self._fh_cache.values():
            try:
                fh.close()
            except Exception:
                pass

    def _register_file(self, file_path: str, group: str):
        group_id = self._group_to_id.get(group)
        if group_id is None:
            group_id = len(self.id_to_group)
            self._group_to_id[group] = group_id
            self.id_to_group.append(group)

        file_id = len(self._file_paths)
        self._file_paths.append(file_path)
        self._path_to_file_id[file_path] = file_id
        self._file_group_ids.append(group_id)
        self._file_formats.append(self._detect_format(file_path))

    def _detect_format(self, file_path: str) -> str:
        p = Path(file_path)
        if p.suffix.lower() == ".jsonl":
            return "jsonl"

        with open(file_path, "rb") as f:
            head = f.read(1024 * 1024)
        head_strip = head.lstrip()
        if not head_strip:
            return "jsonl"
        first = head_strip[:1]
        if first == b"[":
            return "json_array"
        if first == b"{":
            if b"\n{" in head_strip[:200000] or b"\r\n{" in head_strip[:200000]:
                return "jsonl"
            return "json_object"
        return "jsonl"

    def _first_non_ws_offset(self, file_path: str) -> int:
        with open(file_path, "rb") as f:
            off = 0
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                for i, b in enumerate(chunk):
                    if b not in b" \t\r\n":
                        return off + i
                off += len(chunk)
        raise ValueError(f"空文件: {file_path}")

    def _build_offsets_jsonl(self, file_path: str) -> List[int]:
        offsets: List[int] = []
        with open(file_path, "rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(pos)
        return offsets

    def _build_offsets_json_array(self, file_path: str) -> List[int]:
        offsets: List[int] = []
        with open(file_path, "rb") as f:
            in_string = False
            escape = False
            depth = 0
            started = False
            pos = 0

            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break

                for b in chunk:
                    if not started:
                        if b in b" \t\r\n":
                            pos += 1
                            continue
                        if b == ord("["):
                            started = True
                            pos += 1
                            continue
                        raise ValueError(f"文件不是JSON数组: {file_path}")

                    if in_string:
                        if escape:
                            escape = False
                        elif b == ord("\\"):
                            escape = True
                        elif b == ord('"'):
                            in_string = False
                    else:
                        if b == ord('"'):
                            in_string = True
                        elif b == ord("{"):
                            if depth == 0:
                                offsets.append(pos)
                            depth += 1
                        elif b == ord("}"):
                            if depth > 0:
                                depth -= 1
                    pos += 1

        return offsets
        
    def __len__(self):
        return int(self._offsets.shape[0]) if self._offsets is not None else 0
    
    def __getitem__(self, idx):
        """按索引从磁盘读取单条样本（低内存占用）"""
        if self._offsets is None or self._file_ids is None:
            raise RuntimeError("Dataset index not initialized")

        file_id = int(self._file_ids[idx])
        offset = int(self._offsets[idx])
        fmt = self._file_formats[file_id]
        fp = self._file_paths[file_id]

        if fmt == "jsonl":
            item = self._read_one_jsonl(file_id, fp, offset)
        elif fmt == "json_array":
            item = self._read_one_json_array(file_id, fp, offset)
        elif fmt == "json_object":
            item = self._read_one_json_object(file_id, fp, offset)
        else:
            raise ValueError(f"未知数据格式: {fmt} ({fp})")
        sample = {}
        
        # 直接返回token数据
        if "tok_image" in item:
            sample["tok_image"] = item["tok_image"]
        if "tok_image_hsc" in item:
            sample["tok_image_hsc"] = item["tok_image_hsc"]
        if "tok_spectrum_desi" in item:
            sample["tok_spectrum_desi"] = item["tok_spectrum_desi"]
        if "tok_spectrum_sdss" in item:
            sample["tok_spectrum_sdss"] = item["tok_spectrum_sdss"]
        if "tok_flux_i" in item:
            sample["tok_flux_i"] = item["tok_flux_i"]
        if "tok_flux_w1" in item:
            sample["tok_flux_w1"] = item["tok_flux_w1"]
        if "tok_flux_w2" in item:
            sample["tok_flux_w2"] = item["tok_flux_w2"]
        if "tok_flux_w3" in item:
            sample["tok_flux_w3"] = item["tok_flux_w3"]
        if "tok_flux_w4" in item:
            sample["tok_flux_w4"] = item["tok_flux_w4"]
        if "tok_shape_r" in item:
            sample["tok_shape_r"] = item["tok_shape_r"]
        if "tok_shape_e1" in item:
            sample["tok_shape_e1"] = item["tok_shape_e1"]
        if "tok_shape_e2" in item:
            sample["tok_shape_e2"] = item["tok_shape_e2"]
        if "tok_ebv" in item:
            sample["tok_ebv"] = item["tok_ebv"]
        if "tok_z" in item:
            sample["tok_z"] = item["tok_z"]
        if "tok_flux_g" in item:
            sample["tok_flux_g"] = item["tok_flux_g"]
        if "tok_flux_r" in item:
            sample["tok_flux_r"] = item["tok_flux_r"]
        if "tok_flux_z" in item:
            sample["tok_flux_z"] = item["tok_flux_z"]
        if "tok_a_g" in item:
            sample["tok_a_g"] = item["tok_a_g"]
        if "tok_a_r" in item:
            sample["tok_a_r"] = item["tok_a_r"]
        if "tok_a_i" in item:
            sample["tok_a_i"] = item["tok_a_i"]
        if "tok_a_z" in item:
            sample["tok_a_z"] = item["tok_a_z"]
        if "tok_a_y" in item:
            sample["tok_a_y"] = item["tok_a_y"]
        if "tok_mag_g" in item:
            sample["tok_mag_g"] = item["tok_mag_g"]
        if "tok_mag_r" in item:
            sample["tok_mag_r"] = item["tok_mag_r"]
        if "tok_mag_i" in item:
            sample["tok_mag_i"] = item["tok_mag_i"]
        if "tok_mag_z" in item:
            sample["tok_mag_z"] = item["tok_mag_z"]
        if "tok_mag_y" in item:
            sample["tok_mag_y"] = item["tok_mag_y"]
        if "tok_shape11" in item:
            sample["tok_shape11"] = item["tok_shape11"]
        if "tok_shape22" in item:
            sample["tok_shape22"] = item["tok_shape22"]
        if "tok_shape12" in item:
            sample["tok_shape12"] = item["tok_shape12"]
        
        return sample

    def _get_fh(self, file_id: int, file_path: str):
        fh = self._fh_cache.get(file_id)
        if fh is not None:
            if file_id in self._fh_cache_order:
                self._fh_cache_order.remove(file_id)
            self._fh_cache_order.append(file_id)
            return fh

        fh = open(file_path, "rb")
        self._fh_cache[file_id] = fh
        self._fh_cache_order.append(file_id)

        while len(self._fh_cache_order) > self._fh_cache_max:
            evict_id = self._fh_cache_order.pop(0)
            evict_fh = self._fh_cache.pop(evict_id, None)
            if evict_fh is not None:
                try:
                    evict_fh.close()
                except Exception:
                    pass
        return fh

    def _read_one_jsonl(self, file_id: int, file_path: str, offset: int) -> Dict[str, Any]:
        fh = self._get_fh(file_id, file_path)
        fh.seek(offset)
        line = fh.readline()
        if not line:
            raise IndexError(f"读取越界: {file_path}@{offset}")
        return json.loads(line.decode("utf-8"))

    def _read_one_json_object(self, file_id: int, file_path: str, offset: int) -> Dict[str, Any]:
        fh = self._get_fh(file_id, file_path)
        fh.seek(offset)
        data = fh.read()
        if not data:
            raise ValueError(f"空文件: {file_path}")
        obj = json.loads(data.decode("utf-8"))
        if isinstance(obj, list):
            if len(obj) != 1:
                raise ValueError(f"json_object 期望单对象，但读到列表: {file_path}")
            obj = obj[0]
        return obj

    def _read_one_json_array(self, file_id: int, file_path: str, offset: int) -> Dict[str, Any]:
        fh = self._get_fh(file_id, file_path)
        fh.seek(offset)

        buf = bytearray()
        in_string = False
        escape = False
        depth = 0
        started = False

        while True:
            chunk = fh.read(1024 * 64)
            if not chunk:
                break
            buf.extend(chunk)

            i = 0
            while i < len(buf):
                b = buf[i]
                if not started:
                    if b in b" \t\r\n":
                        i += 1
                        continue
                    if b == ord("{"):
                        started = True
                        depth = 1
                        i += 1
                        continue
                    raise ValueError(f"JSON数组项不是对象: {file_path}@{offset}")

                if in_string:
                    if escape:
                        escape = False
                    elif b == ord("\\"):
                        escape = True
                    elif b == ord('"'):
                        in_string = False
                else:
                    if b == ord('"'):
                        in_string = True
                    elif b == ord("{"):
                        depth += 1
                    elif b == ord("}"):
                        depth -= 1
                        if depth == 0:
                            raw = bytes(buf[: i + 1])
                            return json.loads(raw.decode("utf-8"))
                i += 1

            if len(buf) > 1024 * 1024 * 32:
                raise ValueError(f"单条样本过大，无法解析: {file_path}@{offset}")

        raise ValueError(f"无法解析JSON对象: {file_path}@{offset}")


def _stable_int_from_str(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


class GroupedDistributedBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if not (0 <= rank < num_replicas):
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        self._subset_indices = None
        self._groups = None
        self._group_ids = None

        if isinstance(dataset, torch.utils.data.Subset):
            base = dataset.dataset
            self._subset_indices = np.asarray(dataset.indices, dtype=np.int64)
            if hasattr(base, "group_ids"):
                self._group_ids = np.asarray(base.group_ids, dtype=np.int32)[self._subset_indices]
            elif hasattr(base, "groups"):
                base_groups = base.groups
                self._groups = [base_groups[i] for i in self._subset_indices.tolist()]
            else:
                raise ValueError("Subset base dataset must have `group_ids` or `groups` attribute")
        else:
            if hasattr(dataset, "group_ids"):
                self._group_ids = np.asarray(dataset.group_ids, dtype=np.int32)
            elif hasattr(dataset, "groups"):
                self._groups = list(dataset.groups)
            else:
                raise ValueError("Dataset must have `group_ids` or `groups` attribute")

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self) -> int:
        denom = self.num_replicas * self.batch_size
        if self._group_ids is not None:
            counts = np.bincount(self._group_ids.astype(np.int64))
            if self.drop_last:
                return int((counts // denom).sum())
            return int(((counts + denom - 1) // denom).sum())

        total_batches = 0
        group_to_count = {}
        for g in self._groups:
            group_to_count[g] = group_to_count.get(g, 0) + 1
        for count in group_to_count.values():
            if self.drop_last:
                total_batches += count // denom
            else:
                total_batches += (count + denom - 1) // denom
        return int(total_batches)

    def __iter__(self):
        denom = self.num_replicas * self.batch_size

        if self._group_ids is not None:
            group_ids = self._group_ids
            group_to_rank_indices: Dict[int, np.ndarray] = {}
            group_to_ptr: Dict[int, int] = {}
            schedule: List[int] = []

            order = np.argsort(group_ids, kind="stable")
            sorted_group_ids = group_ids[order]
            uniq, first_idx, counts = np.unique(sorted_group_ids, return_index=True, return_counts=True)

            for gid, start, count in zip(uniq.tolist(), first_idx.tolist(), counts.tolist()):
                group_indices = order[start: start + count]
                if self.shuffle:
                    rng = np.random.default_rng(self.seed + self.epoch * 1000 + _stable_int_from_str(str(int(gid))))
                    rng.shuffle(group_indices)

                if self.drop_last:
                    total_size = (len(group_indices) // denom) * denom
                    group_indices = group_indices[:total_size]
                else:
                    total_size = ((len(group_indices) + denom - 1) // denom) * denom
                    padding_size = total_size - len(group_indices)
                    if padding_size > 0:
                        group_indices = np.concatenate([group_indices, group_indices[:padding_size]])

                if total_size == 0:
                    continue

                rank_indices = group_indices[self.rank:total_size:self.num_replicas]
                num_batches = len(rank_indices) // self.batch_size
                if num_batches == 0:
                    continue

                group_to_rank_indices[int(gid)] = rank_indices
                group_to_ptr[int(gid)] = 0
                schedule.extend([int(gid)] * num_batches)

            if self.shuffle:
                rng = random.Random(self.seed + self.epoch * 1000 + 99991)
                rng.shuffle(schedule)

            for gid in schedule:
                p = group_to_ptr[gid]
                arr = group_to_rank_indices[gid]
                batch = arr[p: p + self.batch_size]
                group_to_ptr[gid] = p + self.batch_size
                yield batch.tolist()
            return

        n = len(self._groups)
        indices = list(range(n))
        group_to_indices: Dict[Any, List[int]] = {}
        for i in indices:
            g = self._groups[i]
            group_to_indices.setdefault(g, []).append(i)

        group_to_rank_indices: Dict[Any, List[int]] = {}
        group_to_ptr: Dict[Any, int] = {}
        schedule: List[Any] = []

        for g, group_indices in group_to_indices.items():
            if self.shuffle:
                rng = random.Random(self.seed + self.epoch * 1000 + _stable_int_from_str(str(g)))
                rng.shuffle(group_indices)

            if self.drop_last:
                total_size = (len(group_indices) // denom) * denom
                group_indices = group_indices[:total_size]
            else:
                total_size = ((len(group_indices) + denom - 1) // denom) * denom
                padding_size = total_size - len(group_indices)
                if padding_size > 0:
                    group_indices = group_indices + group_indices[:padding_size]

            if total_size == 0:
                continue

            rank_indices = group_indices[self.rank:total_size:self.num_replicas]
            num_batches = len(rank_indices) // self.batch_size
            if num_batches == 0:
                continue

            group_to_rank_indices[g] = rank_indices
            group_to_ptr[g] = 0
            schedule.extend([g] * num_batches)

        if self.shuffle:
            rng = random.Random(self.seed + self.epoch * 1000 + 99991)
            rng.shuffle(schedule)

        for g in schedule:
            p = group_to_ptr[g]
            batch = group_to_rank_indices[g][p: p + self.batch_size]
            group_to_ptr[g] = p + self.batch_size
            yield batch

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """批处理 collate 函数 - 处理token数据
    
    将token数据打包成batch
    """
    result = {}
    
    # 处理所有token类型的键
    token_keys = [
        "tok_image",
        "tok_image_hsc",
        "tok_spectrum_desi",
        "tok_spectrum_sdss",
        "tok_z",
        "tok_flux_g",
        "tok_flux_r",
        "tok_flux_z",
        "tok_flux_i",
        "tok_flux_w1",
        "tok_flux_w2",
        "tok_flux_w3",
        "tok_flux_w4",
        "tok_shape_r",
        "tok_shape_e1",
        "tok_shape_e2",
        "tok_ebv",
        "tok_a_g",
        "tok_a_r",
        "tok_a_i",
        "tok_a_z",
        "tok_a_y",
        "tok_mag_g",
        "tok_mag_r",
        "tok_mag_i",
        "tok_mag_z",
        "tok_mag_y",
        "tok_shape11",
        "tok_shape22",
        "tok_shape12" 
    ]
    
    batch_size = len(batch)
    
    for key in token_keys:
        # 先收集现有样本的tensor，确定形状信息
        present_tensors = []
        for s in batch:
            if key in s:
                val = s[key]
                if isinstance(val, list):
                    t = torch.LongTensor(val)
                elif isinstance(val, np.ndarray):
                    t = torch.from_numpy(val).long()
                elif isinstance(val, torch.Tensor):
                    t = val.long()
                else:
                    t = torch.LongTensor([val])
                present_tensors.append(t)
        
        if not present_tensors:
            continue
        
        # 序列型token需要padding到统一长度
        is_sequence = key in ["tok_image", "tok_image_hsc","tok_spectrum_desi", "tok_spectrum_sdss"]
        if is_sequence:
            max_len = max(t.shape[0] for t in present_tensors)
        
        tensors_for_all = []
        for s in batch:
            if key in s:
                val = s[key]
                if isinstance(val, list):
                    t = torch.LongTensor(val)
                elif isinstance(val, np.ndarray):
                    t = torch.from_numpy(val).long()
                elif isinstance(val, torch.Tensor):
                    t = val.long()
                else:
                    t = torch.LongTensor([val])
            else:
                if is_sequence:
                    t = torch.full((max_len,), -1, dtype=torch.long)
                else:
                    t = torch.full((1,), -1, dtype=torch.long)
            
            if is_sequence and t.shape[0] < max_len:
                pad = torch.full((max_len - t.shape[0],), -1, dtype=torch.long)
                t = torch.cat([t, pad])
            
            tensors_for_all.append(t)
        
        stacked = torch.stack(tensors_for_all)
        if not is_sequence and stacked.dim() > 1:
            stacked = stacked.squeeze(-1)
        result[key] = stacked
    
    return result


# ===================== 训练相关类 =====================

class MultimodalMaskedLoss(nn.Module):
    """多模态掩码建模损失函数
    
    参考AION论文第5章：使用Cross-Entropy损失函数训练模型
    L = -sum_m [ sum_i (1-M_i^m) * log(p(x_i^m | X_M)) ]
    
    其中：
    - m: 模态索引
    - i: token索引  
    - M_i^m: 掩码（1表示掩码，0表示可见）
    - x_i^m: 原始token
    - X_M: 可见token集合
    """
    
    def __init__(
        self,
        label_smoothing: float = 0.0,
        modality_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.modality_weights = modality_weights or {}
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction='none'
        )
    
    def forward(
        self,
        logits: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算多模态损失
        
        Args:
            logits: 模型输出logits {modality_key: (num_selected, vocab_size)}
                    注意：AION模型返回的logits是扁平化的，只包含被选中的token
            targets: 目标token {modality_key: (B, N) or (B,)}
            masks: 掩码 {modality_key: (B, N) or (B,)}，True表示需要计算损失
        
        Returns:
            total_loss: 总损失
            loss_dict: 各模态损失详情
        """
        total_loss = 0.0
        loss_dict = {}
        total_weight = 0.0
        
        # 调试信息（可在正常运行后删除）
        # print(f"[DEBUG] logits keys: {list(logits.keys())}")
        # print(f"[DEBUG] targets keys: {list(targets.keys())}")
        
        for mod_key, mod_logits in logits.items():
            if mod_key not in targets:
                # print(f"[DEBUG] {mod_key} not in targets, skipping")
                continue
            
            mod_targets = targets[mod_key]
            
            # AION模型返回的logits是扁平化的 (num_selected, vocab_size)
            # 需要将targets也扁平化并筛选对应位置
            if masks is not None and mod_key in masks:
                mask = masks[mod_key]
                # 使用mask筛选targets
                selected_targets = mod_targets[mask]  # (num_selected,)
            else:
                selected_targets = mod_targets.flatten()
            
            # 确保targets是long类型（cross_entropy要求）
            selected_targets = selected_targets.long()
            
            # 确保logits和targets大小匹配
            if mod_logits.shape[0] != selected_targets.shape[0]:
                # 如果大小不匹配，可能是因为模型内部的mask处理不同
                # 尝试使用logits的数量来截断或扩展targets
                num_logits = mod_logits.shape[0]
                num_targets = selected_targets.shape[0]
                
                if num_logits == 0:
                    # 没有logits，跳过
                    continue
                
                if num_targets == 0:
                    # 没有targets，跳过
                    continue
                
                # 取较小的数量
                min_size = min(num_logits, num_targets)
                mod_logits = mod_logits[:min_size]
                selected_targets = selected_targets[:min_size]
            
            # 计算交叉熵损失
            ce = self.ce_loss(mod_logits, selected_targets)  # (num_selected,)
            
            # 过滤掉nan和inf值
            valid_mask = ~(torch.isnan(ce) | torch.isinf(ce))
            if valid_mask.sum() == 0:
                # 所有损失都是nan/inf，跳过这个模态
                continue
            
            ce_valid = ce[valid_mask]
            mod_loss = ce_valid.mean()
            
            # 再次检查结果是否为nan
            if torch.isnan(mod_loss) or torch.isinf(mod_loss):
                continue
            
            # 获取模态权重
            weight = self.modality_weights.get(mod_key, 1.0)
            total_loss += mod_loss * weight
            total_weight += weight
            
            loss_dict[f"loss_{mod_key}"] = mod_loss.item()
        
        # 归一化
        if total_weight > 0:
            total_loss = total_loss / total_weight
        
        loss_dict["total_loss"] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        
        # 确保返回的张量连接到计算图（即使没有有效的损失）
        if isinstance(total_loss, torch.Tensor):
            return total_loss, loss_dict
        else:
            # 创建一个连接到计算图的零损失
            # 使用logits的sum() * 0 来保持梯度流
            first_logits = next(iter(logits.values()))
            zero_loss = first_logits.sum() * 0.0
            return zero_loss, loss_dict


class AIONTrainer:
    """AION模型训练器
    
    实现AION论文中的训练方法：
    - 多模态掩码建模
    - 锚点选择与预算填充采样策略
    - 梯度累积
    - 混合精度训练
    - 学习率调度（Cosine Annealing / OneCycle）
    """
    
    def __init__(
        self,
        model: AION,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        loss_fn: MultimodalMaskedLoss,
        device: torch.device,
        dist_info: Dict[str, Any],
        grad_accum_steps: int = 1,
        use_amp: bool = True,
        max_grad_norm: float = 1.0,
        input_budget: int = 256,
        anchor_ratio_min: float = 0.3,
        anchor_ratio_max: float = 0.7,
        output_budget: int = 128,
        beta_alpha: float = 0.5,
        beta_beta: float = 2.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.dist_info = dist_info
        self.grad_accum_steps = grad_accum_steps
        self.use_amp = use_amp
        self.max_grad_norm = max_grad_norm
        self.input_budget = input_budget
        self.anchor_ratio_min = anchor_ratio_min
        self.anchor_ratio_max = anchor_ratio_max
        self.output_budget = output_budget
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        
        # 混合精度
        self.scaler = GradScaler("cuda") if use_amp else None
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
    
    def _create_input_mask_with_budget(
        self,
        tokens: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], str]:
        """创建基于预算的输入掩码
        
        采样策略：
        1. 设定全局输入预算 Budget（例如256个Token）
        2. 锚点选择：随机选择一个模态作为"锚点"
        3. 锚点采样：从锚点模态中均匀随机采样一定数量的Token
        4. 预算填充：剩余预算从其他模态均匀采样填满
        
        Args:
            tokens: 各模态的token字典 {modality_key: (B, N) or (B,)}
            
        Returns:
            input_masks: 输入掩码 {modality_key: (B, N) or (B,)}，True表示该token被选中作为输入
            anchor_modality: 锚点模态名称
        """
        if not tokens:
            return {}, ""
        
        mod_keys = list(tokens.keys())
        first_tensor = tokens[mod_keys[0]]
        B = first_tensor.shape[0]
        device = first_tensor.device
        
        # 初始化输入掩码（全False，表示都不选中）
        input_masks = {}
        for k, v in tokens.items():
            if v.dim() > 1:
                input_masks[k] = torch.zeros(B, v.shape[1], dtype=torch.bool, device=device)
            else:
                input_masks[k] = torch.zeros(B, dtype=torch.bool, device=device)
        anchor_modality = []
        # 为每个样本独立进行采样
        for b in range(B):
            valid_by_mod = {}
            for mod_key, v in tokens.items():
                if v.dim() > 1:
                    valid_idx = (v[b] != -1).nonzero(as_tuple=False).flatten().tolist()
                else:
                    valid_idx = [0] if v[b].item() != -1 else []
                if len(valid_idx) > 0:
                    valid_by_mod[mod_key] = valid_idx

            if not valid_by_mod:
                continue

            # Step 1: 随机选择锚点模态
            # sdss or desi 数据集不使用红移作为anchor
            if len(list(valid_by_mod.keys())) == 2:
                if valid_by_mod.get('tok_z') is not None:
                    del valid_by_mod['tok_z']
            anchor_mod = random.choice(list(valid_by_mod.keys()))
            anchor_valid_idx = valid_by_mod[anchor_mod]
            anchor_num_tokens = len(anchor_valid_idx)
            
            # 实际可用的预算不能超过总token数
            total_valid_tokens = sum(len(v) for v in valid_by_mod.values())
            actual_budget = min(self.input_budget, total_valid_tokens)
            
            # Step 2: 确定锚点采样数量（在anchor_ratio范围内随机）
            anchor_ratio = random.uniform(self.anchor_ratio_min, self.anchor_ratio_max)
            anchor_budget = min(
                int(actual_budget * anchor_ratio),
                anchor_num_tokens  # 不能超过锚点模态的总token数
            )
            anchor_budget = min(anchor_budget, self.input_budget)
            # Step 3: 从锚点模态均匀随机采样
            anchor_selected = 0
            if anchor_budget > 0:
                if anchor_num_tokens == 1:
                    # 1D模态（如flux），只有一个token
                    input_masks[anchor_mod][b] = True
                    anchor_selected = 1
                else:
                    sampled = random.sample(anchor_valid_idx, anchor_budget)
                    input_masks[anchor_mod][b, sampled] = True
                    anchor_selected = len(sampled)
            
            # Step 4: 计算剩余预算（必须填满到actual_budget）
            remaining_budget = actual_budget - anchor_selected
            
            # Step 5: 优先从其他模态均匀采样，如果不够再从锚点模态补充
            if remaining_budget > 0:
                # 5.1: 首先收集其他模态（非锚点）的所有token
                other_mods_pool = []  # [(mod_key, token_idx), ...]
                for mod_key, valid_idx in valid_by_mod.items():
                    if mod_key == anchor_mod:
                        continue  # 跳过锚点模态
                    for idx in valid_idx:
                        other_mods_pool.append((mod_key, idx))
                
                # 5.2: 从其他模态采样
                if len(other_mods_pool) > 0:
                    num_from_others = min(remaining_budget, len(other_mods_pool))
                    sampled_indices = random.sample(range(len(other_mods_pool)), num_from_others)
                    
                    for idx in sampled_indices:
                        mod_key, token_idx = other_mods_pool[idx]
                        if tokens[mod_key].dim() == 1:
                            input_masks[mod_key][b] = True
                        else:
                            input_masks[mod_key][b, token_idx] = True
                    
                    remaining_budget -= num_from_others
                
                # 5.3: 如果还有剩余预算，从锚点模态未被选中的token中补充
                if remaining_budget > 0:
                    anchor_remaining_pool = []
                    if tokens[anchor_mod].dim() == 1:
                        if not input_masks[anchor_mod][b].item():
                            anchor_remaining_pool.append(0)
                    else:
                        for idx in anchor_valid_idx:
                            if not input_masks[anchor_mod][b, idx].item():
                                anchor_remaining_pool.append(idx)
                    
                    if len(anchor_remaining_pool) > 0:
                        num_from_anchor = min(remaining_budget, len(anchor_remaining_pool))
                        if num_from_anchor > 0:
                            sampled_anchor = random.sample(anchor_remaining_pool, num_from_anchor)
                        
                        for token_idx in sampled_anchor:
                            if tokens[anchor_mod].dim() == 1:
                                input_masks[anchor_mod][b] = True
                            else:
                                input_masks[anchor_mod][b, token_idx] = True
        
            # 返回第一个样本的锚点模态（用于日志）
            anchor_modality.append(anchor_mod)
        
        return input_masks, anchor_modality
    
    def _create_target_mask(
        self,
        tokens: Dict[str, torch.Tensor],
        input_masks: Dict[str, torch.Tensor],
        anchor_modality: List[str],
    ) -> Dict[str, torch.Tensor]:
        """创建目标掩码（基于输出预算和主模态选择策略）
        
        输出Token策略：
        1. 设定全局输出预算（如128个token）
        2. 从候选模态中随机选择一个作为主模态
        3. 主模态的token数不超过总预算
        4. 剩余预算从其他模态通过Beta分布采样填充
        
        Args:
            tokens: 各模态的token字典 {modality_key: (B, N) or (B,)}
            input_masks: 输入掩码, True表示该token被选中作为输入
            anchor_modality: 每个样本的锚点模态名称列表
            
        Returns:
            target_masks: 目标掩码，True表示需要预测
        """
        if not tokens:
            return {}
        
        mod_keys = list(tokens.keys())
        first_tensor = tokens[mod_keys[0]]
        B = first_tensor.shape[0]
        device = first_tensor.device
        
        # 初始化目标掩码（全False）
        target_masks = {}
        for k, v in tokens.items():
            if v.dim() > 1:
                target_masks[k] = torch.zeros(B, v.shape[1], dtype=torch.bool, device=device)
            else:
                target_masks[k] = torch.zeros(B, dtype=torch.bool, device=device)
        
        candidate_cache = []
        candidate_counts = []
        for b in range(B):
            candidate_by_mod = {}
            for mod_key in mod_keys:
                v = tokens[mod_key]
                if v.dim() == 1:
                    is_valid = v[b].item() != -1
                    is_input = (
                        input_masks[mod_key][b].item() if mod_key in input_masks else False
                    )
                    if is_valid and (not is_input):
                        candidate_by_mod[mod_key] = [0]
                else:
                    valid_mask = v[b] != -1
                    if mod_key in input_masks:
                        candidate_mask = valid_mask & (~input_masks[mod_key][b])
                    else:
                        candidate_mask = valid_mask
                    idx = candidate_mask.nonzero(as_tuple=False).flatten().tolist()
                    if len(idx) > 0:
                        candidate_by_mod[mod_key] = idx

            available_mods = list(candidate_by_mod.keys())
            total_candidates = sum(len(v) for v in candidate_by_mod.values())
            candidate_cache.append((candidate_by_mod, available_mods, total_candidates))
            candidate_counts.append(total_candidates)

        min_candidates = min(candidate_counts) if len(candidate_counts) > 0 else 0
        if min_candidates > 0:
            batch_budget = min(min_candidates, max(1, self.output_budget))
        else:
            batch_budget = 0

        for b in range(B):
            if batch_budget == 0:
                continue

            candidate_by_mod, available_mods, total_candidates = candidate_cache[b]
            if total_candidates < batch_budget:
                continue

            actual_budget = batch_budget
            
            # Step 1: 随机选择一个模态作为主模态
            # 优先选择红移模态作为主模态
            if 'tok_z' in available_mods:
                primary_mod = 'tok_z'
            else:
                primary_mod = random.choice(available_mods)
            primary_candidates = candidate_by_mod[primary_mod]

            # Step 2: 使用Beta分布确定主模态的采样比例
            # Beta(alpha, beta) 控制主模态占总预算的比例分布
            beta_sample = np.random.beta(self.beta_alpha, self.beta_beta)
            # 主模态至少占50%，最多占100%
            primary_ratio = 0.5 + beta_sample * 0.7
            primary_ratio = min(primary_ratio, 1)
            
            # 主模态的token数：根据比例计算，但不超过实际可用数量
            primary_budget = min(
                int(actual_budget * primary_ratio),
                len(primary_candidates)
            )
            # 确保至少采样1个
            primary_budget = max(1, primary_budget) if len(primary_candidates) > 0 else 0
            
            # 从主模态中随机采样
            primary_selected = 0
            if primary_budget > 0:
                sampled_primary = random.sample(primary_candidates, primary_budget)
                for token_idx in sampled_primary:
                    if tokens[primary_mod].dim() == 1:
                        target_masks[primary_mod][b] = True
                    else:
                        target_masks[primary_mod][b, token_idx] = True
                primary_selected = len(sampled_primary)
            
            # Step 3: 计算剩余预算（必须填满到actual_budget）
            remaining_budget = actual_budget - primary_selected
            
            # Step 4: 从剩余模态填满剩余预算
            if remaining_budget > 0:
                # 收集所有其他模态的候选token（不包括主模态中已选的）
                other_pool = []  # [(mod_key, token_idx), ...]
                for mod_key in available_mods:
                    for token_idx in candidate_by_mod[mod_key]:
                        # 跳过主模态中已被选中的token
                        if mod_key == primary_mod:
                            if tokens[mod_key].dim() == 1:
                                if target_masks[mod_key][b].item():
                                    continue
                            else:
                                if target_masks[mod_key][b, token_idx].item():
                                    continue

                        # 如果mod_key为标量，则将其添加到target_masks中
                        if tokens[mod_key].dim() == 1 and remaining_budget > 0:
                            if not target_masks[mod_key][b].item():
                                target_masks[mod_key][b] = True
                                remaining_budget -= 1
                        else:
                            other_pool.append((mod_key, token_idx))
                
                
                if len(other_pool) and remaining_budget > 0:
                    # 填满剩余预算
                    fill_count = min(remaining_budget, len(other_pool))
                    
                    # 从其他模态池中随机采样
                    if fill_count > 0:
                        sampled_indices = random.sample(range(len(other_pool)), fill_count)
                        for idx in sampled_indices:
                            mod_key, token_idx = other_pool[idx]
                            if tokens[mod_key].dim() == 1:
                                target_masks[mod_key][b] = True
                            else:
                                target_masks[mod_key][b, token_idx] = True
        
        return target_masks
    
    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict, Dict, Dict, Dict]:
        """准备训练批次 - 直接使用token数据
        
        采用锚点选择与预算填充策略：
        1. 直接使用已加载的token
        2. 使用预算策略选择输入token（锚点+填充）
        3. 剩余token作为预测目标
        
        返回：
            input_tokens: 输入token字典
            target_tokens: 目标token（用于计算损失）
            input_masks: 输入掩码（True表示被选为输入）
            target_masks: 目标掩码（True表示需要预测）
        """
        # 将数据移到设备
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # 直接使用token数据，无需编码
        input_tokens = {}
        
        # 所有token键的映射
        if "tok_image" in batch:
            input_tokens["tok_image"] = batch["tok_image"]
        if "tok_image_hsc" in batch:
            input_tokens["tok_image_hsc"] = batch["tok_image_hsc"]
        if "tok_spectrum_desi" in batch:
            input_tokens["tok_spectrum_desi"] = batch["tok_spectrum_desi"]
        if "tok_spectrum_sdss" in batch:
            input_tokens["tok_spectrum_sdss"] = batch["tok_spectrum_sdss"]
        if "tok_z" in batch:
            input_tokens["tok_z"] = batch["tok_z"]
        if "tok_flux_g" in batch:
            input_tokens["tok_flux_g"] = batch["tok_flux_g"]
        if "tok_flux_r" in batch:
            input_tokens["tok_flux_r"] = batch["tok_flux_r"]
        if "tok_flux_z" in batch:
            input_tokens["tok_flux_z"] = batch["tok_flux_z"]
        if "tok_flux_i" in batch:
            input_tokens["tok_flux_i"] = batch["tok_flux_i"]
        if "tok_mag_g" in batch:
            input_tokens["tok_mag_g"] = batch["tok_mag_g"]
        if "tok_mag_r" in batch:
            input_tokens["tok_mag_r"] = batch["tok_mag_r"]
        if "tok_mag_z" in batch:
            input_tokens["tok_mag_z"] = batch["tok_mag_z"]
        if "tok_mag_i" in batch:
            input_tokens["tok_mag_i"] = batch["tok_mag_i"]
        if "tok_mag_y" in batch:
            input_tokens["tok_mag_y"] = batch["tok_mag_y"]
        if "tok_flux_w1" in batch:
            input_tokens["tok_flux_w1"] = batch["tok_flux_w1"]
        if "tok_flux_w2" in batch:
            input_tokens["tok_flux_w2"] = batch["tok_flux_w2"]
        if "tok_flux_w3" in batch:
            input_tokens["tok_flux_w3"] = batch["tok_flux_w3"]
        if "tok_flux_w4" in batch:
            input_tokens["tok_flux_w4"] = batch["tok_flux_w4"]
        if "tok_shape_r" in batch:
            input_tokens["tok_shape_r"] = batch["tok_shape_r"]
        if "tok_shape_e1" in batch:
            input_tokens["tok_shape_e1"] = batch["tok_shape_e1"]
        if "tok_shape_e2" in batch:
            input_tokens["tok_shape_e2"] = batch["tok_shape_e2"]
        if "tok_ebv" in batch:
            input_tokens["tok_ebv"] = batch["tok_ebv"]
        if "tok_a_g" in batch:
            input_tokens["tok_a_g"] = batch["tok_a_g"]
        if "tok_a_r" in batch:
            input_tokens["tok_a_r"] = batch["tok_a_r"]
        if "tok_a_z" in batch:
            input_tokens["tok_a_z"] = batch["tok_a_z"]
        if "tok_a_i" in batch:
            input_tokens["tok_a_i"] = batch["tok_a_i"]
        if "tok_a_y" in batch:
            input_tokens["tok_a_y"] = batch["tok_a_y"]
        if "tok_shape11" in batch:
            input_tokens["tok_shape11"] = batch["tok_shape11"]
        if "tok_shape22" in batch:
            input_tokens["tok_shape22"] = batch["tok_shape22"]
        if "tok_shape12" in batch:
            input_tokens["tok_shape12"] = batch["tok_shape12"]
        
        if not input_tokens:
            return {}, {}, {}, {}
        
        # 使用预算策略创建输入掩码（锚点选择 + 预算填充）
        input_masks, anchor_mod = self._create_input_mask_with_budget(input_tokens)
        
        # 创建目标掩码（输入掩码的补集）
        target_masks = self._create_target_mask(input_tokens, input_masks, anchor_mod)
        
        # 目标token就是原始token
        target_tokens = {k: v.clone() for k, v in input_tokens.items()}
        
        return input_tokens, target_tokens, input_masks, target_masks
    
    def train_step(self, batch: Dict[str, torch.Tensor], step_in_accum: int = 0) -> Dict[str, float]:
        """单步训练
        
        Args:
            batch: 输入批次
            step_in_accum: 梯度累积中的步骤索引
        
        Returns:
            metrics: 训练指标
        """
        self.model.train()
        
        # 准备数据（使用预算策略）
        input_tokens, target_tokens, input_masks, target_masks = self._prepare_batch(batch)
        
        # 检查batch是否有效 - 需要所有rank同步决定是否跳过
        batch_valid = len(input_tokens) > 0
        if self.dist_info["is_distributed"]:
            # 同步所有rank的batch有效性状态
            valid_tensor = torch.tensor([1 if batch_valid else 0], device=self.device, dtype=torch.int32)
            dist.all_reduce(valid_tensor, op=dist.ReduceOp.MIN)
            batch_valid = valid_tensor.item() > 0
        
        if not batch_valid:
            # 所有rank一起跳过这个batch
            return {
                "total_loss": 0.0,
                "learning_rate": self.scheduler.get_last_lr()[0],
                "skipped": True,
                "optimizer_stepped": False,
                "measured_samples": 0.0,
                "num_encoder_tokens": 0.0,
                "num_decoder_tokens": 0.0,
            }
        
        # 确定是否需要同步梯度
        is_accumulating = (step_in_accum + 1) < self.grad_accum_steps
        
        # DDP 梯度同步控制
        sync_ctx = nullcontext()
        if self.dist_info["is_distributed"] and isinstance(self.model, DDP) and is_accumulating:
            sync_ctx = self.model.no_sync()
        
        with sync_ctx:
            # 混合精度前向
            with autocast("cuda", enabled=self.use_amp):
                # 构建掩码后的输入：只保留被选中的token（input_masks为True的位置）
                # 对于未被选中的位置，设为特殊token（0）
                masked_input = {}
                for mod_key, tok in input_tokens.items():
                    masked_tok = tok.clone()
                    if mod_key in input_masks:
                        # 未被选中的token设为0（掩码）
                        masked_tok[~input_masks[mod_key]] = 0
                    masked_input[mod_key] = masked_tok
                
                # 计算encoder和decoder token数量
                # num_encoder_tokens: 被选中作为输入的token数量（input_masks中True的数量）
                # num_decoder_tokens: 需要预测的token数量（target_masks中True的数量）
                num_encoder_tokens = 0
                num_decoder_tokens = 0
                for mod_key, tok in input_tokens.items():
                    B = tok.shape[0]
                    # encoder tokens: input_masks为True表示被选中作为输入
                    if mod_key in input_masks:
                        mask = input_masks[mod_key]
                        num_encoder_tokens += mask.sum().item()   # 平均每个样本
                    else:
                        # 如果没有mask，所有token都作为输入
                        num_tokens = tok.shape[1] if tok.dim() == 2 else 1
                        num_encoder_tokens += num_tokens
                    # decoder tokens: target_masks为True表示需要预测
                    if mod_key in target_masks:
                        mask = target_masks[mod_key]
                        num_decoder_tokens += mask.sum().item()   # 平均每个样本
                
                num_encoder_tokens = max(1, int(num_encoder_tokens//B))
                num_decoder_tokens = max(1, int(num_decoder_tokens//B))
                
                # 准备input_mask和target_mask（确保都是2D）
                # 模型期望所有mask都是(B, N)形状
                input_mask_2d = {}
                for k, v in input_masks.items():
                    mask = ~v  # 反转：True表示被掩码
                    if mask.dim() == 1:
                        mask = mask.unsqueeze(1)  # (B,) -> (B, 1)
                    input_mask_2d[k] = mask
                
                target_mask_2d = {}
                for k, v in target_masks.items():
                    mask = ~v
                    if mask.dim() == 1:
                        mask = mask.unsqueeze(1)  # (B,) -> (B, 1)
                    target_mask_2d[k] = mask
                
                # 前向传播 - 使用_forward方法
                # 获取底层模型（处理DDP包装）
                base_model = self.model.module if isinstance(self.model, DDP) else self.model
                logits = base_model._forward(
                    input_dict=masked_input,
                    target_mask=target_mask_2d,
                    input_mask=input_mask_2d,
                    num_decoder_tokens=num_decoder_tokens,
                    num_encoder_tokens=num_encoder_tokens,
                )
                
                # 计算损失（只在目标位置计算）
                loss, loss_dict = self.loss_fn(logits, target_tokens, target_masks)
                loss = loss / self.grad_accum_steps
            
            # 检查损失是否为nan/inf - 需要所有rank同步决定
            loss_valid = not (torch.isnan(loss) or torch.isinf(loss))
            if self.dist_info["is_distributed"]:
                valid_tensor = torch.tensor([1 if loss_valid else 0], device=self.device, dtype=torch.int32)
                dist.all_reduce(valid_tensor, op=dist.ReduceOp.MIN)
                loss_valid = valid_tensor.item() > 0
            
            if not loss_valid:
                # 所有rank一起跳过反向传播，但仍需执行参数更新步骤以保持同步
                loss_dict["total_loss"] = float('nan')
                # 创建一个零梯度以保持DDP同步
                dummy_loss = sum(p.sum() * 0.0 for p in self.model.parameters() if p.requires_grad)
                if self.scaler:
                    self.scaler.scale(dummy_loss).backward()
                else:
                    dummy_loss.backward()
            else:
                # 正常反向传播
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
        
        # 在梯度累积完成后更新参数
        if not is_accumulating:
            optimizer_stepped = False
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            
            # 梯度裁剪
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.max_grad_norm
                )
            
            if self.scaler:
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                optimizer_stepped = self.scaler.get_scale() >= scale_before
            else:
                self.optimizer.step()
                optimizer_stepped = True
            
            if optimizer_stepped:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            if optimizer_stepped:
                self.global_step += 1
        
        loss_dict["skipped"] = False
        loss_dict["optimizer_stepped"] = optimizer_stepped if not is_accumulating else False
        loss_dict["measured_samples"] = float(0.0 if loss_dict["skipped"] else 1.0)
        loss_dict["num_encoder_tokens"] = float(num_encoder_tokens)
        loss_dict["num_decoder_tokens"] = float(num_decoder_tokens)
        loss_dict["learning_rate"] = self.scheduler.get_last_lr()[0]
        return loss_dict
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证
        
        Args:
            val_loader: 验证数据加载器
        
        Returns:
            metrics: 验证指标
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        modality_losses = {}
        is_main = self.dist_info.get("is_main_process", False)
        data_iter = tqdm(
            val_loader,
            desc="Validation",
            disable=not is_main,
        )
        
        for batch in data_iter:
            input_tokens, target_tokens, input_masks, target_masks = self._prepare_batch(batch)
            
            if not input_tokens:
                continue
            
            with autocast("cuda", enabled=self.use_amp):
                # 构建掩码后的输入
                masked_input = {}
                for mod_key, tok in input_tokens.items():
                    masked_tok = tok.clone()
                    if mod_key in input_masks:
                        masked_tok[~input_masks[mod_key]] = 0
                    masked_input[mod_key] = masked_tok
                
                # 计算encoder和decoder token数量
                # num_encoder_tokens: 被选中作为输入的token数量（input_masks中True的数量）
                # num_decoder_tokens: 需要预测的token数量（target_masks中True的数量）
                num_encoder_tokens = 0
                num_decoder_tokens = 0
                for mod_key, tok in input_tokens.items():
                    B = tok.shape[0]
                    # encoder tokens: input_masks为True表示被选中作为输入
                    if mod_key in input_masks:
                        mask = input_masks[mod_key]
                        num_encoder_tokens += mask.sum().item()   # 平均每个样本
                    else:
                        # 如果没有mask，所有token都作为输入
                        num_tokens = tok.shape[1] if tok.dim() == 2 else 1
                        num_encoder_tokens += num_tokens
                    # decoder tokens: target_masks为True表示需要预测
                    if mod_key in target_masks:
                        mask = target_masks[mod_key]
                        num_decoder_tokens += mask.sum().item()   # 平均每个样本
                
                num_encoder_tokens = max(1, int(num_encoder_tokens//B))
                num_decoder_tokens = max(1, int(num_decoder_tokens//B))
                
                # 准备input_mask和target_mask（确保都是2D）
                input_mask_2d = {}
                for k, v in input_masks.items():
                    mask = ~v  # 反转：True表示被掩码
                    if mask.dim() == 1:
                        mask = mask.unsqueeze(1)
                    input_mask_2d[k] = mask
                
                target_mask_2d = {}
                for k, v in target_masks.items():
                    mask = ~v
                    if mask.dim() == 1:
                        mask = mask.unsqueeze(1)
                    target_mask_2d[k] = mask
                
                # 获取底层模型（处理DDP包装）
                base_model = self.model.module if isinstance(self.model, DDP) else self.model
                logits = base_model._forward(
                    input_dict=masked_input,
                    target_mask=target_mask_2d,
                    input_mask=input_mask_2d,
                    num_decoder_tokens=num_decoder_tokens,
                    num_encoder_tokens=num_encoder_tokens,
                )
                
                loss, loss_dict = self.loss_fn(logits, target_tokens, target_masks)
            
            batch_loss = loss.item()
            total_loss += batch_loss
            num_batches += 1
            
            for k, v in loss_dict.items():
                if k != "total_loss":
                    modality_losses[k] = modality_losses.get(k, 0) + v
            if is_main:
                avg_loss = total_loss / max(1, num_batches)
                data_iter.set_postfix({"val_loss": f"{avg_loss:.4f}"})
        
        local_num_batches = num_batches
        if self.dist_info["is_distributed"] and dist.is_initialized():
            device = self.device
            loss_tensor = torch.tensor(total_loss, device=device, dtype=torch.float64)
            count_tensor = torch.tensor(num_batches, device=device, dtype=torch.long)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            total_loss = loss_tensor.item()
            num_batches = int(count_tensor.item())
        
        if num_batches == 0:
            return {"val_loss": 0.0}
        
        metrics = {
            "val_loss": total_loss / max(1, num_batches),
        }
        
        for k, v in modality_losses.items():
            metrics[f"val_{k}"] = v / max(1, local_num_batches)
        
        return metrics
    
    def save_checkpoint(
        self, 
        save_path: str, 
        is_best: bool = False,
        extra_state: Optional[Dict] = None,
    ):
        """保存检查点"""
        base_model = self.model.module if isinstance(self.model, DDP) else self.model
        
        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": base_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        
        if self.scaler:
            state["scaler_state_dict"] = self.scaler.state_dict()
        
        if extra_state:
            state.update(extra_state)
        
        torch.save(state, save_path)
        
        if is_best:
            best_path = os.path.join(os.path.dirname(save_path), "best_model.pt")
            torch.save(state, best_path)
    
    def load_checkpoint(self, load_path: str):
        """加载检查点"""
        state = torch.load(load_path, map_location=self.device)
        
        base_model = self.model.module if isinstance(self.model, DDP) else self.model
        base_model.load_state_dict(state["model_state_dict"])
        
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.epoch = state["epoch"] + 1
        self.global_step = state["global_step"]
        self.best_val_loss = state.get("best_val_loss", float('inf'))
        
        if self.scaler and "scaler_state_dict" in state:
            self.scaler.load_state_dict(state["scaler_state_dict"])


# ===================== 主训练函数 =====================

def train(args):
    """主训练函数"""
    
    # 1. 初始化分布式环境
    dist_info = setup_distributed()
    is_main = dist_info["is_main_process"]
    device = torch.device("cuda", dist_info["local_rank"]) if torch.cuda.is_available() else torch.device("cpu")
    
    if is_main:
        print("=" * 80)
        print("AION Distributed Training")
        print("=" * 80)
        print(f"World Size: {dist_info['world_size']}")
        print(f"Rank: {dist_info['rank']}")
        print(f"Local Rank: {dist_info['local_rank']}")
        print(f"Device: {device}")
        print("=" * 80)
    
    # 设置随机种子
    set_seed(args.seed, dist_info["rank"])
    
    # 2. 创建输出目录
    if is_main:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.output_dir, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        
        # 保存配置
        config_data = vars(args).copy()  # 创建参数的副本
        config_data["script_name"] = os.path.basename(__file__)  # 添加当前文件名

        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config_data, f, indent=2)
        
        print(f"Output directory: {output_dir}")
    else:
        output_dir = None
    
    # 广播输出目录路径
    if dist_info["is_distributed"]:
        output_dir_list = [output_dir]
        dist.broadcast_object_list(output_dir_list, src=0)
        output_dir = output_dir_list[0]
    
    # 3. 加载模型
    if is_main:
        print(f"\nLoading model configuration (training from scratch): {args.model_name}")
    
    # 加载预训练模型结构，然后重新初始化权重
    model = AION.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
    )
    
    # 重新初始化所有权重（从零开始训练）
    if is_main:
        print("Reinitializing model weights...")
    
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    model = model.to(device)
    
    # DDP 包装
    if dist_info["is_distributed"]:
        model = DDP(
            model,
            device_ids=[dist_info["local_rank"]],
            output_device=dist_info["local_rank"],
            find_unused_parameters=True,  # AION可能有未使用参数
        )
    
    # 统计参数
    if is_main:
        base_model = model.module if isinstance(model, DDP) else model
        total_params, trainable_params = count_parameters(base_model)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    # 注意：使用预编码的token数据，不需要codec_manager
    
    # 4. 创建数据加载器
    if is_main:
        print(f"\nLoading token dataset: {args.data_path}")
    
    train_dataset = AIONDataset(
        data_path=args.data_path,
        max_samples=args.max_samples,
        world_size=dist_info["world_size"],
        rank=dist_info["rank"],
    )
    
    if dist_info["is_distributed"]:
        train_batch_sampler = GroupedDistributedBatchSampler(
            train_dataset,
            batch_size=args.batch_size,
            num_replicas=dist_info["world_size"],
            rank=dist_info["rank"],
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
    else:
        train_batch_sampler = GroupedDistributedBatchSampler(
            train_dataset,
            batch_size=args.batch_size,
            num_replicas=1,
            rank=0,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # 验证集（可选）- 使用相同数据集的一部分
    val_loader = None
    if args.val_split > 0:
        val_size = int(len(train_dataset) * args.val_split)
        if val_size > 0:
            num_full = len(train_dataset)
            if dist_info["is_distributed"]:
                if dist_info["rank"] == 0:
                    indices = list(range(num_full))
                    random.shuffle(indices)
                else:
                    indices = None
                obj_list = [indices]
                dist.broadcast_object_list(obj_list, src=0)
                indices = obj_list[0]
            else:

                indices = list(range(num_full))
                random.shuffle(indices)
            val_indices = indices[-val_size:]
            val_dataset = torch.utils.data.Subset(train_dataset, val_indices)
            if dist_info["is_distributed"]:
                val_sampler = DistributedSampler(
                    val_dataset,
                    num_replicas=dist_info["world_size"],
                    rank=dist_info["rank"],
                    shuffle=False,
                    drop_last=False,
                )
            else:
                val_sampler = None
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=val_sampler is None,
                sampler=val_sampler,
                num_workers=args.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
            )
    
    # 5. 优化器和调度器
    # 参考论文：使用 AdamW，学习率 1e-4，权重衰减 0.1
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),  # 论文中的设置
    )
    
    # 学习率调度：Cosine Annealing
    # 计算总步数：优先使用命令行参数，否则根据数据集大小计算
    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch else len(train_loader) // args.grad_accum_steps
    if args.max_steps:
        total_steps = args.max_steps
    else:
        total_steps = args.num_epochs * steps_per_epoch
    
    # 确保total_steps至少为1
    total_steps = max(1, total_steps)
    
    if is_main:
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total training steps: {total_steps}")
    
    warmup_steps = 0
    if args.warmup_steps is not None:
        warmup_steps = args.warmup_steps
    elif args.warmup_ratio and args.warmup_ratio > 0:
        warmup_steps = int(total_steps * args.warmup_ratio)
    
    warmup_steps = int(max(0, warmup_steps))
    warmup_steps = min(warmup_steps, max(0, total_steps - 1))
    
    if is_main:
        print(f"Warmup steps: {warmup_steps}")
    
    if warmup_steps > 0:
        eta_min_factor = 0.01
        min_lr_scale = eta_min_factor
        warmup_start_factor = args.warmup_start_factor
        cosine_steps = max(1, total_steps - warmup_steps)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                warmup_progress = current_step / max(1, warmup_steps)
                return warmup_start_factor + (1.0 - warmup_start_factor) * warmup_progress

            cosine_step = min(current_step - warmup_steps, cosine_steps)
            cosine_progress = cosine_step / cosine_steps
            cosine_scale = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
            return min_lr_scale + (1.0 - min_lr_scale) * cosine_scale

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=args.lr*0.01,
        )
    
    # 6. 损失函数
    loss_fn = MultimodalMaskedLoss(
        label_smoothing=args.label_smoothing,
        modality_weights={
            "tok_spectrum_sdss": args.spectrum_weight,
            "tok_spectrum_desi": args.spectrum_weight,
            "tok_image_hsc": args.image_weight,
            "tok_image": args.image_weight,
            "tok_a_g": args.flux_weight,
            "tok_a_r": args.flux_weight,
            "tok_a_i": args.flux_weight,
            "tok_a_z": args.flux_weight,
            "tok_a_y": args.flux_weight,
            "tok_mag_g": args.flux_weight,
            "tok_mag_r": args.flux_weight,
            "tok_mag_i": args.flux_weight,
            "tok_mag_z": args.flux_weight,
            "tok_mag_y": args.flux_weight,
            "tok_shape11": args.flux_weight,
            "tok_shape22": args.flux_weight,
            "tok_shape12": args.flux_weight,
            "tok_z": args.flux_weight,
            "tok_flux_w1": args.flux_weight,
            "tok_flux_w2": args.flux_weight,
            "tok_flux_w3": args.flux_weight,
            "tok_flux_w4": args.flux_weight,
            "tok_shape_r": args.flux_weight,
            "tok_shape_e1": args.flux_weight,
            "tok_shape_e2": args.flux_weight,
            "tok_ebv": args.flux_weight,
            "tok_flux_g": args.flux_weight,
            "tok_flux_r": args.flux_weight,
            "tok_flux_i": args.flux_weight,
            "tok_flux_z": args.flux_weight,
        },
    )
    
    # 7. 创建训练器
    trainer = AIONTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        dist_info=dist_info,
        grad_accum_steps=args.grad_accum_steps,
        use_amp=args.use_amp,
        max_grad_norm=args.max_grad_norm,
        input_budget=args.input_budget,
        anchor_ratio_min=args.anchor_ratio_min,
        anchor_ratio_max=args.anchor_ratio_max,
        output_budget=args.output_budget,
        beta_alpha=args.beta_alpha,
        beta_beta=args.beta_beta,
    )
    
    # 恢复训练
    if args.resume:
        if is_main:
            print(f"\nResuming training from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # 8. 基准测试模式
    if args.benchmark_only:
        return run_benchmark(args, trainer, train_loader, output_dir, dist_info)

    # 9. TensorBoard / WandB
    writer = None
    wandb_run = None
    
    if is_main:
        if args.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter(os.path.join(output_dir, "logs"))
            except Exception as e:
                print(f"TensorBoard initialization failed: {e}")
                writer = None
        
        if args.use_wandb:
            try:
                import wandb
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=args.wandb_run_name or timestamp,
                    config=vars(args),
                )
            except Exception as e:
                print(f"WandB initialization failed: {e}")
    
    # 10. 训练循环
    if is_main:
        print(f"\nStarting training...")
        print(f"Epochs: {args.num_epochs}")
        print(f"Batch Size (per GPU): {args.batch_size}")
        print(f"Total Batch Size: {args.batch_size * dist_info['world_size']}")
        print(f"Gradient accumulation steps: {args.grad_accum_steps}")
        print(f"Learning rate: {args.lr}")
        print(f"Input token budget: {args.input_budget}")
        print(f"Anchor sampling ratio: [{args.anchor_ratio_min}, {args.anchor_ratio_max}]")
        print(f"Output token budget: {args.output_budget}")
        print(f"Beta distribution parameters: alpha={args.beta_alpha}, beta={args.beta_beta}")
    
    for epoch in range(trainer.epoch, args.num_epochs):
        trainer.epoch = epoch
        
        if hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        elif hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        
        # 训练一个epoch
        epoch_loss = 0.0
        num_steps = 0
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            disable=not is_main,
        )
        
        for step, batch in enumerate(progress_bar):
            step_in_accum = step % args.grad_accum_steps
            metrics = trainer.train_step(batch, step_in_accum)
            
            current_loss = metrics.get("total_loss", 0.0)
            
            # 跳过nan损失
            if not np.isnan(current_loss) and not np.isinf(current_loss):
                epoch_loss += current_loss
                num_steps += 1
            
            # 每步都更新进度条（便于调试）
            if num_steps > 0:
                avg_loss = epoch_loss / num_steps
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "cur_loss": f"{current_loss:.4f}",
                    "lr": f"{metrics['learning_rate']:.2e}",
                })
            
            # 日志记录 - 每个梯度更新步骤都记录
            is_grad_step = (step + 1) % args.grad_accum_steps == 0
            if is_grad_step and is_main:
                avg_loss = epoch_loss / max(1, num_steps)
                
                if writer:
                    # 记录平均损失和即时损失
                    writer.add_scalar("train/loss_avg", avg_loss, trainer.global_step)
                    writer.add_scalar("train/loss_instant", current_loss, trainer.global_step)
                    writer.add_scalar("train/lr", metrics["learning_rate"], trainer.global_step)
                    
                    # 记录各模态损失
                    for key, value in metrics.items():
                        if key.startswith("loss_") and key != "total_loss":
                            writer.add_scalar(f"train/{key}", value, trainer.global_step)
                
                if wandb_run:
                    log_dict = {
                        "train/loss_avg": avg_loss,
                        "train/loss_instant": current_loss,
                        "train/lr": metrics["learning_rate"],
                        "train/epoch": epoch + 1,
                        "train/step": trainer.global_step,
                    }
                    # 添加各模态损失
                    for key, value in metrics.items():
                        if key.startswith("loss_") and key != "total_loss":
                            log_dict[f"train/{key}"] = value
                    wandb_run.log(log_dict)
            
            # 检查最大步数 - 需要所有rank同步决定是否退出
            should_break = args.max_steps and trainer.global_step >= args.max_steps
            if dist_info["is_distributed"]:
                # 同步所有rank的退出决定
                break_tensor = torch.tensor([1 if should_break else 0], device=device, dtype=torch.int32)
                dist.all_reduce(break_tensor, op=dist.ReduceOp.MAX)
                should_break = break_tensor.item() > 0
            
            if should_break:
                break
        
        # Epoch 结束
        if dist_info["is_distributed"]:
            dist.barrier()
        
        # 验证
        if val_loader and (epoch + 1) % args.eval_interval == 0:
            if is_main:
                print(f"\nRunning validation...")
            
            val_metrics = trainer.validate(val_loader)
            
            if is_main:
                print(f"Validation loss: {val_metrics['val_loss']:.4f}")
                
                if writer:
                    for k, v in val_metrics.items():
                        writer.add_scalar(f"val/{k}", v, trainer.global_step)
                
                if wandb_run:
                    wandb_run.log({f"val/{k}": v for k, v in val_metrics.items()})
                
                # 保存最佳模型
                if val_metrics["val_loss"] < trainer.best_val_loss:
                    trainer.best_val_loss = val_metrics["val_loss"]
                    trainer.save_checkpoint(
                        os.path.join(output_dir, "checkpoints", f"checkpoint_epoch_{epoch + 1}.pt"),
                        is_best=True,
                    )
        
        # 保存检查点
        if is_main and (epoch + 1) % args.save_interval == 0:
            trainer.save_checkpoint(
                os.path.join(output_dir, "checkpoints", f"checkpoint_epoch_{epoch + 1}.pt"),
            )
            print(f"Checkpoint saved: checkpoint_epoch_{epoch + 1}.pt")
        
        if dist_info["is_distributed"]:
            dist.barrier()
    
    # 11. 训练结束
    if is_main:
        # 保存最终模型
        trainer.save_checkpoint(
            os.path.join(output_dir, "checkpoints", "final_model.pt"),
        )
        
        print("\n" + "=" * 80)
        print("Training complete!")
        print(f"Best validation loss: {trainer.best_val_loss:.4f}")
        print(f"Total training steps: {trainer.global_step}")
        print(f"Output directory: {output_dir}")
        print("=" * 80)
        
        if writer:
            writer.close()
        
        if wandb_run:
            wandb_run.finish()
    
    cleanup_distributed()
    
    return model, trainer


def run_benchmark(args, trainer, train_loader, output_dir: str, dist_info: Dict[str, Any]):
    """运行仅基准测试模式。"""
    is_main = dist_info["is_main_process"]
    warmup_steps = max(0, int(args.benchmark_warmup_steps))
    measure_steps = max(0, int(args.benchmark_measure_steps))
    total_steps = warmup_steps + measure_steps

    if total_steps <= 0:
        raise ValueError("benchmark_warmup_steps + benchmark_measure_steps must be greater than 0")

    if hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "set_epoch"):
        train_loader.batch_sampler.set_epoch(0)
    elif hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
        train_loader.sampler.set_epoch(0)

    if is_main:
        print(f"\nStarting benchmark...")
        print(f"Warmup Steps: {warmup_steps}")
        print(f"Measure Steps: {measure_steps}")
        print(f"Batch Size (per GPU): {args.batch_size}")
        print(f"Global Batch Size: {args.batch_size * dist_info['world_size']}")
        print(f"Gradient accumulation steps: {args.grad_accum_steps}")
        print(f"Input token budget: {args.input_budget}")
        print(f"Output token budget: {args.output_budget}")

    trainer.optimizer.zero_grad(set_to_none=True)

    if dist_info["is_distributed"]:
        dist.barrier()

    data_iter = iter(train_loader)
    measure_elapsed = 0.0
    measured_loss_sum = 0.0
    measured_loss_count = 0
    measured_encoder_tokens = 0.0
    measured_decoder_tokens = 0.0
    measured_samples = 0.0
    measured_optimizer_steps = 0.0
    micro_steps = 0
    pending_progress_updates = 0
    progress_bar = tqdm(
        total=total_steps,
        desc="Benchmark",
        disable=not is_main,
    )

    while micro_steps < total_steps:
        batch, data_iter = _next_batch(data_iter, train_loader)
        step_in_accum = micro_steps % args.grad_accum_steps
        in_measure_window = micro_steps >= warmup_steps

        if in_measure_window:
            synchronize_device()
            step_start = time.perf_counter()

        metrics = trainer.train_step(batch, step_in_accum)

        if in_measure_window:
            synchronize_device()
            measure_elapsed += time.perf_counter() - step_start

            current_loss = metrics.get("total_loss", 0.0)
            if not np.isnan(current_loss) and not np.isinf(current_loss):
                measured_loss_sum += current_loss
                measured_loss_count += 1
            # Track local-rank contributions only; distributed reduction below
            # aggregates them into true global totals exactly once.
            measured_samples += metrics.get("measured_samples", 0.0) * args.batch_size
            measured_encoder_tokens += metrics.get("num_encoder_tokens", 0.0) * args.batch_size
            measured_decoder_tokens += metrics.get("num_decoder_tokens", 0.0) * args.batch_size
            # Optimizer steps are globally synchronized, so count them once via MAX
            # instead of summing one identical step per rank.
            measured_optimizer_steps += 1.0 if metrics.get("optimizer_stepped", False) else 0.0

        micro_steps += 1
        pending_progress_updates += 1
        should_refresh_progress = pending_progress_updates >= 10 or micro_steps == total_steps

        if should_refresh_progress:
            progress_bar.update(pending_progress_updates)
            pending_progress_updates = 0

        if should_refresh_progress and in_measure_window and is_main:
            avg_loss = measured_loss_sum / max(1, measured_loss_count)
            progress_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "lr": f"{metrics.get('learning_rate', 0.0):.2e}",
            })

    progress_bar.close()

    trainer.optimizer.zero_grad(set_to_none=True)

    if dist_info["is_distributed"]:
        dist.barrier()

    elapsed_seconds = _distributed_reduce_scalar(measure_elapsed, op=dist.ReduceOp.MAX)
    global_loss_sum = _distributed_reduce_scalar(measured_loss_sum, op=dist.ReduceOp.SUM)
    global_loss_count = _distributed_reduce_scalar(float(measured_loss_count), op=dist.ReduceOp.SUM)
    global_measured_samples = _distributed_reduce_scalar(measured_samples, op=dist.ReduceOp.SUM)
    global_encoder_tokens = _distributed_reduce_scalar(measured_encoder_tokens, op=dist.ReduceOp.SUM)
    global_decoder_tokens = _distributed_reduce_scalar(measured_decoder_tokens, op=dist.ReduceOp.SUM)
    global_optimizer_steps = _distributed_reduce_scalar(measured_optimizer_steps, op=dist.ReduceOp.MAX)

    samples_per_second_global = global_measured_samples / max(elapsed_seconds, 1e-12)
    samples_per_second_per_gpu = samples_per_second_global / max(1, dist_info["world_size"])
    optimizer_steps_per_second = global_optimizer_steps / max(elapsed_seconds, 1e-12)
    approx_tokens_per_second = samples_per_second_global * (args.input_budget + args.output_budget)
    actual_tokens_per_second = (global_encoder_tokens + global_decoder_tokens) / max(elapsed_seconds, 1e-12)
    global_tokens_per_sec = actual_tokens_per_second
    avg_loss = global_loss_sum / max(1.0, global_loss_count)
    avg_encoder_tokens_per_sample = global_encoder_tokens / max(1.0, global_measured_samples)
    avg_decoder_tokens_per_sample = global_decoder_tokens / max(1.0, global_measured_samples)

    summary = {
        "mode": "benchmark_only",
        "world_size": dist_info["world_size"],
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "batch_size_per_gpu": args.batch_size,
        "global_batch_size": args.batch_size * dist_info["world_size"],
        "grad_accum_steps": args.grad_accum_steps,
        "input_budget": args.input_budget,
        "output_budget": args.output_budget,
        "elapsed_seconds": elapsed_seconds,
        "measured_samples_global": global_measured_samples,
        "samples_per_second_global": samples_per_second_global,
        "samples_per_second_per_gpu": samples_per_second_per_gpu,
        "measured_optimizer_steps_global": global_optimizer_steps,
        "optimizer_steps_per_second": optimizer_steps_per_second,
        "approx_tokens_per_second_global": approx_tokens_per_second,
        "actual_tokens_per_second_global": actual_tokens_per_second,
        "global_tokens_per_sec": global_tokens_per_sec,
        "average_encoder_tokens_per_sample": avg_encoder_tokens_per_sample,
        "average_decoder_tokens_per_sample": avg_decoder_tokens_per_sample,
        "average_loss": avg_loss,
        "output_dir": output_dir,
    }

    if is_main:
        summary_path = os.path.join(output_dir, "benchmark_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 80)
        print("Benchmark complete!")
        print(f"Measured time: {elapsed_seconds:.4f}s")
        print(f"Global throughput: {samples_per_second_global:.2f} samples/s")
        print(f"Per-GPU throughput: {samples_per_second_per_gpu:.2f} samples/s")
        print(f"Optimizer step rate: {optimizer_steps_per_second:.2f} steps/s")
        print(f"Approximate token throughput: {approx_tokens_per_second:.2f} tokens/s")
        print(f"Actual token throughput: {actual_tokens_per_second:.2f} tokens/s")
        print(f"Global token throughput: {global_tokens_per_sec:.2f} tokens/s")
        print(f"Average encoder tokens/sample: {avg_encoder_tokens_per_sample:.2f}")
        print(f"Average decoder tokens/sample: {avg_decoder_tokens_per_sample:.2f}")
        print(f"Average loss: {avg_loss:.4f}")
        print(f"Benchmark summary: {summary_path}")
        print("=" * 80)

    cleanup_distributed()

    return trainer.model, trainer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="AION 模型 DDP 分布式训练")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="polymathic-ai/aion-base",
                        help="预训练模型名称或路径")
    parser.add_argument("--cache_dir", type=str, 
                        default=None,
                        help="模型缓存目录")
    
    # 数据参数
    parser.add_argument("--data_path", type=str, 
                        default="data/train/",
                        help="训练数据路径（JSON token文件或包含多个JSON文件的文件夹）")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="验证集比例（从数据末尾划分）")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大训练样本数")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32,
                        help="每个GPU的批次大小")
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="训练轮数")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="最大训练步数")
    parser.add_argument("--steps_per_epoch", type=int, default=None,
                        help="每轮步数")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="梯度累积步数")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载线程数")
    
    # 优化器参数
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="最小学习率")
    parser.add_argument("--warmup_steps", type=int, default=None,
                        help="线性warmup步数（优先于warmup_ratio）")
    parser.add_argument("--warmup_ratio", type=float, default=0.02,
                        help="线性warmup比例（warmup_steps未设置时生效）")
    parser.add_argument("--warmup_start_factor", type=float, default=2e-1,
                        help="warmup起始学习率倍数（(0,1]，乘以lr）")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="权重衰减")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="梯度裁剪范数")
    
    # 损失函数参数
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="标签平滑")
    parser.add_argument("--spectrum_weight", type=float, default=1.0,
                        help="光谱模态损失权重")
    parser.add_argument("--image_weight", type=float, default=1.0,
                        help="图像模态损失权重")
    parser.add_argument("--flux_weight", type=float, default=3.0,
                        help="测光模态损失权重")
    
    # 掩码/采样参数
    parser.add_argument("--input_budget", type=int, default=256,
                        help="输入token预算（全局输入token数量上限）")
    parser.add_argument("--anchor_ratio_min", type=float, default=0.2,
                        help="锚点模态采样最小比例")
    parser.add_argument("--anchor_ratio_max", type=float, default=0.9,
                        help="锚点模态采样最大比例")
    parser.add_argument("--output_budget", type=int, default=128,
                        help="输出token预算（全局输出token数量上限）")
    parser.add_argument("--beta_alpha", type=float, default=0.5,
                        help="Beta分布alpha参数（控制输出数量分布，越小越向零偏斜）")
    parser.add_argument("--beta_beta", type=float, default=2.0,
                        help="Beta分布beta参数（控制输出数量分布，越大越向零偏斜）")
    parser.add_argument("--benchmark_only", action="store_true",
                        help="仅运行基准测试，不执行完整训练/验证/保存流程")
    parser.add_argument("--benchmark_warmup_steps", type=int, default=200,
                        help="基准测试预热步数")
    parser.add_argument("--benchmark_measure_steps", type=int, default=1000,
                        help="基准测试测量步数")
    
    # 混合精度
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="使用混合精度训练")
    parser.add_argument("--no_amp", action="store_false", dest="use_amp",
                        help="禁用混合精度训练")
    
    # 日志和保存
    parser.add_argument("--output_dir", type=str, default="outputs/aion_training",
                        help="输出目录")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="日志记录间隔")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="检查点保存间隔（epochs）")
    parser.add_argument("--eval_interval", type=int, default=1,
                        help="验证间隔（epochs）")
    
    # TensorBoard / WandB
    parser.add_argument("--use_tensorboard", action="store_true", default=False ,
                        help="使用TensorBoard")
    parser.add_argument("--use_wandb", action="store_true",
                        help="使用WandB")
    parser.add_argument("--wandb_project", type=str, default="aion-training",
                        help="WandB项目名")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="WandB实体")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="WandB运行名称")
    
    # 恢复训练
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练的检查点路径")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
