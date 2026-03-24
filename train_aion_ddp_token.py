"""
AION模型DDP分布式训练脚本
- 采用多模态掩码建模（Multimodal Masked Modeling）
- 使用Cross-Entropy损失函数
- 支持PyTorch DDP分布式训练
- 支持混合精度训练（AMP）
- 支持梯度累积
- 支持TensorBoard/WandB日志记录
- 支持模型检查点保存与恢复

使用方法：
    # 单机多卡训练
    torchrun --standalone --nnodes=1 --nproc_per_node=8 train_aion_ddp_v8_token.py --config configs/aion_base.yaml

    # 快速测试
    torchrun --standalone --nnodes=1 --nproc_per_node=2 train_aion_ddp_v8_token.py --batch_size 4 --max_samples 1000
"""

import os
from pickle import TRUE
import sys
import math
import argparse
import importlib
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from contextlib import nullcontext
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, IterableDataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, LinearLR, SequentialLR
from torch.amp import autocast, GradScaler

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import hashlib

# 全局变量：所有支持的token键
TOKEN_KEYS = [
    'tok_image',
    'tok_spectrum_desi',
    'tok_spectrum_sdss',  
    'tok_flux_i', 
    'tok_flux_w1', 
    'tok_flux_w2',
    'tok_flux_w3', 
    'tok_flux_w4',
    'tok_shape_r', 
    'tok_shape_e1', 
    'tok_shape_e2',
    'tok_ebv', 
    'tok_z',
    'tok_flux_g', 
    'tok_flux_r', 
    'tok_flux_z',
    'tok_image_hsc',
    'tok_a_g',
    'tok_a_r',
    'tok_a_i',
    'tok_a_z',
    'tok_a_y',
    'tok_mag_g',
    'tok_mag_r',
    'tok_mag_i',
    'tok_mag_z',
    'tok_mag_y',
    'tok_shape_11',
    'tok_shape_22',
    'tok_shape_12',
]

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

import psutil

def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process()
    mem_info = process.memory_info()
    return {
        "rss": mem_info.rss / 1024 / 1024 / 1024,  # GB
        "vms": mem_info.vms / 1024 / 1024 / 1024,  # GB
        "available": psutil.virtual_memory().available / 1024 / 1024 / 1024,  # GB
        "percent": psutil.virtual_memory().percent,
    }

def log_memory_usage(prefix: str = "", rank: int = 0):
    """记录内存使用情况"""
    mem = get_memory_usage()
    if rank == 0:
        print(f"[{prefix}] 内存使用: RSS={mem['rss']:.2f}GB, 可用={mem['available']:.2f}GB, 使用率={mem['percent']:.1f}%")

def cleanup_memory(force: bool = False, rank: int = 0, threshold_percent: float = 80.0):
    """清理内存
    
    Args:
        force: 是否强制清理
        rank: 进程排名
        threshold_percent: 内存使用阈值，超过则清理
    """
    mem = get_memory_usage()
    if force or mem['percent'] > threshold_percent:
        if rank == 0:
            print(f"内存使用过高 ({mem['percent']:.1f}%)，执行清理...")
        
        # 清理Python垃圾回收
        import gc
        gc.collect()
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if rank == 0:
            new_mem = get_memory_usage()
            print(f"清理后内存: RSS={new_mem['rss']:.2f}GB, 可用={new_mem['available']:.2f}GB, 使用率={new_mem['percent']:.1f}%")

def setup_distributed():
    """初始化分布式训练环境"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    is_distributed = world_size > 1
    
    if is_distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        torch.backends.cudnn.benchmark = True
    
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


def count_json_samples(file_path: str, rank: int = 0) -> int:
    """使用内存映射技术计算JSON样本数，减少内存使用
    
    使用内存映射技术读取文件，避免一次性加载整个文件到内存
    
    Args:
        file_path: JSON文件路径
        rank: 进程排名
    
    Returns:
        int: 样本数量
    """
    sample_count = 0
    try:
        import mmap
        import json
        
        # 使用内存映射读取文件（只读模式）
        with open(file_path, 'r') as f:
            # 创建只读内存映射
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                # 读取内存映射内容
                content = mm.read().decode('utf-8')
                # 解析JSON
                data = json.loads(content)
                
                # 处理数据格式
                if not isinstance(data, list):
                    data = [data]
                
                # 统计样本数
                sample_count = len(data)
            finally:
                # 关闭内存映射
                mm.close()
        
    except Exception as e:
        print(f"[Rank {rank}] [ERROR] 统计样本数失败: {file_path}, 错误: {e}")
        sample_count = 0
    finally:
        # 手动触发垃圾回收
        import gc
        gc.collect()
        
        # 清理CUDA缓存（如果可用）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return sample_count


# ===================== 数据加载 =====================

class AIONDataset(Dataset):
    """AION训练数据集（JSON token文件加载）
    
    直接加载已编码的token数据，无需codec_manager
    支持分布式训练的数据分片
    支持惰性加载以减少内存占用
    """
    
    def __init__(
        self,
        data_path: str,
        max_samples: Optional[int] = None,
        world_size: int = 1,
        rank: int = 0,
        lazy_loading: bool = False,
    ):
        super().__init__()
        self.data_path = data_path
        self.max_samples = max_samples
        self.world_size = world_size
        self.rank = rank
        self.lazy_loading = lazy_loading
        
        # 加载JSON token数据（支持文件或文件夹）
        print(f"[Rank {rank}] 正在加载token数据: {data_path}")
        path_obj = Path(data_path)
        
        if path_obj.is_file():
            # 单个JSON文件
            print(f"[Rank {rank}] 检测到单个文件")
            with open(data_path, 'r') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
                # 数据分片：每个rank只加载分配给自己的数据部分
                start_idx = (len(data) * rank) // world_size
                end_idx = (len(data) * (rank + 1)) // world_size
                self.data = data[start_idx:end_idx]
                self.groups = ["__single__"] * len(self.data)
        elif path_obj.is_dir():
            # 文件夹：读取所有JSON文件
            print(f"[Rank {rank}] 检测到文件夹，读取所有.json文件")
            json_files = sorted(list(path_obj.rglob("*.json")))
            
            # 打乱文件列表（确保所有rank结果一致）
            random.seed(42)  # 固定种子确保分布式训练一致性
            random.shuffle(json_files)
            print(f"[Rank {rank}] 前5个文件: {[os.path.basename(f) for f in json_files[:5]]}")
            
            if not json_files:
                raise ValueError(f"在文件夹 {data_path} 中未找到任何.json文件")
            
            print(f"[Rank {rank}] 找到 {len(json_files)} 个json文件")
            
            # 第一步：统计所有文件的样本数
            file_sample_counts = []
            total_files = len(json_files)
            cumulative_samples = 0
            
            for i, jf in enumerate(json_files):
                try:
                    # 检查是否达到max_samples
                    if self.max_samples and cumulative_samples >= self.max_samples:
                        print(f"[Rank {rank}] 已达到max_samples限制 ({self.max_samples} 样本)，停止统计")
                        # 填充剩余文件的样本数为0
                        file_sample_counts.extend([0] * (total_files - i))
                        break
                    
                    # 每隔5个文件打印一次详细信息
                    if (i + 1) % 10 == 0:
                        print(f"[Rank {rank}] 正在处理文件 {i+1}/{total_files}: {os.path.basename(jf)}")
                    
                    # 使用内存高效的函数计算样本数
                    sample_count = count_json_samples(jf, self.rank)
                    
                    file_sample_counts.append(sample_count)
                    cumulative_samples += sample_count
                except Exception as e:
                    print(f"[Rank {rank}] 读取文件 {jf} 时出错: {e}")
                    file_sample_counts.append(0)
            
            # 第二步：计算总样本数
            total_samples = sum(file_sample_counts)
            print(f"[Rank {rank}] 总样本数: {total_samples}")
            
            # 第三步：计算每个rank应该分配的样本数范围
            samples_per_rank = total_samples // world_size
            start_sample = rank * samples_per_rank
            end_sample = (rank + 1) * samples_per_rank
            if rank == world_size - 1:
                end_sample = total_samples  # 最后一个rank包含剩余所有样本
            
            # 第四步：基于样本数分配文件
            current_sample = 0
            assigned_files = []
            file_sample_ranges = []  # 记录每个文件的样本范围
            
            for i, (jf, count) in enumerate(zip(json_files, file_sample_counts)):
                if count == 0:
                    continue
                    
                file_start = current_sample
                file_end = current_sample + count
                
                # 检查文件是否与当前rank的样本范围重叠
                if file_end > start_sample and file_start < end_sample:
                    # 计算当前rank在该文件中的样本范围
                    rank_file_start = max(0, start_sample - file_start)
                    rank_file_end = min(count, end_sample - file_start)
                    assigned_files.append(jf)
                    file_sample_ranges.append((rank_file_start, rank_file_end))
                
                current_sample = file_end
                if current_sample >= end_sample:
                    break
            
            print(f"[Rank {rank}] 分配到 {len(assigned_files)} 个文件，样本范围: [{start_sample}, {end_sample})")
            
            if self.lazy_loading:
                # 惰性加载：只存储文件路径和样本范围
                self.assigned_files = assigned_files
                self.file_sample_ranges = file_sample_ranges
                self.path_obj = path_obj
                
                # 计算总样本数
                self.total_samples = sum(end - start for start, end in file_sample_ranges)
                
                # 构建样本到文件的映射
                self.sample_to_file = []
                # 构建groups属性
                self.groups = []
                
                for file_idx, (start, end) in enumerate(file_sample_ranges):
                    jf = assigned_files[file_idx]
                    rel = jf.relative_to(path_obj)
                    group = rel.parts[0] if len(rel.parts) > 1 else "__root__"
                    
                    for sample_idx in range(start, end):
                        self.sample_to_file.append((file_idx, sample_idx))
                        self.groups.append(group)
                
                # 限制样本数
                if max_samples and len(self.sample_to_file) > max_samples:
                    self.sample_to_file = self.sample_to_file[:max_samples]
                    self.groups = self.groups[:max_samples]
                    self.total_samples = len(self.sample_to_file)
                
                print(f"[Rank {rank}] 惰性加载模式，共 {self.total_samples} 条数据")
            else:
                # 传统加载方式：使用内存映射技术加载数据
                # 读取并合并分配的JSON文件
                all_data = []
                all_groups = []
                current_sample = 0
                total_assigned = len(assigned_files)
                
                for i, (jf, (rank_file_start, rank_file_end)) in enumerate(zip(assigned_files, file_sample_ranges)):
                    # 打印文件读取信息和进度
                    print(f"[Rank {rank}] 读取文件 {i+1}/{total_assigned}: {os.path.basename(jf)}")
                    
                    # 打印百分比进度
                    if (i + 1) % 5 == 0 or (i + 1) == total_assigned:
                        progress = (i + 1) / total_assigned * 100
                        print(f"[Rank {rank}] 读取文件进度: {progress:.1f}% ({i+1}/{total_assigned})")
                    
                    try:
                        import mmap
                        import json
                        
                        # 使用内存映射读取文件（只读模式）
                        with open(jf, 'r') as f:
                            # 创建只读内存映射
                            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                            try:
                                # 读取内存映射内容
                                content = mm.read().decode('utf-8')
                                # 解析JSON
                                data = json.loads(content)
                                if not isinstance(data, list):
                                    data = [data]
                                
                                # 只加载分配给自己的样本部分
                                if rank_file_start < rank_file_end:
                                    file_data = data[rank_file_start:rank_file_end]
                                    rel = jf.relative_to(path_obj)
                                    group = rel.parts[0] if len(rel.parts) > 1 else "__root__"
                                    all_data.extend(file_data)
                                    all_groups.extend([group] * len(file_data))
                            finally:
                                # 关闭内存映射
                                mm.close()
                        
                        # 清理内存
                        import gc
                        gc.collect()
                        
                    except Exception as e:
                        print(f"[Rank {rank}] 读取文件 {jf} 时出错: {e}")
                
                self.data = all_data
                self.groups = all_groups
                
                # 限制样本数
                if max_samples and len(self.data) > max_samples:
                    self.data = self.data[:max_samples]
                    self.groups = self.groups[:max_samples]
                
                print(f"[Rank {rank}] 合并后共 {len(self.data)} 条数据")
        else:
            raise ValueError(f"路径不存在或无效: {data_path}")
        
        if not hasattr(self, 'data') and not hasattr(self, 'sample_to_file'):
            raise ValueError(f"在路径 {data_path} 中未找到任何有效数据")
        
        print(f"[Rank {rank}] 数据加载完成")
        
    def __len__(self):
        if hasattr(self, 'total_samples'):
            return self.total_samples
        elif hasattr(self, 'data'):
            return len(self.data)
        return 0
    
    def __getitem__(self, idx):
        """获取单个样本（返回token）"""
        if hasattr(self, 'sample_to_file'):
            # 惰性加载模式
            file_idx, sample_idx = self.sample_to_file[idx]
            jf = self.assigned_files[file_idx]
            
            # 加载文件数据
            with open(jf, 'r') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
                item = data[sample_idx]
        else:
            # 传统模式
            item = self.data[idx]
        
        sample = {}
        
        # 直接返回token数据
        for key in TOKEN_KEYS:
            if key in item:
                sample[key] = item[key]
        
        return sample


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

        if isinstance(dataset, torch.utils.data.Subset):
            self._subset_indices = list(dataset.indices)
            base = dataset.dataset
            if not hasattr(base, "groups"):
                raise ValueError("Subset base dataset must have `groups` attribute")
            base_groups = base.groups
            self._groups = [base_groups[i] for i in self._subset_indices]
        else:
            if not hasattr(dataset, "groups"):
                raise ValueError("Dataset must have `groups` attribute")
            self._groups = list(dataset.groups)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self) -> int:
        total_batches = 0
        group_to_count = {}
        for g in self._groups:
            group_to_count[g] = group_to_count.get(g, 0) + 1

        # 数据已经在Dataset中按rank分片，每个rank只处理自己的数据
        # 所以这里直接使用batch_size，不需要再乘以num_replicas
        denom = self.batch_size
        for count in group_to_count.values():
            if self.drop_last:
                total_batches += count // denom
            else:
                total_batches += (count + denom - 1) // denom
        return total_batches

    def __iter__(self):
        n = len(self._groups)
        indices = list(range(n))
        group_to_indices = {}
        for i in indices:
            g = self._groups[i]
            group_to_indices.setdefault(g, []).append(i)

        all_batches = []
        # 数据已经在Dataset中按rank分片，每个rank只处理自己的数据
        # 所以这里直接使用batch_size，不需要再乘以num_replicas
        denom = self.batch_size

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

            # 数据已经按rank分片，不需要再使用rank步长采样
            # 直接使用所有group_indices
            rank_indices = group_indices
            for i in range(0, len(rank_indices), self.batch_size):
                batch = rank_indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)

        if self.shuffle:
            rng = random.Random(self.seed + self.epoch * 1000 + 99991)
            rng.shuffle(all_batches)

        if self._subset_indices is not None:
            for batch in all_batches:
                yield batch
        else:
            for batch in all_batches:
                yield batch

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """批处理 collate 函数 - 处理token数据
    
    将token数据打包成batch
    优化内存使用：减少临时对象创建，及时释放内存
    """
    result = {}
    
    for key in TOKEN_KEYS:
        # 检查是否有样本包含此key
        has_key = any(key in s for s in batch)
        if not has_key:
            continue
        
        # 序列型token需要padding到统一长度
        is_sequence = key in ['tok_image', 'tok_image_hsc', 'tok_spectrum_desi', 'tok_spectrum_sdss']
        
        if is_sequence:
            # 计算最大长度
            max_len = 0
            for s in batch:
                if key in s:
                    val = s[key]
                    if isinstance(val, (list, np.ndarray, torch.Tensor)):
                        max_len = max(max_len, len(val))
            
            # 直接创建最终tensor，避免中间列表
            batch_size = len(batch)
            stacked = torch.full((batch_size, max_len), -1, dtype=torch.long)
            
            for i, s in enumerate(batch):
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
                    
                    length = min(len(t), max_len)
                    stacked[i, :length] = t[:length]
        else:
            # 非序列型token
            batch_size = len(batch)
            stacked = torch.full((batch_size,), -1, dtype=torch.long)
            
            for i, s in enumerate(batch):
                if key in s:
                    val = s[key]
                    if isinstance(val, (list, np.ndarray, torch.Tensor)):
                        # 取第一个元素
                        if len(val) > 0:
                            stacked[i] = torch.LongTensor([val[0]])[0]
                    else:
                        stacked[i] = torch.LongTensor([val])[0]
        
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
        
        for mod_key, mod_logits in logits.items():
            if mod_key not in targets:
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
        
        # 混合精度训练
        self.scaler = GradScaler('cuda') if use_amp else None
        
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
            primary_ratio = 0.5 + beta_sample * 0.5
            primary_ratio = min(primary_ratio, 1.0)
            
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
        for key in TOKEN_KEYS:
            if key in batch:
                input_tokens[key] = batch[key]
        
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
        
        if not input_tokens:
            return {"total_loss": 0.0}
        
        # 确定是否需要同步梯度
        is_accumulating = (step_in_accum + 1) < self.grad_accum_steps
        
        # DDP 梯度同步控制
        sync_ctx = nullcontext()
        if self.dist_info["is_distributed"] and isinstance(self.model, DDP) and is_accumulating:
            sync_ctx = self.model.no_sync()
        
        with sync_ctx:
            # 混合精度前向
            with autocast('cuda', enabled=self.use_amp):
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
                B = next(iter(input_tokens.values())).shape[0]
                
                for mod_key, tok in input_tokens.items():
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
            
            # 检查损失是否为nan
            if torch.isnan(loss) or torch.isinf(loss):
                # 跳过这个batch，不进行反向传播
                loss_dict["total_loss"] = float('nan')
                loss_dict["learning_rate"] = self.scheduler.get_last_lr()[0]
                
                # 清理内存
                del input_tokens, target_tokens, input_masks, target_masks, masked_input, input_mask_2d, target_mask_2d, logits, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return loss_dict
            
            # 反向传播
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        
        # 在梯度累积完成后更新参数
        if not is_accumulating:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            
            # 梯度裁剪
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.max_grad_norm
                )
            
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
        
        # 清理内存
        del input_tokens, target_tokens, input_masks, target_masks, masked_input, input_mask_2d, target_mask_2d, logits, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
        
        # 强制清理内存，避免内存不足导致的通信问题
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
            
            with autocast('cuda', enabled=self.use_amp):
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
            # 只在主进程(0卡)进行验证结果计算，避免多卡同步问题
            # 其他进程直接返回本地计算的损失
            if self.dist_info.get("rank", 0) != 0:
                return {"val_loss": total_loss / max(1, local_num_batches)}
            
            # 主进程只使用本地计算的损失，不进行集体操作
            # 这样可以避免多卡同步问题
        
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
        self.epoch = state["epoch"]
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
        print("AION 分布式训练")
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
        
        print(f"输出目录: {output_dir}")
    else:
        output_dir = None
    
    # 广播输出目录路径
    if dist_info["is_distributed"]:
        output_dir_list = [output_dir]
        dist.broadcast_object_list(output_dir_list, src=0)
        output_dir = output_dir_list[0]
    
    # 3. 加载模型
    if is_main:
        print(f"\n正在加载模型配置（从零开始训练）: {args.model_name}")
    
    # 加载预训练模型结构，然后重新初始化权重
    model = AION.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
    )
    
    # 重新初始化所有权重（从零开始训练）
    if is_main:
        print("重新初始化模型权重...")
    
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
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
    
    # 注意：使用预编码的token数据，不需要codec_manager
    
    # 4. 创建数据加载器
    if is_main:
        print(f"\n正在加载token数据集: {args.data_path}")
    
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
            # 只有主进程创建验证数据加载器，其他进程不创建
            if dist_info["is_distributed"] and dist_info["rank"] != 0:
                # 其他进程不创建验证数据加载器
                val_loader = None
            else:
                # 主进程创建验证数据加载器
                num_full = len(train_dataset)
                indices = list(range(num_full))
                random.shuffle(indices)
                val_indices = indices[-val_size:]
                val_dataset = torch.utils.data.Subset(train_dataset, val_indices)
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
        print(f"每epoch步数: {steps_per_epoch}")
        print(f"总训练步数: {total_steps}")
    
    warmup_steps = 0
    if args.warmup_steps is not None:
        warmup_steps = args.warmup_steps
    elif args.warmup_ratio and args.warmup_ratio > 0:
        warmup_steps = int(total_steps * args.warmup_ratio)
    
    warmup_steps = int(max(0, warmup_steps))
    warmup_steps = min(warmup_steps, max(0, total_steps - 1))
    
    if is_main:
        print(f"warmup步数: {warmup_steps}")
    
    if warmup_steps > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=args.warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps - warmup_steps),
            eta_min=args.lr*0.01,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
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
            print(f"\n恢复训练: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # 8. TensorBoard / WandB
    writer = None
    wandb_run = None
    
    if is_main:
        if args.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(os.path.join(output_dir, "logs"))
        
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
                print(f"WandB 初始化失败: {e}")
    
    # 9. 训练循环
    if is_main:
        print(f"\n开始训练...")
        print(f"Epochs: {args.num_epochs}")
        print(f"Batch Size (per GPU): {args.batch_size}")
        print(f"Total Batch Size: {args.batch_size * dist_info['world_size']}")
        print(f"梯度累积步数: {args.grad_accum_steps}")
        print(f"学习率: {args.lr}")
        print(f"输入Token预算: {args.input_budget}")
        print(f"锚点采样比例: [{args.anchor_ratio_min}, {args.anchor_ratio_max}]")
        print(f"输出Token预算: {args.output_budget}")
        print(f"Beta分布参数: alpha={args.beta_alpha}, beta={args.beta_beta}")
    
    # 记录初始内存使用
    log_memory_usage("训练开始", dist_info["rank"])
    
    for epoch in range(trainer.epoch, args.num_epochs):
        trainer.epoch = epoch
        
        if hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        elif hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        
        # 训练一个epoch
        epoch_loss = 0.0
        num_steps = 0
        
        # 记录epoch开始的内存使用
        log_memory_usage(f"Epoch {epoch+1} 开始", dist_info["rank"])
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            disable=not is_main,
        )
        
        for step, batch in enumerate(progress_bar):
            # 定期检查内存使用并清理
            if step % 100 == 0:  # 每100步检查一次内存
                cleanup_memory(rank=dist_info["rank"], threshold_percent=75.0)
            
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
            
            # 检查最大步数
            if args.max_steps and trainer.global_step >= args.max_steps:
                break
        
        # Epoch 结束时强制清理内存
        cleanup_memory(force=True, rank=dist_info["rank"])
        log_memory_usage(f"Epoch {epoch+1} 结束", dist_info["rank"])
        
        # Epoch 结束
        # 移除epoch结束的barrier，避免同步问题
        # if dist_info["is_distributed"]:
        #     dist.barrier()
        
        # 验证
        if val_loader and (epoch + 1) % args.eval_interval == 0:
            if is_main:
                print(f"\n运行验证...")
            
            # 验证前强制清理内存
            cleanup_memory(force=True, rank=dist_info["rank"])
            
            # 只有主进程进行验证，其他进程跳过
            if dist_info["is_distributed"] and dist_info["rank"] != 0:
                # 其他进程跳过验证
                val_metrics = {"val_loss": 0.0}
            else:
                # 主进程进行验证
                val_metrics = trainer.validate(val_loader)
            
            if is_main:
                print(f"验证损失: {val_metrics['val_loss']:.4f}")
                
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
            print(f"检查点已保存: checkpoint_epoch_{epoch + 1}.pt")
        
        # 移除保存检查点后的barrier，避免同步问题
        # if dist_info["is_distributed"]:
        #     dist.barrier()
    
    # 10. 训练结束
    if is_main:
        # 保存最终模型
        trainer.save_checkpoint(
            os.path.join(output_dir, "checkpoints", "final_model.pt"),
        )
        
        print("\n" + "=" * 80)
        print("训练完成!")
        print(f"最佳验证损失: {trainer.best_val_loss:.4f}")
        print(f"总训练步数: {trainer.global_step}")
        print(f"输出目录: {output_dir}")
        print("=" * 80)
        
        if writer:
            writer.close()
        
        if wandb_run:
            wandb_run.finish()
    
    cleanup_distributed()
    
    return model, trainer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="AION 模型 DDP 分布式训练")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="polymathic-ai/aion-base",
                        help="预训练模型名称或路径")
    parser.add_argument("--cache_dir", type=str, 
                        default="weight/aion-base/",
                        help="模型缓存目录")
    
    # 数据参数
    parser.add_argument("--data_path", type=str, 
                        default="data/train/",
                        help="训练数据路径（JSON token文件或包含多个JSON文件的文件夹）")
    parser.add_argument("--val_split", type=float, default=0.05,
                        help="验证集比例（从数据末尾划分）")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="最大训练样本数")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=64,
                        help="每个GPU的批次大小")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="最大训练步数")
    parser.add_argument("--steps_per_epoch", type=int, default=None,
                        help="每轮步数")
    parser.add_argument("--grad_accum_steps", type=int, default=2,
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
    parser.add_argument("--weight_decay", type=float, default=0.1,
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
    parser.add_argument("--flux_weight", type=float, default=1.0,
                        help="测光模态损失权重")
    
    # 掩码/采样参数
    parser.add_argument("--input_budget", type=int, default=256,
                        help="输入token预算（全局输入token数量上限）")
    parser.add_argument("--anchor_ratio_min", type=float, default=0.3,
                        help="锚点模态采样最小比例")
    parser.add_argument("--anchor_ratio_max", type=float, default=0.9,
                        help="锚点模态采样最大比例")
    parser.add_argument("--output_budget", type=int, default=128,
                        help="输出token预算（全局输出token数量上限）")
    parser.add_argument("--beta_alpha", type=float, default=1,
                        help="Beta分布alpha参数（控制输出数量分布，越小越向零偏斜）")
    parser.add_argument("--beta_beta", type=float, default=4,
                        help="Beta分布beta参数（控制输出数量分布，越大越向零偏斜）")
    
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
    parser.add_argument("--use_tensorboard", action="store_true", 
                        help="使用TensorBoard")
    parser.add_argument("--use_wandb", action="store_true", default=False,
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
