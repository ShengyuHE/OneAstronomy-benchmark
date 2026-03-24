"""
AION模型红移预测脚本

参考train_aion_ddp_v2.py的数据读取方式和encode_test1.py的红移预测方式
支持从Parquet文件批量预测红移

使用方法：
    python tools/predict_redshift.py --data_path /path/to/data.parquet --output_dir outputs/redshift_pred
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import Optional, Dict, Any, List

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


def load_data(data_path: str, max_samples: Optional[int] = None):
    """加载Parquet数据文件或文件夹
    
    Args:
        data_path: Parquet文件路径或包含Parquet文件的文件夹路径
        max_samples: 最大样本数
    
    Returns:
        list: 文件路径列表或单个文件的数据
    """
    print(f"正在加载数据: {data_path}")
    
    # 检查路径是文件还是文件夹
    path_obj = Path(data_path)
    
    if path_obj.is_file():
        # 单个文件
        print(f"检测到单个文件: {data_path}")
        data = pd.read_parquet(data_path)
        
        if max_samples and len(data) > max_samples:
            print(f"限制样本数量从 {len(data)} 到 {max_samples}")
            data = data.iloc[:max_samples].reset_index(drop=True)
        print(f"数据加载完成，共 {len(data)} 条")
        return data
    elif path_obj.is_dir():
        # 文件夹，递归读取所有parquet文件（包括子文件夹）
        print(f"检测到文件夹: {data_path}")
        # 过滤掉文件名包含"catalog"的文件
        parquet_files = []
        for file_path in path_obj.rglob("*.parquet"):
            # 检查文件名是否包含"catalog"（不区分大小写）
            if "catalog" not in file_path.name.lower():
                parquet_files.append(str(file_path))
        
        if not parquet_files:
            raise ValueError(f"文件夹 {data_path} 及其子文件夹中未找到任何 .parquet 文件（已排除包含'catalog'的文件）")
        
        # 按照文件名排序
        parquet_files.sort(key=lambda x: os.path.basename(x))
        
        print(f"找到 {len(parquet_files)} 个 Parquet 文件（包括子文件夹，已排除包含'catalog'的文件）")
        print(f"数据加载完成，准备逐个处理文件")
        return parquet_files
    else:
        raise ValueError(f"路径 {data_path} 既不是文件也不是文件夹")


def prepare_modalities(row, device="cuda"):
    """准备单个样本的各模态对象"""
    modalities = []
    other_modalities = []
    spectrum = None
    
    def is_valid_value(value):
        """检查值是否有效（非NaN、非无穷大）"""
        return value is not None and not np.isnan(value) and np.isfinite(value)
    # HSC或legacy survey图像数据
    if 'image_flux' in row.index and 'image_band' in row.index:
        
        img_flux = np.array(np.array(row['image_flux'].tolist()).tolist()).astype(np.float32)
        img_flux = np.nan_to_num(img_flux, nan=0.0, posinf=0.0, neginf=0.0)
        # 先在CPU上创建tensor，再移到GPU，避免CUDA缓存问题
        img_flux_tensor = torch.tensor(img_flux, dtype=torch.float32).to(device)
        if len(row['image_band']) == 5:
            image = HSCImage(
                flux=img_flux_tensor.unsqueeze(0),  # 添加batch维度
                # bands=list(row['image_band']),
                bands=["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"],
            )
        else:
            image = LegacySurveyImage(
            flux=img_flux_tensor.unsqueeze(0),  # 添加batch维度
            bands=["DES-G", "DES-R", "DES-I", "DES-Z"],
        )
        modalities.append(image)
    
    #SDSSSpectrum  支持多种光谱列名，按优先级顺序检查 
    spectrum_cols = ['eboss_spectrum', 'boss_spectrum', 'sdss_spectrum', 'segue1_spectrum', 'segue2_spectrum']
    spectrum_col = next((col for col in spectrum_cols if col in row.index), None)
    if spectrum_col:
        try:
            spectrum_data = row[spectrum_col]
            if isinstance(spectrum_data, dict):
                wavelength = np.array(spectrum_data.get('wavelength', spectrum_data.get('lambda', []))).astype(np.float32)
                flux = np.array(spectrum_data['flux']).astype(np.float32)
                ivar = np.array(spectrum_data.get('ivar', np.ones_like(flux))).astype(np.float32)
                mask = np.array(spectrum_data.get('mask', np.ones_like(flux, dtype=bool))).astype(bool)
            else:
                wavelength = np.array([]).astype(np.float32)
                flux = np.array([]).astype(np.float32)
                ivar = np.array([]).astype(np.float32)
                mask = np.array([]).astype(bool)
            
            # 确保所有数组长度一致
            min_length = min(len(wavelength), len(flux), len(ivar), len(mask))
            if min_length > 0:
                wavelength = wavelength[:min_length]
                flux = flux[:min_length]
                ivar = ivar[:min_length]
                mask = mask[:min_length]
            
            valid_wave_indices = (wavelength <= 10000) & (wavelength > 0)
            # 检查过滤后是否还有有效数据
            if np.any(valid_wave_indices):
                order = np.argsort(wavelength[valid_wave_indices])
                wavelength = wavelength[valid_wave_indices][order]
                flux = flux[valid_wave_indices][order]
                ivar = ivar[valid_wave_indices][order]
                mask = mask[valid_wave_indices][order]
                wavelength = np.nan_to_num(wavelength, nan=0.0, posinf=0.0, neginf=0.0)
                flux = np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
                ivar = np.nan_to_num(ivar, nan=0.0, posinf=0.0, neginf=0.0)
                
                # 再次确保所有数组长度一致
                min_length = min(len(wavelength), len(flux), len(ivar), len(mask))
                if min_length > 0:
                    wavelength = wavelength[:min_length]
                    flux = flux[:min_length]
                    ivar = ivar[:min_length]
                    mask = mask[:min_length]
                
                # 创建tensor并移到指定设备
                spectrum = SDSSSpectrum(
                    wavelength=torch.tensor(wavelength, dtype=torch.float32).unsqueeze(0).to(device),
                    flux=torch.tensor(flux, dtype=torch.float32).unsqueeze(0).to(device),
                    ivar=torch.tensor(ivar, dtype=torch.float32).unsqueeze(0).to(device),
                    mask=torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(device),
                )
                modalities.append(spectrum)
        except Exception as e:
            print(f"Error processing SDSS spectrum: {e}")
            # 出错时跳过该光谱，继续处理其他模态
            pass


    # DESISpectrum
    spectrum_cols = ['spectrum', 'dr1_main_spectrum']
    spectrum_col = next((col for col in spectrum_cols if col in row.index), None)
    if spectrum_col:
        try:
            spectrum_data = row[spectrum_col]
            if isinstance(spectrum_data, dict):
                wavelength = np.array(spectrum_data.get('wavelength', spectrum_data.get('lambda', []))).astype(np.float32)
                flux = np.array(spectrum_data['flux']).astype(np.float32)
                ivar = np.array(spectrum_data.get('ivar', np.ones_like(flux))).astype(np.float32)
                mask = np.array(spectrum_data.get('mask', np.ones_like(flux, dtype=bool))).astype(bool)
            else:
                wavelength = np.array([]).astype(np.float32)
                flux = np.array([]).astype(np.float32)
                ivar = np.array([]).astype(np.float32)
                mask = np.array([]).astype(bool)
            
            # 确保所有数组长度一致
            min_length = min(len(wavelength), len(flux), len(ivar), len(mask))
            if min_length > 0:
                wavelength = wavelength[:min_length]
                flux = flux[:min_length]
                ivar = ivar[:min_length]
                mask = mask[:min_length]
            
            valid_wave_indices = (wavelength <= 10000) & (wavelength > 0)
            # 检查过滤后是否还有有效数据
            if np.any(valid_wave_indices):
                order = np.argsort(wavelength[valid_wave_indices])
                wavelength = wavelength[valid_wave_indices][order]
                flux = flux[valid_wave_indices][order]
                ivar = ivar[valid_wave_indices][order]
                mask = mask[valid_wave_indices][order]
                wavelength = np.nan_to_num(wavelength, nan=0.0, posinf=0.0, neginf=0.0)
                flux = np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
                ivar = np.nan_to_num(ivar, nan=0.0, posinf=0.0, neginf=0.0)
                
                # 再次确保所有数组长度一致
                min_length = min(len(wavelength), len(flux), len(ivar), len(mask))
                if min_length > 0:
                    wavelength = wavelength[:min_length]
                    flux = flux[:min_length]
                    ivar = ivar[:min_length]
                    mask = mask[:min_length]
                
                # 创建tensor并移到指定设备
                spectrum = DESISpectrum(
                    wavelength=torch.tensor(wavelength, dtype=torch.float32).unsqueeze(0).to(device),
                    flux=torch.tensor(flux, dtype=torch.float32).unsqueeze(0).to(device),
                    ivar=torch.tensor(ivar, dtype=torch.float32).unsqueeze(0).to(device),
                    mask=torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(device),
                )
                modalities.append(spectrum)
        except Exception as e:
            print(f"Error processing DESI spectrum: {e}")
            # 出错时跳过该光谱，继续处理其他模态
            pass
    
    # HSC消光值
    if 'a_g' in row.index and is_valid_value(row['a_g']):
        modalities.append(HSCAG(value=torch.tensor([row['a_g']], device=device, dtype=torch.float32)))
    if 'a_r' in row.index and is_valid_value(row['a_r']):
        modalities.append(HSCAR(value=torch.tensor([row['a_r']], device=device, dtype=torch.float32)))
    if 'a_i' in row.index and is_valid_value(row['a_i']):
        modalities.append(HSCAI(value=torch.tensor([row['a_i']], device=device, dtype=torch.float32)))
    if 'a_z' in row.index and is_valid_value(row['a_z']):
        modalities.append(HSCAZ(value=torch.tensor([row['a_z']], device=device, dtype=torch.float32)))
    if 'a_y' in row.index and is_valid_value(row['a_y']):
        modalities.append(HSCAY(value=torch.tensor([row['a_y']], device=device, dtype=torch.float32)))
    
    # HSC星等
    if 'g_cmodel_mag' in row.index and is_valid_value(row['g_cmodel_mag']):
        modalities.append(HSCMagG(value=torch.tensor([row['g_cmodel_mag']], device=device, dtype=torch.float32)))
    if 'r_cmodel_mag' in row.index and is_valid_value(row['r_cmodel_mag']):
        modalities.append(HSCMagR(value=torch.tensor([row['r_cmodel_mag']], device=device, dtype=torch.float32)))
    if 'i_cmodel_mag' in row.index and is_valid_value(row['i_cmodel_mag']):
        modalities.append(HSCMagI(value=torch.tensor([row['i_cmodel_mag']], device=device, dtype=torch.float32)))
    if 'z_cmodel_mag' in row.index and is_valid_value(row['z_cmodel_mag']):
        modalities.append(HSCMagZ(value=torch.tensor([row['z_cmodel_mag']], device=device, dtype=torch.float32)))
    if 'y_cmodel_mag' in row.index and is_valid_value(row['y_cmodel_mag']):
        modalities.append(HSCMagY(value=torch.tensor([row['y_cmodel_mag']], device=device, dtype=torch.float32)))
    
    # HSC形状参数
    if 'i_sdssshape_shape11' in row.index and is_valid_value(row['i_sdssshape_shape11']):
        modalities.append(HSCShape11(value=torch.tensor([row['i_sdssshape_shape11']], device=device, dtype=torch.float32)))
    if 'i_sdssshape_shape22' in row.index and is_valid_value(row['i_sdssshape_shape22']):
        modalities.append(HSCShape22(value=torch.tensor([row['i_sdssshape_shape22']], device=device, dtype=torch.float32)))
    if 'i_sdssshape_shape12' in row.index and is_valid_value(row['i_sdssshape_shape12']):
        modalities.append(HSCShape12(value=torch.tensor([row['i_sdssshape_shape12']], device=device, dtype=torch.float32)))
    
    if 'dr1_main_Z' in row.index and is_valid_value(row['dr1_main_Z']):
        other_modalities.append(Z(value=torch.tensor([row['dr1_main_Z']], device=device, dtype=torch.float32)))
    if 'Z' in row.index and is_valid_value(row['Z']):
        other_modalities.append(Z(value=torch.tensor([row['Z']], device=device, dtype=torch.float32)))
    
    if 'FLUX_I' in row.index and is_valid_value(row['FLUX_I']):
        modalities.append(LegacySurveyFluxI(value=torch.tensor([row['FLUX_I']], device=device, dtype=torch.float32)))
    if 'FLUX_W1' in row.index and is_valid_value(row['FLUX_W1']):
        modalities.append(LegacySurveyFluxW1(value=torch.tensor([row['FLUX_W1']], device=device, dtype=torch.float32)))
    if 'FLUX_W2' in row.index and is_valid_value(row['FLUX_W2']):
        modalities.append(LegacySurveyFluxW2(value=torch.tensor([row['FLUX_W2']], device=device, dtype=torch.float32)))
    if 'FLUX_W3' in row.index and is_valid_value(row['FLUX_W3']):
        modalities.append(LegacySurveyFluxW3(value=torch.tensor([row['FLUX_W3']], device=device, dtype=torch.float32)))
    if 'FLUX_W4' in row.index and is_valid_value(row['FLUX_W4']):
        modalities.append(LegacySurveyFluxW4(value=torch.tensor([row['FLUX_W4']], device=device, dtype=torch.float32)))
    if 'SHAPE_R' in row.index and is_valid_value(row['SHAPE_R']):
        modalities.append(LegacySurveyShapeR(value=torch.tensor([row['SHAPE_R']], device=device, dtype=torch.float32)))
    
    if 'SHAPE_E1' in row.index and is_valid_value(row['SHAPE_E1']):
        modalities.append(LegacySurveyShapeE1(value=torch.tensor([row['SHAPE_E1']], device=device, dtype=torch.float32)))
    if 'SHAPE_E2' in row.index and is_valid_value(row['SHAPE_E2']):
        modalities.append(LegacySurveyShapeE2(value=torch.tensor([row['SHAPE_E2']], device=device, dtype=torch.float32)))
    if 'dr10_south_21_EBV' in row.index and is_valid_value(row['dr10_south_21_EBV']):
        modalities.append(LegacySurveyEBV(value=torch.tensor([row['dr10_south_21_EBV']], device=device, dtype=torch.float32)))
    if 'dr10_south_21_FLUX_G' in row.index and is_valid_value(row['dr10_south_21_FLUX_G']):
        modalities.append(LegacySurveyFluxG(value=torch.tensor([row['dr10_south_21_FLUX_G']], device=device, dtype=torch.float32)))
    if 'dr10_south_21_FLUX_R' in row.index and is_valid_value(row['dr10_south_21_FLUX_R']):
        modalities.append(LegacySurveyFluxR(value=torch.tensor([row['dr10_south_21_FLUX_R']], device=device, dtype=torch.float32)))
    if 'dr10_south_21_FLUX_Z' in row.index and is_valid_value(row['dr10_south_21_FLUX_Z']):
        modalities.append(LegacySurveyFluxZ(value=torch.tensor([row['dr10_south_21_FLUX_Z']], device=device, dtype=torch.float32)))
    
    return modalities,other_modalities


def predict_redshift(model, codec_manager, modalities, device="cuda"):
    """预测单个样本的红移"""
    with torch.no_grad():
        try:
            # 确保modalities不为空
            if not modalities:
                raise ValueError("No modalities provided")
            
            # 编码所有模态，只使用有效的模态
            encoded = codec_manager.encode(*modalities)
            # 预测红移
            preds = model(encoded, target_modality=Z)
            
            # 获取概率分布
            pred_probs = torch.softmax(preds["tok_z"][0].squeeze(), 0).detach().cpu().numpy()
            
            # 获取最可能的token索引
            pred_token = np.argmax(pred_probs)
            
            return pred_probs, pred_token
        except Exception as e:
            print(f"Error in predict_redshift: {e}")
            # 返回默认值
            return np.array([1.0]), 0


def token_to_redshift(token, codec_manager, device="cuda"):
    """将token转换为红移值（近似）
    
    通过解码token获得红移值
    """
    try:
        # 创建一个包含该token的字典
        token_dict = {"tok_z": torch.tensor([[token]], device=device, dtype=torch.long)}
        
        # 解码token
        decoded = codec_manager.decode(token_dict, Z)
        
        return decoded.value.item()
    except:
        # 如果解码失败，返回一个估算值
        # AION的红移token通常在0-150范围，对应红移0-6
        return token / 25.0  # 粗略估算


def plot_redshift_prediction(
    pred_probs, 
    true_token=None, 
    pred_token=None,
    save_path=None,
    sample_idx=0,
):
    """绘制红移预测概率分布图"""
    plt.figure(figsize=(10, 6))
    
    plt.plot(pred_probs, color='blue', label='Predicted Probability')
    
    if true_token is not None:
        plt.axvline(true_token, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'True Redshift (token={true_token})')
    
    if pred_token is not None:
        plt.axvline(pred_token, color='green', linestyle=':', alpha=0.7, linewidth=2, label=f'Predicted Redshift (token={pred_token})')
    
    # 根据预测值和真实值的范围设置x轴范围
    if true_token is not None or pred_token is not None:
        # 收集所有相关的token值
        tokens = []
        if true_token is not None:
            tokens.append(true_token)
        if pred_token is not None:
            tokens.append(pred_token)
        # 计算最小值和最大值，添加缓冲区
        min_token = max(0, min(tokens) - 10)
        max_token = min(180, max(tokens) + 10)
        plt.xlim(min_token, max_token)
    else:
        # 如果没有真实值和预测值，使用默认范围
        plt.xlim(0, 150)
    plt.ylim(0, max(pred_probs.max() * 1.2, 0.1))
    plt.xlabel('Tokenized Redshift')
    plt.ylabel('Probability')
    plt.title(f'AION Redshift Prediction - Sample {sample_idx}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main(args):
    """主函数"""
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    
    print("="*80)
    print("AION 红移预测")
    print("="*80)
    print(f"输出目录: {output_dir}")
    
    # 设置设备并重置CUDA状态
    # 暂时强制使用CPU来避免CUDA错误
    device = torch.device("cuda")
    print(f"使用设备: {device}")
    
    # 完全重置CUDA状态
    if torch.cuda.is_available():
        print("正在重置CUDA状态...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        # 清理所有已分配的张量
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print(f"CUDA内存已清理，当前分配: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    
    # 加载模型
    print(f"\n正在加载模型: {args.model_name}")
    model = AION.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
    ).to(device)
    model.eval()
    
    # 如果提供了checkpoint，加载权重
    if args.checkpoint:
        print(f"正在加载checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Checkpoint加载成功")
    
    # 创建Codec Manager
    codec_manager = CodecManager(device=device)
    
    # 清理CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # 加载数据
    data = load_data(args.data_path, args.max_samples)
    
    # 预测结果存储
    results = []
    
    # 批量预测
    print(f"\n开始预测红移...")
    
    # 处理数据
    sample_idx = 0
    total_processed = 0
    
    # 检查返回的是单个文件数据还是文件列表
    if isinstance(data, pd.DataFrame):
        # 单个文件，按照原来的方式处理
        total_samples = len(data)
        plot_indices = set(random.sample(range(total_samples), min(args.num_plot_samples, total_samples)))
        
        for idx in tqdm(range(total_samples), desc="预测进度"):
            row = data.iloc[idx]
            
            # 每处理一定数量的样本清理CUDA缓存
            if idx > 0 and idx % 50 == 0:
                torch.cuda.empty_cache()
            
            # 准备模态数据
            modalities, other_modalities = prepare_modalities(row, device=device)
            
            if not modalities:
                print(f"警告: 样本 {idx} 没有有效的模态数据，跳过")
                continue
            
            # 预测红移
            try:
                pred_probs, pred_token = predict_redshift(model, codec_manager, modalities, device)
                
                # 获取真实红移token（如果存在）
                true_token = None
                true_redshift = None
                if other_modalities:
                    true_redshift = other_modalities[0].value.item()
                    # 编码真实红移获取token
                    true_z_obj = Z(value=torch.tensor([true_redshift], device=device, dtype=torch.float32))
                    true_token_dict = codec_manager.encode(true_z_obj)
                    true_token = true_token_dict["tok_z"].item()
                
                # 将token转换为红移值
                pred_redshift = token_to_redshift(pred_token, codec_manager, device)
                
                # 保存结果
                result = {
                    'sample_idx': sample_idx,
                    'pred_token': pred_token,
                    'pred_redshift': pred_redshift,
                    'true_token': true_token,
                    'true_redshift': true_redshift,
                    'pred_confidence': pred_probs[pred_token],
                }
                results.append(result)
                
                # 保存随机挑选的样本的预测图
                if idx in plot_indices:
                    plot_path = os.path.join(output_dir, "plots", f"sample_{sample_idx:04d}.png")
                    plot_redshift_prediction(
                        pred_probs,
                        true_token=true_token,
                        pred_token=pred_token,
                        save_path=plot_path,
                        sample_idx=sample_idx,
                    )
                
                sample_idx += 1
                total_processed += 1
                
            except Exception as e:
                print(f"警告: 样本 {idx} 预测失败: {str(e)}")
                # 出错时清理CUDA缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
    else:
        # 多个文件，逐个处理
        total_files = len(data)
        print(f"开始处理 {total_files} 个文件...")
        
        for file_idx, file_path in enumerate(tqdm(data, desc="文件处理进度")):
            try:
                # 读取当前文件
                file_data = pd.read_parquet(file_path)
                
                # 限制样本数量
                if args.max_samples and total_processed >= args.max_samples:
                    print(f"已达到最大样本数 {args.max_samples}，停止处理更多文件")
                    break
                
                # 计算当前文件可处理的样本数
                remaining_samples = args.max_samples - total_processed if args.max_samples else len(file_data)
                if remaining_samples <= 0:
                    break
                
                # 限制当前文件的样本数
                if remaining_samples < len(file_data):
                    file_data = file_data.iloc[:remaining_samples].reset_index(drop=True)
                
                print(f"处理文件 {file_idx+1}/{total_files}: {file_path}，共 {len(file_data)} 条数据")
                
                # 为当前文件生成随机样本索引列表，用于保存预测图
                file_plot_indices = set(random.sample(range(len(file_data)), min(args.num_plot_samples // total_files + 1, len(file_data))))
                
                # 处理当前文件的每个样本
                for row_idx, row in file_data.iterrows():
                    # 每处理一定数量的样本清理CUDA缓存
                    if total_processed > 0 and total_processed % 50 == 0:
                        torch.cuda.empty_cache()
                    
                    # 准备模态数据
                    modalities, other_modalities = prepare_modalities(row, device=device)
                    
                    if not modalities:
                        print(f"警告: 文件 {file_idx+1} 中的样本 {row_idx} 没有有效的模态数据，跳过")
                        continue
                    
                    # 预测红移
                    try:
                        pred_probs, pred_token = predict_redshift(model, codec_manager, modalities, device)
                        
                        # 获取真实红移token（如果存在）
                        true_token = None
                        true_redshift = None
                        if other_modalities:
                            true_redshift = other_modalities[0].value.item()
                            # 编码真实红移获取token
                            true_z_obj = Z(value=torch.tensor([true_redshift], device=device, dtype=torch.float32))
                            true_token_dict = codec_manager.encode(true_z_obj)
                            true_token = true_token_dict["tok_z"].item()
                        
                        # 将token转换为红移值
                        pred_redshift = token_to_redshift(pred_token, codec_manager, device)
                        
                        # 保存结果
                        result = {
                            'sample_idx': sample_idx,
                            'file_idx': file_idx,
                            'file_path': file_path,
                            'pred_token': pred_token,
                            'pred_redshift': pred_redshift,
                            'true_token': true_token,
                            'true_redshift': true_redshift,
                            'pred_confidence': pred_probs[pred_token],
                        }
                        results.append(result)
                        
                        # 保存随机挑选的样本的预测图
                        if row_idx in file_plot_indices:
                            plot_path = os.path.join(output_dir, "plots", f"sample_{sample_idx:04d}.png")
                            plot_redshift_prediction(
                                pred_probs,
                                true_token=true_token,
                                pred_token=pred_token,
                                save_path=plot_path,
                                sample_idx=sample_idx,
                            )
                        
                        sample_idx += 1
                        total_processed += 1
                        
                        # 检查是否达到最大样本数
                        if args.max_samples and total_processed >= args.max_samples:
                            print(f"已达到最大样本数 {args.max_samples}，停止处理")
                            break
                        
                    except Exception as e:
                        print(f"警告: 文件 {file_idx+1} 中的样本 {row_idx} 预测失败: {str(e)}")
                        # 出错时清理CUDA缓存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    
            except Exception as e:
                print(f"警告: 处理文件 {file_path} 失败: {str(e)}")
                continue

    # 保存预测结果
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(output_dir, "redshift_predictions.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\n预测结果已保存到: {results_csv_path}")
    
    # 计算统计指标
    if 'true_redshift' in results_df.columns and results_df['true_redshift'].notna().any():
        valid_results = results_df[results_df['true_redshift'].notna()]
        
        # 计算误差
        errors = np.abs(valid_results['pred_redshift'] - valid_results['true_redshift'])
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(errors**2))
        
        print("\n" + "="*80)
        print("预测统计")
        print("="*80)
        print(f"有效预测数量: {len(valid_results)}/{len(results_df)}")
        print(f"平均绝对误差 (MAE): {mae:.4f}")
        print(f"均方根误差 (RMSE): {rmse:.4f}")
        
        # 绘制预测vs真实值散点图
        plt.figure(figsize=(10, 10))
        plt.scatter(valid_results['true_redshift'], valid_results['pred_redshift'], alpha=0.5, s=20)
        
        # 绘制y=x参考线
        min_z = min(valid_results['true_redshift'].min(), valid_results['pred_redshift'].min())
        max_z = max(valid_results['true_redshift'].max(), valid_results['pred_redshift'].max())
        plt.plot([min_z, max_z], [min_z, max_z], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('True Redshift')
        plt.ylabel('Predicted Redshift')
        plt.title(f'Redshift Prediction vs True Values\nMAE={mae:.4f}, RMSE={rmse:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        scatter_path = os.path.join(output_dir, "redshift_scatter.png")
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"散点图已保存到: {scatter_path}")
        
        # 绘制误差分布直方图
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.title(f'Redshift Prediction Error Distribution\nMAE={mae:.4f}')
        plt.grid(True, alpha=0.3)
        
        error_hist_path = os.path.join(output_dir, "error_distribution.png")
        plt.savefig(error_hist_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"误差分布图已保存到: {error_hist_path}")
        
        # 保存统计信息
        stats_path = os.path.join(output_dir, "statistics.txt")
        with open(stats_path, 'w') as f:
            f.write(f"红移预测统计信息\n")
            f.write(f"{'='*80}\n")
            f.write(f"数据路径: {args.data_path}\n")
            f.write(f"模型: {args.model_name}\n")
            if args.checkpoint:
                f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"预测时间: {timestamp}\n")
            f.write(f"\n总样本数: {len(data)}\n")
            f.write(f"成功预测数: {len(results_df)}\n")
            f.write(f"有真实值的样本数: {len(valid_results)}\n")
            f.write(f"\n平均绝对误差 (MAE): {mae:.6f}\n")
            f.write(f"均方根误差 (RMSE): {rmse:.6f}\n")
            f.write(f"最大误差: {errors.max():.6f}\n")
            f.write(f"最小误差: {errors.min():.6f}\n")
            f.write(f"中位数误差: {np.median(errors):.6f}\n")
        print(f"统计信息已保存到: {stats_path}")
    
    print("\n" + "="*80)
    print("预测完成!")
    print("="*80)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="AION 红移预测脚本")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="polymathic-ai/aion-base",
                        help="预训练模型名称或路径")
    parser.add_argument("--cache_dir", type=str,
                        default="weight/aion-base/",
                        help="模型缓存目录")
    parser.add_argument("--checkpoint", type=str, 
                        default="data/best_model.pt",
                        # default=None,
                        help="模型checkpoint路径（可选）")
    
    # 数据参数
    parser.add_argument("--data_path", type=str, required=False,
                        default="data/val/",
                        help="输入数据路径（Parquet文件或包含Parquet文件的文件夹）")
    parser.add_argument("--max_samples", type=int, default=2000,
                        help="最大预测样本数")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="outputs/redshift_predictions_with_spectrum_ls_desi",
                        help="输出目录")
    parser.add_argument("--num_plot_samples", type=int, default=20,
                        help="保存预测图的样本数量")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
