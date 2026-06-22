import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
import numpy as np
from tqdm import tqdm
import logging
import json
import re

class FineGrainedDataset(Dataset):
    """细粒度漏洞检测数据集
    从每个文件的第一行"Original Code => [行号]"部分识别真正的漏洞行，
    标记对应行号的代码为漏洞行(1)，其他行为非漏洞行(0)。
    """
    def __init__(self, data_dir, tokenizer, max_length=512, is_train=True):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        
        self.line_codes = []
        self.line_labels = []
        self.line_contexts = []  # 存储每行代码的上下文信息
        self.func_ids = []  # 存储每行代码对应的函数ID
        self.slice_ids = []  # 存储每行代码对应的切片ID
        self.line_numbers = []  # 存储每行代码的行号
        
        self._load_data()
        
    def _load_data(self):
        """加载数据"""
        logging.info(f"正在加载{'训练' if self.is_train else '验证/测试'}数据...")
        
        # 获取所有行级数据文件
        line_files = [f for f in os.listdir(self.data_dir) if f.endswith('_contexts.txt')]
        
        for file in tqdm(line_files, desc="加载行级数据"):
            # 从文件名中提取标签和ID信息
            parts = file.split('_')
            label = int(parts[0])
            func_id = parts[1]
            slice_id = parts[2] if len(parts) > 2 else None
            
            # 只处理函数级标签为1的样本
            if label != 1:
                continue
                
            file_path = os.path.join(self.data_dir, file)
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析文件内容
            sections = content.split('==================================================')
            
            # 提取原始代码和漏洞行号
            original_code_section = sections[0].strip()
            
            # 解析漏洞行号
            first_line = original_code_section.split('\n')[0]
            if first_line.startswith('Original Code => '):
                try:
                    # 提取漏洞行号列表
                    vuln_lines_str = first_line.replace('Original Code => ', '')
                    vuln_lines = eval(vuln_lines_str)  # 将字符串转换为列表
                    
                    # 如果漏洞行为空，跳过此样本
                    if not vuln_lines:
                        # logging.info(f"跳过漏洞行为空的样本: {file}")
                        continue
                except:
                    logging.warning(f"无法解析漏洞行: {first_line} in {file}")
                    continue
            else:
                logging.warning(f"文件格式不正确: {file}")
                continue
            
            # 提取原始代码行
            code_lines = original_code_section.split('\n')[1:]
            
            # 处理每一行及其上下文
            line_number = 1  # 从1开始计数
            for line_info in sections[1:]:
                if not line_info.strip():
                    continue
                    
                lines = line_info.strip().split('\n')
                if len(lines) < 2:
                    continue
                
                # 提取行号和代码
                line_header = lines[0]
                if not line_header.startswith('Line '):
                    continue
                    
                try:
                    # 提取行号
                    line_number = int(line_header.split(':')[0].replace('Line ', ''))
                    line_code = lines[0].split(': ', 1)[1] if len(lines[0].split(': ', 1)) > 1 else ""
                except:
                    logging.warning(f"无法解析行号: {line_header} in {file}")
                    continue
                
                # 确定是否为漏洞行
                is_vuln_line = line_number in vuln_lines
                
                # 提取上下文信息
                operation_context = []
                dependence_context = []
                surrounding_context = []
                
                current_context = None
                for line in lines[1:]:
                    if line.startswith('Operation Context:'):
                        current_context = operation_context
                    elif line.startswith('Dependence Context:'):
                        current_context = dependence_context
                    elif line.startswith('Surrounding Context:'):
                        current_context = surrounding_context
                    elif line.strip() and current_context is not None:
                        current_context.append(line.strip())
                
                # 合并上下文信息
                context = {
                    'operation': '\n'.join(operation_context),
                    'dependence': '\n'.join(dependence_context),
                    'surrounding': '\n'.join(surrounding_context)
                }
                
                # 添加到数据集
                self.line_codes.append(line_code)
                self.line_labels.append(1 if is_vuln_line else 0)  # 根据实际漏洞行设置标签
                self.line_contexts.append(context)
                self.func_ids.append(func_id)
                self.slice_ids.append(slice_id)
                self.line_numbers.append(line_number)
        
        logging.info(f"加载完成: {len(self.line_codes)}行代码")
        
        # 统计标签分布
        line_label_counts = {0: 0, 1: 0}
        for label in self.line_labels:
            line_label_counts[label] += 1
            
        logging.info(f"行标签分布: 非漏洞={line_label_counts[0]}, 漏洞={line_label_counts[1]}")
    
    def __len__(self):
        return len(self.line_codes)
    
    def __getitem__(self, idx):
        # 获取行代码和标签
        line_code = self.line_codes[idx]
        line_label = self.line_labels[idx]
        context = self.line_contexts[idx]
        func_id = self.func_ids[idx]
        slice_id = self.slice_ids[idx]
        line_number = self.line_numbers[idx]
        
        # 构造输入文本 - 使用CodeBERT的格式
        input_text = f"{line_code} [SEP] Operation Context: {context['operation']} [SEP] Dependence Context: {context['dependence']} [SEP] Surrounding Context: {context['surrounding']}"
        
        # 编码
        encoded = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # 保留行号作为内部信息，但不在输入文本中使用
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': torch.tensor(line_label, dtype=torch.float),  # float用于BCELoss
            'func_id': func_id,
            'slice_id': slice_id,
            'line_number': line_number  # 保留行号信息用于评估时按函数聚合
        }
    
    def get_label_distribution(self):
        """返回数据集标签分布统计"""
        line_label_counts = {0: 0, 1: 0}
        for label in self.line_labels:
            line_label_counts[label] += 1
            
        return {
            'line': line_label_counts
        }

def collate_fn(batch):
    """自定义批处理函数"""
    result = {}
    for key in batch[0].keys():
        if key in ['func_id', 'slice_id', 'line_number']:
            result[key] = [item[key] for item in batch]
        else:
            result[key] = torch.stack([item[key] for item in batch])

    return result

def create_fine_dataloaders(data_root, tokenizer, batch_size=8, max_length=512, num_workers=4, pin_memory=True, prefetch_factor=4):
    """创建细粒度数据加载器"""
    # 当num_workers=0时，prefetch_factor必须为None
    if num_workers == 0:
        prefetch_factor = None

    # 创建数据集
    train_dataset = FineGrainedDataset(
        os.path.join(data_root, 'train'),
        tokenizer,
        max_length=max_length,
        is_train=True
    )

    val_dataset = FineGrainedDataset(
        os.path.join(data_root, 'val'),
        tokenizer,
        max_length=max_length,
        is_train=False
    )
    
    test_dataset = FineGrainedDataset(
        os.path.join(data_root, 'test'),
        tokenizer,
        max_length=max_length,
        is_train=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader


class FineGrainedDatasetWithCoarse(Dataset):
    """带粗粒度特征缓存的细粒度漏洞检测数据集

    用于交互模式，加载预先生成的粗粒度模型特征缓存。
    """
    def __init__(self, data_dir, coarse_cache_dir, tokenizer, max_length=512, is_train=True, max_slices=5):
        self.data_dir = data_dir
        self.coarse_cache_dir = coarse_cache_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.max_slices = max_slices  # 最大切片数量

        self.line_codes = []
        self.line_labels = []
        self.line_contexts = []
        self.func_file_ids = []  # 完整的函数文件ID (用于查找缓存)
        self.line_numbers = []

        self._load_data()

    def _load_data(self):
        """加载数据"""
        logging.info(f"正在加载{'训练' if self.is_train else '验证/测试'}数据 (交互模式)...")

        # 获取所有行级数据文件
        line_files = [f for f in os.listdir(self.data_dir) if f.endswith('_contexts.txt')]

        for file in tqdm(line_files, desc="加载行级数据"):
            # 从文件名中提取标签
            parts = file.split('_')
            label = int(parts[0])

            # 只处理函数级标签为1的样本
            if label != 1:
                continue

            # 提取函数文件ID (去掉 _contexts.txt 后缀)
            func_file_id = file.replace('_contexts.txt', '')

            # 检查缓存文件是否存在
            cache_path = os.path.join(self.coarse_cache_dir, f"{func_file_id}.pt")
            if not os.path.exists(cache_path):
                logging.warning(f"缓存文件不存在，跳过: {cache_path}")
                continue

            file_path = os.path.join(self.data_dir, file)

            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 解析文件内容
            sections = content.split('==================================================')

            # 提取原始代码和漏洞行号
            original_code_section = sections[0].strip()

            # 解析漏洞行号
            first_line = original_code_section.split('\n')[0]
            if first_line.startswith('Original Code => '):
                try:
                    vuln_lines_str = first_line.replace('Original Code => ', '')
                    vuln_lines = eval(vuln_lines_str)

                    if not vuln_lines:
                        continue
                except:
                    logging.warning(f"无法解析漏洞行: {first_line} in {file}")
                    continue
            else:
                logging.warning(f"文件格式不正确: {file}")
                continue

            # 处理每一行及其上下文
            for line_info in sections[1:]:
                if not line_info.strip():
                    continue

                lines = line_info.strip().split('\n')
                if len(lines) < 2:
                    continue

                # 提取行号和代码
                line_header = lines[0]
                if not line_header.startswith('Line '):
                    continue

                try:
                    line_number = int(line_header.split(':')[0].replace('Line ', ''))
                    line_code = lines[0].split(': ', 1)[1] if len(lines[0].split(': ', 1)) > 1 else ""
                except:
                    logging.warning(f"无法解析行号: {line_header} in {file}")
                    continue

                # 确定是否为漏洞行
                is_vuln_line = line_number in vuln_lines

                # 提取上下文信息
                operation_context = []
                dependence_context = []
                surrounding_context = []

                current_context = None
                for line in lines[1:]:
                    if line.startswith('Operation Context:'):
                        current_context = operation_context
                    elif line.startswith('Dependence Context:'):
                        current_context = dependence_context
                    elif line.startswith('Surrounding Context:'):
                        current_context = surrounding_context
                    elif line.strip() and current_context is not None:
                        current_context.append(line.strip())

                context = {
                    'operation': '\n'.join(operation_context),
                    'dependence': '\n'.join(dependence_context),
                    'surrounding': '\n'.join(surrounding_context)
                }

                # 添加到数据集
                self.line_codes.append(line_code)
                self.line_labels.append(1 if is_vuln_line else 0)
                self.line_contexts.append(context)
                self.func_file_ids.append(func_file_id)
                self.line_numbers.append(line_number)

        logging.info(f"加载完成: {len(self.line_codes)}行代码")

        # 统计标签分布
        line_label_counts = {0: 0, 1: 0}
        for label in self.line_labels:
            line_label_counts[label] += 1

        logging.info(f"行标签分布: 非漏洞={line_label_counts[0]}, 漏洞={line_label_counts[1]}")

    def __len__(self):
        return len(self.line_codes)

    def __getitem__(self, idx):
        # 获取行代码和标签
        line_code = self.line_codes[idx]
        line_label = self.line_labels[idx]
        context = self.line_contexts[idx]
        func_file_id = self.func_file_ids[idx]
        line_number = self.line_numbers[idx]

        # 构造输入文本
        input_text = f"{line_code} [SEP] Operation Context: {context['operation']} [SEP] Dependence Context: {context['dependence']} [SEP] Surrounding Context: {context['surrounding']}"

        # 编码
        encoded = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # 加载粗粒度缓存
        cache_path = os.path.join(self.coarse_cache_dir, f"{func_file_id}.pt")
        cache_data = torch.load(cache_path, map_location='cpu')

        func_prob = torch.tensor(cache_data['func_prob'], dtype=torch.float)
        func_features = cache_data['func_features']  # [hidden_dim]

        # 处理切片特征和切片-行对齐
        slices = cache_data['slices']
        num_slices = min(len(slices), self.max_slices)
        hidden_dim = func_features.size(0)

        # 初始化切片特征和概率
        slice_features = torch.zeros(self.max_slices, hidden_dim)
        slice_probs = torch.zeros(self.max_slices)
        slice_line_mask = torch.zeros(self.max_slices)  # 标记哪些切片包含当前行

        for i, slice_info in enumerate(slices[:num_slices]):
            slice_features[i] = slice_info['slice_features']
            slice_probs[i] = slice_info['slice_prob']

            # 检查当前行是否在该切片中
            if line_number in slice_info['line_numbers']:
                slice_line_mask[i] = 1.0

        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': torch.tensor(line_label, dtype=torch.float),
            'func_file_id': func_file_id,
            'line_number': line_number,
            # 粗粒度特征
            'func_prob': func_prob,
            'func_features': func_features,
            'slice_probs': slice_probs,
            'slice_features': slice_features,
            'slice_line_mask': slice_line_mask  # 切片-行对齐掩码
        }

    def get_label_distribution(self):
        """返回数据集标签分布统计"""
        line_label_counts = {0: 0, 1: 0}
        for label in self.line_labels:
            line_label_counts[label] += 1

        return {
            'line': line_label_counts
        }


def collate_fn_with_coarse(batch):
    """带粗粒度特征的批处理函数"""
    result = {}
    for key in batch[0].keys():
        if key in ['func_file_id', 'line_number']:
            result[key] = [item[key] for item in batch]
        else:
            result[key] = torch.stack([item[key] for item in batch])

    return result


def create_fine_dataloaders_with_coarse(data_root, coarse_cache_dir, tokenizer, batch_size=8, max_length=512, num_workers=4, pin_memory=True, prefetch_factor=4, max_slices=5):
    """创建带粗粒度特征的细粒度数据加载器"""
    # 当num_workers=0时，prefetch_factor必须为None
    if num_workers == 0:
        prefetch_factor = None

    # 创建数据集
    train_dataset = FineGrainedDatasetWithCoarse(
        os.path.join(data_root, 'train'),
        os.path.join(coarse_cache_dir, 'train'),
        tokenizer,
        max_length=max_length,
        is_train=True,
        max_slices=max_slices
    )

    val_dataset = FineGrainedDatasetWithCoarse(
        os.path.join(data_root, 'val'),
        os.path.join(coarse_cache_dir, 'val'),
        tokenizer,
        max_length=max_length,
        is_train=False,
        max_slices=max_slices
    )

    test_dataset = FineGrainedDatasetWithCoarse(
        os.path.join(data_root, 'test'),
        os.path.join(coarse_cache_dir, 'test'),
        tokenizer,
        max_length=max_length,
        is_train=False,
        max_slices=max_slices
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn_with_coarse
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn_with_coarse
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn_with_coarse
    )

    return train_loader, val_loader, test_loader