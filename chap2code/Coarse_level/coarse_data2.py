import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
import numpy as np
from tqdm import tqdm
import logging
import psutil

class CoarseGrainedDataset(Dataset):
    """粗粒度漏洞检测数据集"""
    def __init__(self, data_dir, func_tokenizer, slice_tokenizer, max_length=512, is_train=True, pos_neg_ratio=2.0, max_slices_per_func=10, use_cache=True):
        self.data_dir = data_dir
        self.func_tokenizer = func_tokenizer
        self.slice_tokenizer = slice_tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.pos_neg_ratio = pos_neg_ratio  # 正负样本比例
        self.max_slices_per_func = max_slices_per_func  # 每个函数的最大切片数量
        self.use_cache = use_cache  # 是否使用缓存
        
        self.func_codes = []
        self.func_labels = []
        self.func_slices = []  # 每个函数对应的切片列表
        self.func_slice_labels = []  # 每个函数对应的切片标签列表
        
        self._load_data()
        
    def _process_file(self, file):
        """处理单个文件"""
        file_path = os.path.join(self.data_dir, file)
        
        # 从文件名获取函数级标签 (0 or 1)
        filename_no_ext = file[:-4]
        func_label = int(filename_no_ext.split('_')[0])
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 提取函数原代码
        func_code = []
        for line in lines:
            if line.startswith("Original Code"):
                continue
            if line.startswith("—"):
                break
            if ': ' in line:  # 处理带行号的代码行
                try:
                    _, code_content = line.split(': ', 1)
                    func_code.append(code_content.strip())
                except:
                    continue
        
        if not func_code:
            return None
            
        # 提取切片代码
        current_slice = []
        slice_header = ""
        slice_codes = []
        slice_labels = []
        slice_lengths = []  # 记录每个切片的长度
        
        for line in lines:
            if line.startswith('[Slice'):
                if current_slice:  # 保存前一个切片
                    # 确定当前切片标签
                    slice_label = 1 if "=> vul" in slice_header else 0
                    
                    # 只保留漏洞切片或长度大于2行的非漏洞切片
                    if slice_label == 1 or len(current_slice) > 2:
                        slice_code = '\n'.join(current_slice)
                        slice_codes.append(slice_code)
                        slice_labels.append(slice_label)
                        slice_lengths.append(len(current_slice))  # 记录切片长度
                    
                    current_slice = []
                slice_header = line
            elif not line.startswith('—') and not line.startswith('Original Code'):
                current_slice.append(line.strip())
        
        # 保存最后一个切片
        if current_slice:
            slice_label = 1 if "=> vul" in slice_header else 0
            if slice_label == 1 or len(current_slice) > 2:
                slice_code = '\n'.join(current_slice)
                slice_codes.append(slice_code)
                slice_labels.append(slice_label)
                slice_lengths.append(len(current_slice))  # 记录切片长度
        
        # 如果没有切片，创建一个空切片
        if not slice_codes:
            # 确保函数代码是字符串而非列表
            func_code_str = '\n'.join(func_code) if isinstance(func_code, list) else func_code
            slice_codes.append(func_code_str)
            slice_labels.append(func_label)
            # 计算行数时确保是字符串
            slice_lengths.append(len(func_code_str.split('\n')))
        
        # 如果切片数量超过限制，选择长度最长的max_slices_per_func个切片
        if len(slice_codes) > self.max_slices_per_func:
            # 将切片索引和长度组合在一起
            slice_length_pairs = list(zip(range(len(slice_codes)), slice_lengths))
            # 按长度排序（降序）
            slice_length_pairs.sort(key=lambda x: x[1], reverse=True)
            # 只保留前max_slices_per_func个切片
            selected_indices = [idx for idx, _ in slice_length_pairs[:self.max_slices_per_func]]
            slice_codes = [slice_codes[i] for i in selected_indices]
            slice_labels = [slice_labels[i] for i in selected_indices]
        
        # 添加函数代码和标签
        func_code_str = '\n'.join(func_code) if isinstance(func_code, list) else func_code
        self.func_codes.append(func_code_str)
        self.func_labels.append(func_label)
        self.func_slices.append(slice_codes)
        self.func_slice_labels.append(slice_labels)
        
        return len(self.func_codes) - 1
        
    def _load_data(self):
        """加载数据"""
        logging.info(f"正在加载{'训练' if self.is_train else '验证/测试'}数据...")
        
        # 检查是否存在缓存文件
        cache_dir = os.path.join(os.path.dirname(self.data_dir), 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        # 生成缓存文件名
        cache_name = f"{os.path.basename(self.data_dir)}_{self.pos_neg_ratio}_{self.max_slices_per_func}.pkl"
        cache_path = os.path.join(cache_dir, cache_name)
        
        # 如果是训练集且存在缓存文件，则直接加载
        if self.is_train and self.use_cache and os.path.exists(cache_path):
            logging.info(f"从缓存加载数据: {cache_path}")
            import pickle
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                self.func_codes = cache_data['func_codes']
                self.func_labels = cache_data['func_labels']
                self.func_slices = cache_data['func_slices']
                self.func_slice_labels = cache_data['func_slice_labels']
            logging.info(f"缓存加载完成: {len(self.func_codes)}个函数")
            return
        
        # 获取所有函数文件
        func_files = [f for f in os.listdir(self.data_dir) if f.endswith('.txt')]
        
        # 处理每个文件
        for file in tqdm(func_files, desc="加载函数和切片"):
            try:
                self._process_file(file)
            except Exception as e:
                logging.error(f"处理文件 {file} 时出错: {e}")
        
        # 根据正负样本比例选择样本
        positive_indices = [i for i, label in enumerate(self.func_labels) if label == 1]
        negative_indices = [i for i, label in enumerate(self.func_labels) if label == 0]
        
        # 计算应该保留的负样本数量
        num_negative = min(len(negative_indices), int(len(positive_indices) * self.pos_neg_ratio))
        
        # 随机选择负样本
        selected_negative = np.random.choice(negative_indices, num_negative, replace=False)
        
        # 合并正负样本索引
        selected_indices = list(positive_indices) + list(selected_negative)
        
        # 更新数据
        self.func_codes = [self.func_codes[i] for i in selected_indices]
        self.func_labels = [self.func_labels[i] for i in selected_indices]
        self.func_slices = [self.func_slices[i] for i in selected_indices]
        self.func_slice_labels = [self.func_slice_labels[i] for i in selected_indices]
        
        # 保存缓存
        if self.is_train and self.use_cache:
            logging.info(f"保存数据到缓存: {cache_path}")
            import pickle
            cache_data = {
                'func_codes': self.func_codes,
                'func_labels': self.func_labels,
                'func_slices': self.func_slices,
                'func_slice_labels': self.func_slice_labels
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        
        logging.info(f"加载完成: {len(self.func_codes)}个函数")
        logging.info(f"正样本: {len(positive_indices)}, 负样本: {len(selected_negative)}")
        
        # 统计标签分布
        func_label_counts = {0: 0, 1: 0}
        for label in self.func_labels:
            func_label_counts[label] += 1
            
        logging.info(f"函数标签分布: 非漏洞={func_label_counts[0]}, 漏洞={func_label_counts[1]}")
    
    def __len__(self):
        return len(self.func_codes)
    
    def get_label_distribution(self, level='func'):
        """获取标签分布
        
        Args:
            level: 'func' 或 'slice'，表示获取函数级还是切片级的标签分布
        """
        if level == 'func':
            # 计算函数级标签分布
            label_counts = {0: 0, 1: 0}
            for label in self.func_labels:
                label_counts[label] += 1
            return label_counts
        elif level == 'slice':
            # 计算切片级标签分布
            label_counts = {0: 0, 1: 0}
            for slice_labels in self.func_slice_labels:
                for label in slice_labels:
                    label_counts[label] += 1
            return label_counts
        else:
            raise ValueError(f"不支持的级别: {level}")
            
    def __getitem__(self, idx):
        # 获取函数代码和标签
        func_code = self.func_codes[idx]
        func_label = self.func_labels[idx]
        
        # 获取函数对应的切片
        slice_codes = self.func_slices[idx]
        slice_labels = self.func_slice_labels[idx]
        
        # 对函数代码进行编码
        func_encoded = self.func_tokenizer.encode_plus(
            func_code,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # 记录实际切片数量
        actual_slices = len(slice_codes)
        
        # 确保切片数量不超过最大限制
        if len(slice_codes) > self.max_slices_per_func:
            slice_codes = slice_codes[:self.max_slices_per_func]
            slice_labels = slice_labels[:self.max_slices_per_func]
            actual_slices = self.max_slices_per_func
        
        # 对所有切片一次性进行编码
        slice_encoded = self.slice_tokenizer.batch_encode_plus(
            slice_codes,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # 如果切片数量不足，用零张量填充到指定数量
        if actual_slices < self.max_slices_per_func:
            # 创建填充张量
            pad_input_ids = torch.zeros(
                (self.max_slices_per_func - actual_slices, self.max_length), 
                dtype=torch.long
            )
            pad_attention_mask = torch.zeros(
                (self.max_slices_per_func - actual_slices, self.max_length), 
                dtype=torch.long
            )
            
            # 拼接原始张量和填充张量
            slice_input_ids = torch.cat([slice_encoded['input_ids'], pad_input_ids], dim=0)
            slice_attention_mask = torch.cat([slice_encoded['attention_mask'], pad_attention_mask], dim=0)
            
            # 填充标签
            slice_labels.extend([0] * (self.max_slices_per_func - actual_slices))
        else:
            slice_input_ids = slice_encoded['input_ids']
            slice_attention_mask = slice_encoded['attention_mask']
        
        # 创建切片索引，确保长度和填充后的切片数量一致
        slice_indices = torch.zeros(self.max_slices_per_func, dtype=torch.long)
        slice_indices[:actual_slices] = torch.arange(actual_slices)
        
        return {
            'func_input_ids': func_encoded['input_ids'].squeeze(0),
            'func_attention_mask': func_encoded['attention_mask'].squeeze(0),
            'func_label': torch.tensor(func_label),
            'slice_input_ids': slice_input_ids,
            'slice_attention_mask': slice_attention_mask,
            'slice_labels': torch.tensor(slice_labels),
            'slice_indices': slice_indices
        }

def custom_collate_fn(batch):
    """自定义的collate函数，确保切片输入是三维的"""
    # 函数级张量直接堆叠
    func_input_ids = torch.stack([item['func_input_ids'] for item in batch])
    func_attention_mask = torch.stack([item['func_attention_mask'] for item in batch])
    func_label = torch.stack([item['func_label'] for item in batch])
    
    # 切片级张量直接堆叠 - 它们已经是正确的形状 [batch_size, max_slices_per_func, seq_len]
    slice_input_ids = torch.stack([item['slice_input_ids'] for item in batch])
    slice_attention_mask = torch.stack([item['slice_attention_mask'] for item in batch])
    slice_labels = torch.stack([item['slice_labels'] for item in batch])
    slice_indices = torch.stack([item['slice_indices'] for item in batch])
    
    return {
        'func_input_ids': func_input_ids,
        'func_attention_mask': func_attention_mask,
        'func_label': func_label,
        'slice_input_ids': slice_input_ids,
        'slice_attention_mask': slice_attention_mask,
        'slice_labels': slice_labels,
        'slice_indices': slice_indices
    }

def get_optimal_num_workers():
    """获取最优的num_workers数量"""
    # 获取CPU核心数
    cpu_count = os.cpu_count()
    
    # 获取可用内存（GB）
    available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)
    
    # 计算建议的worker数量
    # 基于CPU核心数：取1/4到1/2
    cpu_based = max(1, cpu_count // 4)
    
    # 基于内存：每2GB内存分配一个worker
    memory_based = max(1, int(available_memory // 2))
    
    # 取两者中的较小值
    optimal_workers = min(cpu_based, memory_based)
    # print(f"建议的num_workers: {optimal_workers} (CPU: {cpu_based}, 内存: {memory_based})")
    
    return optimal_workers

def create_coarse_dataloaders(data_root, func_tokenizer, slice_tokenizer, batch_size=8, max_length=512, num_workers=None, pos_neg_ratio=2.0, max_slices_per_func=10, use_cache=True, pin_memory=True, prefetch_factor=2):
    """创建粗粒度数据加载器"""
    # 如果没有指定num_workers，自动计算最优值
    if num_workers is None:
        num_workers = get_optimal_num_workers()
    
    # 创建数据集
    train_dataset = CoarseGrainedDataset(
        os.path.join(data_root, 'train'),
        func_tokenizer,
        slice_tokenizer,
        max_length=max_length,
        is_train=True,
        pos_neg_ratio=pos_neg_ratio,
        max_slices_per_func=max_slices_per_func,
        use_cache=use_cache
    )
    
    val_dataset = CoarseGrainedDataset(
        os.path.join(data_root, 'val'),
        func_tokenizer,
        slice_tokenizer,
        max_length=max_length,
        is_train=False,
        max_slices_per_func=max_slices_per_func,
        use_cache=use_cache
    )
    
    test_dataset = CoarseGrainedDataset(
        os.path.join(data_root, 'test'),
        func_tokenizer,
        slice_tokenizer,
        max_length=max_length,
        is_train=False,
        max_slices_per_func=max_slices_per_func,
        use_cache=use_cache
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader, test_loader