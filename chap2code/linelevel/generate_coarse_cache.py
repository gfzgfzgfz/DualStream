"""
粗粒度模型特征缓存生成脚本

功能：运行粗粒度模型，为每个函数生成并保存特征缓存，供细粒度模型交互模式使用。

缓存结构 (每个函数一个 .pt 文件):
{
    'func_id': str,                    # 函数文件ID (不含扩展名)
    'func_prob': float,                # 函数级漏洞概率
    'func_features': Tensor[768],      # 函数特征向量
    'slices': [                        # 切片列表
        {
            'slice_idx': int,          # 切片索引
            'slice_prob': float,       # 切片漏洞概率
            'slice_features': Tensor[768],  # 切片特征向量
            'line_numbers': List[int]  # 该切片包含的行号列表
        },
        ...
    ]
}
"""

import os
import sys
import re
import torch
import argparse
import logging
from tqdm import tqdm
from transformers import RobertaTokenizer

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Coarse_level.coarse_model import CoarseGrainedModel
from Coarse_level.coarse_data import CoarseGrainedDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def extract_slice_line_numbers(slice_text):
    """从切片文本中提取行号列表

    切片格式示例:
    status_t SampleTable::setCompositionTimeToSampleParams( #1
    size_t numEntries = U32_AT(&header[4]); #13

    返回: [1, 13, ...]
    """
    line_numbers = []
    # 匹配行末的 #数字 模式
    pattern = r'#(\d+)\s*$'

    for line in slice_text.split('\n'):
        match = re.search(pattern, line.strip())
        if match:
            line_numbers.append(int(match.group(1)))

    return line_numbers


def parse_coarse_file(file_path):
    """解析粗粒度数据文件，提取切片及其行号信息

    返回: {
        'func_code': str,
        'slices': [
            {'code': str, 'label': int, 'line_numbers': List[int]},
            ...
        ]
    }
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 分割原始代码和切片部分
    parts = content.split('————————————————————————————')

    # 提取函数代码
    func_code_section = parts[0].strip()
    func_lines = []
    for line in func_code_section.split('\n'):
        if line.startswith('Original Code'):
            continue
        if ': ' in line:
            try:
                _, code_content = line.split(': ', 1)
                func_lines.append(code_content.strip())
            except:
                continue
    func_code = '\n'.join(func_lines)

    # 提取切片信息
    slices = []
    for part in parts[1:]:
        part = part.strip()
        if not part:
            continue

        lines = part.split('\n')
        if not lines:
            continue

        # 解析切片头部
        header = lines[0]
        if not header.startswith('[Slice'):
            continue

        # 确定切片标签
        slice_label = 1 if '=> vul' in header else 0

        # 提取切片代码
        slice_code_lines = lines[1:]
        slice_code = '\n'.join(slice_code_lines)

        # 提取行号
        line_numbers = extract_slice_line_numbers(slice_code)

        slices.append({
            'code': slice_code,
            'label': slice_label,
            'line_numbers': line_numbers
        })

    return {
        'func_code': func_code,
        'slices': slices
    }


def generate_cache_for_split(
    coarse_data_dir,
    fine_data_dir,
    coarse_model,
    func_tokenizer,
    slice_tokenizer,
    output_dir,
    split,
    device,
    max_length=512,
    max_slices_per_func=10
):
    """为指定数据集分割生成缓存

    只为细粒度数据集中存在的函数生成缓存（通过文件名匹配）
    """

    coarse_split_dir = os.path.join(coarse_data_dir, split)
    fine_split_dir = os.path.join(fine_data_dir, split)
    split_output_dir = os.path.join(output_dir, split)
    os.makedirs(split_output_dir, exist_ok=True)

    # 获取细粒度数据集中的函数ID列表（只处理漏洞函数，即label=1的样本）
    fine_files = [f for f in os.listdir(fine_split_dir) if f.endswith('_contexts.txt')]
    fine_func_ids = set()
    for f in fine_files:
        # 细粒度文件名: {label}_{project}_{CVE}_{commit}_{idx}_contexts.txt
        # 对应粗粒度文件: {label}_{project}_{CVE}_{commit}_{idx}.txt
        # 只处理label=1的漏洞函数（与fine_data.py保持一致）
        if not f.startswith('1_'):
            continue
        func_id = f.replace('_contexts.txt', '')
        fine_func_ids.add(func_id)

    logging.info(f"细粒度数据集 {split} 包含 {len(fine_func_ids)} 个漏洞函数")

    # 只处理细粒度数据集中存在的函数
    files_to_process = [f"{func_id}.txt" for func_id in fine_func_ids
                        if os.path.exists(os.path.join(coarse_split_dir, f"{func_id}.txt"))]

    logging.info(f"将处理 {len(files_to_process)} 个粗粒度文件")

    coarse_model.eval()

    with torch.no_grad():
        for file in tqdm(files_to_process, desc=f"生成 {split} 缓存"):
            file_path = os.path.join(coarse_split_dir, file)
            func_id = file[:-4]  # 去掉 .txt 后缀

            try:
                # 解析文件
                parsed = parse_coarse_file(file_path)
                func_code = parsed['func_code']
                slices = parsed['slices']

                if not func_code or not slices:
                    logging.debug(f"跳过空文件: {file}")
                    continue

                # 编码函数
                func_encoded = func_tokenizer.encode_plus(
                    func_code,
                    add_special_tokens=True,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )

                # 编码切片
                slice_codes = [s['code'] for s in slices[:max_slices_per_func]]
                actual_slices = len(slice_codes)

                if actual_slices == 0:
                    # 如果没有切片，使用函数代码作为切片
                    slice_codes = [func_code]
                    slices = [{'code': func_code, 'label': 0, 'line_numbers': []}]
                    actual_slices = 1

                slice_encoded = slice_tokenizer.batch_encode_plus(
                    slice_codes,
                    add_special_tokens=True,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )

                # 填充切片到固定数量
                if actual_slices < max_slices_per_func:
                    pad_input_ids = torch.zeros(
                        (max_slices_per_func - actual_slices, max_length),
                        dtype=torch.long
                    )
                    pad_attention_mask = torch.zeros(
                        (max_slices_per_func - actual_slices, max_length),
                        dtype=torch.long
                    )
                    slice_input_ids = torch.cat([slice_encoded['input_ids'], pad_input_ids], dim=0)
                    slice_attention_mask = torch.cat([slice_encoded['attention_mask'], pad_attention_mask], dim=0)
                else:
                    slice_input_ids = slice_encoded['input_ids'][:max_slices_per_func]
                    slice_attention_mask = slice_encoded['attention_mask'][:max_slices_per_func]

                # 准备模型输入
                func_input_ids = func_encoded['input_ids'].to(device)
                func_attention_mask = func_encoded['attention_mask'].to(device)
                slice_input_ids = slice_input_ids.unsqueeze(0).to(device)  # [1, num_slices, seq_len]
                slice_attention_mask = slice_attention_mask.unsqueeze(0).to(device)

                # 运行模型
                outputs = coarse_model(
                    func_input_ids,
                    func_attention_mask,
                    slice_input_ids,
                    slice_attention_mask
                )

                # 提取特征
                func_prob = outputs['func_probs'].squeeze().cpu().item()
                func_features = outputs['func_features'].squeeze().cpu()
                slice_probs = outputs['slice_probs'].squeeze().cpu()
                selected_slice_features = outputs['selected_slice_features'].squeeze().cpu()
                selected_indices = outputs['selected_indices'].squeeze().cpu().tolist()

                # 构建切片信息
                cache_slices = []
                num_selected = min(len(selected_indices), selected_slice_features.size(0))

                for i in range(num_selected):
                    orig_idx = selected_indices[i]
                    if orig_idx < len(slices):
                        slice_info = slices[orig_idx]
                        cache_slices.append({
                            'slice_idx': orig_idx,
                            'slice_prob': slice_probs[i].item() if i < slice_probs.size(0) else 0.0,
                            'slice_features': selected_slice_features[i],
                            'line_numbers': slice_info['line_numbers']
                        })

                # 保存缓存
                cache_data = {
                    'func_id': func_id,
                    'func_prob': func_prob,
                    'func_features': func_features,
                    'slices': cache_slices
                }

                cache_path = os.path.join(split_output_dir, f"{func_id}.pt")
                torch.save(cache_data, cache_path)

            except Exception as e:
                logging.error(f"处理文件 {file} 时出错: {e}")
                continue

    logging.info(f"{split} 集缓存生成完成")


def main():
    parser = argparse.ArgumentParser(description='生成粗粒度模型特征缓存')

    parser.add_argument('--coarse_data_dir', type=str, required=True,
                        help='粗粒度数据集目录')
    parser.add_argument('--fine_data_dir', type=str, required=True,
                        help='细粒度数据集目录（只为其中存在的函数生成缓存）')
    parser.add_argument('--coarse_model_path', type=str, required=True,
                        help='训练好的粗粒度模型路径')
    parser.add_argument('--func_codebert_path', type=str, required=True,
                        help='函数级CodeBERT路径')
    parser.add_argument('--slice_codebert_path', type=str, required=True,
                        help='切片级CodeBERT路径')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='缓存输出目录')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val', 'test'],
                        help='要处理的数据集分割')
    parser.add_argument('--max_length', type=int, default=512,
                        help='最大序列长度')
    parser.add_argument('--max_slices_per_func', type=int, default=10,
                        help='每个函数的最大切片数')
    parser.add_argument('--hidden_dim', type=int, default=768,
                        help='隐藏层维度')
    parser.add_argument('--k', type=int, default=5,
                        help='选择的切片数量')

    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载tokenizer
    logging.info("加载tokenizer...")
    func_tokenizer = RobertaTokenizer.from_pretrained(args.func_codebert_path,local_files_only=True)
    slice_tokenizer = RobertaTokenizer.from_pretrained(args.slice_codebert_path,local_files_only=True)

    # 加载粗粒度模型
    logging.info("加载粗粒度模型...")
    coarse_model = CoarseGrainedModel(
        func_codebert_path=args.func_codebert_path,
        slice_codebert_path=args.slice_codebert_path,
        hidden_dim=args.hidden_dim,
        k=args.k
    )

    # 加载模型权重
    checkpoint = torch.load(args.coarse_model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        coarse_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        coarse_model.load_state_dict(checkpoint)

    coarse_model = coarse_model.to(device)
    coarse_model.eval()
    logging.info("粗粒度模型加载完成")

    # 为每个分割生成缓存
    for split in args.splits:
        generate_cache_for_split(
            coarse_data_dir=args.coarse_data_dir,
            fine_data_dir=args.fine_data_dir,
            coarse_model=coarse_model,
            func_tokenizer=func_tokenizer,
            slice_tokenizer=slice_tokenizer,
            output_dir=args.output_dir,
            split=split,
            device=device,
            max_length=args.max_length,
            max_slices_per_func=args.max_slices_per_func
        )

    logging.info("所有缓存生成完成!")


if __name__ == '__main__':
    main()
