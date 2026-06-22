import os
import torch
import logging
import numpy as np
from torch import nn
from tqdm import tqdm
from datetime import datetime
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split , WeightedRandomSampler
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.amp import autocast, GradScaler
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 确保使用正确的GPU

class CodeDataset(Dataset):
    def __init__(self, data_dir, tokenizer, is_func=True, max_length=128):
        self.tokenizer = tokenizer
        self.is_func = is_func
        self.max_length = max_length
        self.examples = []
        self.labels = []
        
        files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
        for file in tqdm(files, desc="Loading data"):
            with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 从文件名获取函数级标签 (0 or 1)
            filename_no_ext = file[:-4]
            func_label = int(filename_no_ext.split('_')[0])
            
            if is_func:
                # 提取函数原代码(带行号,需要处理)
                code = []
                for line in lines:
                    if line.startswith("Original Code"):
                        continue
                    if line.startswith("—"):
                        break
                    if ': ' in line:  # 处理带行号的代码行
                        try:
                            _, code_content = line.split(': ', 1)
                            code.append(code_content.strip())
                        except:
                            continue
                if code:
                    self.examples.append('\n'.join(code))
                    self.labels.append(func_label)
            else:
                # 提取切片代码(直接使用,不需要处理行号)
                current_slice = []
                slice_header = ""
                for line in lines:
                    if line.startswith('[Slice'):
                        if current_slice:  # 保存前一个切片
                            self.examples.append('\n'.join(current_slice))
                            slice_label = 1 if "=> vul" in slice_header else 0
                            self.labels.append(slice_label)
                            current_slice = []
                        slice_header = line
                    elif not line.startswith('—') and not line.startswith('Original Code'):
                        current_slice.append(line.strip())
                # 保存最后一个切片
                if current_slice:
                    self.examples.append('\n'.join(current_slice))
                    slice_label = 1 if "=> vul" in slice_header else 0
                    self.labels.append(slice_label)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # 对代码进行编码
        encodings = self.tokenizer(
            self.examples[idx],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx])
        }

    def get_label_distribution(self):
        """返回数据集标签分布统计"""
        label_counts = {0: 0, 1: 0}
        for label in self.labels:
            label_counts[label] += 1
        return label_counts

def compute_metrics(labels, preds):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# 设置日志
def setup_logger(save_dir):
    # 固定日志文件名
    log_file = os.path.join(save_dir, 'codebert_func_slice_fintune_openssl_ffmpeg.log')
    
    logger = logging.getLogger('CodeBERT_Finetune')
    logger.setLevel(logging.INFO)
    
    # 如果 logger 中已存在处理器，就不再重复添加
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode='a')  # 'a' 为追加写模式
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    metrics = compute_metrics(all_labels, all_preds)
    return metrics

def finetune_codebert(data_root, output_dir, is_func=True, codebert_path=None, log_dir=None, config=None):
    # 设置日志
    if log_dir is None:
        log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(log_dir)

    # 定义配置参数
    if config is None:
        config = {
            "max_seq_length": 128,
            "batch_size": 256,
            "learning_rate": 1e-5,
            "num_train_epochs": 50,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "early_stopping_patience": 3,
            "gradient_accumulation_steps": 1
        }

    # 记录实验配置
    logger.info("="*50)
    logger.info("实验配置:")
    logger.info(f"数据根目录: {data_root}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"任务类型: {'函数级' if is_func else '切片级'}")
    logger.info(f"设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info("模型参数:")
    logger.info(f" - 最大序列长度: {config['max_seq_length']}")
    logger.info(f" - 批次大小: {config['batch_size']}")
    logger.info(f" - 学习率: {config['learning_rate']}")
    logger.info(f" - 训练轮数: {config['num_train_epochs']}")
    logger.info("="*50)

    # 检查并创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 初始化分词器和模型
    if codebert_path is None:
        codebert_path = "/root/autodl-tmp/model/models__microsoft__codebert_base/snapshots/3b0952feddeffad0063f274080e3c23d75e7eb39"

    logger.info(f"加载CodeBERT模型: {codebert_path}")
    tokenizer = RobertaTokenizer.from_pretrained(codebert_path)

    try:
        model = RobertaForSequenceClassification.from_pretrained(
            codebert_path,
            num_labels=2,
            local_files_only=True
        )
    except Exception as e:
        logger.warning(f"标准加载失败: {e}")
        logger.info("尝试使用torch.load直接加载权重...")

        from transformers import RobertaConfig
        model_config = RobertaConfig.from_pretrained(codebert_path)
        model_config.num_labels = 2
        model = RobertaForSequenceClassification(model_config)

        model_path = os.path.join(codebert_path, "pytorch_model.bin")
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)

        if 'classifier.out_proj.weight' in state_dict:
            del state_dict['classifier.out_proj.weight']
            del state_dict['classifier.out_proj.bias']
        if 'classifier.dense.weight' in state_dict:
            del state_dict['classifier.dense.weight']
            del state_dict['classifier.dense.bias']

        model.load_state_dict(state_dict, strict=False)
        logger.info("权重加载成功")

    # 检查并设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 准备数据集 - 从train/val/test目录加载
    logger.info("加载数据集...")
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')
    test_dir = os.path.join(data_root, 'test')

    if not os.path.exists(train_dir):
        raise ValueError(f"训练集目录不存在: {train_dir}")
    if not os.path.exists(val_dir):
        raise ValueError(f"验证集目录不存在: {val_dir}")

    train_dataset = CodeDataset(train_dir, tokenizer, is_func, config['max_seq_length'])
    eval_dataset = CodeDataset(val_dir, tokenizer, is_func, config['max_seq_length'])
    test_dataset = None
    if os.path.exists(test_dir):
        test_dataset = CodeDataset(test_dir, tokenizer, is_func, config['max_seq_length'])

    # 统计训练集标签分布
    train_label_dist = train_dataset.get_label_distribution()
    train_pos_count = train_label_dist[1]
    train_neg_count = train_label_dist[0]

    logger.info(f"训练集标签分布:")
    logger.info(f" - 训练集负样本(0): {train_neg_count}")
    logger.info(f" - 训练集正样本(1): {train_pos_count}")
    logger.info(f" - 训练集正负比例: 1:{train_neg_count/train_pos_count:.2f}" if train_pos_count > 0 else " - 无正样本")

    val_label_dist = eval_dataset.get_label_distribution()
    logger.info(f"验证集标签分布:")
    logger.info(f" - 验证集负样本(0): {val_label_dist[0]}")
    logger.info(f" - 验证集正样本(1): {val_label_dist[1]}")

    if test_dataset:
        test_label_dist = test_dataset.get_label_distribution()
        logger.info(f"测试集标签分布:")
        logger.info(f" - 测试集负样本(0): {test_label_dist[0]}")
        logger.info(f" - 测试集正样本(1): {test_label_dist[1]}")

    # 使用加权损失函数
    criterion = nn.CrossEntropyLoss()

    # 获取正负样本索引 - 直接从train_dataset获取
    train_labels = train_dataset.labels
    pos_indices = [i for i, label in enumerate(train_labels) if label == 1]
    neg_indices = [i for i, label in enumerate(train_labels) if label == 0]

    # 下采样比例设置 - 尝试保持正负样本比例为1:3或1:5
    target_ratio = 5  # 可以调整为3-5之间
    target_neg_count = min(train_neg_count, train_pos_count * target_ratio)

    # 随机选择负样本
    random.seed(42)  # 设置随机种子确保可重复性
    sampled_neg_indices = random.sample(neg_indices, target_neg_count)

    # 合并正样本和采样后的负样本
    balanced_indices = pos_indices + sampled_neg_indices
    random.shuffle(balanced_indices)  # 随机打乱顺序

    # 创建平衡后的子集
    balanced_subset = torch.utils.data.Subset(train_dataset, balanced_indices)
    
    # 计算下采样后的比例
    balanced_pos_count = len(pos_indices)
    balanced_neg_count = len(sampled_neg_indices)
    
    logger.info(f"下采样后训练集分布:")
    logger.info(f" - 下采样后负样本(0): {balanced_neg_count}")
    logger.info(f" - 下采样后正样本(1): {balanced_pos_count}")
    logger.info(f" - 下采样后正负比例: 1:{balanced_neg_count/balanced_pos_count:.2f}")
    logger.info(f" - 下采样后训练集大小: {len(balanced_subset)}")
    
    # 使用下采样后的数据集创建DataLoader
    train_loader = DataLoader(
        balanced_subset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=16,
        pin_memory=True
    )

    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=16,
        pin_memory=True
    )

    logger.info(f"数据集大小:")
    logger.info(f" - 训练集(原始): {len(train_dataset)}")
    logger.info(f" - 训练集(下采样后): {len(balanced_subset)}")
    logger.info(f" - 验证集: {len(eval_dataset)}")
    if test_dataset:
        logger.info(f" - 测试集: {len(test_dataset)}")
    
    # 优化器和学习率调度器
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    total_steps = len(train_loader) * config['num_train_epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        total_steps=total_steps,
        pct_start=warmup_steps/total_steps
    )
    
    # 更新GradScaler初始化
    scaler = GradScaler('cuda')
    accumulation_steps = config.get('gradient_accumulation_steps', 1)
    
    # 训练循环
    logger.info("开始训练...")
    best_f1 = 0
    patience_counter = 0
    best_model_path = os.path.join(output_dir, "best_model.pt")
    
    for epoch in range(config['num_train_epochs']):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_train_epochs']}")
        for step, batch in enumerate(progress_bar):
            # 将数据移动到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 使用混合精度训练
            with autocast(device_type='cuda'):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                # 如果使用自定义损失函数而不是模型内置损失
                if criterion:
                    logits = outputs.logits
                    loss = criterion(logits, labels) / accumulation_steps
                else:
                    loss = outputs.loss / accumulation_steps
            
            # 使用scaler进行反向传播
            scaler.scale(loss).backward()
            
            if (step + 1) % accumulation_steps == 0:
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # 更新权重
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            epoch_loss += loss.item() * accumulation_steps
            
            # 更新进度条
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # 计算平均损失
        avg_train_loss = epoch_loss / len(train_loader)
        
        # 评估模型
        eval_metrics = evaluate(model, eval_loader, device)
        
        # 记录结果
        logger.info(f"Epoch {epoch+1}/{config['num_train_epochs']}:valLoss: {avg_train_loss:.4f},acc: {eval_metrics['accuracy']:.4f},f1: {eval_metrics['f1']:.4f},precision: {eval_metrics['precision']:.4f},recall: {eval_metrics['recall']:.4f}")

        # 检查模型是否有提升
        current_f1 = eval_metrics['f1']
        if current_f1 > best_f1:
            best_f1 = current_f1
            patience_counter = 0
            
            # 保存最佳模型
            logger.info(f"F1提升，保存模型...")
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            logger.info(f"F1未提升，当前早停计数: {patience_counter}/{config['early_stopping_patience']}")
            
            # 检查早停
            if patience_counter >= config['early_stopping_patience']:
                logger.info(f"触发早停，停止训练！")
                break
        
        # 定期清理缓存
        def clear_gpu_cache():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        # 在每个epoch结束时调用
        clear_gpu_cache()
    
    # 加载最佳模型并保存
    model.load_state_dict(torch.load(best_model_path))
    logger.info("训练完成!")

    # 最终评估
    logger.info("="*50)
    logger.info("最终验证集评估结果:")
    final_metrics = evaluate(model, eval_loader, device)
    for metric, value in final_metrics.items():
        logger.info(f"{metric}: {value:.4f}")

    # 如果有测试集，也进行评估
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=16,
            pin_memory=True
        )
        logger.info("="*50)
        logger.info("测试集评估结果:")
        test_metrics = evaluate(model, test_loader, device)
        for metric, value in test_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
    
    # 保存模型
    logger.info("保存最终模型...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"模型已保存至: {output_dir}")
    logger.info("="*50)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="微调CodeBERT模型（使用raw_slice数据集）")
    parser.add_argument("--data_root", type=str, default="/root/rivermind-data/datasets/bigvul/coarse_level/raw_slice", help="数据根目录（包含train/val/test）")
    parser.add_argument("--func_output_dir", type=str, default="./func_best_model_raw_slice", help="函数级模型输出目录")
    parser.add_argument("--slice_output_dir", type=str, default="./slice_best_model_raw_slice", help="切片级模型输出目录")
    parser.add_argument("--codebert_path", type=str, default="/root/rivermind-fs/models__microsoft__codebert_base/snapshots/3b0952feddeffad0063f274080e3c23d75e7eb39", help="预训练CodeBERT路径")
    parser.add_argument("--log_dir", type=str, default="./logs", help="日志目录")
    parser.add_argument("--max_seq_length", type=int, default=128, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=256, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
    parser.add_argument("--num_epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--skip_func", action="store_true", help="跳过函数级微调")
    parser.add_argument("--skip_slice", action="store_true", help="跳过切片级微调")

    args = parser.parse_args()

    # 配置参数
    config = {
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_epochs,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "early_stopping_patience": 5,
        "gradient_accumulation_steps": 1
    }

    # 微调函数级CodeBERT
    if not args.skip_func:
        print("\n" + "="*80)
        print("开始微调函数级CodeBERT（raw_slice数据集）")
        print("="*80)
        finetune_codebert(
            data_root=args.data_root,
            output_dir=args.func_output_dir,
            is_func=True,
            codebert_path=args.codebert_path,
            log_dir=args.log_dir,
            config=config
        )

    # 微调切片级CodeBERT
    if not args.skip_slice:
        print("\n" + "="*80)
        print("开始微调切片级CodeBERT（raw_slice数据集）")
        print("="*80)
        finetune_codebert(
            data_root=args.data_root,
            output_dir=args.slice_output_dir,
            is_func=False,
            codebert_path=args.codebert_path,
            log_dir=args.log_dir,
            config=config
        )