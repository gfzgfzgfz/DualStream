import os
import torch
import logging
import numpy as np
from torch import nn
from tqdm import tqdm
from datetime import datetime
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.amp import autocast, GradScaler
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 确保使用正确的GPU

class LineDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=512, pos_neg_ratio=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pos_neg_ratio = pos_neg_ratio
        self.examples = []
        self.labels = []

        # 加载数据
        self._load_data(data_dir)

        # 平衡正负样本（如果需要）
        if self.pos_neg_ratio is not None and self.pos_neg_ratio > 0:
            self._balance_samples()
        else:
            # 记录原始数据集统计信息
            pos_count = sum(self.labels)
            neg_count = len(self.labels) - pos_count
            logging.info(f"使用原始数据集（未采样）:")
            logging.info(f" - 正样本: {pos_count}")
            logging.info(f" - 负样本: {neg_count}")
            logging.info(f" - 正负比例: 1:{neg_count/pos_count:.2f}" if pos_count > 0 else " - 正负比例: 无正样本")
        
    def _load_data(self, data_dir):
        """加载数据"""
        logging.info("正在加载数据...")
        
        # 获取所有行级数据文件
        line_files = [f for f in os.listdir(data_dir) if f.endswith('_contexts.txt')]
        
        for file in tqdm(line_files, desc="加载行级数据"):
            # 从文件名中提取标签和ID信息
            parts = file.split('_')
            label = int(parts[0])
            func_id = parts[1]
            slice_id = parts[2] if len(parts) > 2 else None
            
            # 只处理函数级标签为1的样本
            if label != 1:
                continue
                
            file_path = os.path.join(data_dir, file)
            
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
                
                # 构造输入文本
                input_text = f"{line_code} [SEP]{' '.join(operation_context)} [SEP]{' '.join(dependence_context)} [SEP]{' '.join(surrounding_context)}"
                
                # 添加到数据集
                self.examples.append(input_text)
                self.labels.append(1 if is_vuln_line else 0)
    
    def _balance_samples(self):
        """平衡正负样本"""
        # 统计正负样本
        pos_indices = [i for i, label in enumerate(self.labels) if label == 1]
        neg_indices = [i for i, label in enumerate(self.labels) if label == 0]
        
        # 计算目标负样本数量
        target_neg_count = int(len(pos_indices) * self.pos_neg_ratio)
        
        # 随机选择负样本
        random.seed(42)
        sampled_neg_indices = random.sample(neg_indices, min(target_neg_count, len(neg_indices)))
        
        # 合并正样本和采样后的负样本
        balanced_indices = pos_indices + sampled_neg_indices
        random.shuffle(balanced_indices)
        
        # 更新数据集
        self.examples = [self.examples[i] for i in balanced_indices]
        self.labels = [self.labels[i] for i in balanced_indices]
        
        # 记录统计信息
        pos_count = sum(self.labels)
        neg_count = len(self.labels) - pos_count
        logging.info(f"平衡后数据集统计:")
        logging.info(f" - 正样本: {pos_count}")
        logging.info(f" - 负样本: {neg_count}")
        logging.info(f" - 正负比例: 1:{neg_count/pos_count:.2f}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        # 对文本进行编码
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

def compute_metrics(labels, preds):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def setup_logger(save_dir):
    # 固定日志文件名
    log_file = os.path.join(save_dir, 'line_codebert_fintune.log')
    
    logger = logging.getLogger('LineCodeBERT_Finetune')
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
    
    progress_bar = tqdm(dataloader, desc='Evaluating')
    
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            progress_bar.set_postfix({'processed': f"{len(all_preds)}/{len(dataloader.dataset)}"})
    
    metrics = compute_metrics(all_labels, all_preds)
    return metrics

def finetune_line_codebert(train_dir, val_dir, test_dir, output_dir, pos_neg_ratio=None):
    # 设置日志
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(log_dir)
    
    # 定义配置参数
    config = {
        "max_seq_length": 256,
        "batch_size": 128,
        "learning_rate": 5e-5,
        "num_train_epochs": 50,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "early_stopping_patience": 10,
        "gradient_accumulation_steps": 4
    }
    
    # 记录实验配置
    logger.info("="*50)
    logger.info("实验配置:")
    logger.info(f"训练数据目录: {train_dir}")
    logger.info(f"验证数据目录: {val_dir}")
    logger.info(f"测试数据目录: {test_dir}")
    logger.info(f"输出目录: {output_dir}")
    if pos_neg_ratio is None or pos_neg_ratio <= 0:
        logger.info(f"正负样本比例: 使用原始数据集分布（不采样）")
    else:
        logger.info(f"正负样本比例: 1:{pos_neg_ratio}")
    logger.info(f"设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info("模型参数:")
    for k, v in config.items():
        logger.info(f" - {k}: {v}")
    logger.info("="*50)
    
    # 检查并创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化分词器和模型
    model_dir = "/root/rivermind-fs/models__microsoft__codebert_base/snapshots/3b0952feddeffad0063f274080e3c23d75e7eb39"
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)

    # 由于 torch 版本限制，手动加载模型
    from transformers import RobertaConfig
    model_config = RobertaConfig.from_pretrained(model_dir)
    model_config.num_labels = 2
    model = RobertaForSequenceClassification(model_config)

    # 手动加载权重
    state_dict = torch.load(
        os.path.join(model_dir, "pytorch_model.bin"),
        map_location="cpu",
        weights_only=False  # 明确设置为 False 以绕过安全检查
    )

    # 为 state_dict 的键添加 'roberta.' 前缀
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = f"roberta.{key}"
        new_state_dict[new_key] = value

    # 加载权重到模型（忽略分类头的不匹配，因为我们会重新训练）
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    # 检查并设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 准备数据集
    logger.info("加载训练数据集...")
    train_dataset = LineDataset(train_dir, tokenizer, config["max_seq_length"], pos_neg_ratio)
    
    logger.info("加载验证数据集...")
    val_dataset = LineDataset(val_dir, tokenizer, config["max_seq_length"], pos_neg_ratio)  # 使用相同的正负样本比例
    
    logger.info("加载测试数据集...")
    test_dataset = LineDataset(test_dir, tokenizer, config["max_seq_length"], pos_neg_ratio)  # 使用相同的正负样本比例
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    # 优化器和学习率调度器
    # 准备带权重衰减的优化器
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': config["weight_decay"]},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}
    ]
    
    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=config["learning_rate"],
        eps=1e-8
    )
    
    total_steps = len(train_loader) * config["num_train_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    
    # 使用线性预热的学习率调度器
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 创建混合精度训练的scaler
    scaler = GradScaler()
    
    # 训练循环
    logger.info("开始训练...")
    best_f1 = 0
    patience_counter = 0
    best_model_path = os.path.join(output_dir, "1_1_best_model_64_5e_5.pt")
    
    for epoch in range(config["num_train_epochs"]):
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
                loss = outputs.loss / config["gradient_accumulation_steps"]
            
            # 使用scaler进行反向传播
            scaler.scale(loss).backward()
            
            if (step + 1) % config["gradient_accumulation_steps"] == 0:
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # 更新权重
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            epoch_loss += loss.item() * config["gradient_accumulation_steps"]
            
            # 更新进度条
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        # 计算平均损失
        avg_train_loss = epoch_loss / len(train_loader)
        
        # 评估模型
        val_metrics = evaluate(model, val_loader, device)
        
        # 记录结果
        logger.info(f"Epoch {epoch+1}/{config['num_train_epochs']}:")
        logger.info(f" - 训练损失: {avg_train_loss:.4f}")
        logger.info(f" - 验证准确率: {val_metrics['accuracy']:.4f}")
        logger.info(f" - 验证F1值: {val_metrics['f1']:.4f}")
        logger.info(f" - 验证精确率: {val_metrics['precision']:.4f}")
        logger.info(f" - 验证召回率: {val_metrics['recall']:.4f}")
        
        # 检查模型是否有提升
        current_f1 = val_metrics['f1']
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 加载最佳模型并保存
    model.load_state_dict(torch.load(best_model_path))
    logger.info("训练完成!")
    
    # 在测试集上评估
    logger.info("在测试集上评估模型...")
    test_metrics = evaluate(model, test_loader, device)
    logger.info("测试集结果:")
    for metric, value in test_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # 保存模型
    logger.info("保存最终模型...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"模型已保存至: {output_dir}")
    logger.info("="*50)

if __name__ == "__main__":
    # 微调行级CodeBERT
    finetune_line_codebert(
        train_dir="/root/rivermind-data/datasets/bigvul/line_level/datasets/train",
        val_dir="/root/rivermind-data/datasets/bigvul/line_level/datasets/val",
        test_dir="/root/rivermind-data/datasets/bigvul/line_level/datasets/test",
        output_dir="/root/rivermind-data/model/bigvul/line_codebert_finetuned",
        pos_neg_ratio=None  # 使用原始数据集分布，不进行采样
    )