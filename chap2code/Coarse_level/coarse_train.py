import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import logging
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader


from coarse_model import CoarseGrainedModel
from coarse_data import create_coarse_dataloaders
from coarse_data_fast import create_pretokenized_dataloaders

def setup_logger(log_dir):
    """设置日志"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, f'coarse_trainV2_bigvul_ablation_full.log')

    # 配置日志
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

class WeightedBCELoss(nn.Module):
    """加权二元交叉熵loss"""
    def __init__(self, pos_weight=None):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, probs, targets):
        # 模型输出已经过 sigmoid，这里直接对概率计算 BCE，避免二次 sigmoid
        probs = probs.clamp(min=1e-7, max=1 - 1e-7)
        bce_loss = F.binary_cross_entropy(probs, targets.float(), reduction='none')

        # 应用正样本权重
        if self.pos_weight is not None:
            weights = torch.ones_like(targets, dtype=torch.float)
            weights[targets == 1] = self.pos_weight
            bce_loss = bce_loss * weights

        return bce_loss.mean()

class FocalLoss(nn.Module):
    """焦点loss"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, probs, targets):
        # 模型输出已经过 sigmoid，直接使用概率，不再重复 sigmoid
        probs = probs.clamp(min=1e-7, max=1 - 1e-7)

        # 计算二元交叉熵loss
        bce_loss = F.binary_cross_entropy(probs, targets.float(), reduction='none')

        # 计算焦点权重
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        # 应用焦点权重
        focal_loss = bce_loss * focal_weight

        return focal_loss.mean()

def train_epoch(model, train_loader, criterion_func, criterion_slice, optimizer, scheduler, device, scaler, logger, epoch, args, gradient_accumulation_steps=4):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    func_preds = []
    func_labels = []
    slice_preds = []
    slice_labels = []
    
    progress_bar = tqdm(train_loader, desc=f"train  Epoch {epoch}")
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar):
        # 将数据移到设备上
        func_input_ids = batch['func_input_ids'].to(device)
        func_attention_mask = batch['func_attention_mask'].to(device)
        func_labels_batch = batch['func_label'].to(device)
        slice_input_ids = batch['slice_input_ids'].to(device)
        slice_attention_mask = batch['slice_attention_mask'].to(device)
        slice_labels_batch = batch['slice_labels'].to(device)
        slice_indices = batch['slice_indices'].to(device)
        
        # 前向传播
        with autocast(device_type=device.type if hasattr(device, 'type') else 'cuda'):
            outputs = model(func_input_ids, func_attention_mask, slice_input_ids, slice_attention_mask, slice_indices)
            
            func_probs = outputs['func_probs']
            slice_probs = outputs['slice_probs']
            
            # 计算func loss
            func_loss = criterion_func(func_probs, func_labels_batch.unsqueeze(1).float())
            
            # 计算slice loss - 只计算有效切片的loss
            # slice_probs的形状是[batch_size, k]，其中k是选择的切片数量
            # 需要使用selected_indices来获取对应的标签
            batch_size, num_selected_slices = slice_probs.size()
            selected_indices = outputs['selected_indices']  # [batch_size, k]

            # 使用selected_indices从slice_labels_batch中获取对应的标签
            # slice_labels_batch的形状是[batch_size, max_slices_per_func]
            # 需要使用gather操作来获取选中切片的标签
            selected_slice_labels = torch.gather(slice_labels_batch, 1, selected_indices)  # [batch_size, k]

            # 创建有效切片的掩码 - 向量化版本
            # 使用gather操作获取对应的slice_indices值
            gathered_slice_indices = torch.gather(slice_indices, 1, selected_indices)
            valid_mask = gathered_slice_indices != 0

            # 只计算有效切片的loss
            if valid_mask.any():
                valid_slice_probs = slice_probs[valid_mask]
                valid_slice_labels = selected_slice_labels[valid_mask].float()

                slice_loss = criterion_slice(valid_slice_probs, valid_slice_labels)
            else:
                slice_loss = torch.tensor(0.0, device=device)
            
            # 组合loss，使用权重参数平衡func 和slice loss
            alpha = args.func_loss_weight  # func loss权重
            beta = args.slice_loss_weight  # slice loss权重

            func_loss = func_loss.mean() if func_loss.dim() > 0 else func_loss
            slice_loss = slice_loss.mean() if slice_loss.dim() > 0 else slice_loss

            loss = alpha * func_loss + beta * slice_loss

            # 反事实正则化（稀疏采样 + Epoch调度）
            if args.use_cf_reg and epoch >= args.cf_start_epoch and \
               (epoch % args.cf_epoch_interval == 0) and (step % args.cf_batch_interval == 0):
                # 找出有漏洞的样本
                vuln_mask = func_labels_batch == 1

                if vuln_mask.sum() > 0:
                    # 随机采样：只对部分有漏洞样本做反事实传播
                    vuln_indices = torch.where(vuln_mask)[0]
                    sample_size = max(1, int(len(vuln_indices) * args.cf_sample_ratio))
                    sampled_indices = vuln_indices[torch.randperm(len(vuln_indices))[:sample_size]]

                    if len(sampled_indices) > 0:
                        # 创建masked的切片输入（全零表示无切片信息）
                        slice_input_ids_masked = torch.zeros_like(slice_input_ids)
                        slice_attention_mask_masked = torch.zeros_like(slice_attention_mask)

                        # 反事实前向传播
                        outputs_cf = model(
                            func_input_ids,
                            func_attention_mask,
                            slice_input_ids_masked,
                            slice_attention_mask_masked,
                            slice_indices
                        )

                        func_probs_cf = outputs_cf['func_probs']

                        # 计算正则化损失（只对采样的有漏洞样本）
                        probs_orig = func_probs.squeeze(-1)  # [batch_size]
                        probs_cf = func_probs_cf.squeeze(-1)  # [batch_size]

                        # 核心公式：希望 probs_orig > probs_cf + margin
                        # 即有切片信息时的预测概率应该显著高于无切片信息时
                        cf_reg_loss = torch.relu(
                            probs_cf[sampled_indices] - probs_orig[sampled_indices] + args.cf_margin
                        ).mean()

                        # 添加到总损失
                        loss = loss + args.cf_weight * cf_reg_loss

            if loss.dim() > 0:
                loss = loss.mean()

            loss = loss / gradient_accumulation_steps
        
        # 反向传播
        scaler.scale(loss).backward()
        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1 == len(train_loader)):
            # 先反缩放梯度，再裁剪，最后更新
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        
        # 记录loss
        total_loss += loss.item() * gradient_accumulation_steps
        
        # 收集func 预测结果
        func_pred = (func_probs > 0.5).int().cpu().numpy()
        func_preds.extend(func_pred)
        func_labels.extend(func_labels_batch.cpu().numpy())
        
        # 收集slice 预测结果 - 向量化版本
        slice_pred = (slice_probs > 0.5).int().cpu()
        selected_indices_cpu = outputs['selected_indices'].cpu()

        # 使用gather获取对应的标签和slice_indices
        selected_slice_labels_cpu = torch.gather(slice_labels_batch.cpu(), 1, selected_indices_cpu)
        gathered_slice_indices_cpu = torch.gather(slice_indices.cpu(), 1, selected_indices_cpu)

        # 创建有效掩码
        valid_mask_cpu = gathered_slice_indices_cpu != 0

        # 只保留有效的预测和标签
        if valid_mask_cpu.any():
            valid_preds = slice_pred[valid_mask_cpu].numpy()
            valid_labels = selected_slice_labels_cpu[valid_mask_cpu].numpy()
            slice_preds.extend(valid_preds.tolist())
            slice_labels.extend(valid_labels.tolist())
        
        # 更新进度条
        progress_bar.set_postfix({'loss': f"{loss.item() * gradient_accumulation_steps:.4f}"})
        if step % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 计算指标
    func_acc = accuracy_score(func_labels, func_preds)
    func_precision = precision_score(func_labels, func_preds, average='binary', zero_division=0)
    func_recall = recall_score(func_labels, func_preds, average='binary', zero_division=0)
    func_f1 = f1_score(func_labels, func_preds, average='binary', zero_division=0)
    func_tn, func_fp, func_fn, func_tp = confusion_matrix(func_labels, func_preds, labels=[0, 1]).ravel()
    func_fnr = func_fn / (func_fn + func_tp) if (func_fn + func_tp) > 0 else 0.0
    func_fpr = func_fp / (func_fp + func_tn) if (func_fp + func_tn) > 0 else 0.0
    
    # 确保切片预测和标签数量一致
    if len(slice_labels) == 0:
        logger.warning("没有收集到任何有效切片的预测和标签")
        slice_acc, slice_precision, slice_recall, slice_f1 = 0, 0, 0, 0
    else:
        assert len(slice_labels) == len(slice_preds), f"切片标签数量({len(slice_labels)})和预测数量({len(slice_preds)})不一致"
        
        slice_acc = accuracy_score(slice_labels, slice_preds)
        slice_precision = precision_score(slice_labels, slice_preds, average='binary', zero_division=0)
        slice_recall = recall_score(slice_labels, slice_preds, average='binary', zero_division=0)
        slice_f1 = f1_score(slice_labels, slice_preds, average='binary', zero_division=0)
    
    # 记录训练结果
    logger.info(
        f"Epoch {epoch} - train loss: {total_loss/len(train_loader):.4f}, "
        f"func : acc={func_acc:.4f}, precision={func_precision:.4f}, recall={func_recall:.4f}, F1={func_f1:.4f}, FNR={func_fnr:.4f}, FPR={func_fpr:.4f}, "
        f"slice (参考): acc={slice_acc:.4f}, precision={slice_precision:.4f}, recall={slice_recall:.4f}, F1={slice_f1:.4f}, 切片数量: {len(slice_preds)}"
    )
    
    return total_loss/len(train_loader), func_acc, func_precision, func_recall, func_f1, slice_acc, slice_precision, slice_recall, slice_f1

def evaluate(model, eval_loader, criterion_func, criterion_slice, device, logger, args, mode="验证"):
    """评估模型"""
    model.eval()
    total_loss = 0
    func_preds = []
    func_labels = []
    slice_preds = []
    slice_labels = []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"{mode}"):
            # 将数据移到设备上
            func_input_ids = batch['func_input_ids'].to(device)
            func_attention_mask = batch['func_attention_mask'].to(device)
            func_labels_batch = batch['func_label'].to(device)
            slice_input_ids = batch['slice_input_ids'].to(device)
            slice_attention_mask = batch['slice_attention_mask'].to(device)
            slice_labels_batch = batch['slice_labels'].to(device)
            slice_indices = batch['slice_indices'].to(device)
            
            # 前向传播
            outputs = model(func_input_ids, func_attention_mask, slice_input_ids, slice_attention_mask, slice_indices)
            
            func_probs = outputs['func_probs']
            slice_probs = outputs['slice_probs']
            
            # 计算func loss
            func_loss = criterion_func(func_probs, func_labels_batch.unsqueeze(1).float())
            
            # 计算slice loss - 只计算有效切片的loss
            # slice_probs的形状是[batch_size, k]，其中k是选择的切片数量
            # 需要使用selected_indices来获取对应的标签
            batch_size, num_selected_slices = slice_probs.size()
            selected_indices = outputs['selected_indices']  # [batch_size, k]

            # 使用selected_indices从slice_labels_batch中获取对应的标签
            selected_slice_labels = torch.gather(slice_labels_batch, 1, selected_indices)  # [batch_size, k]

            # 创建有效切片的掩码 - 向量化版本
            gathered_slice_indices = torch.gather(slice_indices, 1, selected_indices)
            valid_mask = gathered_slice_indices != 0

            # 只计算有效切片的loss
            if valid_mask.any():
                valid_slice_probs = slice_probs[valid_mask]
                valid_slice_labels = selected_slice_labels[valid_mask].float()
                slice_loss = criterion_slice(valid_slice_probs, valid_slice_labels)
            else:
                slice_loss = torch.tensor(0.0, device=device)
            
            # 组合loss，使用权重参数平衡func 和slice loss
            alpha = args.func_loss_weight  # func loss权重
            beta = args.slice_loss_weight  # slice loss权重
            loss = alpha * func_loss + beta * slice_loss
            
            total_loss += loss.item()
            
            # 收集func 预测结果
            func_pred = (func_probs > 0.5).int().cpu().numpy()
            func_preds.extend(func_pred)
            func_labels.extend(func_labels_batch.cpu().numpy())
            
            # 收集slice 预测结果 - 向量化版本
            slice_pred = (slice_probs > 0.5).int().cpu()
            selected_indices_cpu = outputs['selected_indices'].cpu()

            # 使用gather获取对应的标签和slice_indices
            selected_slice_labels_cpu = torch.gather(slice_labels_batch.cpu(), 1, selected_indices_cpu)
            gathered_slice_indices_cpu = torch.gather(slice_indices.cpu(), 1, selected_indices_cpu)

            # 创建有效掩码
            valid_mask_cpu = gathered_slice_indices_cpu != 0

            # 只保留有效的预测和标签
            if valid_mask_cpu.any():
                valid_preds = slice_pred[valid_mask_cpu].numpy()
                valid_labels = selected_slice_labels_cpu[valid_mask_cpu].numpy()
                slice_preds.extend(valid_preds.tolist())
                slice_labels.extend(valid_labels.tolist())
    
    # 计算指标
    func_acc = accuracy_score(func_labels, func_preds)
    func_precision = precision_score(func_labels, func_preds, average='binary', zero_division=0)
    func_recall = recall_score(func_labels, func_preds, average='binary', zero_division=0)
    func_f1 = f1_score(func_labels, func_preds, average='binary', zero_division=0)
    func_tn, func_fp, func_fn, func_tp = confusion_matrix(func_labels, func_preds, labels=[0, 1]).ravel()
    func_fnr = func_fn / (func_fn + func_tp) if (func_fn + func_tp) > 0 else 0.0
    func_fpr = func_fp / (func_fp + func_tn) if (func_fp + func_tn) > 0 else 0.0
    
    # 确保切片预测和标签数量一致
    if len(slice_labels) == 0:
        logger.warning(f"{mode}没有收集到任何有效切片的预测和标签")
        slice_acc, slice_precision, slice_recall, slice_f1 = 0, 0, 0, 0
    else:
        assert len(slice_labels) == len(slice_preds), f"切片标签数量({len(slice_labels)})和预测数量({len(slice_preds)})不一致"
        
        slice_acc = accuracy_score(slice_labels, slice_preds)
        slice_precision = precision_score(slice_labels, slice_preds, average='binary', zero_division=0)
        slice_recall = recall_score(slice_labels, slice_preds, average='binary', zero_division=0)
        slice_f1 = f1_score(slice_labels, slice_preds, average='binary', zero_division=0)
    
    # 记录评估结果
    logger.info(
        f"{mode} - loss: {total_loss/len(eval_loader):.4f}, "
        f"func : acc={func_acc:.4f}, precision={func_precision:.4f}, recall={func_recall:.4f}, F1={func_f1:.4f}, FNR={func_fnr:.4f}, FPR={func_fpr:.4f}, "
        f"slice : acc={slice_acc:.4f}, precision={slice_precision:.4f}, recall={slice_recall:.4f}, F1={slice_f1:.4f}, 切片数量: {len(slice_preds)}"
    )
    
    return total_loss/len(eval_loader), func_acc, func_precision, func_recall, func_f1, slice_acc, slice_precision, slice_recall, slice_f1

def plot_metrics(train_metrics, val_metrics, output_dir):
    """绘制训练指标"""
    # 创建图表
    plt.figure(figsize=(15, 10))
    
    # 绘制loss曲线
    plt.subplot(2, 2, 1)
    plt.plot(train_metrics['loss'], label='train loss')
    plt.plot(val_metrics['loss'], label='val loss')
    plt.title('loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制func F1曲线
    plt.subplot(2, 2, 2)
    plt.plot(train_metrics['func_f1'], label='train F1')
    plt.plot(val_metrics['func_f1'], label='val F1')
    plt.title('func F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.grid(True)
    
    # 绘制slice F1曲线
    plt.subplot(2, 2, 3)
    plt.plot(train_metrics['slice_f1'], label='train F1')
    plt.plot(val_metrics['slice_f1'], label='val F1')
    plt.title('slice F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.grid(True)
    
    # 绘制func 和slice F1对比
    plt.subplot(2, 2, 4)
    plt.plot(val_metrics['func_f1'], label='func F1')
    plt.plot(val_metrics['slice_f1'], label='slice F1')
    plt.title('func vs slice F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(output_dir, 'coarse_metrics.png'))
    plt.close()



def train_coarse_model(args, logger):
    """训练粗粒度模型"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")
    
    # 加载预训练分词器
    func_codebert_path = args.func_codebert_path
    slice_codebert_path = args.slice_codebert_path
    func_tokenizer = RobertaTokenizer.from_pretrained(func_codebert_path)
    slice_tokenizer = RobertaTokenizer.from_pretrained(slice_codebert_path)
    
    # 创建数据加载器 - 使用预tokenize的数据
    use_pretokenized = args.use_pretokenized if hasattr(args, 'use_pretokenized') else True

    if use_pretokenized:
        logger.info("使用预tokenize数据加载器（快速模式）")
        train_loader, val_loader, test_loader = create_pretokenized_dataloaders(
            args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_length=args.max_length,
            max_slices_per_func=args.max_slices_per_func,
            pin_memory=True,
            persistent_workers=True
        )
    else:
        logger.info("使用实时tokenize数据加载器（慢速模式）")
        train_loader, val_loader, test_loader = create_coarse_dataloaders(
            args.data_root,
            func_tokenizer,
            slice_tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_workers=args.num_workers,
            pos_neg_ratio=args.pos_neg_ratio,
            max_slices_per_func=args.max_slices_per_func,
            use_cache=True
        )
    
    # 创建模型
    model = CoarseGrainedModel(
        func_codebert_path=func_codebert_path,
        slice_codebert_path=slice_codebert_path,
        hidden_dim=args.hidden_dim,
        k=args.k,
        lambda_mmr=args.lambda_mmr,
        ablation_mode=args.ablation_mode
    ).to(device)

    # 记录消融实验模式
    logger.info(f"消融实验模式: {args.ablation_mode}")

    # 冻结CodeBERT参数
    for param in model.func_encoder.codebert.parameters():
        param.requires_grad = False
    for param in model.slice_encoder.codebert.parameters():
        param.requires_grad = False

    # 如果使用GPU，启用cudnn基准测试
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # 创建loss函数
    if args.loss_type == 'weighted':
        train_dataset = train_loader.dataset
        func_label_dist = train_dataset.get_label_distribution('func')
        slice_label_dist = train_dataset.get_label_distribution('slice')
        
        func_pos_weight = func_label_dist[0] / func_label_dist[1] if func_label_dist[1] > 0 else 1.0
        
        # 为slice 预测使用更高的正样本权重，不基于实际分布
        # 这样可以更好地处理无漏洞函数中的大量负样本
        slice_pos_weight = args.slice_balance_weight * 5.0  # 使用固定倍数而不是基于分布
        
        criterion_func = WeightedBCELoss(pos_weight=func_pos_weight)
        criterion_slice = WeightedBCELoss(pos_weight=slice_pos_weight)
        
        logger.info(f"use WeightedBCELoss: func_pos_weight={func_pos_weight:.2f}, slice_pos_weight={slice_pos_weight:.2f} ")
    elif args.loss_type == 'focal':
        # 为slice 预测使用更强的焦点loss参数
        criterion_func = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        criterion_slice = FocalLoss(alpha=args.focal_alpha * args.slice_balance_weight * 2.0, gamma=args.focal_gamma + 1.0)
        
        logger.info(f"use FocalLoss: func alpha={args.focal_alpha}, gamma={args.focal_gamma}")
        logger.info(f"use FocalLoss: slice alpha={args.focal_alpha * args.slice_balance_weight * 2.0}, gamma={args.focal_gamma + 1.0}")
    else:
        criterion_func = nn.BCELoss()
        # 为slice 使用更高的正样本权重（对概率直接计算，使用 WeightedBCELoss）
        criterion_slice = WeightedBCELoss(pos_weight=args.slice_balance_weight * 5.0)

        logger.info(f"use BCELoss slice_balance_weight={args.slice_balance_weight * 5.0}")
    
    # 创建优化器 - 只优化未冻结的参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 创建学习率调度器
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    # 创建混合精度训练的scaler
    scaler = GradScaler()
    
    # 记录训练参数
    logger.info("train parameters:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    
    # 训练循环
    best_val_f1 = 0
    train_metrics = {
        'loss': [], 'func_acc': [], 'func_precision': [], 'func_recall': [], 'func_f1': [],
        'slice_acc': [], 'slice_precision': [], 'slice_recall': [], 'slice_f1': []
    }
    val_metrics = {
        'loss': [], 'func_acc': [], 'func_precision': [], 'func_recall': [], 'func_f1': [],
        'slice_acc': [], 'slice_precision': [], 'slice_recall': [], 'slice_f1': []
    }
    
    # 启用梯度检查点
    if hasattr(model, 'enable_gradient_checkpointing'):
        model.enable_gradient_checkpointing()
    
    no_improvement_count = 0
    PATIENCE = 5
    
    for epoch in range(1, args.epochs + 1):
        # 训练一个epoch
        train_loss, train_func_acc, train_func_precision, train_func_recall, train_func_f1, \
        train_slice_acc, train_slice_precision, train_slice_recall, train_slice_f1 = train_epoch(
            model, train_loader, criterion_func, criterion_slice, optimizer, scheduler, device, scaler, logger, epoch, args,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
        
        # 记录训练指标
        train_metrics['loss'].append(train_loss)
        train_metrics['func_acc'].append(train_func_acc)
        train_metrics['func_precision'].append(train_func_precision)
        train_metrics['func_recall'].append(train_func_recall)
        train_metrics['func_f1'].append(train_func_f1)
        train_metrics['slice_acc'].append(train_slice_acc)
        train_metrics['slice_precision'].append(train_slice_precision)
        train_metrics['slice_recall'].append(train_slice_recall)
        train_metrics['slice_f1'].append(train_slice_f1)
        
        # 验证
        val_loss, val_func_acc, val_func_precision, val_func_recall, val_func_f1, \
        val_slice_acc, val_slice_precision, val_slice_recall, val_slice_f1 = evaluate(
            model, val_loader, criterion_func, criterion_slice, device, logger, args
        )
        
        # 记录验证指标
        val_metrics['loss'].append(val_loss)
        val_metrics['func_acc'].append(val_func_acc)
        val_metrics['func_precision'].append(val_func_precision)
        val_metrics['func_recall'].append(val_func_recall)
        val_metrics['func_f1'].append(val_func_f1)
        val_metrics['slice_acc'].append(val_slice_acc)
        val_metrics['slice_precision'].append(val_slice_precision)
        val_metrics['slice_recall'].append(val_slice_recall)
        val_metrics['slice_f1'].append(val_slice_f1)
        
        # 绘制指标
        plot_metrics(train_metrics, val_metrics, args.output_dir)
        
        # 保存最佳模型（仅使用函数级 F1）
        val_f1 = val_func_f1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            logger.info(f"find better model! func F1: {val_f1:.4f}")
            
            # 保存模型
            model_save_path = os.path.join(args.output_dir, 'best_model')
            os.makedirs(model_save_path, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_val_f1': best_val_f1,
                'args': args
            }, os.path.join(model_save_path, 'checkpoint.pt'))
            logger.info(f"model saved at {model_save_path}")
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        if no_improvement_count >= PATIENCE:
            logger.info(f"early stopping at epoch {epoch} due to no improvement")
            break
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 加载最佳模型并在测试集上评估
    logger.info("load best model and evaluate on test set...")
    best_model_path = os.path.join(args.output_dir, 'best_model', 'checkpoint.pt')
    checkpoint = torch.load(best_model_path,weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 在测试集上评估
    test_loss, test_func_acc, test_func_precision, test_func_recall, test_func_f1, \
    test_slice_acc, test_slice_precision, test_slice_recall, test_slice_f1 = evaluate(
        model, test_loader, criterion_func, criterion_slice, device, logger, args, mode="test"
    )
    
    # 记录最终结果
    logger.info("final test results:")
    logger.info(
        f"func : acc={test_func_acc:.4f}, precision={test_func_precision:.4f}, "
        f"recall={test_func_recall:.4f}, F1={test_func_f1:.4f}"
    )
    logger.info(
        f"slice : acc={test_slice_acc:.4f}, precision={test_slice_precision:.4f}, "
        f"recall={test_slice_recall:.4f}, F1={test_slice_f1:.4f}"
    )
    
    return model, func_tokenizer, slice_tokenizer

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="func_slice level training")
    
    # 数据参数
    parser.add_argument("--data_root", type=str, default='E:\\zsm\\MGVD_exp\\Coarse_level\\dataset\\raw_slice', help="数据根目录")
    parser.add_argument("--output_dir", type=str, default="./coarse_outputV2", help="输出目录")
    parser.add_argument("--func_codebert_path", type=str, default="E:\\zsm\\ZSM_EXP\\Coarse_level\\func_best_model", help="func CodeBERT模型路径")
    parser.add_argument("--slice_codebert_path", type=str, default="E:\\zsm\\ZSM_EXP\\Coarse_level\\slice_best_model", help="slice CodeBERT模型路径")
    
    # 模型参数
    parser.add_argument("--hidden_dim", type=int, default=768, help="隐藏层维度")
    parser.add_argument("--k", type=int, default=3, help="选择的切片数量")
    parser.add_argument("--lambda_mmr", type=float, default=0.7, help="MMR多样性控制参数")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--max_length", type=int, default=256, help="最大序列长度")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载器工作进程数")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--warmup_steps", type=int, default=100, help="预热步数")
    parser.add_argument("--loss_type", type=str, choices=['standard', 'weighted', 'focal'], default='weighted', help="loss函数类型")
    parser.add_argument("--focal_alpha", type=float, default=0.25, help="焦点lossalpha参数")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="焦点lossgamma参数")
    parser.add_argument("--pos_neg_ratio", type=float, default=1.0, help="正负样本比例")
    
    # slice 学习参数
    parser.add_argument("--slice_loss_weight", type=float, default=0.3, help="slice loss权重")
    parser.add_argument("--func_loss_weight", type=float, default=0.7, help="func loss权重")
    parser.add_argument("--max_slices_per_func", type=int, default=10, help="每个函数最大切片数量")
    parser.add_argument("--slice_balance_weight", type=float, default=2.0, help="slice 正样本权重倍数")
    
    # 性能优化参数
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="梯度累积步数")
    parser.add_argument("--fp16", action="store_true", help="是否使用混合精度训练")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="是否使用梯度检查点")
    parser.add_argument("--pin_memory", action="store_true", help="是否使用固定内存")
    parser.add_argument("--prefetch_factor", type=int, default=4, help="数据预取因子")
    parser.add_argument("--use_pretokenized", action="store_true", help="是否使用预tokenize数据（推荐）")

    # 反事实正则化参数
    parser.add_argument("--use_cf_reg", action="store_true", help="是否使用反事实正则化")
    parser.add_argument("--cf_start_epoch", type=int, default=5, help="从第几个epoch开始反事实正则化")
    parser.add_argument("--cf_epoch_interval", type=int, default=2, help="每N个epoch启用一次反事实正则化")
    parser.add_argument("--cf_batch_interval", type=int, default=5, help="每N个batch启用一次反事实正则化")
    parser.add_argument("--cf_sample_ratio", type=float, default=0.3, help="反事实采样比例(0-1)")
    parser.add_argument("--cf_weight", type=float, default=0.15, help="反事实正则化权重")
    parser.add_argument("--cf_margin", type=float, default=0.2, help="反事实正则化margin")

    # 消融实验参数
    parser.add_argument("--ablation_mode", type=str, default='full',
                       choices=['func_only', 'slice_only', 'no_fusion', 'full'],
                       help="消融实验模式: func_only(只用函数), slice_only(只用切片), no_fusion(无融合), full(完整模型)")

    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logger(args.output_dir)
    logger.info("training start...")
    
    # 训练模型
    model, func_tokenizer, slice_tokenizer = train_coarse_model(args, logger)
    
    logger.info("training end!") 