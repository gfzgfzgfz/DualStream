"""
消融实验训练脚本

支持6种消融配置:
- E1 (base): 独立模式，无粗粒度信息
- E2 (func_feat): 仅使用函数特征增强
- E3 (slice_attn): 仅使用切片注意力
- E4 (prob_guide): 仅使用概率引导
- E5 (full_no_reg): 完整模型但无正则化
- E6 (full): 完整模型

使用方法:
    # E1 基线 (不需要缓存)
    python ablation_train.py --ablation_mode base --data_root ./debug_data --output_dir ./ablation_output/E1_base

    # E2-E6 (需要缓存)
    python ablation_train.py --ablation_mode func_feat --data_root ./debug_data --coarse_cache_dir ./coarse_cache --output_dir ./ablation_output/E2_func_feat
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import logging
import json
import argparse
from datetime import datetime
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# 添加项目根目录到系统路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from line_level.ablation_model import AblationModel
from line_level.fine_data import create_fine_dataloaders, create_fine_dataloaders_with_coarse


def setup_logger(output_dir, exp_name):
    """设置日志"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_file = os.path.join(output_dir, f'{exp_name}_training.log')

    # 清除之前的handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def calculate_metrics(y_true, y_pred, sample_ids=None, threshold=0.5):
    """计算评估指标

    Args:
        y_true: 真实标签数组
        y_pred: 预测概率数组
        sample_ids: 每行对应的样本ID列表（用于按样本计算Top-K）
        threshold: 分类阈值

    返回: precision, recall, f1, accuracy, fnr, fpr, iou, top_k_accuracy
    """
    from collections import defaultdict

    y_pred_binary = (y_pred > threshold).astype(int)

    # 基础指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_binary, average='binary', zero_division=0
    )
    accuracy = np.mean(y_true == y_pred_binary)

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred_binary)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        if len(np.unique(y_true)) == 1:
            if y_true[0] == 0:
                tn = len(y_true)
            else:
                tp = len(y_true)

    # FNR和FPR
    fnr = fn / (fn + tp + 1e-10)
    fpr = fp / (fp + tn + 1e-10)

    # IoU
    intersection = np.sum((y_true == 1) & (y_pred_binary == 1))
    union = np.sum((y_true == 1) | (y_pred_binary == 1))
    iou = intersection / (union + 1e-10)

    # Top-K准确率 (K=1,3,5,10) - 按样本计算
    top_k_results = {}
    if sample_ids is not None:
        # 按样本分组计算Top-K
        sample_data = defaultdict(lambda: {'y_true': [], 'y_pred': []})
        for i, sid in enumerate(sample_ids):
            sample_data[sid]['y_true'].append(y_true[i])
            sample_data[sid]['y_pred'].append(y_pred[i])

        for k in [1, 3, 5, 10]:
            top_k_scores = []
            for sid, data in sample_data.items():
                s_true = np.array(data['y_true'])
                s_pred = np.array(data['y_pred'])

                # 只对有漏洞的样本计算Top-K
                if np.sum(s_true) == 0:
                    continue

                if len(s_true) >= k:
                    top_k_indices = np.argsort(s_pred)[-k:]
                    # Top-K中命中的漏洞行比例
                    top_k_hit = np.sum(s_true[top_k_indices]) / min(k, np.sum(s_true))
                    top_k_scores.append(min(top_k_hit, 1.0))

            if top_k_scores:
                top_k_results[f'top_{k}_accuracy'] = np.mean(top_k_scores)
            else:
                top_k_results[f'top_{k}_accuracy'] = 0.0
    else:
        # 兼容旧逻辑（全局计算，不推荐）
        for k in [1, 3, 5, 10]:
            if len(y_true) >= k:
                top_k_indices = np.argsort(y_pred)[-k:]
                top_k_accuracy = np.mean(y_true[top_k_indices] == 1)
                top_k_results[f'top_{k}_accuracy'] = top_k_accuracy
            else:
                top_k_results[f'top_{k}_accuracy'] = 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'fnr': fnr,
        'fpr': fpr,
        'iou': iou,
        **top_k_results,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }


def train_epoch(model, train_loader, optimizer, scheduler, device, scaler,
                ablation_mode, epoch, gradient_accumulation_steps=1):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_reg_loss = 0
    all_probs = []
    all_targets = []
    all_sample_ids = []

    criterion = nn.BCEWithLogitsLoss()
    needs_cache = ablation_mode in AblationModel.MODES_NEED_CACHE

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')

    for batch_idx, batch in enumerate(progress_bar):
        # 准备输入数据
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            if needs_cache:
                # 需要粗粒度缓存的模式
                func_features = batch['func_features'].to(device)
                func_probs = batch['func_prob'].to(device)
                slice_features = batch['slice_features'].to(device)
                slice_probs = batch['slice_probs'].to(device)
                slice_line_mask = batch['slice_line_mask'].to(device)

                outputs = model(
                    input_ids, attention_mask,
                    func_features=func_features,
                    slice_features=slice_features,
                    slice_probs=slice_probs,
                    func_probs=func_probs,
                    slice_line_mask=slice_line_mask
                )
            else:
                # E1 base模式，不需要缓存
                outputs = model(input_ids, attention_mask)

            # 计算损失
            line_logits = outputs['line_logits']
            loss = criterion(line_logits, labels)

            # 添加正则化损失
            reg_loss = outputs['reg_loss'].mean()
            loss = loss + reg_loss
            total_reg_loss += reg_loss.item()

            loss = loss / gradient_accumulation_steps

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度累积
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * gradient_accumulation_steps

        # 收集预测结果
        all_probs.extend(outputs['line_probs'].detach().cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        # 收集样本ID
        if 'func_file_id' in batch:
            all_sample_ids.extend(batch['func_file_id'])
        elif 'func_id' in batch:
            all_sample_ids.extend(batch['func_id'])

        # 更新进度条
        progress_bar.set_postfix({
            'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })

    # 计算指标
    sample_ids = all_sample_ids if all_sample_ids else None
    metrics = calculate_metrics(np.array(all_targets), np.array(all_probs), sample_ids)

    return {
        'loss': total_loss / len(train_loader),
        'reg_loss': total_reg_loss / len(train_loader),
        'metrics': metrics
    }


def evaluate(model, data_loader, device, ablation_mode):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_probs = []
    all_targets = []
    all_sample_ids = []

    criterion = nn.BCEWithLogitsLoss()
    needs_cache = ablation_mode in AblationModel.MODES_NEED_CACHE

    progress_bar = tqdm(data_loader, desc='Evaluating')

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            if needs_cache:
                func_features = batch['func_features'].to(device)
                func_probs = batch['func_prob'].to(device)
                slice_features = batch['slice_features'].to(device)
                slice_probs = batch['slice_probs'].to(device)
                slice_line_mask = batch['slice_line_mask'].to(device)

                outputs = model(
                    input_ids, attention_mask,
                    func_features=func_features,
                    slice_features=slice_features,
                    slice_probs=slice_probs,
                    func_probs=func_probs,
                    slice_line_mask=slice_line_mask
                )
            else:
                outputs = model(input_ids, attention_mask)

            line_logits = outputs['line_logits']
            loss = criterion(line_logits, labels)
            total_loss += loss.item()

            all_probs.extend(outputs['line_probs'].cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            # 收集样本ID
            if 'func_file_id' in batch:
                all_sample_ids.extend(batch['func_file_id'])
            elif 'func_id' in batch:
                all_sample_ids.extend(batch['func_id'])

            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    sample_ids = all_sample_ids if all_sample_ids else None
    metrics = calculate_metrics(np.array(all_targets), np.array(all_probs), sample_ids)

    return {
        'loss': total_loss / len(data_loader),
        'metrics': metrics
    }


def test(model, test_loader, device, ablation_mode):
    """测试模型"""
    model.eval()
    all_targets = []
    all_probs = []
    all_sample_ids = []

    needs_cache = ablation_mode in AblationModel.MODES_NEED_CACHE
    progress_bar = tqdm(test_loader, desc='Testing')

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            if needs_cache:
                func_features = batch['func_features'].to(device)
                func_probs = batch['func_prob'].to(device)
                slice_features = batch['slice_features'].to(device)
                slice_probs = batch['slice_probs'].to(device)
                slice_line_mask = batch['slice_line_mask'].to(device)

                outputs = model(
                    input_ids, attention_mask,
                    func_features=func_features,
                    slice_features=slice_features,
                    slice_probs=slice_probs,
                    func_probs=func_probs,
                    slice_line_mask=slice_line_mask
                )
            else:
                outputs = model(input_ids, attention_mask)

            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(outputs['line_probs'].cpu().numpy())
            # 收集样本ID
            if 'func_file_id' in batch:
                all_sample_ids.extend(batch['func_file_id'])
            elif 'func_id' in batch:
                all_sample_ids.extend(batch['func_id'])

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    sample_ids = all_sample_ids if all_sample_ids else None
    metrics = calculate_metrics(all_targets, all_probs, sample_ids)

    return {
        'metrics': metrics,
        'predictions': (all_probs > 0.5).astype(int),
        'probabilities': all_probs,
        'targets': all_targets
    }


def save_results(results, output_dir, exp_name):
    """保存实验结果"""
    results_file = os.path.join(output_dir, f'{exp_name}_results.json')

    # 转换numpy类型为Python原生类型
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj

    with open(results_file, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)

    logging.info(f"结果已保存到: {results_file}")


def run_ablation_experiment(args):
    """运行消融实验"""
    # 实验名称映射
    exp_names = {
        'base': 'E1_base',
        'func_feat': 'E2_func_feat',
        'slice_attn': 'E3_slice_attn',
        'prob_guide': 'E4_prob_guide',
        'full_no_reg': 'E5_full_no_reg',
        'full': 'E6_full'
    }
    exp_name = exp_names[args.ablation_mode]

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 设置日志
    setup_logger(args.output_dir, exp_name)
    logging.info(f"=" * 60)
    logging.info(f"消融实验: {exp_name}")
    logging.info(f"模式: {args.ablation_mode}")
    logging.info(f"设备: {device}")
    logging.info(f"=" * 60)

    # GPU优化
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # 加载tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.codebert_path)

    # 创建数据加载器
    needs_cache = args.ablation_mode in AblationModel.MODES_NEED_CACHE

    if needs_cache:
        if not args.coarse_cache_dir:
            raise ValueError(f"模式 {args.ablation_mode} 需要粗粒度缓存，请提供 --coarse_cache_dir")

        logging.info(f"使用粗粒度缓存: {args.coarse_cache_dir}")
        train_loader, val_loader, test_loader = create_fine_dataloaders_with_coarse(
            args.data_root,
            args.coarse_cache_dir,
            tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            max_slices=args.max_slices
        )
    else:
        logging.info("使用独立模式数据加载器 (无粗粒度缓存)")
        train_loader, val_loader, test_loader = create_fine_dataloaders(
            args.data_root,
            tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None
        )

    # 创建模型
    model = AblationModel(
        codebert_path=args.codebert_path,
        hidden_dim=args.hidden_dim,
        alpha=args.alpha,
        delta=args.delta,
        lambda_threshold=args.lambda_threshold,
        reg_weight=args.reg_weight,
        dropout=args.dropout,
        ablation_mode=args.ablation_mode
    ).to(device)

    # 优化器
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon,
        betas=(0.9, 0.999)
    )

    # 学习率调度器
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    # 混合精度训练
    scaler = GradScaler()

    # 记录训练参数
    logging.info("训练参数:")
    for k, v in vars(args).items():
        logging.info(f"  {k}: {v}")

    # 训练循环
    best_val_f1 = 0
    best_epoch = 0
    train_history = []
    val_history = []

    for epoch in range(args.epochs):
        # 训练
        train_results = train_epoch(
            model, train_loader, optimizer, scheduler, device, scaler,
            args.ablation_mode, epoch,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )

        # 验证
        val_results = evaluate(model, val_loader, device, args.ablation_mode)

        # 记录历史
        train_history.append(train_results)
        val_history.append(val_results)

        # 日志 - 输出完整指标
        logging.info(f"Epoch {epoch + 1}/{args.epochs}")
        train_m = train_results['metrics']
        logging.info(f"  Train - Loss: {train_results['loss']:.4f}, "
                     f"Precision: {train_m['precision']:.4f}, Recall: {train_m['recall']:.4f}, "
                     f"F1: {train_m['f1']:.4f}, IoU: {train_m['iou']:.4f}, "
                     f"FNR: {train_m['fnr']:.4f}, FPR: {train_m['fpr']:.4f}, "
                     f"Top-1: {train_m['top_1_accuracy']:.4f}, Top-3: {train_m['top_3_accuracy']:.4f}, "
                     f"Top-5: {train_m['top_5_accuracy']:.4f}, Top-10: {train_m['top_10_accuracy']:.4f}")
        val_m = val_results['metrics']
        logging.info(f"  Val   - Loss: {val_results['loss']:.4f}, "
                     f"Precision: {val_m['precision']:.4f}, Recall: {val_m['recall']:.4f}, "
                     f"F1: {val_m['f1']:.4f}, IoU: {val_m['iou']:.4f}, "
                     f"FNR: {val_m['fnr']:.4f}, FPR: {val_m['fpr']:.4f}, "
                     f"Top-1: {val_m['top_1_accuracy']:.4f}, Top-3: {val_m['top_3_accuracy']:.4f}, "
                     f"Top-5: {val_m['top_5_accuracy']:.4f}, Top-10: {val_m['top_10_accuracy']:.4f}")

        # 保存最佳模型
        if val_results['metrics']['f1'] > best_val_f1:
            best_val_f1 = val_results['metrics']['f1']
            best_epoch = epoch + 1
            model_save_path = os.path.join(args.output_dir, f'{exp_name}_best_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_val_f1': best_val_f1,
                'ablation_mode': args.ablation_mode,
                'args': vars(args)
            }, model_save_path)
            logging.info(f"  保存最佳模型，验证集F1: {best_val_f1:.4f}")

    # 加载最佳模型进行测试
    logging.info(f"\n加载最佳模型 (Epoch {best_epoch}) 进行测试...")
    checkpoint = torch.load(os.path.join(args.output_dir, f'{exp_name}_best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_results = test(model, test_loader, device, args.ablation_mode)

    # 记录测试结果
    logging.info(f"\n{'=' * 60}")
    logging.info(f"测试集结果 ({exp_name}):")
    logging.info(f"{'=' * 60}")
    logging.info(f"  Precision:      {test_results['metrics']['precision']:.4f}")
    logging.info(f"  Recall:         {test_results['metrics']['recall']:.4f}")
    logging.info(f"  F1:             {test_results['metrics']['f1']:.4f}")
    logging.info(f"  IoU:            {test_results['metrics']['iou']:.4f}")
    logging.info(f"  FNR:            {test_results['metrics']['fnr']:.4f}")
    logging.info(f"  FPR:            {test_results['metrics']['fpr']:.4f}")
    logging.info(f"  Top-1 Accuracy: {test_results['metrics']['top_1_accuracy']:.4f}")
    logging.info(f"  Top-3 Accuracy: {test_results['metrics']['top_3_accuracy']:.4f}")
    logging.info(f"  Top-5 Accuracy: {test_results['metrics']['top_5_accuracy']:.4f}")
    logging.info(f"  Top-10 Accuracy:{test_results['metrics']['top_10_accuracy']:.4f}")
    logging.info(f"{'=' * 60}")

    # 保存完整结果
    final_results = {
        'experiment': exp_name,
        'ablation_mode': args.ablation_mode,
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        'test_metrics': test_results['metrics'],
        'args': vars(args),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    save_results(final_results, args.output_dir, exp_name)

    return final_results


def main():
    parser = argparse.ArgumentParser(description='消融实验训练脚本')

    # 消融模式
    parser.add_argument('--ablation_mode', type=str, required=True,
                        choices=['base', 'func_feat', 'slice_attn', 'prob_guide', 'full_no_reg', 'full'],
                        help='消融模式: base(E1), func_feat(E2), slice_attn(E3), prob_guide(E4), full_no_reg(E5), full(E6)')

    # 数据参数
    parser.add_argument('--data_root', type=str, required=True, help='数据根目录')
    parser.add_argument('--coarse_cache_dir', type=str, default=None, help='粗粒度特征缓存目录 (E2-E6需要)')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--codebert_path', type=str, required=True, help='CodeBERT模型路径')

    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=768, help='隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--alpha', type=float, default=0.3, help='全局语义增强系数 (E2,E5,E6)')
    parser.add_argument('--delta', type=float, default=0.5, help='先验概率调节系数 (E4,E5,E6)')
    parser.add_argument('--lambda_threshold', type=float, default=0.2, help='动态阈值调节系数 (E4,E5,E6)')
    parser.add_argument('--reg_weight', type=float, default=0.1, help='正则化损失权重 (仅E6)')
    parser.add_argument('--max_slices', type=int, default=5, help='每个样本使用的最大切片数')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列长度')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作进程数')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--warmup_steps', type=int, default=500, help='预热步数')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Adam epsilon')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='梯度累积步数')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='数据预取因子')

    args = parser.parse_args()

    # 运行实验
    run_ablation_experiment(args)


if __name__ == '__main__':
    main()
