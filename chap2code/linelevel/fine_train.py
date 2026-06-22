import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
import sys
from torch.amp import autocast, GradScaler

# 添加项目根目录到系统路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

# 导入本地模块
from .fine_model import FineGrainedModel
from .fine_data import create_fine_dataloaders, create_fine_dataloaders_with_coarse

def setup_logger(output_dir):
    """设置日志"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    log_file = os.path.join(output_dir, 'training.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

class WeightedBCELoss(nn.Module):
    """带权重的二元交叉熵损失"""
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, pred, target):
        # 计算正样本权重
        if self.pos_weight is None:
            pos_weight = torch.tensor([target.sum() / (len(target) - target.sum())])
        else:
            pos_weight = self.pos_weight
            
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)(pred, target.float())

def calculate_metrics(y_true, y_pred, threshold=0.5):
    """计算评估指标"""
    y_pred_binary = (y_pred > threshold).astype(int)

    # 基础指标
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_binary, average='binary', zero_division=0)
    accuracy = np.mean(y_true == y_pred_binary)

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred_binary)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # 处理只有一个类别的情况
        tn, fp, fn, tp = 0, 0, 0, 0
        if len(np.unique(y_true)) == 1:
            if y_true[0] == 0:
                tn = len(y_true)
            else:
                tp = len(y_true)

    # 计算FNR和FPR
    fnr = fn / (fn + tp + 1e-10)  # False Negative Rate = FN / (FN + TP)
    fpr = fp / (fp + tn + 1e-10)  # False Positive Rate = FP / (FP + TN)

    # 计算IoU
    intersection = np.sum((y_true == 1) & (y_pred_binary == 1))
    union = np.sum((y_true == 1) | (y_pred_binary == 1))
    iou = intersection / (union + 1e-10)

    # 计算top-k准确率
    k = min(3, len(y_true))
    top_k_indices = np.argsort(y_pred)[-k:]
    top_k_accuracy = np.mean(y_true[top_k_indices] == 1)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'fnr': fnr,
        'fpr': fpr,
        'iou': iou,
        'top_k_accuracy': top_k_accuracy,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }

def train_epoch(model, train_loader, optimizer, scheduler, device, scaler, logging, epoch, use_interaction=False, gradient_accumulation_steps=1):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_reg_loss = 0
    all_probs = []
    all_targets = []

    # 创建损失函数 - 使用BCEWithLogitsLoss，在autocast下更安全
    criterion = nn.BCEWithLogitsLoss()

    # 创建进度条
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')

    for batch_idx, batch in enumerate(progress_bar):
        # 准备输入数据
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 使用自动混合精度训练
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            if use_interaction:
                # 交互模式 - 使用缓存的粗粒度特征
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

                # 使用logits计算损失
                line_logits = outputs['line_logits']
                loss = criterion(line_logits, labels)

                # 添加正则化损失
                reg_loss = outputs['reg_loss'].mean()
                loss = loss + reg_loss
                total_reg_loss += reg_loss.item()
            else:
                # 独立模式
                outputs = model(input_ids, attention_mask)
                line_logits = outputs['line_logits']
                loss = criterion(line_logits, labels)

            loss = loss / gradient_accumulation_steps

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度累积
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 更新参数
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        # 记录损失
        total_loss += loss.item() * gradient_accumulation_steps

        # 收集预测结果（使用probs用于指标计算）
        all_probs.extend(outputs['line_probs'].detach().cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

        # 更新进度条
        progress_bar.set_postfix({
            'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })

        # 每100个batch记录一次日志
        if (batch_idx + 1) % 100 == 0:
            logging.info(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}, "
                        f"Loss: {loss.item() * gradient_accumulation_steps:.4f}, "
                        f"LR: {scheduler.get_last_lr()[0]:.2e}")

    # 计算指标
    metrics = calculate_metrics(np.array(all_targets), np.array(all_probs))

    result = {
        'loss': total_loss / len(train_loader),
        'line_metrics': metrics
    }

    if use_interaction:
        result['reg_loss'] = total_reg_loss / len(train_loader)

    return result

def evaluate(model, data_loader, device, use_interaction=False):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_reg_loss = 0
    all_probs = []
    all_targets = []

    # 创建损失函数
    criterion = nn.BCEWithLogitsLoss()

    # 创建进度条
    progress_bar = tqdm(data_loader, desc='Evaluating')

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # 准备输入数据
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            if use_interaction:
                # 交互模式 - 使用缓存的粗粒度特征
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

                # 使用logits计算损失
                line_logits = outputs['line_logits']
                loss = criterion(line_logits, labels)

                # 添加正则化损失
                reg_loss = outputs['reg_loss'].mean()
                loss = loss + reg_loss
                total_reg_loss += reg_loss.item()
            else:
                # 独立模式评估
                outputs = model(input_ids, attention_mask)
                line_logits = outputs['line_logits']
                loss = criterion(line_logits, labels)

            total_loss += loss.item()

            # 收集预测结果（使用probs）
            all_probs.extend(outputs['line_probs'].cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

            # 更新进度条
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

            # 每100个batch记录一次日志
            if (batch_idx + 1) % 100 == 0:
                logging.info(f"Batch {batch_idx + 1}/{len(data_loader)}, "
                            f"Loss: {loss.item():.4f}")

    # 计算指标
    metrics = calculate_metrics(np.array(all_targets), np.array(all_probs))

    result = {
        'loss': total_loss / len(data_loader),
        'line_metrics': metrics
    }

    if use_interaction:
        result['reg_loss'] = total_reg_loss / len(data_loader)

    return result

def test(model, test_loader, device, use_interaction=False):
    """测试模型"""
    model.eval()
    all_targets = []
    all_probs = []

    # 创建进度条
    progress_bar = tqdm(test_loader, desc='Testing')

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # 准备输入数据
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            if use_interaction:
                # 交互模式 - 使用缓存的粗粒度特征
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
                # 独立模式测试
                outputs = model(input_ids, attention_mask)

            # 获取预测概率
            line_probs = outputs['line_probs']

            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(line_probs.cpu().numpy())

            # 更新进度条
            progress_bar.set_postfix({'processed': f"{batch_idx + 1}/{len(test_loader)}"})

            # 每100个batch记录一次日志
            if (batch_idx + 1) % 100 == 0:
                logging.info(f"Batch {batch_idx + 1}/{len(test_loader)}")

    # 计算指标
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    metrics = calculate_metrics(all_targets, all_probs)

    return {
        'line_metrics': metrics,
        'predictions': (all_probs > 0.5).astype(int),
        'probabilities': all_probs,
        'targets': all_targets
    }

def plot_metrics(train_metrics, val_metrics, output_dir):
    """绘制训练指标"""
    metrics = ['loss', 'f1', 'precision', 'recall', 'accuracy']
    
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        plt.plot(train_metrics[metric], label='Train')
        plt.plot(val_metrics[metric], label='Validation')
        plt.title(f'{metric.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    plt.close()

def train_fine_model(args):
    """训练细粒度模型"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 设置日志
    setup_logger(args.output_dir)
    logging.info(f"使用设备: {device}")

    # 如果使用GPU，启用cudnn基准测试
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # 加载tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.codebert_path)

    # 创建数据加载器
    if args.use_interaction:
        # 交互模式 - 使用带粗粒度缓存的数据加载器
        if not args.coarse_cache_dir:
            raise ValueError("交互模式下必须提供粗粒度缓存目录 (--coarse_cache_dir)")

        logging.info(f"使用交互模式，加载粗粒度缓存: {args.coarse_cache_dir}")
        train_loader, val_loader, test_loader = create_fine_dataloaders_with_coarse(
            args.data_root,
            args.coarse_cache_dir,
            tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=args.prefetch_factor,
            max_slices=args.max_slices
        )
    else:
        # 独立模式
        logging.info("使用独立模式")
        train_loader, val_loader, test_loader = create_fine_dataloaders(
            args.data_root,
            tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=args.prefetch_factor
        )

    # 创建模型
    model = FineGrainedModel(
        codebert_path=args.codebert_path,
        hidden_dim=args.hidden_dim,
        alpha=args.alpha,
        delta=args.delta,
        lambda_threshold=args.lambda_threshold,
        reg_weight=args.reg_weight,
        use_interaction=args.use_interaction,
        dropout=args.dropout
    ).to(device)

    # 准备优化器和学习率调度器
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

    # 计算总训练步数
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps

    # 创建学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    # 创建混合精度训练的scaler
    scaler = GradScaler()

    # 记录训练参数
    logging.info("训练参数:")
    for k, v in vars(args).items():
        logging.info(f"  {k}: {v}")

    # 训练循环
    best_val_f1 = 0
    metrics = ['precision', 'recall', 'f1', 'accuracy', 'fnr', 'fpr', 'iou', 'top_k_accuracy', 'tp', 'fp', 'tn', 'fn']
    train_metrics = {metric: [] for metric in metrics}
    val_metrics = {metric: [] for metric in metrics}
    train_metrics['loss'] = []
    val_metrics['loss'] = []

    for epoch in range(args.epochs):
        # 训练
        train_results = train_epoch(
            model, train_loader, optimizer, scheduler, device, scaler, logging, epoch,
            use_interaction=args.use_interaction,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )

        # 验证
        val_results = evaluate(
            model, val_loader, device,
            use_interaction=args.use_interaction
        )

        # 记录指标
        train_metrics['loss'].append(train_results['loss'])
        val_metrics['loss'].append(val_results['loss'])
        for metric in metrics:
            if metric in train_results['line_metrics']:
                train_metrics[metric].append(train_results['line_metrics'][metric])
                val_metrics[metric].append(val_results['line_metrics'][metric])

        # 记录日志
        logging.info(f"Epoch {epoch + 1}/{args.epochs}")
        log_msg = f"Train - Loss: {train_results['loss']:.4f}, F1: {train_results['line_metrics']['f1']:.4f}, FNR: {train_results['line_metrics']['fnr']:.4f}, FPR: {train_results['line_metrics']['fpr']:.4f}"
        if args.use_interaction and 'reg_loss' in train_results:
            log_msg += f", RegLoss: {train_results['reg_loss']:.4f}"
        logging.info(log_msg)

        log_msg = f"Val - Loss: {val_results['loss']:.4f}, F1: {val_results['line_metrics']['f1']:.4f}, FNR: {val_results['line_metrics']['fnr']:.4f}, FPR: {val_results['line_metrics']['fpr']:.4f}"
        logging.info(log_msg)

        # 保存最佳模型
        if val_results['line_metrics']['f1'] > best_val_f1:
            best_val_f1 = val_results['line_metrics']['f1']
            model_save_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_val_f1': best_val_f1,
                'args': args
            }, model_save_path)
            logging.info(f"保存最佳模型，验证集F1: {best_val_f1:.4f}")

    # 绘制训练指标
    plot_metrics(train_metrics, val_metrics, args.output_dir)

    # 加载最佳模型进行测试
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    test_results = test(
        model, test_loader, device,
        use_interaction=args.use_interaction
    )

    # 记录测试结果
    logging.info("测试集结果:")
    logging.info(f"F1: {test_results['line_metrics']['f1']:.4f}")
    logging.info(f"Precision: {test_results['line_metrics']['precision']:.4f}")
    logging.info(f"Recall: {test_results['line_metrics']['recall']:.4f}")
    logging.info(f"Accuracy: {test_results['line_metrics']['accuracy']:.4f}")
    logging.info(f"FNR: {test_results['line_metrics']['fnr']:.4f}")
    logging.info(f"FPR: {test_results['line_metrics']['fpr']:.4f}")
    logging.info(f"IoU: {test_results['line_metrics']['iou']:.4f}")
    logging.info(f"Top-k Accuracy: {test_results['line_metrics']['top_k_accuracy']:.4f}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='细粒度漏洞检测模型训练')

    # 数据参数
    parser.add_argument("--data_root", type=str, default='./debug_data', help="数据根目录")
    parser.add_argument("--output_dir", type=str, default="./fine_output", help="输出目录")
    parser.add_argument('--codebert_path', type=str, default='/mnt/e/zsm/remote/model/bigvul/line_codebert_finetuned', help='CodeBERT模型路径')

    # 交互模式参数
    parser.add_argument('--use_interaction', action='store_true', help='是否使用粗细粒度交互')
    parser.add_argument('--coarse_cache_dir', type=str, default=None, help='粗粒度特征缓存目录（交互模式必需）')
    parser.add_argument('--max_slices', type=int, default=5, help='每个样本使用的最大切片数')

    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=768, help='隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--alpha', type=float, default=0.3, help='全局语义增强系数')
    parser.add_argument('--delta', type=float, default=0.5, help='先验概率调节系数')
    parser.add_argument('--lambda_threshold', type=float, default=0.2, help='动态阈值调节系数')
    parser.add_argument('--reg_weight', type=float, default=0.1, help='正则化损失权重')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列长度')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作进程数')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--warmup_steps', type=int, default=500, help='预热步数')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Adam优化器的epsilon参数')

    # 性能优化参数
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='梯度累积步数')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='数据预取因子')

    args = parser.parse_args()

    # 验证参数
    if args.use_interaction and not args.coarse_cache_dir:
        raise ValueError("交互模式下必须提供粗粒度缓存目录 (--coarse_cache_dir)")

    train_fine_model(args) 