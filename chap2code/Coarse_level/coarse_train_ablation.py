import os
import argparse

import coarse_train as ct
import coarse_data_ablation as cda

ct.create_coarse_dataloaders = cda.create_coarse_dataloaders

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="func_slice level training (ablation data sampling)")

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

    # 切片采样参数（消融版数据加载专用）
    parser.add_argument("--slice_neg_per_pos", type=float, default=cda.DEFAULT_SLICE_NEG_PER_POS,
                        help="切片负样本/正样本比例（如4表示1:4）")

    args = parser.parse_args()

    # 应用切片采样比例
    cda.DEFAULT_SLICE_NEG_PER_POS = args.slice_neg_per_pos

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置日志
    logger = ct.setup_logger(args.output_dir)
    logger.info("training start (ablation data sampling)...")

    # 训练模型
    ct.train_coarse_model(args, logger)

    logger.info("training end!")
