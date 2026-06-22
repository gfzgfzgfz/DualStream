# 快速开始指南 (Quick Start)

本指南帮助您快速运行实验并复现论文结果。

---

## ⚡ 5分钟快速开始

### 1️⃣ 环境安装

```bash
# 创建环境
conda create -n vuldet python=3.8 -y
conda activate vuldet

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.30.0 numpy pandas scikit-learn tqdm matplotlib
```

### 2️⃣ 数据准备

```bash
cd chap2code

# 数据预处理（确保数据集文件夹中有 full_data_vul_lines_all.csv）
python process_data/get_raw_data.py
python process_data/normalize.py
```

### 3️⃣ 训练模型

**函数级检测**：
```bash
cd Coarse_level
python coarse_train.py \
    --data_root ../../dataset \
    --batch_size 16 \
    --epochs 30
```

**语句级检测**：
```bash
cd ../linelevel

# 先生成函数级特征缓存
python generate_coarse_cache.py \
    --coarse_model_path ../Coarse_level/output/best_model/checkpoint.pt

# 训练语句级模型
python fine_train.py \
    --coarse_cache_dir ./coarse_cache \
    --batch_size 16 \
    --epochs 30
```

---

## 📊 复现论文实验

### BigVul数据集实验

**1. 函数级对比实验（Table 5）**
```bash
cd chap2code/Coarse_level
python coarse_train.py --dataset bigvul
```

预期结果：F1 = 49.76%, Precision = 43.06%, Recall = 58.94%

**2. 语句级对比实验（Table 7）**
```bash
cd chap2code/linelevel
python fine_train.py --dataset bigvul
```

预期结果：IoU = 55.68%, Top-10 Acc = 73.28%

**3. 函数级消融实验（Table 9）**
```bash
cd chap2code/Coarse_level

# 测试各组件
python coarse_train_ablation.py --ablation_type no_slice      # w/o 切片输入
python coarse_train_ablation.py --ablation_type no_dualstream # w/o 双流编码器
python coarse_train_ablation.py --ablation_type no_filter     # w/o 切片过滤
python coarse_train_ablation.py --ablation_type no_fusion     # w/o 特征融合
```

**4. 语句级消融实验（Table 10-11）**
```bash
cd chap2code/linelevel

# 核心组件消融（Table 10）
python ablation_train.py --ablation_type no_context  # w/o Context
python ablation_train.py --ablation_type no_fusion   # w/o Fusion
python ablation_train.py --ablation_type no_gate     # w/o Gate

# 上下文组合消融（Table 11）
python ablation_train.py --context_config stmt_only       # Stmt Only
python ablation_train.py --context_config stmt_attr       # Stmt + Attr
python ablation_train.py --context_config stmt_attr_surr  # Stmt + Attr + Surr
python ablation_train.py --context_config stmt_attr_dep   # Stmt + Attr + Dep
python ablation_train.py --context_config full            # Full (所有上下文)
```

**5. 级联策略实验（Table 12）**
```bash
cd chap2code/linelevel

python cascade_experiment.py --strategy coarse_only  # Coarse Only
python cascade_experiment.py --strategy fine_only    # Fine Only
python cascade_experiment.py --strategy cascade      # Coarse→Fine
```

### FFmpeg+OpenSSL数据集实验

```bash
# 函数级（Table 6）
cd chap2code/Coarse_level
python coarse_train.py --dataset ffmpeg_openssl

# 语句级（Table 8）
cd chap2code/linelevel
python fine_train.py --dataset ffmpeg_openssl
```

预期结果：
- 函数级：F1 = 85.39%, Precision = 80.85%, Recall = 90.48%
- 语句级：IoU = 85.27%, Top-10 Acc = 91.22%

---

## 🎯 关键参数说明

### 训练参数

| 参数 | 默认值 | 说明 | 论文中的值 |
|-----|-------|------|-----------|
| `--batch_size` | 16 | 批次大小 | 16 |
| `--max_length` | 512 | 最大序列长度 | 512 |
| `--learning_rate` | 2e-5 | 学习率 | 2×10⁻⁵ |
| `--epochs` | 30 | 训练轮数 | 30 |
| `--hidden_dim` | 768 | 隐藏层维度 | 768 |
| `--k` | 3 | 切片保留数量 | 3 |
| `--context_window` | 2 | 周围上下文窗口 | ±2 |

### GPU内存需求

| 配置 | 最小GPU内存 | 推荐GPU |
|-----|-----------|---------|
| batch_size=8, max_length=256 | 8GB | RTX 2080 |
| batch_size=16, max_length=512 | 16GB | RTX 3090 |
| batch_size=32, max_length=512 | 24GB | RTX 4090 |

如果GPU内存不足：
```bash
# 减小批次大小
python coarse_train.py --batch_size 8

# 减小序列长度
python coarse_train.py --max_length 256

# 启用梯度累积
python coarse_train.py --batch_size 8 --gradient_accumulation_steps 2
```

---

## 📁 输出文件说明

### 函数级模型输出

```
Coarse_level/output/
├── best_model/
│   ├── checkpoint.pt              # 最佳模型权重
│   ├── func_tokenizer/            # 函数编码器tokenizer
│   └── slice_tokenizer/           # 切片编码器tokenizer
├── checkpoints/
│   ├── epoch_1.pt                 # 各轮检查点
│   ├── epoch_2.pt
│   └── ...
└── logs/
    ├── training.log               # 训练日志
    └── results.json               # 评估结果
```

### 语句级模型输出

```
linelevel/output/
├── best_model/
│   └── checkpoint.pt              # 最佳模型权重
├── coarse_cache/
│   ├── train_cache.pt             # 训练集函数级特征缓存
│   ├── val_cache.pt               # 验证集函数级特征缓存
│   └── test_cache.pt              # 测试集函数级特征缓存
└── logs/
    ├── training.log               # 训练日志
    └── results.json               # 评估结果
```

---

## 🔧 常见问题快速解决

### 问题1：CUDA out of memory

```bash
# 方案1：减小批次
python coarse_train.py --batch_size 8

# 方案2：减小序列长度
python coarse_train.py --max_length 256

# 方案3：使用CPU（速度较慢）
python coarse_train.py --device cpu
```

### 问题2：找不到数据集

```bash
# 确认数据集位置
ls -la 数据集/full_data_vul_lines_all.csv

# 如果数据集不在默认位置，指定路径
python coarse_train.py --data_root /path/to/your/dataset
```

### 问题3：预训练模型下载失败

```bash
# 使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载后指定本地路径
python coarse_train.py \
    --func_codebert_path /path/to/codebert-base \
    --slice_codebert_path /path/to/codebert-base
```

### 问题4：训练速度慢

```bash
# 启用混合精度训练
python coarse_train.py --fp16

# 减少数据加载进程
python coarse_train.py --num_workers 4

# 使用预tokenized数据
python preprocess_tokenize.py  # 先生成缓存
python coarse_train.py --use_cache
```

---

## 📊 评估已训练模型

如果你已经有训练好的模型，直接评估：

```bash
# 函数级评估
cd chap2code/Coarse_level
python evaluate.py \
    --model_path ./output/best_model/checkpoint.pt \
    --data_root ../../dataset \
    --split test

# 语句级评估
cd chap2code/linelevel
python evaluate.py \
    --model_path ./output/best_model/checkpoint.pt \
    --coarse_cache_dir ./coarse_cache \
    --split test
```

---

## 💡 使用预训练模型推理

检测单个函数是否有漏洞：

```python
from chap2code.Coarse_level.coarse_model import CoarseGrainedModel
import torch

# 加载模型
model = CoarseGrainedModel.from_pretrained('path/to/checkpoint.pt')
model.eval()

# 待检测的函数
func_code = """
int vulnerable_function(char *input) {
    char buffer[32];
    strcpy(buffer, input);  // 可能的缓冲区溢出
    return 0;
}
"""

# 推理
result = model.predict(func_code)
print(f"漏洞概率: {result['probability']:.2%}")
if result['is_vulnerable']:
    print(f"检测到漏洞！可疑行: {result['suspicious_lines']}")
```

---

## 📞 获取帮助

遇到问题？

1. 查看详细文档：[README.md](README.md)
2. 检查训练日志：`output/logs/training.log`
3. 提交Issue：包含错误信息和运行环境

---

## ✅ 检查清单

训练前请确认：

- [ ] Python 3.8+ 环境已安装
- [ ] PyTorch + CUDA 已正确配置
- [ ] 数据集文件存在且格式正确
- [ ] 有足够的GPU内存（至少8GB）
- [ ] 有足够的磁盘空间（至少50GB）

---

**祝实验顺利！** 🚀
