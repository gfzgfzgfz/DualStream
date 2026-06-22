# DualStream-based Vulnerability Detection 实验指南

本项目为论文《DualStream-based Vulnerability Detection for Multi-Granularity Feature Fusion》的完整实验代码和数据集。

---

## 📋 目录结构

```
张丝敏毕设数据集及代码/
├── chap2code/                    # 论文相关代码（主要）
│   ├── Coarse_level/            # 函数级漏洞检测
│   │   ├── coarse_model.py      # 核心模型（双流编码器、切片过滤、层次融合）
│   │   ├── coarse_train.py      # 训练脚本
│   │   ├── coarse_train_ablation.py  # 消融实验
│   │   ├── coarse_data.py       # 数据加载
│   │   └── ...
│   ├── linelevel/               # 语句级漏洞检测
│   │   ├── fine_model.py        # 核心模型（多维上下文、粗细粒度交互）
│   │   ├── fine_train.py        # 训练脚本
│   │   ├── fine_data.py         # 数据加载
│   │   ├── ablation_train.py    # 消融实验
│   │   └── generate_coarse_cache.py  # 生成函数级特征缓存
│   ├── process_data/            # 数据预处理
│   │   ├── get_raw_data.py      # 原始数据处理
│   │   └── normalize.py         # 数据标准化
│   ├── slice/                   # 程序切片提取
│   │   └── get_func_slices.py   # 基于敏感点的切片提取
│   └── joernAnalysis/           # 程序依赖图提取
│       └── joern_parse.py       # 使用Joern提取AST/CFG/DFG
├── chap3code/                   # 非论文相关（KLEE符号执行等）
├── 数据集/                       # 实验数据
│   ├── full_data_vul_lines_all.csv  # BigVul数据集（CSV格式）
│   ├── 2015-10-27-ffmpeg-v1-2-2.zip # FFmpeg源码
│   └── 2015-10-27-openssl-v1-0-1e.zip # OpenSSL源码
└── README.md                    # 本文件
```

---

## 🔧 环境配置

### 1. Python环境

推荐使用 Conda 创建虚拟环境：

```bash
# 创建Python 3.8环境
conda create -n vuldet python=3.8
conda activate vuldet
```

### 2. 安装依赖

```bash
# 基础深度学习框架
pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Transformers和预训练模型
pip install transformers==4.30.0
pip install tokenizers==0.13.3

# 数据处理和科学计算
pip install numpy pandas scikit-learn
pip install tqdm matplotlib seaborn

# 代码分析工具（用于提取程序依赖图）
pip install tree-sitter
pip install networkx
```

### 3. 下载预训练模型

本项目基于 CodeBERT 预训练模型：

```bash
# 从Hugging Face下载
# 或者在代码中指定模型路径，首次运行会自动下载
```

预训练模型路径：
- 函数编码器：`microsoft/codebert-base`
- 切片编码器：`microsoft/codebert-base`

### 4. 安装Joern（可选，用于程序依赖图提取）

```bash
# 下载Joern（用于静态代码分析）
# https://github.com/joernio/joern/releases
# 解压后添加到环境变量
```

---

## 📊 数据集说明

### 数据集概览

本项目使用两个数据集：

1. **BigVul数据集** (`full_data_vul_lines_all.csv`)
   - 来源：348个开源C/C++项目（2002-2019年）
   - 函数总数：101,568个（5,260个漏洞函数）
   - 语句总数：99,301条（15,457条漏洞语句）
   - 漏洞类型：91种不同的CWE类型
   - 标注：函数级标签 + 行级标签

2. **FFmpeg+OpenSSL数据集**
   - FFmpeg v1.2.2: `2015-10-27-ffmpeg-v1-2-2.zip`
   - OpenSSL v1.0.1e: `2015-10-27-openssl-v1-0-1e.zip`
   - 来源：NVD 2022年公开的CVE条目
   - 函数总数：4,762个（1,044个漏洞函数）

### 数据集格式

**BigVul CSV文件字段**：
- `commit_id`: Git提交ID
- `CWE ID`: CWE分类
- `CVE ID`: CVE编号
- `project`: 项目名称
- `vul`: 是否有漏洞（0/1）
- `func_before`: 修复前的函数代码
- `func_after`: 修复后的函数代码
- `vul_lines`: 漏洞行号列表

---

## 🚀 快速开始

### 步骤1：数据预处理

```bash
cd chap2code

# 1. 处理原始数据（从CSV提取函数和切片）
python process_data/get_raw_data.py

# 2. 数据标准化和清洗
python process_data/normalize.py
```

**预处理后的数据结构**：
```
dataset/
├── train/              # 训练集
│   ├── 0_func_1.txt   # 非漏洞函数（文件名：label_func_id.txt）
│   ├── 1_func_2.txt   # 漏洞函数
│   └── ...
├── val/                # 验证集
└── test/               # 测试集
```

**数据文件格式示例**：
```
Original Code
————————————————————————————————————————
1: int vulnerable_function(char *input) {
2:     char buffer[32];
3:     strcpy(buffer, input);  // 漏洞行
4:     return 0;
5: }
————————————————————————————————————————
[Slice 1: API Call - strcpy => vul]
2:     char buffer[32];
3:     strcpy(buffer, input);
————————————————————————————————————————
[Slice 2: Pointer Operation]
1: int vulnerable_function(char *input) {
3:     strcpy(buffer, input);
```

### 步骤2：提取程序切片（可选，如果数据预处理未包含）

```bash
cd chap2code/slice

# 基于四类漏洞敏感点提取切片：
# - API函数调用
# - 指针解引用
# - 数组下标访问
# - 算术表达式
python get_func_slices.py \
    --input_dir ../../dataset/train \
    --output_dir ../../dataset/train_with_slices
```

### 步骤3：训练函数级检测模型

```bash
cd chap2code/Coarse_level

# 训练函数级模型（双流编码器 + 层次融合）
python coarse_train.py \
    --data_root ../../dataset \
    --output_dir ./output \
    --func_codebert_path microsoft/codebert-base \
    --slice_codebert_path microsoft/codebert-base \
    --batch_size 16 \
    --max_length 512 \
    --epochs 30 \
    --learning_rate 2e-5 \
    --k 3
```

**训练参数说明**：
- `--data_root`: 数据集根目录
- `--output_dir`: 模型输出目录
- `--func_codebert_path`: 函数编码器预训练模型路径
- `--slice_codebert_path`: 切片编码器预训练模型路径
- `--batch_size`: 批次大小（默认16，根据GPU内存调整）
- `--max_length`: 最大序列长度（默认512）
- `--epochs`: 训练轮数（默认30）
- `--learning_rate`: 学习率（默认2e-5）
- `--k`: 保留的切片数量（默认3）

**预期输出**：
```
output/
├── best_model/
│   └── checkpoint.pt      # 最佳模型权重
├── checkpoints/
│   └── epoch_*.pt         # 每轮检查点
└── logs/
    └── training.log       # 训练日志
```

### 步骤4：生成函数级特征缓存（用于语句级检测）

```bash
cd chap2code/linelevel

# 为训练集、验证集、测试集生成函数级特征
python generate_coarse_cache.py \
    --coarse_model_path ../Coarse_level/output/best_model/checkpoint.pt \
    --data_root ../../dataset \
    --output_dir ./coarse_cache
```

**生成的缓存文件**：
```
coarse_cache/
├── train_cache.pt       # 训练集函数级特征
├── val_cache.pt         # 验证集函数级特征
└── test_cache.pt        # 测试集函数级特征
```

### 步骤5：训练语句级检测模型

```bash
cd chap2code/linelevel

# 训练语句级模型（多维上下文 + 粗细粒度交互）
python fine_train.py \
    --data_root ../../dataset \
    --coarse_cache_dir ./coarse_cache \
    --output_dir ./output \
    --codebert_path microsoft/codebert-base \
    --batch_size 16 \
    --max_length 512 \
    --epochs 30 \
    --learning_rate 2e-5 \
    --context_window 2
```

**训练参数说明**：
- `--coarse_cache_dir`: 函数级特征缓存目录
- `--context_window`: 周围上下文窗口大小（默认±2行）

---

## 📈 评估模型

### 函数级检测评估

```bash
cd chap2code/Coarse_level

python evaluate.py \
    --model_path ./output/best_model/checkpoint.pt \
    --data_root ../../dataset \
    --split test
```

**评估指标**（对应论文Table 5-6）：
- Precision (P)
- Recall (R)
- Accuracy (A)
- F1 Score
- False Positive Rate (FPR)
- False Negative Rate (FNR)

### 语句级检测评估

```bash
cd chap2code/linelevel

python evaluate.py \
    --model_path ./output/best_model/checkpoint.pt \
    --data_root ../../dataset \
    --coarse_cache_dir ./coarse_cache \
    --split test
```

**评估指标**（对应论文Table 7-8）：
- IoU (Intersection over Union)
- Top-5 Accuracy
- Top-10 Accuracy

---

## 🧪 消融实验

### 函数级消融实验（对应论文Table 9）

测试不同组件的贡献：

```bash
cd chap2code/Coarse_level

# w/o 切片输入
python coarse_train_ablation.py --ablation_type no_slice

# w/o 双流编码器
python coarse_train_ablation.py --ablation_type no_dualstream

# w/o 切片过滤
python coarse_train_ablation.py --ablation_type no_filter

# w/o 特征融合
python coarse_train_ablation.py --ablation_type no_fusion
```

### 语句级消融实验（对应论文Table 10-11）

```bash
cd chap2code/linelevel

# w/o Context（无多维上下文）
python ablation_train.py --ablation_type no_context

# w/o Fusion（无函数级先验融合）
python ablation_train.py --ablation_type no_fusion

# w/o Gate（无门控机制）
python ablation_train.py --ablation_type no_gate

# 上下文组合实验
python ablation_train.py --context_config stmt_only          # 仅语句
python ablation_train.py --context_config stmt_attr          # 语句+属性
python ablation_train.py --context_config stmt_attr_surr     # 语句+属性+周围
python ablation_train.py --context_config stmt_attr_dep      # 语句+属性+依赖
python ablation_train.py --context_config full               # 全部上下文
```

---

## 🔍 级联检测实验（对应论文Table 12）

测试不同检测策略的效果：

```bash
cd chap2code/linelevel

# Coarse Only - 仅函数级检测
python cascade_experiment.py --strategy coarse_only

# Fine Only - 仅语句级检测
python cascade_experiment.py --strategy fine_only

# Coarse→Fine - 级联检测
python cascade_experiment.py --strategy cascade
```

---

## 📝 模型推理示例

### 检测单个函数

```python
import torch
from transformers import RobertaTokenizer
from chap2code.Coarse_level.coarse_model import CoarseGrainedModel

# 加载模型
model = CoarseGrainedModel(
    func_codebert_path='microsoft/codebert-base',
    slice_codebert_path='microsoft/codebert-base'
)
checkpoint = torch.load('path/to/checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 准备输入
func_code = """
int vulnerable_function(char *input) {
    char buffer[32];
    strcpy(buffer, input);
    return 0;
}
"""

slice_codes = [
    "char buffer[32];\nstrcpy(buffer, input);",
    # ... 更多切片
]

# 函数级检测
with torch.no_grad():
    outputs = model(func_input_ids, func_attention_mask,
                   slice_input_ids, slice_attention_mask)
    
    func_prob = outputs['func_prob'].item()
    print(f"函数漏洞概率: {func_prob:.4f}")
    
    if func_prob > 0.5:
        print("检测到漏洞函数！")
        
        # 语句级定位
        line_probs = get_line_predictions(func_code, outputs)
        for line_no, prob in enumerate(line_probs, 1):
            if prob > 0.5:
                print(f"  漏洞行 {line_no}: {prob:.4f}")
```

---

## 📊 实验结果复现

### BigVul数据集预期结果

**函数级检测**（对应论文Table 5）：
- Precision: 43.06%
- Recall: 58.94%
- F1 Score: 49.76%
- Accuracy: 93.85%

**语句级检测**（对应论文Table 7）：
- IoU: 55.68%
- Top-5 Acc: 71.42%
- Top-10 Acc: 73.28%

### FFmpeg+OpenSSL数据集预期结果

**函数级检测**（对应论文Table 6）：
- Precision: 80.85%
- Recall: 90.48%
- F1 Score: 85.39%
- Accuracy: 93.18%

**语句级检测**（对应论文Table 8）：
- IoU: 85.27%
- Top-5 Acc: 88.53%
- Top-10 Acc: 91.22%

---

## 🛠️ 常见问题

### 1. GPU内存不足

**问题**：`CUDA out of memory`

**解决方案**：
```bash
# 减小批次大小
python coarse_train.py --batch_size 8

# 减小最大序列长度
python coarse_train.py --max_length 256

# 启用梯度累积
python coarse_train.py --gradient_accumulation_steps 2
```

### 2. 数据加载过慢

**问题**：数据加载时间过长

**解决方案**：
```bash
# 减少数据加载进程数
python coarse_train.py --num_workers 4

# 使用预tokenized数据（需要先生成缓存）
python preprocess_tokenize.py --data_root ../../dataset
```

### 3. CodeBERT模型下载失败

**问题**：无法从Hugging Face下载

**解决方案**：
```bash
# 方法1：使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com
pip install -U huggingface_hub

# 方法2：手动下载模型
# 1. 从 https://huggingface.co/microsoft/codebert-base 下载
# 2. 解压到本地目录
# 3. 指定本地路径：--func_codebert_path /path/to/codebert-base
```

### 4. Joern安装问题

**问题**：无法安装Joern

**解决方案**：
- Joern仅用于程序依赖图提取（可选）
- 如果已有预处理的数据，可以跳过此步骤
- 或者使用替代工具（如tree-sitter）进行代码分析

---

## 📚 论文对应关系

| 论文章节 | 代码文件 | 说明 |
|---------|---------|------|
| 4.1.1 DualStream编码器 | `coarse_model.py` (FunctionEncoder, SliceEncoder) | 双流预训练模型 |
| 4.1.2 函数级检测 | `coarse_model.py` (SliceFilter, HierarchicalAttention) | 切片过滤+层次融合 |
| 4.2.1 多维上下文编码 | `fine_model.py` (LineEncoder) | 属性+周围+依赖上下文 |
| 4.2.2 粗细粒度交互 | `fine_model.py` (CoarseFineInteraction) | 风险先验引导 |
| 5.1 数据集 | `process_data/` | BigVul + FFmpeg/OpenSSL |
| 6.1 整体性能对比 | `coarse_train.py`, `fine_train.py` | Table 5-8 |
| 6.2 函数级消融 | `coarse_train_ablation.py` | Table 9 |
| 6.3 语句级消融 | `ablation_train.py` | Table 10-11 |
| 6.4 级联策略 | `generate_coarse_cache.py` | Table 12 |
| 6.5 案例分析 | `visualize.py` | Fig. 9 |

---

## 🤝 贡献

本项目为学术研究代码，如有问题请提交Issue。

---

## 📄 引用

如果本代码对您的研究有帮助，请引用我们的论文：

```bibtex
@article{tao2026dualstream,
  title={DualStream-based Vulnerability Detection for Multi-Granularity Feature Fusion},
  author={Tao, Wenxin and Su, Xiaohong and Gao, Fangzheng and Zheng, Yu and Gao, Wei},
  journal={Knowledge-Based Systems},
  year={2026}
}
```

---

## 📧 联系方式

- 作者：Wenxin Tao
- 邮箱：taowenxin@hit.edu.cn
- 单位：Harbin Institute of Technology

---

**最后更新**: 2026年6月
