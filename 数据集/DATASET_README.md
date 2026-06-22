# 数据集下载说明

本项目使用两个漏洞检测数据集。文件已通过Git LFS上传到仓库，可以直接克隆获取，也可以通过以下方式下载。

---

## 📥 数据集下载方式

### 🚀 方式1：直接从GitHub克隆（推荐）

所有数据集文件已通过Git LFS上传到仓库，克隆时会自动下载：

```bash
# 克隆仓库（包含所有数据集）
git clone https://github.com/gfzgfzgfz/DualStream.git
cd DualStream

# 数据集已自动下载到 数据集/ 文件夹
ls -lh 数据集/
```

**如果只需要代码，不需要数据集**：
```bash
# 跳过LFS文件下载
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/gfzgfzgfz/DualStream.git
```

---

### 📦 方式2：百度网盘下载（国内用户推荐）

**百度网盘链接**: https://pan.baidu.com/s/1Tw2GaD39ZuGvllMTPmZ5iQ?pwd=5x92

**提取码**: `5x92`

**包含文件**:
- `full_data_vul_lines_all.csv` (748 MB) - BigVul数据集
- `2015-10-27-ffmpeg-v1-2-2.zip` (114 MB) - FFmpeg v1.2.2
- `2015-10-27-openssl-v1-0-1e.zip` (119 MB) - OpenSSL v1.0.1e

**下载后放置位置**:
```bash
# 将下载的文件放到项目的数据集文件夹
DualStream/数据集/
├── full_data_vul_lines_all.csv
├── 2015-10-27-ffmpeg-v1-2-2.zip
└── 2015-10-27-openssl-v1-0-1e.zip
```

---

### 🌐 方式3：从官方源下载

### 1. BigVul 数据集

**文件名**: `full_data_vul_lines_all.csv`

**大小**: ~748 MB

**官方源下载**:
```bash
# 克隆BigVul仓库
git clone https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset.git

# 或者直接下载CSV文件
wget https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset/raw/master/MSR_data_cleaned.csv
# 重命名为
mv MSR_data_cleaned.csv full_data_vul_lines_all.csv
```

**数据集信息**:
- 函数总数: 101,568 (5,260个漏洞函数)
- 语句总数: 99,301 (15,457条漏洞语句)
- 项目数量: 348个开源C/C++项目
- 时间范围: 2002-2019年
- 漏洞类型: 91种CWE分类

---

### 2. FFmpeg + OpenSSL 数据集

**文件名**: 
- `2015-10-27-ffmpeg-v1-2-2.zip` (~114 MB)
- `2015-10-27-openssl-v1-0-1e.zip` (~119 MB)

**官方源下载**:

**FFmpeg v1.2.2**:
```bash
# 从官方源下载
wget https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n1.2.2.zip
mv n1.2.2.zip 2015-10-27-ffmpeg-v1-2-2.zip
```

**OpenSSL v1.0.1e**:
```bash
# 从官方源下载
wget https://github.com/openssl/openssl/archive/refs/tags/OpenSSL_1_0_1e.zip
mv OpenSSL_1_0_1e.zip 2015-10-27-openssl-v1-0-1e.zip
```

**数据集信息**:
- FFmpeg: 3,478个源文件, 637个测试文件
- OpenSSL: 2,203个源文件, 636个测试文件
- 函数总数: 4,762 (1,044个漏洞函数)
- 漏洞来源: NVD 2022年公开的CVE条目

---

## 📁 完整目录结构

下载完成后，`数据集/` 目录应该包含：

```
数据集/
├── full_data_vul_lines_all.csv           # BigVul数据集
├── 2015-10-27-ffmpeg-v1-2-2.zip          # FFmpeg源码
├── 2015-10-27-openssl-v1-0-1e.zip        # OpenSSL源码
└── DATASET_README.md                     # 本文件
```

---

## 🔄 数据预处理

下载数据集后，运行预处理脚本：

```bash
cd chap2code

# 1. 提取和处理原始数据
python process_data/get_raw_data.py

# 2. 数据标准化
python process_data/normalize.py
```

预处理后会生成：
```
dataset/
├── train/              # 训练集 (80%)
├── val/                # 验证集 (10%)
└── test/               # 测试集 (10%)
```

---

## 📊 数据集格式说明

### BigVul CSV格式

| 字段 | 说明 | 示例 |
|------|------|------|
| `commit_id` | Git提交ID | `b000da128b5fb519...` |
| `CWE ID` | CWE分类 | `CWE-264` |
| `CVE ID` | CVE编号 | `CVE-2015-8467` |
| `project` | 项目名称 | `samba` |
| `vul` | 是否有漏洞 | `0` (安全) / `1` (漏洞) |
| `func_before` | 修复前函数代码 | `int func() {...}` |
| `func_after` | 修复后函数代码 | `int func() {...}` |
| `vul_lines` | 漏洞行号列表 | `[23, 24]` |

### 预处理后的文件格式

每个函数保存为单独的txt文件：`{label}_func_{id}.txt`

**文件内容结构**:
```
Original Code
————————————————————————————————————————
1: int vulnerable_function(char *input) {
2:     char buffer[32];
3:     strcpy(buffer, input);
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

---

## 🔍 验证数据集完整性

运行验证脚本检查数据集是否正确：

```bash
cd chap2code
python verify_dataset.py
```

预期输出：
```
✓ BigVul数据集存在
✓ FFmpeg数据集存在
✓ OpenSSL数据集存在
✓ 数据集格式正确
✓ 文件数量匹配

数据集统计:
- 总函数数: 101,568
- 漏洞函数: 5,260 (5.18%)
- 安全函数: 96,308 (94.82%)
```

---

## ❓ 常见问题

**Q: 数据集下载太慢怎么办？**

A: 可以使用镜像源或下载工具：
```bash
# 使用aria2加速下载
aria2c -x 16 -s 16 <download_url>

# 或使用国内镜像（如果有）
```

**Q: 如何使用自己的数据集？**

A: 参考 `chap2code/process_data/custom_dataset.py` 中的格式说明，将你的数据转换为相同格式。

**Q: 数据集预处理需要多久？**

A: 
- BigVul预处理: ~30分钟 (取决于CPU性能)
- FFmpeg+OpenSSL预处理: ~15分钟

---

## 📚 数据集引用

如果使用这些数据集，请引用原始论文：

**BigVul**:
```bibtex
@inproceedings{fan2020ac,
  title={A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries},
  author={Fan, Jiahao and Li, Yi and Wang, Shaohua and Nguyen, Tien N},
  booktitle={Proceedings of MSR},
  year={2020}
}
```

**FFmpeg & OpenSSL**:
来自NVD (National Vulnerability Database) 2022年公开的CVE条目

---

## 📧 获取帮助

如果数据集下载或处理遇到问题：

1. 检查网络连接
2. 确认磁盘空间充足 (至少5GB)
3. 查看 `chap2code/process_data/README.md`
4. 提交Issue并附上错误日志

---

**最后更新**: 2026年6月
