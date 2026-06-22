import os
import pandas as pd
import numpy as np
import subprocess
import shutil
from functools import partial
from multiprocessing import Pool

def read_from_file(fold):
    """
    从给定目录遍历文件，将目录名识别为标签(如 Vul->1 ，No-Vul->0)，读取每个文件的内容到 DataFrame 中
    :param fold: 文件所在的根目录
    :return: 包含目标标签和函数文本的 DataFrame
    """
    label_list = []
    function_list = []
    for root, dirs, files in os.walk(fold):
        # 遍历 root 下的所有文件
        for file in files:
            dir = root.split("/")[-1]  # 根据所在目录判断标签
            if dir == "Vul":
                label = 1
                label_list.append(label)
            elif dir == "No-Vul":
                label = 0
                label_list.append(label)
            with open(os.path.join(root, file), "r") as f:
                function = f.read()
                function_list.append(function)
    data = {"target": label_list, "func": function_list}
    return pd.DataFrame(data)


def drop(data_frame: pd.DataFrame, keys):
    """
    从 DataFrame 中删除指定的列
    :param data_frame: 要处理的 DataFrame
    :param keys: 要删除的列名列表
    """
    for key in keys:
        del data_frame[key]


def slice_frame(data_frame: pd.DataFrame, size: int):
    """
    将 DataFrame 按照固定大小进行分块并分组
    :param data_frame: 要分块的 DataFrame
    :param size: 每块的行数
    :return: 一个分组后的对象
    """
    data_frame_size = len(data_frame)
    return data_frame.groupby(np.arange(data_frame_size) // size)


def to_files(data_frame: pd.DataFrame, out_path, code_type):
    """
    将 DataFrame 中的每一行写入 *.c 文件，按照索引命名
    :param data_frame: 包含代码内容的 DataFrame
    :param out_path: 输出文件夹路径
    :param code_type: 指定写入 'raw' 或 'normalize' 字段
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for idx, row in data_frame.iterrows():
        project = row['project']
        cve_id = row['CVE ID'] if not pd.isna(row['CVE ID']) else 'noCVE'
        commit_id = row['commit_id']
        vul = row['vul']
        # 在文件名末尾添加行索引
        file_name = f"{vul}_{project}_{cve_id}_{commit_id}_{idx}.c"

        with open(os.path.join(out_path, file_name), 'w') as f:
            try:
                if code_type == "raw":
                    f.write(row.raw)
                elif code_type == "normalize":
                    f.write(row.normalize)
            except:
                print(f"处理索引 {idx} 时出错")
                print(f"函数名: {row.func_name}")
                print(row)


def joern_parse(file, out_fold, file_name):
    """
    调用 joern-parse 命令，将 C/C++ 源码转换为 .bin 格式的 CPG (Code Property Graph)
    :param file: 要转换的 .c 文件夹路径
    :param out_fold: 输出bin文件存储目录
    :param file_name: 转换后的文件名(不包括后缀)
    :return: 生成的 .bin 文件名
    """
    out_file = file_name + ".bin"
    out_path = os.path.join(out_fold, out_file)

    # 设置环境变量给 shell 脚本使用
    os.environ['file'] = str(file)
    os.environ['outPath'] = str(out_path)

    process = subprocess.Popen('joern-parse $file --out $outPath',
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               shell=True, close_fds=True)
    output = process.communicate()
    print(output)

    return out_file


def joern_parse_task(raw_file_fold, temp_file_fold, parse_result_fold, code_type):
    """
    分批读取 CSV 数据，每次将一小批代码写入到临时目录，然后调用 joern_parse 转换为 CPG
    :param raw_file_fold: 包含 full_data.csv 的目录
    :param temp_file_fold: 用于存放临时 .c 文件的目录
    :param parse_result_fold: 生成的 .bin 文件所在目录
    :param code_type: 指定处理哪一列的代码 ('raw' 或 'normalize')
    """
    # 读取包含代码的 CSV
    raw = pd.read_csv(os.path.join(raw_file_fold, "full_data_vul_lines_all.csv"))
    # 将 DataFrame 分块，每块大小500行
    slices = slice_frame(raw, size=500)
    # 将 groupby 格式处理成 (s, slice_data) 的列表
    slices = [(s, slice.apply(lambda x: x)) for s, slice in slices]

    # 收集已经生成的 .bin 文件，避免重复
    cpg_files = []
    for root, dirs, files in os.walk(parse_result_fold):
        for file in files:
            cpg_files.append(str(file))

    # 对每块数据进行处理
    for s, slice in slices:
        # 将代码写为 .c 文件
        to_files(slice, temp_file_fold, code_type)
        for c_file in os.listdir(temp_file_fold):
            file_name = os.path.splitext(c_file)[0]
            c_file_path = os.path.join(temp_file_fold, c_file)
            cpg_file = joern_parse(c_file_path, parse_result_fold, file_name)
            cpg_files.append(cpg_file)


def joern_graph_task(parse_result, raw_graph_fold):
    """
    调用 joern 的脚本(如 njf.sc)，将生成的 .bin 文件解析为 Graph
    :param parse_result: .bin 文件路径
    :param raw_graph_fold: 解析出的图文件存储目录
    """
    joern_path = "/home/tao/joern/"
    script_file = "/root/MGVD-master/build_graph/njf.sc"

    # 设置脚本执行所需的参数
    params = f"cpgFile={parse_result},outDir={raw_graph_fold}"
    os.environ['params'] = str(params)
    os.environ['script_file'] = str(script_file)

    # 调用 joern 命令将 .bin 文件转成图数据
    process = subprocess.Popen('joern --script $script_file --params=$params',
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               shell=True, close_fds=True)
    output = process.communicate()
    # 去除后缀只保留文件名
    name = parse_result.split('/')[-1].split('.')[0]
    print(name)
    return


def process_func_multi(file_fold, save_fold, code_type, flag, temp_file_fold=None):
    """
    根据标志位决定执行 parse 阶段或 graph 阶段
    :param file_fold: 输入文件夹(可能是 .csv 所在或 .bin 所在)
    :param save_fold: 输出文件夹(存放 .bin 或图文件)
    :param code_type: 处理 'raw' 还是 'normalize' 代码
    :param flag: "parse" 或 "graph" 表示所执行的任务
    :param temp_file_fold: 若处于 parse 阶段，用于存放临时 .c 文件
    """
    if "parse" == flag:
        joern_parse_task(file_fold, temp_file_fold, save_fold, code_type)

    if "graph" == flag:
        # 批量将 .bin 文件转换为图
        input_path_list = []
        for root, dirs, files in os.walk(file_fold):
            for file in files:
                input_path = os.path.join(root, file)
                input_path_list.append(input_path)
                joern_graph_task(input_path, save_fold)

# 以下代码可以输出dot文件
def bin_to_image(bin_file: str, dot_dir: str):
    if not os.path.exists(dot_dir):
        os.makedirs(dot_dir)

    # 直接使用bin文件的基础名称
    base_name = os.path.splitext(os.path.basename(bin_file))[0]
    
    dot_file = os.path.join(dot_dir, base_name)

    try:
        # 使用新的命令格式
        subprocess.run(
            f"joern-export --repr cpg14 {bin_file} --out {dot_file}",
            shell=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error exporting {bin_file}: {e}")
        return

    print(f"DOT file saved at: {dot_file}")

def convert_all_bins_to_images(bin_dir: str, dot_dir: str):
    if not os.path.exists(dot_dir):
        os.makedirs(dot_dir)

    # 直接遍历bin_dir下的所有.bin文件
    for file in os.listdir(bin_dir):
        if file.endswith(".bin"):
            bin_file = os.path.join(bin_dir, file)
            bin_to_image(bin_file, dot_dir)

    # print(f"All images generated at: {image_output_dir}")


def bin_to_json(bin_file: str, json_dir: str):
    """
    将单个bin文件转换为json格式
    :param bin_file: .bin文件路径
    :param json_dir: json文件输出根目录
    """
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    # 使用bin文件名作为目录名
    base_name = os.path.splitext(os.path.basename(bin_file))[0]
    # 为每个bin文件创建独立的json文件目录
    bin_json_dir = os.path.join(json_dir, base_name)
    if not os.path.exists(bin_json_dir):
        os.makedirs(bin_json_dir)
    
    # 设置输出json文件路径
    json_file = os.path.join(bin_json_dir, f"{base_name}.json")
    
    # 设置脚本文件路径
    script_file = "/root/MGVD_exp/joernAnalysis/getjson.sc"
    
    try:
        # 调用joern执行脚本
        subprocess.run(
            f'joern --script {script_file} --params cpgFile={bin_file},outFile={json_file}',
            shell=True,
            check=True
        )
        print(f"JSON file saved at: {json_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting {bin_file} to JSON: {e}")
        return

def convert_all_bins_to_jsons(bin_dir: str, json_dir: str):
    """
    将bin目录下的所有.bin文件转换为json文件
    :param bin_dir: 存放.bin文件的目录
    :param json_dir: json文件输出目录
    """
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    # 遍历bin目录下的所有.bin文件
    for file in os.listdir(bin_dir):
        if file.endswith(".bin"):
            bin_file = os.path.join(bin_dir, file)
            bin_to_json(bin_file, json_dir)

def convert_all_bins_to_jsons(bin_dir: str, json_dir: str, cleanup_dir: str = None, cleanup_interval: int = 10, num_workers: int = 4):
    """
    将bin目录下的所有.bin文件转换为json文件，并定时删除指定目录
    多线程处理，自动跳过已存在的JSON文件
    :param bin_dir: 存放.bin文件的目录
    :param json_dir: json文件输出目录
    :param cleanup_dir: 要定时删除的目录，如不需要删除则设为None
    :param cleanup_interval: 每处理多少个文件后清理一次指定目录
    :param num_workers: 并行处理的线程数
    """
    import shutil
    from concurrent.futures import ThreadPoolExecutor
    import threading
    
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    # 遍历bin目录下的所有.bin文件
    files = [f for f in os.listdir(bin_dir) if f.endswith(".bin")]
    total_files = len(files)
    
    # 用于跟踪已处理的文件数量
    processed_count = [0]
    lock = threading.Lock()
    
    def process_file(file_idx, file):
        bin_file = os.path.join(bin_dir, file)
        base_name = os.path.splitext(os.path.basename(bin_file))[0]
        
        # 检查JSON文件是否已存在
        bin_json_dir = os.path.join(json_dir, base_name)
        json_file = os.path.join(bin_json_dir, f"{base_name}.json")
        
        if os.path.exists(json_file):
            print(f"跳过已存在的JSON文件: {json_file}")
            with lock:
                processed_count[0] += 1
                current_count = processed_count[0]
            return
        
        # 执行转换
        bin_to_json(bin_file, json_dir)
        
        # 更新处理计数并检查是否需要清理
        with lock:
            processed_count[0] += 1
            current_count = processed_count[0]
            
            # 定期清理目录
            if cleanup_dir and os.path.exists(cleanup_dir) and current_count % cleanup_interval == 0:
                print(f"正在永久删除目录: {cleanup_dir}")
                try:
                    shutil.rmtree(cleanup_dir)
                    print(f"目录 {cleanup_dir} 已清理 (已处理 {current_count}/{total_files} 个文件)")
                except Exception as e:
                    print(f"清理目录 {cleanup_dir} 时出错: {e}")
    
    # 使用线程池并行处理文件
    print(f"开始处理 {total_files} 个bin文件，使用 {num_workers} 个线程...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_file, idx, file) for idx, file in enumerate(files)]
        
    print(f"所有文件处理完成，共处理 {processed_count[0]} 个文件")

if __name__ == "__main__":
    # ======== OpenSSL 数据集配置 ========
    # 输入：已有的 bin 文件目录
    bin_dir = "/root/MGVD-master/dataset/openssl/raw/parse_result"

    # 输出：JSON和DOT文件目录（存到硬盘）
    dot_dir = "/mnt/e/zsm/openssl/json_dot_files"

    # 转换所有bin文件到DOT文件
    convert_all_bins_to_images(bin_dir, dot_dir)

    # 转换所有bin文件到JSON文件
    convert_all_bins_to_jsons(bin_dir, dot_dir)