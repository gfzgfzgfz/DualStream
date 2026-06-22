import os
import re
import json
import pydot
import hashlib
import pandas as pd
from tqdm import tqdm
import pickle

class NodeData:
    def __init__(self, node_id, label, node_type, code, line_nums):
        self.id = node_id
        self.label = label  
        self.node_type = node_type
        self.raw_code = code
        self.line_nums = line_nums
        self.is_vulnerable = False
        self.edges = {}

    def ddg_predecessors(self):
        preds = []
        for edge_key, edge_obj in self.edges.items():
            if edge_obj.node_out == self.id and edge_obj.type == "DDG":
                preds.append(edge_obj.node_in)
        return list(set(preds))

    def ddg_successors(self):
        succs = []
        for edge_key, edge_obj in self.edges.items():
            if edge_obj.node_in == self.id and edge_obj.type == "DDG":
                succs.append(edge_obj.node_out)
        return list(set(succs))

class EdgeData:
    def __init__(self, node_in, node_out, edge_type):
        self.node_in = node_in
        self.node_out = node_out 
        self.type = edge_type

def parse_dot_and_json(dot_path, json_path):
    """
    解析dot文件和json文件,构建节点字典
    """
    # 读取json文件获取节点-行号映射
    with open(json_path) as f:
        json_data = json.load(f)
    
    # 构建节点ID到行号的映射
    node_to_lines = {}
    for item in json_data:
        if "id" in item and "lineNumber" in item:
            node_to_lines[str(item["id"])] = item["lineNumber"]
    
    # 解析dot文件
    graph = pydot.graph_from_dot_file(dot_path)[0]
    
    node_dict = {}
    # 处理节点
    for node in graph.get_nodes():
        node_id = node.get_name().strip('"')
        label = node.get_attributes()["label"].strip('"')
        
        # 解析label "(类型,内容[,完整代码])"
        match = re.match(r"\((.*?),(.*?)(?:,(.*?))?\)", label)
        if match:
            node_type = match.group(1)
            content = match.group(2)
            code = match.group(3) if match.group(3) else content
            
            # 获取行号
            line_nums = []
            if node_id in node_to_lines:
                line_nums.append(node_to_lines[node_id])
                
            node_data = NodeData(node_id, label, node_type, code, line_nums)
            node_dict[node_id] = node_data
            
    # 处理边
    for edge in graph.get_edges():
        source = edge.get_source().strip('"')
        target = edge.get_destination().strip('"')
        edge_type = edge.get_attributes()["label"].strip('"').split(":")[0]  # 取:前的类型
        
        if source in node_dict and target in node_dict:
            edge_data = EdgeData(source, target, edge_type)
            edge_key = f"{source}@{target}"
            node_dict[source].edges[edge_key] = edge_data
            
    return node_dict

def get_file_info_from_path(file_path):
    """
    从文件路径中提取信息
    OpenSSL格式: {vul}_{project}_{CVE_ID}_{commit_id}/0-cpg.dot
    """
    # 获取文件夹名
    dir_path = os.path.dirname(file_path)
    folder_name = os.path.basename(dir_path)

    # 解析文件夹名: vul_project_CVE_commit
    parts = folder_name.split('_')
    if len(parts) < 4:
        return None, None, None

    vul_label = parts[0]
    project = parts[1]
    cve_id = parts[2]
    commit_id = '_'.join(parts[3:])  # commit_id可能包含下划线

    return cve_id, commit_id, vul_label

def get_code_and_vul_lines(source_data, cve_id, commit_id, filename, vul_label, code_type="raw"):
    """
    从CSV文件中获取代码和漏洞行信息
    OpenSSL数据集使用vul标签而不是索引进行匹配
    """
    # 在CSV中查找匹配的行（使用vul标签）
    matching_row = source_data[
        (source_data['CVE ID'] == cve_id) &
        (source_data['commit_id'] == commit_id) &
        (source_data['vul'] == int(vul_label))
    ]

    if not matching_row.empty:
        matched_row = matching_row.iloc[0]
        dataTag = str(matched_row.vul)
        code = matched_row.raw if code_type == "raw" else matched_row.normalize
        vul_lines_str = matched_row.vul_lines

        # 解析漏洞行号
        vul_lines = []
        if pd.notna(vul_lines_str):
            vul_lines = [int(x) for x in re.findall(r'\d+', vul_lines_str)]

        return code, vul_lines
    else:
        print(f"No matching row found for: CVE={cve_id}, commit={commit_id}, vul={vul_label}")
        return None, []

def parse_dot_and_json(dot_path, json_path, source_data):
    """
    解析dot文件、json文件和CSV数据,构建节点字典
    """
    # 从文件路径获取信息（OpenSSL格式）
    cve_id, commit_id, vul_label = get_file_info_from_path(dot_path)
    if cve_id is None:
        return None, None, None

    filename = os.path.basename(dot_path)

    # 获取代码和漏洞行信息
    code, vulnerable_lines = get_code_and_vul_lines(source_data, cve_id, commit_id, filename, vul_label)
    if code is None:
        return None, None, None
        
    # 将代码分割成行
    code_lines = code.split('\n')
    
    # 读取json文件获取节点-行号映射
    with open(json_path) as f:
        json_data = json.load(f)
    
    # 构建节点ID到行号的映射
    node_to_lines = {}
    for item in json_data:
        if "id" in item and "lineNumber" in item:
            node_to_lines[str(item["id"])] = item["lineNumber"]
    
    try:
        # 先读取dot文件内容
        with open(dot_path, 'r') as f:
            dot_content = f.read()
            
        # 替换图名中的特殊字符
        dot_content = re.sub(r'digraph\s+[^{]+{', 'digraph G {', dot_content)
        
        # 使用修改后的内容创建图
        graphs = pydot.graph_from_dot_data(dot_content)
        if not graphs:
            print(f"Failed to parse dot file: {dot_path}")
            return None, None, None
        graph = graphs[0]
    except Exception as e:
        print(f"Error parsing dot file {dot_path}: {str(e)}")
        return None, None, None
    
    node_dict = {}
    # 处理节点
    for node in graph.get_nodes():
        node_id = node.get_name().strip('"')
        label = node.get_attributes()["label"].strip('"')
        
        # 解析label "(类型,内容[,完整代码])"
        match = re.match(r"\((.*?),(.*?)(?:,(.*?))?\)", label)
        if match:
            node_type = match.group(1)
            content = match.group(2)
            code = match.group(3) if match.group(3) else content
            
            # 获取行号
            line_nums = []
            if node_id in node_to_lines:
                line_num = node_to_lines[node_id]
                line_nums.append(line_num)
                
            node_data = NodeData(node_id, label, node_type, code, line_nums)
            
            # 标记漏洞节点
            if any(ln in vulnerable_lines for ln in line_nums):
                node_data.is_vulnerable = True
                
            node_dict[node_id] = node_data
            
    # 处理边
    for edge in graph.get_edges():
        source = edge.get_source().strip('"')
        target = edge.get_destination().strip('"')
        edge_type = edge.get_attributes()["label"].strip('"').split(":")[0]
        
        if source in node_dict and target in node_dict:
            edge_data = EdgeData(source, target, edge_type)
            edge_key = f"{source}@{target}"
            node_dict[source].edges[edge_key] = edge_data
            
    return node_dict, code_lines, vulnerable_lines
def get_pointer_nodes(node_dict):
    """
    获取指针操作相关的节点作为切片起点
    """
    start_nodes = []
    pointer_keywords = ["->", "*", "&"]
    
    for node_id, node in node_dict.items():
        # 检查代码中是否包含指针操作
        if any(keyword in node.raw_code for keyword in pointer_keywords):
            start_nodes.append(node_id)
    return start_nodes

def get_array_nodes(node_dict):
    """
    获取数组访问相关的节点作为切片起点
    """
    start_nodes = []
    for node_id, node in node_dict.items():
        # 检查是否是数组访问节点
        if "indirectIndexAccess" in node.node_type or "[" in node.raw_code:
            start_nodes.append(node_id)
    return start_nodes

def get_sensitive_api_nodes(node_dict):
    """
    获取敏感API调用节点作为切片起点
    """
    try:
        # 使用脚本所在目录的绝对路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pkl_path = os.path.join(script_dir, "sensitive_func.pkl")
        with open(pkl_path, "rb") as f:
            sensitive_apis = pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load sensitive APIs from pkl ({e}), using default list")
        # 使用默认的敏感API列表作为备选
        sensitive_apis = [
            "memcpy", "strcpy", "strncpy", "malloc", "free",
            "sprintf", "vsprintf", "gets", "scanf"
        ]
    start_nodes = []
    
    for node_id, node in node_dict.items():
        if any(api in node.raw_code for api in sensitive_apis):
            start_nodes.append(node_id)
    return start_nodes

def get_integer_overflow_nodes(node_dict):
    """
    获取可能发生整数溢出的节点作为切片起点
    """
    start_nodes = []
    arithmetic_ops = ["+", "-", "*", "/", "<<", ">>"]
    
    for node_id, node in node_dict.items():
        if any(op in node.raw_code for op in arithmetic_ops):
            start_nodes.append(node_id)
    return start_nodes

def program_slice(node_dict, start_node_id):
    """
    从起点节点开始进行程序切片
    """
    if start_node_id not in node_dict:
        return None
        
    visited = set()
    slice_list = []
    
    def dfs(node_id):
        if node_id in visited:
            return
        visited.add(node_id)
        
        node = node_dict[node_id]
        slice_list.append(node)
        
        # 向前切片：获取数据依赖的前驱节点
        for pred_id in node.ddg_predecessors():
            dfs(pred_id)
            
        # 向后切片：获取数据依赖的后继节点
        for succ_id in node.ddg_successors():
            dfs(succ_id)
    
    dfs(start_node_id)
    return slice_list

def to_dot_and_text(slice_list, code_lines, output_dir, base_filename, idx):
    """
    将切片转换为DOT格式和文本格式
    """
    # 创建新的图
    graph = pydot.Dot(graph_type='digraph')
    
    # 添加节点和边
    added_nodes = set()
    added_edges = set()
    
    for node in slice_list:
        # 添加节点
        if node.id not in added_nodes:
            graph.add_node(pydot.Node(node.id, label=node.label))
            added_nodes.add(node.id)
        
        # 添加边
        for edge_key, edge_data in node.edges.items():
            if edge_key not in added_edges and \
               edge_data.node_in in added_nodes and \
               edge_data.node_out in added_nodes:
                graph.add_edge(pydot.Edge(
                    edge_data.node_in,
                    edge_data.node_out,
                    label=edge_data.type
                ))
                added_edges.add(edge_key)
    
    # 收集切片中的代码行
    lines_set = set()
    line_vul_map = {}  # 记录每行是否是漏洞行
    
    for node in slice_list:
        for ln in node.line_nums:
            if 0 < ln <= len(code_lines):
                lines_set.add(ln)
                if node.is_vulnerable:
                    line_vul_map[ln] = True
    
    # 生成代码切片文本
    slice_code_lines = []
    pure_code_lines = []  # 用于生成哈希值的纯代码行
    for ln in sorted(lines_set):
        code_line = code_lines[ln - 1].strip()
        # 带标记的完整代码行
        if ln in line_vul_map:
            slice_code_lines.append(f"{code_line} #{ln} vul")
        else:
            slice_code_lines.append(f"{code_line} #{ln}")
        # 纯代码行（用于生成哈希值）
        pure_code_lines.append(code_line)
    
    slice_code = "\n".join(slice_code_lines)
    
    # 生成切片哈希值（只基于纯代码内容）
    pure_code = "\n".join(pure_code_lines)
    slice_hash = hashlib.md5(pure_code.encode('utf-8')).hexdigest()
    
    return graph.to_string(), slice_code, slice_hash

def main():
    # ======== OpenSSL 数据集配置 ========
    # 读取CSV数据
    source_csv_path = "/root/MGVD-master/dataset/openssl/full_data_vul_lines.csv"
    source_data = pd.read_csv(source_csv_path)

    # 设置输入输出目录（存到硬盘）
    input_dir = "/mnt/e/zsm/openssl/json_dot_files"
    output_dir = "/mnt/e/zsm/openssl/slices_out/"
    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        for dir_name in tqdm(dirs, desc="Processing folders"):
            # 构建0-cpg.dot的完整路径
            dot_path = os.path.join(root, dir_name, "0-cpg.dot")
            # 查找目录中的json文件
            json_files = [f for f in os.listdir(os.path.join(root, dir_name)) 
                         if f.endswith('.json')]
            if not json_files:
                print(f"No json file found in {dir_name}")
                continue
                
            json_path = os.path.join(root, dir_name, json_files[0])  # 使用找到的第一个json文件
            
            if not os.path.exists(dot_path) or not os.path.exists(json_path):
                print(f"Missing dot or json file in {dir_name}")
                continue
            
            # 解析文件
            base_filename = dir_name  # 使用文件夹名作为基础文件名
            node_dict, code_lines, vulnerable_lines = parse_dot_and_json(
                dot_path, json_path, source_data
            )
            
            if node_dict is None:
                continue
                
            # 获取切片起点
            pointer_nodes = get_pointer_nodes(node_dict)
            array_nodes = get_array_nodes(node_dict)
            api_nodes = get_sensitive_api_nodes(node_dict)
            int_overflow_nodes = get_integer_overflow_nodes(node_dict)
            all_start_nodes = pointer_nodes + array_nodes + api_nodes + int_overflow_nodes
            
            # 创建输出文件
            out_txt_path = os.path.join(output_dir, f"{base_filename}.txt")
            with open(out_txt_path, "w", encoding="utf-8") as fout:
                # 写入原始代码
                fout.write(f"Original Code => {vulnerable_lines}\n")
                for i, line in enumerate(code_lines, 1):
                    fout.write(f"{i}: {line}\n")
                fout.write("\n————————————————————————————\n")
                
                # 收集有效切片
                valid_slices = []
                seen_slices = set()
                
                for start_node in all_start_nodes:
                    slice_list = program_slice(node_dict, start_node)
                    if slice_list is None:
                        continue
                        
                    dot_str, slice_code, slice_hash = to_dot_and_text(
                        slice_list, code_lines, output_dir, base_filename, len(valid_slices) + 1
                    )
                    
                    # 检查是否是重复切片
                    if slice_hash in seen_slices:
                        continue
                    
                    # 获取切片中的行数
                    lines_set = set()
                    for node in slice_list:
                        for ln in node.line_nums:
                            if 0 < ln <= len(code_lines):
                                lines_set.add(ln)
                    
                    # 跳过单行切片
                    if len(lines_set) <= 1:
                        continue
                        
                    seen_slices.add(slice_hash)
                    vul_in_slice = any(node.is_vulnerable for node in slice_list)
                    valid_slices.append({
                        'code': slice_code,
                        'label': "vul" if vul_in_slice else "no_vul"
                    })
                
                # 写入有效切片，使用连续的序号
                for idx, slice_info in enumerate(valid_slices, 1):
                    fout.write(f"[Slice {idx}] => {slice_info['label']}\n")
                    fout.write(slice_info['code'] + "\n")
                    fout.write("\n————————————————————————————\n")

if __name__ == "__main__":
    main()
