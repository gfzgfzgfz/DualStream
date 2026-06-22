import os
import re
import json
import pydot
import hashlib
import pandas as pd
from tqdm import tqdm
import pickle
import threading

# 全局pydot解析锁（pyparsing不是线程安全的）
_pydot_lock = threading.Lock()

class NodeData:
    def __init__(self, node_id, label, node_type, code, line_nums):
        self.id = node_id
        self.label = label  
        self.node_type = node_type
        self.raw_code = code
        self.line_nums = line_nums
        self.is_vulnerable = False
        self.edges = {}

    def ast_children(self):
        """获取AST子节点"""
        children = []
        for edge_key, edge_obj in self.edges.items():
            if edge_obj.node_in == self.id and edge_obj.type == "AST":
                children.append(edge_obj.node_out)
        return list(set(children))

    def ddg_predecessors(self):
        """获取DDG前驱节点"""
        preds = []
        for edge_key, edge_obj in self.edges.items():
            if edge_obj.node_out == self.id and edge_obj.type == "DDG":
                preds.append(edge_obj.node_in)
        return list(set(preds))

    def ddg_successors(self):
        """获取DDG后继节点"""
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

def get_file_info_from_path(file_path):
    """
    从文件路径中提取信息
    OpenSSL格式: {vul}_{project}_{CVE_ID}_{commit_id}/0-cpg.dot
    """
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
    matching_row = source_data[
        (source_data['CVE ID'] == cve_id) &
        (source_data['commit_id'] == commit_id) &
        (source_data['vul'] == int(vul_label))
    ]

    if not matching_row.empty:
        matched_row = matching_row.iloc[0]
        code = matched_row.raw if code_type == "raw" else matched_row.normalize
        vul_lines_str = matched_row.vul_lines

        vul_lines = []
        if pd.notna(vul_lines_str):
            vul_lines = [int(x) for x in re.findall(r'\d+', vul_lines_str)]

        return code, vul_lines
    else:
        print(f"No matching row found for: CVE={cve_id}, commit={commit_id}, vul={vul_label}")
        return None, []

def parse_dot_and_json(dot_path, json_path, source_data):
    """解析dot文件、json文件和CSV数据,构建节点字典"""
    # 获取信息（OpenSSL格式）
    cve_id, commit_id, vul_label = get_file_info_from_path(dot_path)
    if cve_id is None:
        return None, None, None, None

    filename = os.path.basename(dot_path)

    # 获取代码和漏洞行信息
    code, vulnerable_lines = get_code_and_vul_lines(source_data, cve_id, commit_id, filename, vul_label)
    if code is None:
        return None, None, None, None
        
    # 将代码分割成行
    code_lines = code.split('\n')
    
    # 读取json文件获取节点-行号映射
    with open(json_path) as f:
        json_data = json.load(f)
    
    node_to_lines = {}
    line_to_nodes = {}  # 新增：记录每行对应的所有节点
    
    for item in json_data:
        if "id" in item and "lineNumber" in item:
            node_id = str(item["id"])
            line_num = item["lineNumber"]
            
            # 更新节点到行号的映射
            if node_id not in node_to_lines:
                node_to_lines[node_id] = []
            node_to_lines[node_id].append(line_num)
            
            # 更新行号到节点的映射
            if line_num not in line_to_nodes:
                line_to_nodes[line_num] = []
            line_to_nodes[line_num].append(node_id)
    
    try:
        # 使用锁保护pydot解析（pyparsing不是线程安全的）
        with _pydot_lock:
            # 读取并预处理dot文件
            with open(dot_path, 'r', encoding='utf-8') as f:
                dot_content = f.read()
            dot_content = re.sub(r'digraph\s+[^{]+{', 'digraph G {', dot_content)

            graphs = pydot.graph_from_dot_data(dot_content)
            if not graphs:
                print(f"Failed to parse dot file: {dot_path}")
                return None, None, None, None
            graph = graphs[0]
    except Exception as e:
        print(f"Error parsing dot file {dot_path}: {str(e)}")
        return None, None, None, None
    
    node_dict = {}
    # 处理节点
    for node in graph.get_nodes():
        node_id = node.get_name().strip('"')
        try:
            label = node.get_attributes()["label"].strip('"')
        except KeyError:
            continue
            
        # 改进正则表达式以更准确地匹配括号内容
        try:
            # 首先匹配最外层的括号对
            outer_match = re.match(r"\((.*)\)$", label)
            if not outer_match:
                continue
                
            # 然后分割括号内的内容
            parts = outer_match.group(1).split(',', 2)  # 最多分割2次
            if len(parts) >= 2:
                node_type = parts[0].strip()
                content = parts[1].strip()
                # 如果有第三部分，则使用它作为代码，否则使用content
                code = parts[2].strip() if len(parts) > 2 else content
                
                line_nums = []
                if node_id in node_to_lines:
                    line_num = node_to_lines[node_id]
                    line_nums.append(line_num)
                    
                node_data = NodeData(node_id, label, node_type, code, line_nums)
                node_dict[node_id] = node_data
                
                if any(ln in vulnerable_lines for ln in line_nums):
                    node_data.is_vulnerable = True
        except Exception as e:
            print(f"Error parsing node label: {label}")
            continue
            
    # 处理边
    for edge in graph.get_edges():
        source = edge.get_source().strip('"')
        target = edge.get_destination().strip('"')
        try:
            edge_type = edge.get_attributes()["label"].strip('"').split(":")[0]
        except KeyError:
            continue
            
        if source in node_dict and target in node_dict:
            edge_data = EdgeData(source, target, edge_type)
            edge_key = f"{source}@{target}"
            node_dict[source].edges[edge_key] = edge_data
            
    # 确保每行代码都有对应的节点
    for line_num in range(1, len(code_lines) + 1):
        if line_num not in line_to_nodes:
            # 查找可能的父节点
            for node_id, node in node_dict.items():
                if node.line_nums:  # 确保节点有行号
                    # 处理嵌套列表的情况
                    all_line_nums = []
                    for ln in node.line_nums:
                        if isinstance(ln, list):
                            all_line_nums.extend(ln)
                        else:
                            all_line_nums.append(ln)
                    
                    if all_line_nums:  # 确保有行号
                        node_min_line = min(int(ln) for ln in all_line_nums)
                        node_max_line = max(int(ln) for ln in all_line_nums)
                        if node_min_line <= line_num <= node_max_line:
                            if line_num not in line_to_nodes:
                                line_to_nodes[line_num] = []
                            line_to_nodes[line_num].append(node_id)
    
    return node_dict, code_lines, vulnerable_lines, line_to_nodes

def get_operation_context(node_dict, node_id):
    """获取操作上下文：当前语句的AST节点及其子节点的类型和内容"""
    if node_id not in node_dict:
        return ["[PAD]"]
        
    context = []
    node = node_dict[node_id]
    
    # 添加当前节点的表示
    context.append(f"{node.node_type}:{node.raw_code}")
    
    # 获取AST子节点
    for child_id in node.ast_children():
        child_node = node_dict.get(child_id)
        if child_node:
            context.append(f"{child_node.node_type}:{child_node.raw_code}")
    
    return context

def get_complete_statement(node_id, code_lines, node_dict):
    """
    获取节点对应的完整语句，处理跨行代码的情况
    """
    node = node_dict[node_id]
    if not node.line_nums:
        return None
    
    # 处理嵌套列表的情况
    all_line_nums = []
    for ln in node.line_nums:
        if isinstance(ln, list):
            all_line_nums.extend(ln)
        else:
            all_line_nums.append(ln)
    
    if not all_line_nums:
        return None
        
    # 获取最小和最大行号，处理跨行代码
    min_line = min(int(ln) for ln in all_line_nums)
    max_line = max(int(ln) for ln in all_line_nums)
    
    if 0 < min_line <= len(code_lines) and 0 < max_line <= len(code_lines):
        # 如果是跨行代码，合并这些行
        if min_line == max_line:
            return code_lines[min_line - 1].strip()
        else:
            # 合并多行代码，保持缩进
            lines = code_lines[min_line - 1:max_line]
            # 获取第一行的缩进级别
            first_line_indent = len(lines[0]) - len(lines[0].lstrip())
            # 处理后续行的缩进
            processed_lines = []
            for i, line in enumerate(lines):
                if i == 0:
                    processed_lines.append(line.strip())
                else:
                    # 移除多余的缩进
                    line_content = line.strip()
                    if line_content:  # 忽略空行
                        processed_lines.append(" " * first_line_indent + line_content)
            return '\n'.join(processed_lines)
    return None

def get_dependence_context(node_dict, node_id, code_lines, window=3):
    """
    获取依赖上下文：通过PDG向前和向后切片获取相关语句
    window: 切片步长
    """
    if node_id not in node_dict:
        return ["[PAD]"]
        
    context = []
    visited = set()
    
    def traverse_dependencies(curr_id, depth=0, forward=True, path=None):
        if path is None:
            path = []
        
        if depth >= window or curr_id in visited:
            return
            
        visited.add(curr_id)
        curr_node = node_dict[curr_id]
        
        # 获取完整语句
        stmt = get_complete_statement(curr_id, code_lines, node_dict)
        if stmt:
            # 获取当前节点的行号范围
            all_line_nums = []
            for ln in curr_node.line_nums:
                if isinstance(ln, list):
                    all_line_nums.extend(ln)
                else:
                    all_line_nums.append(ln)
            min_line = min(int(ln) for ln in all_line_nums)
            
            new_path = path + [(min_line, stmt)]
            if forward:
                context.append(new_path)
            else:
                context.insert(0, new_path)
        
        # 获取依赖节点
        next_nodes = (curr_node.ddg_successors() if forward 
                     else curr_node.ddg_predecessors())
        for next_id in next_nodes:
            if stmt:
                new_path = path + [(min_line, stmt)]
            else:
                new_path = path
            traverse_dependencies(next_id, depth+1, forward, new_path)
    
    # 向前和向后遍历依赖
    traverse_dependencies(node_id, forward=False)  # 向前切片
    visited.clear()
    traverse_dependencies(node_id, forward=True)   # 向后切片
    
    # 展平路径并按行号排序
    flat_context = []
    for path in context:
        flat_context.extend(path)
    
    # 去重并按行号排序
    seen = set()
    sorted_context = []
    for line_num, stmt in sorted(flat_context, key=lambda x: x[0]):
        if stmt not in seen:
            seen.add(stmt)
            sorted_context.append(stmt)
    
    return sorted_context if sorted_context else ["[PAD]"]

def get_surrounding_context(code_lines, line_num, window=3):
    """
    获取周围上下文：前后k行代码，保持原始顺序
    """
    if not code_lines or line_num <= 0:
        return ["[PAD]"]
        
    context = []
    # 先添加前面的行
    for i in range(max(0, line_num - window - 1), line_num - 1):
        line = code_lines[i].strip()
        if line:
            context.append(line)
            
    # 再添加后面的行
    for i in range(line_num, min(len(code_lines), line_num + window)):
        line = code_lines[i].strip()
        if line:
            context.append(line)
    
    return context if context else ["[PAD]"]

def write_contexts_to_file(output_dir, base_filename, code_lines, node_dict, line_to_nodes,vulnerable_lines):
    """将代码及其上下文写入文件，合并同一行的上下文"""
    out_path = os.path.join(output_dir, f"{base_filename}_contexts.txt")
    
    with open(out_path, "w", encoding="utf-8") as f:
        # 写入原始代码
        f.write(f"Original Code => {vulnerable_lines}\n")
        for i, line in enumerate(code_lines, 1):
            f.write(f"{i}: {line}\n")
        f.write("\n" + "="*50 + "\n\n")
        
        # 遍历每一行代码
        for line_num, line in enumerate(code_lines, 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue
                
            f.write(f"Line {line_num}: {line}\n")
            f.write("-" * 30 + "\n")
            
            # 使用预先计算的行号到节点的映射
            line_nodes = line_to_nodes.get(line_num, [])
            
            # 如果没有找到相关节点，创建一个虚拟节点来获取上下文
            if not line_nodes:
                # 获取周围上下文（这个总是可以获取的）
                surr_context = get_surrounding_context(code_lines, line_num)
                
                f.write("Operation Context:\n")
                f.write("  [No operation context available]\n")
                
                f.write("\nDependence Context:\n")
                f.write("  [No dependence context available]\n")
                
                f.write("\nSurrounding Context:\n")
                for item in surr_context:
                    f.write(f"  {item}\n")
            else:
                # 合并该行所有节点的上下文
                all_op_contexts = set()
                all_dep_contexts = set()
                all_surr_contexts = set()
                
                # 收集所有节点的上下文
                for node_id in line_nodes:
                    op_context = get_operation_context(node_dict, node_id)
                    dep_context = get_dependence_context(node_dict, node_id, code_lines)
                    surr_context = get_surrounding_context(code_lines, line_num)
                    
                    all_op_contexts.update(op_context)
                    all_dep_contexts.update(dep_context)
                    all_surr_contexts.update(surr_context)
                
                # 移除填充标记
                if "[PAD]" in all_op_contexts:
                    all_op_contexts.remove("[PAD]")
                if "[PAD]" in all_dep_contexts:
                    all_dep_contexts.remove("[PAD]")
                if "[PAD]" in all_surr_contexts:
                    all_surr_contexts.remove("[PAD]")
                
                # 写入合并后的上下文信息
                f.write("Operation Context:\n")
                for item in sorted(all_op_contexts):
                    f.write(f"  {item}\n")
                    
                f.write("\nDependence Context:\n")
                for item in sorted(all_dep_contexts):
                    f.write(f"  {item}\n")
                    
                f.write("\nSurrounding Context:\n")
                for item in sorted(all_surr_contexts):
                    f.write(f"  {item}\n")
            
            f.write("\n" + "="*50 + "\n\n")

def main():
    # ======== OpenSSL 数据集配置 ========
    # 读取CSV数据
    source_csv_path = "/root/MGVD-master/dataset/openssl/full_data_vul_lines.csv"
    source_data = pd.read_csv(source_csv_path)

    # 设置输入输出目录（存到硬盘）
    input_dir = "/mnt/e/zsm/openssl/json_dot_files"
    output_dir = "/mnt/e/zsm/openssl/contexts_out/"
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历所有子文件夹
    for root, dirs, files in os.walk(input_dir):
        for dir_name in tqdm(dirs, desc="Processing folders"):
            # 构建文件路径
            dot_path = os.path.join(root, dir_name, "0-cpg.dot")
            json_files = [f for f in os.listdir(os.path.join(root, dir_name)) 
                         if f.endswith('.json')]
            
            if not json_files:
                print(f"No json file found in {dir_name}")
                continue
                
            json_path = os.path.join(root, dir_name, json_files[0])
            
            if not os.path.exists(dot_path) or not os.path.exists(json_path):
                print(f"Missing dot or json file in {dir_name}")
                continue
            
            # 解析文件
            node_dict, code_lines, vulnerable_lines, line_to_nodes = parse_dot_and_json(
                dot_path, json_path, source_data
            )
            
            if node_dict is None:
                continue
            
            # 提取并保存上下文信息
            write_contexts_to_file(output_dir, dir_name, code_lines, node_dict, line_to_nodes,vulnerable_lines)

if __name__ == "__main__":
    main()