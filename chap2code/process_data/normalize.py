import re
import codecs
import os
import pandas as pd

# 下面定义了若干不可变集合，用来存储语言关键字、已知函数名等，以便后续做处理时可以过滤或跳过
from typing import List

keywords = frozenset({'__asm', '__builtin', '__cdecl', '__declspec', '__except', '__export', '__far16', '__far32',
                      '__fastcall', '__finally', '__import', '__inline', '__int16', '__int32', '__int64', '__int8',
                      '__leave', '__optlink', '__packed', '__pascal', '__stdcall', '__system', '__thread', '__try',
                      '__unaligned', '_asm', '_Builtin', '_Cdecl', '_declspec', '_except', '_Export', '_Far16',
                      '_Far32', '_Fastcall', '_finally', '_Import', '_inline', '_int16', '_int32', '_int64',
                      '_int8', '_leave', '_Optlink', '_Packed', '_Pascal', '_stdcall', '_System', '_try', 'alignas',
                      'alignof', 'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor', 'bool', 'break', 'case',
                      'catch', 'char', 'char16_t', 'char32_t', 'class', 'compl', 'const', 'const_cast', 'constexpr',
                      'continue', 'decltype', 'default', 'delete', 'do', 'double', 'dynamic_cast', 'else', 'enum',
                      'explicit', 'export', 'extern', 'false', 'final', 'float', 'for', 'friend', 'goto', 'if',
                      'inline', 'int', 'long', 'mutable', 'namespace', 'new', 'noexcept', 'not', 'not_eq', 'nullptr',
                      'operator', 'or', 'or_eq', 'override', 'private', 'protected', 'public', 'register',
                      'reinterpret_cast', 'return', 'short', 'signed', 'sizeof', 'static', 'static_assert',
                      'static_cast', 'struct', 'switch', 'template', 'this', 'thread_local', 'throw', 'true', 'try',
                      'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using', 'virtual', 'void', 'volatile',
                      'wchar_t', 'while', 'xor', 'xor_eq', 'NULL', 'printf', 'STR'})
main_set = frozenset({'main'})    # 已知main函数名，排除处理
main_args = frozenset({'argc', 'argv'}) # C/C++中main函数参数，排除处理

# 定义了运算符集合，后面会根据这些进行分隔或替换
operators3 = {'<<=', '>>='}
operators2 = {
    '->', '++', '--', '**',
    '!~', '<<', '>>', '<=', '>=',
    '==', '!=', '&&', '||', '+=',
    '-=', '*=', '/=', '%=', '&=', '^=', '|='
}
operators1 = {
    '(', ')', '[', ']', '.',
    '+', '&',
    '%', '<', '>', '^', '|',
    '=', ',', '?', ':',
    '{', '}', '!', '~'
}

# 将集合中的每一个元素转义后，用 | 拼接成正则表达式，用于对代码进行分割匹配
def to_regex(lst):
    return r'|'.join([f"({re.escape(el)})" for el in lst])

regex_split_operators = to_regex(operators3) + to_regex(operators2) + to_regex(operators1)

def merge_braces(code_lines):
    """
    合并单独的大括号到上一行，并处理###vul标识
    Args:
        code_lines: 原始代码行列表
    Returns:
        merged_lines: 合并后的代码行列表
    """
    merged_lines = []
    i = 0
    while i < len(code_lines):
        current_line = code_lines[i].rstrip()
        
        # 检查是否是单独的大括号行（可能带有###vul标识）
        current_line_stripped = current_line.replace(" ###vul", "").strip()
        if current_line_stripped in {'{', '}'}:
            # 如果不是第一行，则合并到上一行
            if merged_lines:
                prev_line = merged_lines[-1]
                # 检查当前行和上一行是否有漏洞标识
                prev_has_vul = " ###vul" in prev_line
                curr_has_vul = " ###vul" in current_line
                
                # 移除漏洞标识（如果有）
                prev_line = prev_line.replace(" ###vul", "").rstrip()
                current_line = current_line_stripped
                
                # 合并行，确保在大括号前添加空格
                if current_line == '{':
                    merged_line = prev_line + ' ' + current_line
                else:  # 处理 }
                    merged_line = prev_line + current_line
                
                # 如果任一行有漏洞标识，则在合并后添加漏洞标识
                if prev_has_vul or curr_has_vul:
                    merged_line += " ###vul"
                
                merged_lines[-1] = merged_line
            else:
                # 如果是第一行，则单独保留
                merged_lines.append(current_line)
        else:
            # 非大括号行
            merged_lines.append(current_line)
        
        i += 1
    
    return merged_lines

# 去除注释，包括 /*...*/ 块注释和 // 单行注释
def _removeComments(source) -> []:
    in_block = False
    new_source = []
    for line in source:
        i = 0
        if not in_block:
            newline = []
        while i < len(line):
            # 遇到 "/*" 则进入块注释状态，直到 "*/" 才结束
            if line[i:i + 2] == '/*' and not in_block:
                in_block = True
                i += 1
            elif line[i:i + 2] == '*/' and in_block:
                in_block = False
                i += 1
            # 遇到 "//" 则跳过本行剩余内容
            elif not in_block and line[i:i + 2] == '//':
                break
            # 非注释则正常保留代码字符
            elif not in_block:
                newline.append(line[i])
            i += 1
        # 如果本行不在块注释状态，则将处理后的内容写入结果
        if newline and not in_block:
            new_source.append("".join(newline))
    return new_source

# 将传入的代码片段(以行列表为单位)进行清洗：包括替换变量、函数名为符号，并去除多余字符
def clean_gadget(gadget):
    fun_symbols = {}  # 用于保存已替换过的函数名 -> 符号
    var_symbols = {}  # 用于保存已替换过的变量名 -> 符号
    fun_count = 1
    var_count = 1

    # 匹配函数名（在几个字符组成后紧跟一个左括号）
    rx_fun = re.compile(r'\b([_A-Za-z]\w*)\b(?=\s*\()')
    # 匹配变量名（排除了像形如 函数( 调用的情况）
    rx_var = re.compile(r'\b([_A-Za-z]\w*)\b((?!\s*\**\w+))(?!\s*\()')

    cleaned_gadget = []

    for line in gadget:
        # 去掉所有非ASCII字符
        ascii_line = re.sub(r'[^\x00-\x7f]', r'', line)
        # 将十六进制数替换为 "HEX"
        hex_line = re.sub(r'0[xX][0-9a-fA-F]+', "HEX", ascii_line)
        # 根据正则找出行内所有函数和变量名，用于后续统一替换
        user_fun = rx_fun.findall(hex_line)
        user_var = rx_var.findall(hex_line)

        # 替换函数名
        for fun_name in user_fun:
            # 不在main_set和keywords中才认为是用户定义的函数
            if len({fun_name}.difference(main_set)) != 0 and len({fun_name}.difference(keywords)) != 0:
                if fun_name not in fun_symbols.keys():
                    fun_symbols[fun_name] = 'FUN' + str(fun_count)
                    fun_count += 1
                # 用上面统计好的符号替换对应函数名
                hex_line = re.sub(r'\b(' + fun_name + r')\b(?=\s*\()', fun_symbols[fun_name], hex_line)

        # 替换变量名
        for var_name in user_var:
            if len({var_name[0]}.difference(keywords)) != 0 and len({var_name[0]}.difference(main_args)) != 0:
                if var_name[0] not in var_symbols.keys():
                    var_symbols[var_name[0]] = 'VAR' + str(var_count)
                    var_count += 1
                # 用符号替换变量名，但排除有函数调用等形式的情况
                hex_line = re.sub(r'\b(' + var_name[0] + r')\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()',
                                  var_symbols[var_name[0]], hex_line)

        cleaned_gadget.append(hex_line)
    return cleaned_gadget

# 此函数会遍历 data_path 下所有文件，对每个文件做normalize操作，然后将清洗后的结果写回 store_path
def normalize_code(data_path, store_path):
    files = os.listdir(data_path)
    files_num = len(files)
    count = 0
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    for file in files:
        count = count + 1
        print("\r", end="")
        # 打印进度
        print("Process progress: {}%: ".format(count / files_num * 100), end="")
        path = data_path + '/' + file
        with open(path, "r") as f1:
            code = f1.read()
            gadget: List[str] = []
            # 将所有字符串字面量替换为"STR"
            no_str_lit_line = re.sub(r'["]([^"\\\n]|\\.|\\\n)*["]', '"STR"', code)
            # 将所有字符字面量去掉
            no_char_lit_line = re.sub(r"'.*?'", "", no_str_lit_line)
            code = no_char_lit_line

            # 按行拆分，并去掉空行
            for line in code.splitlines():
                if line == '':
                    continue
                stripped = line.strip()
                gadget.append(stripped)
            # 去掉注释
            clean = _removeComments(gadget)
            # 清洗后重新组织
            clean = clean_gadget(clean)

            # 将清洗结果写入新文件
            with open(store_path + "/" + file, 'w', encoding='utf-8') as f2:
                f2.writelines([line + '\n' for line in clean])

# 与 normalize_code 类似，只不过从 csv 中读取数据，用于清洗并输出到新的 csv 中
def normalize_code_csv(data_path, store_path):
    """与原来相同，但在处理代码前先合并大括号"""
    data = pd.read_csv(data_path, low_memory=False)
    files_num = data.shape[0]
    count = 0

    normalize_code = []
    rc_raw_code = []
    for index, row in data.iterrows():
        count = count + 1
        print("\r", end="")
        print("Process progress: {}%: ".format(count / files_num * 100), end="")
        
        raw_code = row['func_before']
        code = row['func_before']
        
        # 处理大括号合并
        try:
            code_lines = code.splitlines()
            merged_lines = merge_braces(code_lines)
            code = '\n'.join(merged_lines)
        except Exception as e:
            print(f"\nError merging braces: {e}")
            continue

        gadget: List[str] = []
        # 删除字符串字面量
        try:
            no_str_lit_line = re.sub(r'["]([^"\\\n]|\\.|\\\n)*["]', '"STR"', code)
        except:
            print(code)
            continue
            
        # 删除字符字面量
        no_char_lit_line = re.sub(r"'.*?'", "", no_str_lit_line)
        code = no_char_lit_line

        # 按行处理并去掉空行
        for line in code.splitlines():
            if line == '':
                continue
            stripped = line.strip()
            gadget.append(stripped)

        # 去注释并清洗
        clean = _removeComments(gadget)
        clean = clean_gadget(clean)

        # 将清洗后的结果拼回单个字符串
        normalize = ""
        for line in clean:
            normalize = normalize + line + '\n'
        normalize_code.append(normalize)

        # 同时将原始代码去注释保存下来
        raw_lines = []
        try:
            # 对原始代码也进行大括号合并
            raw_code_lines = raw_code.splitlines()
            merged_raw_lines = merge_braces(raw_code_lines)
            raw_code = '\n'.join(merged_raw_lines)
            
            for line in raw_code.splitlines():
                if line == '':
                    continue
                stripped = line.strip()
                raw_lines.append(stripped)
        except:
            print(raw_code)
            continue
            
        rc_raw_lines = _removeComments(raw_lines)
        rc = ""
        for line in rc_raw_lines:
            rc = rc + line + '\n'
        rc_raw_code.append(rc)

    # 将新增的两列写回csv
    data["raw"] = rc_raw_code
    data["normalize"] = normalize_code
    # 更新漏洞行信息
    data = find_vul_lines(data)
    data.to_csv(os.path.join(store_path, 'full_data_vul_lines_all.csv'))

def find_vul_lines(df):
    """
    找到 raw 和 normalize 中末尾为 "###vul" 的行，统计其所在行数并去除标记，
    新增一列存储行号。
    """
    vul_lines = []
    df['raw_label'] = df['raw']
    for idx, row in df.iterrows():
        raw_code = str(row['raw']) if not pd.isna(row['raw']) else ""
        normalize_code = str(row['normalize']) if not pd.isna(row['normalize']) else ""

        raw_lines = raw_code.splitlines()
        normalize_lines = normalize_code.splitlines()
        # 存储每行的 vul_lines
        row_vul_lines = []
        # 找到 raw 中末尾为 "###vul" 的行
        for i, line in enumerate(raw_lines):
            if line.endswith(" ###vul"):
                raw_lines[i] = line[:-7]  # 去除 " ###vul" 标记
                row_vul_lines.append(i + 1)  # 行号从 1 开始

        # 找到 normalize 中末尾为 "###vul" 的行
        for i, line in enumerate(normalize_lines):
            if re.search(r' ###VAR\d+$', line):
                normalize_lines[i] = re.sub(r' ###VAR\d+$', '', line)  # 去除 " ###VAR" 标记
                # row_vul_lines.append(i + 1)  # 行号从 1 开始

        # 更新 raw 和 normalize
        df.at[idx, 'raw'] = "\n".join(raw_lines)
        df.at[idx, 'normalize'] = "\n".join(normalize_lines)
        vul_lines.append(row_vul_lines)

    # 新增一列存储行号
    df['vul_lines'] = vul_lines
    return df

def main():
    raw_csv = "./bigvul/bigvul_raw_data_all.csv"
    store_path = "./bigvul"
    normalize_code_csv(raw_csv, store_path)

if __name__ == "__main__":
    main()


