""" 
从bigvul原数据集中获取部分数据进行测试
对要使用的字段进行了筛选
对删除的语句进行了行级的标注
"""
import pandas as pd
import difflib

def extract_columns(src_csv):
    """
    先从原始 CSV 中仅保留指定列，
    并先取 80 条 vul==1 和 20 条 vul==0 数据，
    然后返回处理后的 DataFrame
    """
    df = pd.read_csv(src_csv, low_memory=False)

    # 仅保留指定列并返回
    df_sampled = df[[
        "commit_id", 
        "CWE ID", 
        "CVE ID", 
        "Summary", 
        "project", 
        "Vulnerability Classification", 
        "vul", 
        "func_before", 
        "func_after", 
        "lines_before", 
        "lines_after",
        "del_lines"
    ]]
    return df_sampled

def mark_removed_code(df):
    """
    仅对 vul == 1 的行进行处理：
    使用 difflib 比较 func_before 和 func_after，找到被删除的代码行，
    并在这些被删除的行在 func_before 中的末尾添加 " ###vul" 标记。
    """
    updated_func_befores = []

    for idx, row in df.iterrows():
        if row['vul'] == 1:
            before = str(row['func_before']) if not pd.isna(row['func_before']) else ""
            after = str(row['func_after']) if not pd.isna(row['func_after']) else ""

            before_lines = before.splitlines()
            after_lines = after.splitlines()

            # 使用 difflib 的 SequenceMatcher 来获取详细的操作码
            matcher = difflib.SequenceMatcher(None, before_lines, after_lines)
            opcodes = matcher.get_opcodes()

            # 存储修改后的 func_before
            modified_before = []
            
            for tag, i1, i2, j1, j2 in opcodes:
                if tag == 'equal':
                    # 相同的行，直接添加
                    modified_before.extend(before_lines[i1:i2])
                elif tag == 'delete':
                    # 被删除的行，在原位置添加标记
                    for line in before_lines[i1:i2]:
                        modified_before.append(f"{line} ###vul")
                elif tag == 'replace':
                    # 替换操作，表示有行被删除且有新行被添加
                    for line in before_lines[i1:i2]:
                        modified_before.append(f"{line} ###vul")
                    # 添加新的行
                    modified_before.extend(after_lines[j1:j2])


            # 将修改后的 func_before 添加到列表
            modified_before_str = "\n".join(modified_before)
            updated_func_befores.append(modified_before_str)
        else:
            # 如果 vul != 1，保持 func_before 不变
            updated_func_befores.append(row['func_before'])
    
    # 更新 DataFrame 中的 func_before 列
    df['func_before'] = updated_func_befores
    return df

if __name__ == "__main__":
    src_csv = "./bigvul/MSR_data_cleaned.csv"
    dst_csv = "./bigvul/bigvul_raw_data_all.csv"

    df_extracted = extract_columns(src_csv)
    df_marked = mark_removed_code(df_extracted)
    df_marked.to_csv(dst_csv, index=False)