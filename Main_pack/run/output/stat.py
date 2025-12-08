import os

def count_total_data_rows(root_dir, suffix=".txt"):
    total_lines = 0
    file_count = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(suffix):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        if len(lines) > 1:
                            data_lines = len(lines) - 1  # 除去列名行
                            total_lines += data_lines
                            file_count += 1
                except Exception as e:
                    print(f"读取文件失败：{file_path}，错误：{e}")

    print(f"共处理文件数：{file_count}")
    print(f"所有文件的数据行总数（不含列名）：{total_lines}")
    return total_lines


# 使用示例
if __name__ == "__main__":
    root_directory = "./bin_with_case_del"  # 替换为你的目录路径
    count_total_data_rows(root_directory)
