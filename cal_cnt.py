import os
from pathlib import Path
import argparse
def count_files_in_directory(directory_path, file_extension=None, recursive=False):
    """
    统计指定目录下的文件数量
    
    参数:
        directory_path (str): 要统计的目录路径
        file_extension (str, optional): 如果指定，只统计特定扩展名的文件（如 '.png'）
        recursive (bool, optional): 是否递归统计子目录中的文件
        
    返回:
        int: 文件数量
    """
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"错误：目录 '{directory_path}' 不存在")
        return 0
    
    if not directory.is_dir():
        print(f"错误：'{directory_path}' 不是一个目录")
        return 0
    
    count = 0
    
    if recursive:
        # 递归方式统计
        for root, dirs, files in os.walk(directory):
            if file_extension:
                # 只统计指定扩展名的文件
                count += sum(1 for file in files if file.lower().endswith(file_extension.lower()))
            else:
                # 统计所有文件
                count += len(files)
    else:
        # 非递归方式，只统计当前目录
        if file_extension:
            count = len([f for f in directory.iterdir() if f.is_file() and f.suffix.lower() == file_extension.lower()])
        else:
            count = len([f for f in directory.iterdir() if f.is_file()])
    
    return count

# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='统计目录文件数量')
    parser.add_argument('-p', '--path', type=str, default='output_extract', help='要统计的目录路径')
    args = parser.parse_args()
    
    dir = args.path
    # 统计指定目录及其子目录下的所有文件
    print(f"{dir}目录及子目录文件数量: {count_files_in_directory(dir, recursive=True)}")
    
    # 统计指定目录及其子目录下的所有PNG文件
    print(f"{dir}目录及子目录PNG文件数量: {count_files_in_directory(dir, '.png', recursive=True)}")