import os
import re
import shutil
from pathlib import Path

def extract_and_rename_images():
    # 创建输出目录
    output_extract_dir = Path("output_extract")
    output_extract_dir.mkdir(exist_ok=True)
    
    # 遍历output目录下所有以Rank开头的目录
    output_dir = Path("output")
    rank_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("Rank")]
    
    for rank_dir in rank_dirs:
        # 从目录名提取信息
        match = re.match(r"Rank_id(\d+)-id(\d+)_(\d+)_(\d+)", rank_dir.name)
        if not match:
            print(f"目录名称格式不匹配: {rank_dir.name}")
            continue
        
        id_a, id_b, c, d = match.groups()
        
        # 获取该目录下的所有png图像
        images = [f for f in rank_dir.glob("*.png")]
        
        if not images:
            print(f"在 {rank_dir.name} 中没有找到图像")
            continue
        
        # 尝试从文件名中提取数值进行排序
        def get_number(filename):
            # 假设文件名格式为 "数字.png"，例如 "02289.png"
            match = re.match(r'(\d+)\.png$', filename.name)
            if match:
                return int(match.group(1))
            # 如果找不到数字，则使用文件修改时间作为备选排序标准
            return os.path.getmtime(filename)
        
        # 按数值大小排序并取最小的图像
        images.sort(key=get_number)
        best_image = images[0]
        
        # 新的文件名格式: id6_id2_0005_00120.png
        new_filename = f"id{id_b}_id{id_a}_{c}_{d}.png"
        
        # 复制并重命名文件到输出目录
        shutil.copy2(best_image, output_extract_dir / new_filename)
        print(f"已处理: {rank_dir.name} -> {new_filename}, 使用文件: {best_image.name}")

if __name__ == "__main__":
    extract_and_rename_images()
    print("所有图像处理完成，结果保存在 output_extract 目录")