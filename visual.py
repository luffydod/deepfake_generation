import os
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import random
import argparse

def visualize_comparison(src_dir="data/src", targ_dir="data/targ", output_dir="output_extract", num_samples=1):
    """
    随机抽取图像在所有三个目录中都存在的图片进行可视化对比
    
    Args:
        src_dir: 源图像目录
        targ_dir: 目标图像目录
        output_dir: 输出图像目录
        num_samples: 要抽取的样本数量
    
    Returns:
        成功可视化的图像数量
    """
    # 确保输出目录存在
    vis_dir = "visualization"
    os.makedirs(vis_dir, exist_ok=True)
    
    # 获取output_dir中的所有图片
    output_files = glob.glob(os.path.join(output_dir, "*.png"))
    if not output_files:
        print(f"错误：{output_dir}目录中没有找到图片")
        return 0
    
    # 随机打乱文件列表
    random.shuffle(output_files)
    
    # 记录成功可视化的数量
    successful_vis = 0
    
    # 查找同时在三个目录都存在的图片
    for output_file in output_files:
        if successful_vis >= num_samples:
            break
            
        filename = os.path.basename(output_file)
        src_file = os.path.join(src_dir, filename)
        targ_file = os.path.join(targ_dir, filename)
        
        if os.path.exists(src_file) and os.path.exists(targ_file):
            # 读取三个图片
            src_img = cv2.imread(src_file)
            targ_img = cv2.imread(targ_file)
            output_img = cv2.imread(output_file)
            
            # 确保所有图片被正确加载
            if src_img is None or targ_img is None or output_img is None:
                print(f"警告：无法读取图像: {filename}")
                continue
            
            # 确保所有图片具有相同的尺寸
            target_size = 256
            if src_img.shape[0] != target_size or src_img.shape[1] != target_size:
                src_img = cv2.resize(src_img, (target_size, target_size))
            if targ_img.shape[0] != target_size or targ_img.shape[1] != target_size:
                targ_img = cv2.resize(targ_img, (target_size, target_size))
            if output_img.shape[0] != target_size or output_img.shape[1] != target_size:
                output_img = cv2.resize(output_img, (target_size, target_size))
            
            # 创建一个大画布，水平排列三个图像
            canvas = np.zeros((target_size, target_size * 3, 3), dtype=np.uint8)
            canvas[:, :target_size] = src_img
            canvas[:, target_size:2*target_size] = targ_img
            canvas[:, 2*target_size:] = output_img
            
            # 添加标签
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            font_color = (255, 255, 255)
            
            # 为每个图像添加标签
            cv2.putText(canvas, "Source", (10, 30), font, font_scale, font_color, font_thickness)
            cv2.putText(canvas, "Target", (target_size + 10, 30), font, font_scale, font_color, font_thickness)
            cv2.putText(canvas, "Output", (2 * target_size + 10, 30), font, font_scale, font_color, font_thickness)
            
            # 在底部添加文件名
            name_y = target_size - 15
            cv2.putText(canvas, filename, (target_size, name_y), font, 0.5, font_color, 1, cv2.LINE_AA)
            
            # 保存可视化结果
            vis_path = os.path.join(vis_dir, f"comparison_{filename}")
            cv2.imwrite(vis_path, canvas)
            
            print(f"已保存可视化对比图像到: {vis_path}")
            
            # 展示图像
            plt.figure(figsize=(15, 5))
            plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title(f"Comparasion: {filename}")
            plt.tight_layout()
            plt.show()
            
            successful_vis += 1
    
    if successful_vis == 0:
        print("未找到同时存在于所有三个目录中的图片")
    else:
        print(f"成功可视化了 {successful_vis} 张图片")
    
    return successful_vis

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='可视化对比源图像、目标图像和输出图像')
    parser.add_argument('--src', type=str, default='data/src', help='源图像目录')
    parser.add_argument('--targ', type=str, default='data/targ', help='目标图像目录')
    parser.add_argument('--output', type=str, default='output_extract', help='输出图像目录')
    parser.add_argument('--num', type=int, default=1, help='要抽取的样本数量')
    args = parser.parse_args()

    # 进行可视化对比
    visualize_comparison(
        src_dir=args.src,
        targ_dir=args.targ,
        output_dir=args.output,
        num_samples=args.num
    )

if __name__ == "__main__":
    main()