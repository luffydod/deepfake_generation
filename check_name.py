import os

def check_image_correspondence():
    # 读取图片列表文件
    try:
        with open('data/image_list.txt', 'r') as f:
            expected_images = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("错误：找不到文件 data/image_list.txt")
        return
    
    # 获取output_extract目录中的所有图片文件
    try:
        actual_images = os.listdir('output_extract/')
    except FileNotFoundError:
        print("错误：找不到目录 output_extract/")
        return
    
    # 检查数量是否一致
    if len(expected_images) != 1000 or len(actual_images) != 1000:
        print(f"错误：文件数量不匹配")
        print(f"image_list.txt 中有 {len(expected_images)} 个名称")
        print(f"output_extract/ 目录中有 {len(actual_images)} 个文件")
    
    # 对实际图片列表排序以便比较
    actual_images_set = set(actual_images)
    expected_images_set = set(expected_images)
    
    # 检查不匹配的情况
    missing_in_folder = expected_images_set - actual_images_set
    extra_in_folder = actual_images_set - expected_images_set
    
    if missing_in_folder or extra_in_folder:
        print("检测到不匹配情况：")
        
        if missing_in_folder:
            print("\n在 image_list.txt 中存在但在 output_extract/ 目录中缺失的文件:")
            for img in sorted(missing_in_folder):
                print(f"  - {img}")
        
        if extra_in_folder:
            print("\n在 output_extract/ 目录中存在但在 image_list.txt 中缺失的文件:")
            for img in sorted(extra_in_folder):
                print(f"  - {img}")
    else:
        print("成功：output_extract/ 目录下的所有图片名称与 data/image_list.txt 中的名称完全一一对应。")

if __name__ == "__main__":
    check_image_correspondence()