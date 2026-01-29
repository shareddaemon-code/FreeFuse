#!/usr/bin/env python3
"""
为assets文件夹中的所有图片添加白色背景
"""

from PIL import Image
import os

def add_white_background(image_path, padding=20):
    """给图片添加白色背景和padding"""
    img = Image.open(image_path)
    
    # 如果图片有透明通道，先处理
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        # 创建白色背景
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    # 添加白色padding
    new_width = img.width + 2 * padding
    new_height = img.height + 2 * padding
    new_img = Image.new('RGB', (new_width, new_height), (255, 255, 255))
    new_img.paste(img, (padding, padding))
    
    return new_img

def process_assets_folder(assets_dir='assets', padding=20):
    """处理assets文件夹中的所有图片"""
    supported_formats = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    
    for filename in os.listdir(assets_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in supported_formats:
            image_path = os.path.join(assets_dir, filename)
            print(f"Processing: {image_path}")
            
            new_img = add_white_background(image_path, padding)
            
            # 保存为PNG以保持质量
            output_path = os.path.join(assets_dir, os.path.splitext(filename)[0] + '.png')
            new_img.save(output_path, 'PNG')
            print(f"Saved: {output_path}")
            
            # 如果原文件不是png，删除原文件
            if ext != '.png' and os.path.exists(image_path):
                os.remove(image_path)
                print(f"Removed original: {image_path}")

if __name__ == '__main__':
    process_assets_folder('assets', padding=20)
    print("\nDone! All images now have white backgrounds.")
