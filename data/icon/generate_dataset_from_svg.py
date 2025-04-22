import cairosvg
import numpy as np
from PIL import Image
import os
from pathlib import Path
import random
import colorsys
import xml.etree.ElementTree as ET
import re

import json
from pycocotools import mask as mask_util
import json

# BEFORE RUNNING: Download ./OmniSVG_MMSVG_Icon/MMSVG-Icon_dataset.json from https://huggingface.co/datasets/OmniSVG/MMSVG-Icon

with open('./OmniSVG_MMSVG_Icon/MMSVG-Icon_dataset.json') as f:
    data = json.load(f)
    for i in range(2000, 2200):
        svg = data[i]['svg']
        # replace default fill to black to ensure rendering
        svg = svg.replace('fill=\"\"', 'fill=\"#000000\"')
        with open(f'./OmniSVG_MMSVG_Icon/test_200/{i}.svg', 'w') as f:
            f.write(svg)

def extract_colors_from_svg(svg_path):
    """
    从SVG文件中提取颜色
    """
    colors = set()
    colors.add((0, 0, 0))
    
    # 常见的颜色关键字映射到RGB值
    import json
    color_keywords = json.load(open('./color_keywords.json', 'r'))
    COLOR_KEYWORDS = {}
    for color_keyword in color_keywords:
        name = list(color_keyword.keys())[0]
        rgb = tuple(list(color_keyword.values())[0])
        COLOR_KEYWORDS[name] = rgb
    
    def parse_rgb_string(rgb_str):
        """解析RGB/RGBA字符串"""
        try:
            # 移除 rgb( 或 rgba( 和结尾的 )
            values = rgb_str.split('(')[1].rstrip(')').split(',')
            values = [v.strip().rstrip('%') for v in values]
            
            rgb_values = []
            for v in values[:3]:  # 只处理RGB部分
                if '%' in v:
                    rgb_values.append(int(float(v) * 255 / 100))
                else:
                    rgb_values.append(int(float(v)))
            
            return tuple(rgb_values)
        except:
            return None

    def parse_hsl_string(hsl_str):
        """解析HSL/HSLA字符串"""
        try:
            # 移除 hsl( 或 hsla( 和结尾的 )
            values = hsl_str.split('(')[1].rstrip(')').split(',')
            values = [v.strip().rstrip('%') for v in values]
            
            h = float(values[0]) / 360
            s = float(values[1].rstrip('%')) / 100
            l = float(values[2].rstrip('%')) / 100
            
            # 转换HSL为RGB
            rgb = colorsys.hls_to_rgb(h, l, s)
            return tuple(int(x * 255) for x in rgb)
        except:
            return None

    # 读取SVG文件
    tree = ET.parse(svg_path)
    root = tree.getroot()
    svg_str = ET.tostring(root, encoding='unicode')
    
    # 定义颜色模式的正则表达式
    color_patterns = {
        'hex': r'#[0-9a-fA-F]{3,6}',
        'rgb': r'rgb\([^)]+\)',
        'rgba': r'rgba\([^)]+\)',
        'hsl': r'hsl\([^)]+\)',
        'hsla': r'hsla\([^)]+\)',
        'keyword': r'(?:fill|stroke)="([a-zA-Z]+)"'
    }
    
    # 查找所有颜色值
    for pattern_name, pattern in color_patterns.items():
        matches = re.findall(pattern, svg_str)
        
        for match in matches:
            if 'none' in str(match).lower() or 'transparent' in str(match).lower():
                continue
                
            if pattern_name == 'hex':
                colors.add(hex_to_rgb(match))
            elif pattern_name in ['rgb', 'rgba']:
                rgb = parse_rgb_string(match)
                if rgb:
                    colors.add(rgb)
            elif pattern_name in ['hsl', 'hsla']:
                rgb = parse_hsl_string(match)
                if rgb:
                    colors.add(rgb)
            elif pattern_name == 'keyword':
                if isinstance(match, tuple):
                    color_name = match[0].lower()
                else:
                    color_name = match.lower()
                if color_name in COLOR_KEYWORDS:
                    colors.add(COLOR_KEYWORDS[color_name])
    return colors


def hex_to_rgb(hex_color):
    """
    将HEX颜色转换为RGB
    """
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join(c + c for c in hex_color)
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hsv(rgb):
    """
    将RGB转换为HSV
    """
    return colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)

def is_color_similar(color1, color2, threshold=0.2):
    """
    判断两个颜色是否相似
    """
    hsv1 = rgb_to_hsv(color1)
    hsv2 = rgb_to_hsv(color2)
    
    # 计算HSV空间中的距离
    h_diff = min(abs(hsv1[0] - hsv2[0]), 1 - abs(hsv1[0] - hsv2[0]))
    s_diff = abs(hsv1[1] - hsv2[1])
    v_diff = abs(hsv1[2] - hsv2[2])
    
    return (h_diff + s_diff + v_diff) / 3 < threshold

def generate_background_color(icon_colors):
    """
    生成背景颜色，避开图标颜色
    使用HSV颜色空间生成柔和的颜色
    """
    max_attempts = 50
    
    # choose white color in 0.5 probability

    if random.random() < 0.5:
        rgb = (255, 255, 255)  # 白色
        is_similar = False
        for icon_color in icon_colors:
            if is_color_similar(rgb, icon_color):
                is_similar = True
                break
        if not is_similar:
            return rgb

    for _ in range(max_attempts):
        # 生成HSV颜色
        h = random.random()  # 色相：0-1
        s = random.triangular(0.1, 0.5, 0.3)  # 饱和度：偏向低饱和度
        v = random.triangular(0.8, 1.0, 0.9)  # 明度：偏向高明度
        
        # 转换为RGB
        rgb = tuple(int(x * 255) for x in colorsys.hsv_to_rgb(h, s, v))
        
        # 检查是否与图标颜色相似
        is_similar = False
        for icon_color in icon_colors:
            if is_color_similar(rgb, icon_color):
                is_similar = True
                break
        
        if not is_similar:
            return rgb
            
    # 如果多次尝试都失败，返回安全的默认颜色
    return (240, 240, 240)  # 浅灰色

def modify_svg_content(svg_content):
    
    # 查找所有opacity相关属性
    opacity_pattern = r'(fill-opacity|stroke-opacity|opacity)="([0-9.]+)"'
    
    def replace_opacity(match):
        attr, value = match.groups()
        if float(value) > 0:
            return f'{attr}="1"'  # 将半透明改为完全透明
        return match.group()

    modified_content = re.sub(opacity_pattern, replace_opacity, svg_content)
    
    return modified_content


def binary_mask_to_rle(binary_mask):
    """将二值mask转换为RLE编码"""
    rle = mask_util.encode(np.asfortranarray(binary_mask))
    rle['counts'] = rle['counts'].decode('utf-8')  # 将bytes转换为string
    return rle

def create_icon_dataset_with_color(
    svg_dir, 
    output_dir, 
    canvas_size=(256, 256),
    icon_size_range=(64, 128),
    rotation_range=(-30, 30),
    num_variations=3
):
    """
    创建带有智能背景颜色的图标数据集
    """
    images_dir = Path(output_dir) / 'images'
    masks_dir = Path(output_dir) / 'masks'
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    img_size = (256, 256)
    all_data = {
        'images': [],
        'annotations': []
    }
    annotation_id = 0

    for img_idx, svg_file in enumerate(Path(svg_dir).glob('*.svg')):
        # 提取SVG中的颜色
        svg_colors = extract_colors_from_svg(svg_file)
        icon_colors = []
        for color in svg_colors:
            if '#' in color:
                icon_colors.append(hex_to_rgb(color.split('"')[1] if '"' in color else color))
            else:
                icon_colors.append(color)
            # 可以添加其他颜色格式的处理

        for variation in range(num_variations):
            # 随机大小
            icon_size = random.randint(icon_size_range[0], icon_size_range[1])
            
            # 生成临时PNG
            temp_png = 'temp.png'
            cairosvg.svg2png(
                url=str(svg_file),
                write_to=temp_png,
                output_width=icon_size,
                output_height=icon_size
            )

            temp_png_a1 = 'temp_a1.png'
            with open(svg_file, 'r') as f:
                svg_content = f.read()
            modified_svg_content = modify_svg_content(svg_content).encode('utf-8')
            cairosvg.svg2png(
                bytestring=modified_svg_content,
                write_to=temp_png_a1,
                output_width=icon_size,
                output_height=icon_size,
            )

            # 读取图标
            icon = Image.open(temp_png).convert('RGBA')
            icon_a1 = Image.open(temp_png_a1).convert('RGBA')
            
            # 随机旋转
            rotation = random.uniform(rotation_range[0], rotation_range[1])
            icon = icon.rotate(rotation, expand=True, resample=Image.BICUBIC)
            
            # 生成背景颜色并创建背景
            bg_color = generate_background_color(icon_colors)
            background = Image.new('RGBA', canvas_size, bg_color)
            
            # 随机位置
            x = random.randint(0, canvas_size[0] - icon.size[0])
            y = random.randint(0, canvas_size[1] - icon.size[1])
            
            # 粘贴图标
            background.paste(icon, (x, y), icon)
            
            # 创建mask
            mask = Image.new('L', canvas_size, 0)
            alpha = np.array(icon_a1.split()[3])  # 获取 alpha 通道
            alpha = np.where(alpha > 128, 255, 0)  # 二值化
            mask.paste(Image.fromarray(alpha.astype(np.uint8)), (x, y))
            
            # 保存文件
            output_name = f"{svg_file.stem}_var{variation}"
            background.convert('RGB').save(images_dir / f'{output_name}.png')

            image_info = {
                'id': img_idx,
                'file_name': str(images_dir / f'{output_name}.png'),
                'width': img_size[1],
                'height': img_size[0]
            }
            all_data['images'].append(image_info)

            #mask.save(masks_dir / f'{output_name}.png')
            mask = np.array(mask)
            y_indices, x_indices = np.where(mask > 0)
            if len(x_indices) > 0 and len(y_indices) > 0:
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                bbox = [int(x_min), int(y_min), 
                        int(x_max - x_min), int(y_max - y_min)]
                
                # 创建标注信息
                annotation = {
                    'id': annotation_id,
                    'image_id': img_idx,
                    'category_id': 1,  # 可以根据需要设置类别
                    'bbox': bbox,
                    'area': int(np.sum(mask)),
                    'segmentation': binary_mask_to_rle(mask),
                }
                all_data['annotations'].append(annotation)
                annotation_id += 1
            
            os.remove(temp_png)
            print(f"Processed: {output_name} with background color {bg_color}")
    with open(os.path.join(output_dir, 'annotations.json'), 'w') as f:
        json.dump(all_data, f)

# 使用示例
create_icon_dataset_with_color(
    svg_dir="./OmniSVG_MMSVG_Icon/train_2k",
    output_dir="./train",
    canvas_size=(512, 512),
    icon_size_range=(128, 256),
    rotation_range=(0, 0),
    num_variations=1
)
