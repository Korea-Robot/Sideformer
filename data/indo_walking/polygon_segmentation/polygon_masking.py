import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import json
from PIL import Image

# --- 설정 ---
base_dir = './'
xml_files = glob.glob(os.path.join(base_dir, 'Polygon_*', '*.xml'))
mapping_txt = os.path.join(base_dir, 'class_mapping.txt')

# === 1. 전체 XML에서 클래스 수집 ===
class_names = set()
for xml_path in xml_files:
    tree = ET.parse(xml_path)
    for image in tree.getroot().findall('image'):
        for poly in image.findall('polygon'):
            class_names.add(poly.attrib['label'])

class_names = sorted(list(class_names))
class_to_id = {name: i + 1 for i, name in enumerate(class_names)}  # 0: 배경

# === 2. 폴더별 마스킹 ===
for xml_path in xml_files:
    folder_dir = os.path.dirname(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # mask/ 디렉토리 생성
    mask_dir = os.path.join(folder_dir, 'mask')
    os.makedirs(mask_dir, exist_ok=True)

    for image in root.findall('image'):
        img_name = image.attrib['name']
        img_w = int(image.attrib['width'])
        img_h = int(image.attrib['height'])

        # Grayscale 마스크 초기화
        mask = np.zeros((img_h, img_w), dtype=np.uint8)

        for poly in image.findall('polygon'):
            label = poly.attrib['label']
            class_id = class_to_id[label]

            polygon = []
            for pt in poly.attrib['points'].split(';'):
                if ',' in pt:
                    x_str, y_str = pt.split(',')
                    x = min(max(int(float(x_str)), 0), img_w - 1)
                    y = min(max(int(float(y_str)), 0), img_h - 1)
                    polygon.append([x, y])

            if len(polygon) >= 3:
                pts_np = np.array([polygon], dtype=np.int32)
                cv2.fillPoly(mask, [pts_np], color=class_id)

        # 저장
        base_name = os.path.splitext(img_name)[0]
        save_path = os.path.join(mask_dir, f"{base_name}_mask.png")
        Image.fromarray(mask, mode='L').save(save_path)
        print(f"저장됨: {save_path}")

# === 3. 클래스 매핑 저장 ===
with open(mapping_txt, 'w') as f:
    f.write("class_id\tclass_name\n")
    f.write("0\tBACKGROUND\n")
    for name, idx in class_to_id.items():
        f.write(f"{idx}\t{name}\n")

print(f"클래스 매핑 저장: {mapping_txt}")

# --- 4. 클래스 매핑 JSON 저장 ---
# 고정된 색상 맵 생성 (기존 코드에서 추가)
np.random.seed(42)
color_map = np.random.randint(0, 256, (len(class_to_id) + 1, 3), dtype=np.uint8)
color_map[0] = [0, 0, 0]  # background

mapping_json = {}
mapping_json["0"] = {
    "class_name": "BACKGROUND",
    "color": color_map[0].tolist()
}
for class_name, class_id in class_to_id.items():
    mapping_json[str(class_id)] = {
        "class_name": class_name,
        "color": color_map[class_id].tolist()
    }

json_path = os.path.join(base_dir, "class_mapping.json")
with open(json_path, 'w') as jf:
    json.dump(mapping_json, jf, indent=4)
print(f"클래스 매핑 JSON 저장: {json_path}")