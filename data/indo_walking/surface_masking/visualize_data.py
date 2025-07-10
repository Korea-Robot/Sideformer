import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from PIL import Image
import os
import cv2
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    TrainingArguments,
    Trainer
)

import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# ===================================================================
# 사용자께서 제공해주신 코드 (수정 없이 그대로 사용)
# ===================================================================

class SurfaceSegmentationDataset(Dataset):
    """전처리된 이미지/마스크 경로를 받아 PyTorch 텐서로 변환하는 클래스"""
    def __init__(self, data_list: list, target_size: tuple, is_train: bool):
        self.data_list = data_list
        self.target_size = target_size
        self.is_train = is_train

        if self.is_train:
            self.image_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.target_size),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
            ])

        if self.is_train:
            self.mask_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ])
        else:
            self.mask_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ])

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> dict:
        item = self.data_list[idx]
        image_path = item['image_path']
        mask_path = item['mask_path']

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.is_train:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            image = self.image_transform(image)
            torch.manual_seed(seed)
            mask = self.mask_transform(mask)
        else:
            image = self.image_transform(image)
            mask = self.mask_transform(mask)

        mask = mask.squeeze(0).long()
        return {'pixel_values': image, 'labels': mask}

# --- 설정 ---
metadata_path = "./processed_dataset/metadata.json"
batch_size = 32
target_size = (512, 512)
num_workers = 16 # 환경에 맞게 조절하세요

# metadata.json 로드 및 데이터 로더 생성
with open(metadata_path, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

valid_dataset = SurfaceSegmentationDataset(metadata['valid_data'], target_size, is_train=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


# ===================================================================
# 데이터 시각화 및 저장 코드 (수정됨)
# ===================================================================

ID_TO_COLOR = [
    [0, 0, 0],         # 0: background - Black
    [255, 0, 0],       # 1: class 1 - Red
    [0, 255, 0],       # 2: class 2 - Green
    [0, 0, 255],       # 3: class 3 - Blue
    [255, 255, 0],     # 4: class 4 - Yellow
]

def visualize_and_save_batch(images, masks, output_dir, num_to_show=10):
    """
    데이터로더에서 가져온 배치를 시각화하고 파일로 저장합니다.
    
    Args:
        images (torch.Tensor): 이미지 텐서 (B, C, H, W)
        masks (torch.Tensor): 마스크 텐서 (B, H, W)
        output_dir (str): 시각화 결과를 저장할 디렉토리 경로
        num_to_show (int): 저장할 샘플의 최대 개수
    """
    count = min(images.size(0), num_to_show)
    
    for i in range(count):
        # 1. 텐서를 NumPy 배열로 변환
        image_np = (images[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask_np = masks[i].numpy().astype(np.uint8)

        # 2. 컬러 마스크 생성
        mask_color = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
        for class_id, color in enumerate(ID_TO_COLOR):
            mask_color[mask_np == class_id] = color

        # 3. 이미지와 마스크 중첩
        overlayed_image = cv2.addWeighted(image_np, 0.6, mask_color, 0.4, 0)
        
        # 4. Matplotlib으로 이미지들 표시
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        ax1.imshow(image_np)
        ax1.set_title(f'Original Image #{i+1}')
        ax1.axis('off')
        
        ax2.imshow(mask_color)
        ax2.set_title(f'Ground Truth Mask #{i+1}')
        ax2.axis('off')
        
        ax3.imshow(overlayed_image)
        ax3.set_title(f'Overlayed Image #{i+1}')
        ax3.axis('off')
        
        plt.tight_layout()
        
        # 5. 이미지 파일로 저장
        save_path = os.path.join(output_dir, f'result_sample_{i+1}.png')
        plt.savefig(save_path)
        plt.close(fig)  # 메모리 누수 방지를 위해 창을 닫아줍니다.
        
        print(f"✅ 시각화 결과 저장 완료: {save_path}")


# --- 시각화 및 저장 실행 ---
if __name__ == '__main__':
    # 결과를 저장할 디렉토리 설정
    output_directory = "visualization_results"
    
    # 디렉토리가 없으면 생성
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"📂 '{output_directory}' 디렉토리를 생성했습니다.")

    # 데이터 로더에서 한 배치를 가져옵니다.
    data_iter = iter(valid_loader)
    batch = next(data_iter)
    
    pixel_values = batch['pixel_values']
    labels = batch['labels']
    
    # 가져온 배치에서 10개의 샘플을 시각화하고 저장합니다.
    print(f"\n🚀 데이터셋에서 {min(10, batch_size)}개의 샘플을 시각화하고 저장합니다...")
    visualize_and_save_batch(
        pixel_values, 
        labels, 
        output_dir=output_directory, 
        num_to_show=10
    )