## inference.py 

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from PIL import Image
import os
import cv2 # 텍스트 추가를 위해 OpenCV 임포트
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    TrainingArguments,
    Trainer
)

import json
import cv2

from torch.utils.data import Dataset, DataLoader

# from surface_dataset import SurfaceSegmentationDataset

# surface_dataset.py - 수정된 버전

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np
from typing import List, Dict, Tuple

class SurfaceSegmentationDataset(Dataset):
    """전처리된 이미지/마스크 경로를 받아 PyTorch 텐서로 변환하는 클래스"""
    def __init__(self, data_list: List[Dict], target_size: Tuple[int, int], is_train: bool):
        self.data_list = data_list
        self.target_size = target_size
        self.is_train = is_train

        # 이미지 변환 - 정규화 제거 (DirectSegFormer에서 처리)
        if self.is_train:
            self.image_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.target_size),
                # 세그멘테이션에서는 색상 변화를 최소화
                transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 색상 변화 최소화
                # transforms.RandomHorizontalFlip(p=0.3),  # 확률 감소
                transforms.ToTensor(),  # 0-1 범위로만 변환
                # 정규화 제거! DirectSegFormer에서 처리
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.target_size),
                transforms.ToTensor(),  # 0-1 범위로만 변환
                # 정규화 제거!
            ])

        # 마스크 변환 - 수평 뒤집기 추가 (이미지와 동일하게)
        if self.is_train:
            self.mask_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.NEAREST),
                # transforms.RandomHorizontalFlip(p=0.3),  # 이미지와 동일한 확률
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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # metadata.json에 저장된 파일 경로 읽기
        item = self.data_list[idx]
        image_path = item['image_path']
        mask_path = item['mask_path']

        # 이미지와 마스크 로드
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 동일한 시드로 랜덤 변환 적용 (이미지와 마스크가 같은 변환을 받도록)
        if self.is_train:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            image = self.image_transform(image)
            torch.manual_seed(seed)
            mask = self.mask_transform(mask)
        else:
            image = self.image_transform(image)
            mask = self.mask_transform(mask)

        # 마스크 텐서 형태 변경: (1, H, W) -> (H, W), Long 타입으로 변환
        mask = mask.squeeze(0).long()

        return {'pixel_values': image, 'labels': mask}

# --- 설정 ---
# 전처리된 데이터 정보가 담긴 metadata.json 파일 경로
metadata_path = "./processed_dataset/metadata.json"

# 학습 파라미터
batch_size = 32
target_size = (512, 512)
num_workers = 16  # CPU 코어 수에 맞게 조절

"""
metadata.json을 읽어 학습/검증 데이터로더를 생성
"""
with open(metadata_path, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# 데이터셋 인스턴스 생성
valid_dataset = SurfaceSegmentationDataset(metadata['valid_data'], target_size, is_train=False)

valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


# --- 사용자 설정 ---
# 비교할 모델의 체크포인트 경로 리스트 (오름차순으로 정렬)
MODEL_PATHS = [
    "ckpts/seg_model_epoch_1.pth",   # 예: 초기 epoch
    "ckpts/seg_model_epoch_5.pth",   # 예: 중간 epoch
    "ckpts/seg_model_epoch_30.pth"   # 예: 최종 epoch
]
# 각 모델에 해당하는 라벨
EPOCH_LABELS = ["Epoch 1", "Epoch 5", "Epoch 30"]

# 결과 이미지를 저장할 디렉터리
OUTPUT_DIR = "inference_progress_results"
# 저장할 이미지 개수
NUM_IMAGES_TO_SAVE = 10
# GPU 설정
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# --- 이전 코드의 클래스 및 변수 정의가 필요합니다 ---
# DirectSegFormer 모델 클래스 정의
class DirectSegFormer(nn.Module):
    def __init__(self, pretrained_model_name="nvidia/mit-b0", num_classes=7):
        super().__init__()
        self.original_model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    def forward(self, x):
        x = (x - self.mean) / self.std
        outputs = self.original_model(pixel_values=x)
        return outputs.logits

# --- 추론 및 시각화 코드 ---

# 1. 설정 검증 및 디렉터리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"결과 이미지는 '{OUTPUT_DIR}' 디렉터리에 저장됩니다.")

for path in MODEL_PATHS:
    if not os.path.exists(path):
        print(f"🚨 에러: 모델 파일을 찾을 수 없습니다 -> {path}")
        exit()

# 2. 시각화를 위한 헬퍼 함수
palette = {0: (0, 0, 0), 1: (255, 255, 0), 2: (0, 255, 0), 3: (100, 100, 100), 4: (255, 0, 0), 5: (0, 0, 255), 6: (255, 0, 255)}

def mask_to_rgb(mask_np, color_palette):
    rgb_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    for class_idx, color in color_palette.items():
        rgb_mask[mask_np == class_idx] = color
    return rgb_mask

def add_label_to_image(image, label):
    """OpenCV를 사용하여 이미지에 텍스트 라벨을 추가하는 함수"""
    # PIL 이미지를 OpenCV 포맷(BGR)으로 변환
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # 텍스트 설정
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, 30)  # 이미지 좌측 상단 좌표
    font_scale = 1
    font_color = (255, 255, 255)  # 흰색
    thickness = 2
    # 이미지에 텍스트 추가
    cv2.putText(img_cv, label, position, font, font_scale, font_color, thickness, cv2.LINE_AA)
    # 다시 PIL 포맷(RGB)으로 변환하여 반환
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

# 3. 모델 구조 로드 (가중치는 루프 안에서 교체)
model = DirectSegFormer(num_classes=7)
model.to(DEVICE)
model.eval()

# 4. 추론 및 이미지 생성 루프
saved_count = 0
with torch.no_grad():
    for batch_data in valid_loader:
        if saved_count >= NUM_IMAGES_TO_SAVE:
            break

        # 배치에서 첫 번째 이미지만 사용
        image_tensor = batch_data['pixel_values'][0].unsqueeze(0).to(DEVICE)
        gt_mask_tensor = batch_data['labels'][0]

        # 예측 결과를 저장할 리스트
        prediction_images = []

        # 지정된 각 모델에 대해 추론 수행
        for i, model_path in enumerate(MODEL_PATHS):
            # 현재 epoch의 모델 가중치 로드
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            
            # 추론
            logits = model(image_tensor)
            upsampled_logits = F.interpolate(logits, size=gt_mask_tensor.shape[-2:], mode='bilinear', align_corners=False)
            pred_mask = torch.argmax(upsampled_logits, dim=1).squeeze(0) # (1, H, W) -> (H, W)

            # NumPy 배열로 변환 후 RGB 마스크 생성 및 라벨 추가
            pred_mask_np = pred_mask.cpu().numpy()
            pred_mask_rgb = mask_to_rgb(pred_mask_np, palette)
            labeled_pred_img = add_label_to_image(Image.fromarray(pred_mask_rgb), EPOCH_LABELS[i])
            prediction_images.append(np.array(labeled_pred_img))
        
        # 원본 이미지와 정답(GT) 마스크 준비 및 라벨 추가
        img_np = (np.transpose(image_tensor.squeeze(0).cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8)
        gt_mask_np = gt_mask_tensor.numpy()
        gt_mask_rgb = mask_to_rgb(gt_mask_np, palette)

        labeled_original = add_label_to_image(Image.fromarray(img_np), "Original")
        labeled_gt = add_label_to_image(Image.fromarray(gt_mask_rgb), "Ground Truth")

        # [원본, 정답, 예측1, 예측2, 예측3] 순서로 이미지 리스트 생성
        all_images = [np.array(labeled_original), np.array(labeled_gt)] + prediction_images
        
        # 모든 이미지를 가로로 연결
        comparison_img = np.concatenate(all_images, axis=1)

        # 최종 결과 이미지 저장
        save_path = os.path.join(OUTPUT_DIR, f"progress_comparison_{saved_count + 1}.png")
        Image.fromarray(comparison_img).save(save_path)
        
        print(f"[{saved_count + 1}/{NUM_IMAGES_TO_SAVE}] 진행 과정 비교 이미지 저장 완료: {save_path}")
        saved_count += 1

print(f"\n✅ 모든 작업이 완료되었습니다. 총 {saved_count}개의 이미지가 '{OUTPUT_DIR}'에 저장되었습니다.")