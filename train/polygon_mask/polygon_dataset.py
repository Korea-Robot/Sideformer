import os
import json
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PolygonSegmentationDataset(Dataset):
    """
    폴리곤 분할 데이터를 로드하고 변환하는 PyTorch Dataset 클래스.

    Args:
        root_dir (str): 'Polygon_0001', 'Polygon_0002' 등이 포함된 최상위 디렉토리 경로.
        is_train (bool): True이면 학습용 데이터 증강을 적용하고, False이면 검증용 변환을 적용합니다.
        target_size (Tuple[int, int]): 이미지를 리사이즈할 목표 크기.
    """
    def __init__(self, root_dir: str, is_train: bool, target_size: Tuple[int, int] = (512, 512)):
        self.root_path = Path(root_dir)
        self.is_train = is_train
        self.target_size = target_size

        # 데이터 파일 쌍 (이미지, 마스크)을 찾습니다.
        self.data_pairs = self._get_data_pairs()
        if not self.data_pairs:
            raise IOError(f"'{root_dir}' 디렉토리에서 유효한 이미지/마스크 쌍을 찾을 수 없습니다.")

        # --- 이미지 변환 파이프라인 ---
        # 학습 시에는 데이터 증강(augmentation)을 적용합니다.
        if self.is_train:
            self.image_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.target_size),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
                ], p=0.5),
                # transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        # 검증 시에는 데이터 증강을 적용하지 않습니다.
        else:
            self.image_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # --- 마스크 변환 파이프라인 ---
        # 마스크는 픽셀값이 클래스 ID이므로 보간법으로 NEAREST를 사용해야 합니다.
        if self.is_train:
            self.mask_transform = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.NEAREST),
                # transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])
        else:
            self.mask_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.target_size),
                transforms.ToTensor()
            ])

    def _get_data_pairs(self) -> List[Dict[str, Path]]:
        """root_dir를 탐색하여 모든 (이미지, 마스크) 파일 쌍을 찾습니다."""
        data_pairs = []
        for poly_dir in self.root_path.iterdir():
            # 'Polygon_XXXX' 형태의 디렉토리인지 확인
            if poly_dir.is_dir() and poly_dir.name.startswith('Polygon_'):
                mask_dir = poly_dir / 'mask'
                if not mask_dir.exists():
                    continue

                # 디렉토리 내의 모든 jpg 이미지 검색
                for image_path in poly_dir.glob('*.jpg'):
                    # 해당 이미지에 대한 마스크 파일 경로 생성
                    mask_filename = f"{image_path.stem}_mask.png"
                    mask_path = mask_dir / mask_filename

                    # 이미지와 마스크 파일이 모두 존재할 경우에만 리스트에 추가
                    if mask_path.exists():
                        data_pairs.append({'image': image_path, 'mask': mask_path})
        return data_pairs

    def __len__(self) -> int:
        """데이터셋의 총 샘플 수를 반환합니다."""
        return len(self.data_pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """지정된 인덱스(idx)의 데이터 샘플(이미지, 마스크)을 로드하고 변환하여 반환합니다."""
        item = self.data_pairs[idx]

        # 이미지 로드 (OpenCV: BGR -> RGB로 변환)
        image = cv2.imread(str(item['image']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 마스크 로드 (단일 채널, 그레이스케일)
        # mask = cv2.imread(str(item['mask']), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(item['mask']), cv2.IMREAD_UNCHANGED)


        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        mask = torch.from_numpy(mask).long()  # (H, W)

        # 학습 시, 이미지와 마스크에 동일한 랜덤 변환(예: 좌우 반전)을 적용
        if self.is_train:
            seed = torch.seed() # 동일한 변환을 위한 시드 생성
            torch.manual_seed(seed)
            image = self.image_transform(image)
            torch.manual_seed(seed)
            # mask = self.mask_transform(mask)
        else:
            image = self.image_transform(image)
            # mask = self.mask_transform(mask)

        # 마스크 텐서의 차원을 (1, H, W) -> (H, W)로 변경하고,
        # 손실 계산을 위해 Long 타입으로 변환합니다.

        # mask = mask.squeeze(0).long()
        # mask=mask.squeeze(0)

        return {
            'pixel_values': image,
            'labels': mask
        }

if __name__ == '__main__':
    # --- 데이터셋 사용 예제 ---
    
    # 1. 설정
    ROOT_DIRECTORY = '~/data/indo_walking/polygon_segmentation'
    ROOT_DIRECTORY = "/home/work/data/indo_walking/polygon_segmentation"
    CLASS_MAPPING_FILE = os.path.join(ROOT_DIRECTORY, 'class_mapping.json')
    BATCH_SIZE = 4
    
    # 2. 클래스 정보 로드 (시각화 등에 사용 가능)
    try:
        with open(CLASS_MAPPING_FILE, 'r') as f:
            class_info = json.load(f)
        print(f"✅ 총 {len(class_info)}개의 클래스 정보를 로드했습니다.")
    except FileNotFoundError:
        print(f"🚨 클래스 매핑 파일('{CLASS_MAPPING_FILE}')을 찾을 수 없습니다.")
        class_info = None

    # 3. 학습용 데이터셋 및 데이터로더 생성
    train_dataset = PolygonSegmentationDataset(root_dir=ROOT_DIRECTORY, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # 4. 검증용 데이터셋 및 데이터로더 생성
    valid_dataset = PolygonSegmentationDataset(root_dir=ROOT_DIRECTORY, is_train=False)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"\n- 총 데이터 샘플 수: {len(train_dataset)}")
    print(f"- 학습용 배치 수: {len(train_loader)}")
    print(f"- 검증용 배치 수: {len(valid_loader)}")
    
    # 5. 데이터로더에서 한 배치를 가져와서 확인
    if len(train_loader) > 0:
        batch = next(iter(train_loader))
        images = batch['pixel_values']
        masks = batch['labels']
        
        print("\n--- 데이터 배치 확인 ---")
        print(f"이미지 텐서 모양 (Batch, Channels, Height, Width): {images.shape}")
        print(f"마스크 텐서 모양 (Batch, Height, Width): {masks.shape}")
        print(f"이미지 텐서 타입: {images.dtype}")
        print(f"마스크 텐서 타입: {masks.dtype}")
        print(f"한 배치 내 마스크 클래스 ID 분포: {torch.unique(masks)}")

        breakpoint()