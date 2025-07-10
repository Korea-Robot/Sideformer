import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os

# --- PyTorch 데이터셋 클래스 ---
class SurfaceSegmentationDataset(Dataset):
    """전처리된 이미지/마스크 경로를 받아 PyTorch 텐서로 변환하는 클래스"""
    def __init__(self, data_list: List[Dict], target_size: Tuple[int, int], is_train: bool, data_base_path: str):
        self.data_list = data_list
        self.target_size = target_size
        self.is_train = is_train
        self.data_base_path = data_base_path # 데이터셋의 기본 경로

        # 이미지 변환 (학습 시에만 Augmentation 적용)
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.target_size),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
            ], p=0.5) if self.is_train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # metadata.json에 저장된 파일 경로 읽기
        item = self.data_list[idx]
        
        # 기본 경로와 상대 경로를 조합하여 절대 경로 생성
        image_path = os.path.join(self.data_base_path, item['image_path'])
        mask_path = os.path.join(self.data_base_path, item['mask_path'])

        # 이미지와 마스크 로드
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"이미지 파일을 로드할 수 없습니다: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"마스크 파일을 로드할 수 없습니다: {mask_path}")

        # 이미지 변환 적용
        image = self.image_transform(image)

        # 마스크 리사이즈 및 텐서 변환
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        mask = torch.from_numpy(mask).long()  # (H, W), Long 타입으로 변환

        return {'pixel_values': image, 'labels': mask}


# --- 데이터로더 생성 함수 ---
def create_dataloaders(metadata_path: str, batch_size: int, target_size: Tuple[int, int], num_workers: int) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    metadata.json을 읽어 학습/검증 데이터로더를 생성
    """
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # metadata.json 파일이 위치한 디렉토리 경로를 구함
    data_base_path = os.path.dirname(os.path.dirname(metadata_path)) # processed_dataset의 상위 폴더

    # 데이터셋 인스턴스 생성
    train_dataset = SurfaceSegmentationDataset(metadata['train_data'], target_size, is_train=True, data_base_path=data_base_path)
    valid_dataset = SurfaceSegmentationDataset(metadata['valid_data'], target_size, is_train=False, data_base_path=data_base_path)

    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, valid_loader, metadata


# --- 시각화 함수 (결과 확인용) ---
def visualize_sample(dataset: Dataset, idx: int, metadata: Dict):
    """데이터셋의 샘플을 시각화하여 데이터 로딩이 잘 되었는지 확인"""
    class_colors = {int(k): v for k, v in metadata['class_colors'].items()}
    sample = dataset[idx]
    image_tensor = sample['pixel_values']
    mask_tensor = sample['labels']

    # 이미지 텐서 Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image_tensor * std + mean
    image = image.permute(1, 2, 0).numpy().clip(0, 1)

    # 마스크 텐서를 컬러 이미지로 변환
    mask_colored = np.zeros((*mask_tensor.shape, 3), dtype=np.uint8)
    for class_idx, color in class_colors.items():
        mask_colored[mask_tensor == class_idx] = color

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image)
    axes[0].set_title('Image')
    axes[0].axis('off')
    axes[1].imshow(mask_colored)
    axes[1].set_title('Mask')
    axes[1].axis('off')
    # plt.show()
    savefig('sampleimagemaskdata.png')


# --- 메인 실행 부분 ---
if __name__ == "__main__":
    # --- 설정 ---
    # 전처리된 데이터 정보가 담긴 metadata.json 파일 경로
    METADATA_PATH = "/home/work/data/indo_walking/surface_masking/processed_dataset/metadata.json"
    
    # 학습 파라미터
    BATCH_SIZE = 8
    TARGET_SIZE = (512, 512)
    NUM_WORKERS = 4  # CPU 코어 수에 맞게 조절

    # --- 실행 ---
    try:
        print(f"'{METADATA_PATH}' 파일을 이용해 데이터로더를 생성합니다.")
        train_loader, valid_loader, metadata = create_dataloaders(
            metadata_path=METADATA_PATH,
            batch_size=BATCH_SIZE,
            target_size=TARGET_SIZE,
            num_workers=NUM_WORKERS
        )

        print("\n🎉 데이터로더 생성 완료!")
        print(f"  - 학습용 데이터 수: {len(train_loader.dataset)}")
        print(f"  - 검증용 데이터 수: {len(valid_loader.dataset)}")
        print(f"  - 학습용 배치 수: {len(train_loader)}")
        print(f"  - 검증용 배치 수: {len(valid_loader)}")

        # 첫 번째 학습 배치 정보 확인
        print("\n-- 첫 학습 배치 샘플 정보 --")
        sample_batch = next(iter(train_loader))
        print(f"  - 이미지 배치 형태: {sample_batch['pixel_values'].shape}")
        print(f"  - 마스크 배치 형태: {sample_batch['labels'].shape}")

        mask_sample = sample_batch['labels']

        # breakpoint()

        # 검증 데이터셋의 첫 번째 샘플 시각화로 확인
        print("\n[결과 확인] 검증 데이터셋의 첫 번째 샘플을 시각화합니다...")
        visualize_sample(dataset=valid_loader.dataset, idx=0, metadata=metadata)

    except FileNotFoundError as e:
        print(f"🚨 오류: 파일을 찾을 수 없습니다.")
        print(e)
        print("경로가 올바른지 확인해주세요.")
    except Exception as e:
        print(f"🚨 예기치 않은 오류 발생: {e}")