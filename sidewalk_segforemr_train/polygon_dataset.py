# polygon_dataset.py
import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
from torchvision import transforms


class PolygonSegmentationDataset(Dataset):
    """
    Polygon Segmentation 데이터셋을 위한 커스텀 Dataset 클래스
    
    Args:
        root_dir (str): 데이터셋 루트 디렉토리 경로
        class_mapping_file (str): 클래스 매핑 파일 경로 (json 또는 txt)
        split (str): 'train' 또는 'val' 데이터셋 분할
        transform (callable, optional): 데이터 변환 함수
        target_size (tuple, optional): 이미지 리사이즈 크기 (height, width)
    """
    
    def __init__(
        self, 
        root_dir: str,
        class_mapping_file: str,
        split: str = 'train',
        transform=None,
        target_size: Optional[Tuple[int, int]] = (512, 512)
    ):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_size = target_size
        
        # 클래스 매핑 로드
        self.id2label, self.label2id, self.num_labels = self._load_class_mapping(class_mapping_file)
        
        # 데이터 파일 경로 수집
        self.data_pairs = self._collect_data_pairs()
        
        # 데이터셋 분할
        self.data_pairs = self._split_dataset()
        

        # image transformation 데이터 
        self.image_transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor()
        ])

        self.mask_transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(degrees=15),
            transforms.ToTensor()  # float tensor로 변환되지만 mask이므로 후처리 필요
        ])

        print(f"{split.upper()} 데이터셋: {len(self.data_pairs)}개")
        print(f"클래스 수: {self.num_labels}")
        print(f"이미지 크기: {self.target_size}")
    
    def _load_class_mapping(self, class_mapping_file: str) -> Tuple[Dict, Dict, int]:
        """클래스 매핑 파일을 로드하여 id2label, label2id 딕셔너리 생성"""
        if class_mapping_file.endswith('.json'):
            with open(class_mapping_file, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            id2label = {int(k): v for k, v in mapping.items()}
        
        elif class_mapping_file.endswith('.txt'):
            id2label = {}
            with open(class_mapping_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[1:]:  # 첫 번째 라인은 헤더 스킵
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        class_id = int(parts[0])
                        class_name = parts[1]
                        id2label[class_id] = class_name
        else:
            raise ValueError("클래스 매핑 파일은 .json 또는 .txt 형식이어야 합니다.")
        
        label2id = {v: k for k, v in id2label.items()}
        num_labels = len(id2label)
        
        return id2label, label2id, num_labels
    
    def _collect_data_pairs(self) -> List[Tuple[str, str]]:
        """이미지와 마스크 파일 경로 쌍을 수집"""
        data_pairs = []
        
        # 모든 Polygon_ 폴더 탐색
        for folder_name in os.listdir(self.root_dir):
            if folder_name.startswith('Polygon_'):
                folder_path = os.path.join(self.root_dir, folder_name)
                mask_folder = os.path.join(folder_path, 'mask')
                
                if not os.path.exists(mask_folder):
                    continue
                
                # 마스크 파일들 탐색
                for mask_file in os.listdir(mask_folder):
                    if mask_file.endswith('_mask.png'):
                        # 원본 이미지 파일명 추출
                        image_file = mask_file.replace('_mask.png', '.jpg')
                        image_path = os.path.join(folder_path, image_file)
                        mask_path = os.path.join(mask_folder, mask_file)
                        
                        # 두 파일이 모두 존재하는지 확인
                        if os.path.exists(image_path) and os.path.exists(mask_path):
                            data_pairs.append((image_path, mask_path))
        
        return data_pairs
    
    def _split_dataset(self) -> List[Tuple[str, str]]:
        """데이터셋을 train/val로 분할 (8:2 비율)"""
        total_samples = len(self.data_pairs)
        train_size = int(0.95 * total_samples)
        
        # 재현 가능한 분할을 위해 정렬
        self.data_pairs.sort()
        
        if self.split == 'train':
            return self.data_pairs[:train_size]
        else:  # val
            return self.data_pairs[train_size:]
    
    def _load_and_preprocess_image(self, image_path: str) -> Image.Image:
        """이미지 로드 및 전처리"""
        image = Image.open(image_path).convert('RGB')
        
        if self.target_size:
            image = image.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
        
        
        # image  : <PIL.Image.Image image mode=RGB size=512x512 at 0x7A0F90213A60> => torch.tensor
        return self.image_transform(image)
    
    def _load_and_preprocess_mask(self, mask_path: str) -> Image.Image:
        """마스크 로드 및 전처리"""
        mask = Image.open(mask_path)
        
        # 마스크가 RGB인 경우 L로 변환
        if mask.mode != 'L':
            mask = mask.convert('L')
        
        if self.target_size:
            mask = mask.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)
        
        # mask <PIL.Image.Image image mode=L size=512x512 at 0x7A0F903B9150> => torch.tensor
        mask = self.mask_transform(mask)
        return mask.long()
    
    def __len__(self) -> int:
        return len(self.data_pairs)
    
    def __getitem__(self, idx: int) -> Dict:
        """데이터셋 아이템 반환"""
        image_path, mask_path = self.data_pairs[idx]
        
        # 이미지와 마스크 로드
        image = self._load_and_preprocess_image(image_path)
        mask = self._load_and_preprocess_mask(mask_path).squeeze(dim=0)

        
        # 딕셔너리 형태로 반환 (HuggingFace 스타일)
        sample = {
            'pixel_values': image,
            'labels': mask,
            'image_path': image_path,
            'mask_path': mask_path
        }
        
        # transform이 있으면 적용
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def get_class_info(self) -> Tuple[Dict, Dict, int]:
        """클래스 정보 반환"""
        return self.id2label, self.label2id, self.num_labels
    
    def visualize_sample(self, idx: int, save_path: Optional[str] = None):
        """샘플 시각화"""
        import matplotlib.pyplot as plt
        
        sample = self[idx]
        image = sample['pixel_values']
        mask = sample['labels']
        
        # PIL Image를 numpy array로 변환
        if hasattr(image, 'convert'):
            image = np.array(image)
        if hasattr(mask, 'convert'):
            mask = np.array(mask)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(mask, cmap='tab20')
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


def create_polygon_datasets(
    root_dir: str,
    class_mapping_file: str,
    target_size: Tuple[int, int] = (512, 512)
) -> Tuple[PolygonSegmentationDataset, PolygonSegmentationDataset]:
    """
    훈련용과 검증용 데이터셋을 생성하는 팩토리 함수
    
    Args:
        root_dir (str): 데이터셋 루트 디렉토리
        class_mapping_file (str): 클래스 매핑 파일 경로
        target_size (tuple): 이미지 리사이즈 크기
    
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    train_dataset = PolygonSegmentationDataset(
        root_dir=root_dir,
        class_mapping_file=class_mapping_file,
        split='train',
        target_size=target_size
    )
    
    val_dataset = PolygonSegmentationDataset(
        root_dir=root_dir,
        class_mapping_file=class_mapping_file,
        split='val',
        target_size=target_size
    )
    
    return train_dataset, val_dataset


if __name__ == "__main__":
    # 테스트 코드
    root_dir = "/home/work/data/indo_walking/polygon_segmentation"
    class_mapping_file = "/home/work/data/indo_walking/polygon_segmentation/class_mapping.txt"
    
    # 데이터셋 생성
    train_dataset, val_dataset = create_polygon_datasets(
        root_dir=root_dir,
        class_mapping_file=class_mapping_file,
        target_size=(512, 512)
    )
    
    # 첫 번째 샘플 확인
    sample = train_dataset[0]
    print(f"이미지 경로: {sample['image_path']}")
    print(f"마스크 경로: {sample['mask_path']}")
    print(f"이미지 크기: {sample['pixel_values'].shape}")

    
    print(f"마스크 크기: {sample['labels'].shape}")
    
    breakpoint()
    # 클래스 정보 출력
    id2label, label2id, num_labels = train_dataset.get_class_info()
    print(f"\n클래스 수: {num_labels}")
    print("클래스 매핑:")
    for class_id, class_name in list(id2label.items())[:10]:  # 처음 10개만 출력
        print(f"  {class_id}: {class_name}")
    
    breakpoint()
    
    """
    (Pdb) sample['pixel_values'].shape
    torch.Size([3, 512, 512])
    (Pdb) sample['labels'].shape
    torch.Size([1, 512, 512])
    """
    
    # # 샘플 시각화 (matplotlib 필요)
    # try:
    #     train_dataset.visualize_sample(0, "sample_visualization.png")
    #     print("\n샘플 이미지가 'sample_visualization.png'로 저장되었습니다.")
    # except ImportError:
    #     print("\n시각화를 위해서는 matplotlib가 필요합니다.")
    
    # breakpoint()