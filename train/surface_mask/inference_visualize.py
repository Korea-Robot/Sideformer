import os
import json
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm # 진행 상황 표시
import sys

# surface_dataset.py가 같은 디렉토리에 있거나 경로가 설정되어 있어야 합니다.
try:
    from surface_dataset import SurfaceSegmentationDataset
except ImportError:
    print("오류: surface_dataset.py를 찾을 수 없습니다.")
    print("train_ddp.py와 같은 위치에 있거나 Python 경로에 추가되었는지 확인하세요.")
    sys.exit(1)

# 글로벌 변수로 클래스 매핑 정의 (원본 코드와 동일)
CLASS_TO_IDX = {
    'background': 0, 'caution_zone': 1, 'bike_lane': 2, 'alley': 3,
    'roadway': 4, 'braille_guide_blocks': 5, 'sidewalk': 6
}

# 글로벌 collate 함수 정의 (원본 코드와 동일)
def custom_collate_fn(batch):
    try:
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        return {
            'pixel_values': pixel_values,
            'labels': labels
        }
    except Exception as e:
        print(f"Collate function error: {e}")
        return {
            'pixel_values': batch[0]['pixel_values'].unsqueeze(0),
            'labels': batch[0]['labels'].unsqueeze(0)
        }

def create_colormap(num_classes):
    """클래스별 고유 색상 생성 (원본 코드와 동일)"""
    colors = []
    for i in range(num_classes):
        hue = i / num_classes
        saturation = 0.8
        value = 0.9
        c = value * saturation
        x = c * (1 - abs((hue * 6) % 2 - 1))
        m = value - c
        if hue < 1/6: r, g, b = c, x, 0
        elif hue < 2/6: r, g, b = x, c, 0
        elif hue < 3/6: r, g, b = 0, c, x
        elif hue < 4/6: r, g, b = 0, x, c
        elif hue < 5/6: r, g, b = x, 0, c
        else: r, g, b = c, 0, x
        colors.append([r + m, g + m, b + m])
    colors[0] = [0, 0, 0]
    return ListedColormap(colors)

def create_validation_loader():
    """검증 데이터 로더 생성 함수 (DDP 없이)"""
    metadata_path = "/home/work/data/indo_walking/surface_masking/processed_dataset/metadata.json"
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    data_base_path = os.path.dirname(os.path.dirname(metadata_path))
    target_size = (512, 512)
    batch_size = 16 # 시각화 및 추론 시에는 배치 크기를 조절할 수 있습니다.

    valid_dataset = SurfaceSegmentationDataset(
        metadata['valid_data'], 
        target_size, 
        is_train=False, 
        data_base_path=data_base_path
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, # 검증 시에는 섞지 않음
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    return valid_loader

def visualize_and_save(model, valid_loader, device, save_dir="inference_results", num_samples=10):
    """
    추론 결과를 시각화하고 저장하는 함수 (단일 GPU용)
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    colormap = create_colormap(len(CLASS_TO_IDX))
    
    processed_samples = 0
    
    with torch.no_grad():
        # tqdm을 사용하여 진행률 표시
        for i, data in enumerate(tqdm(valid_loader, desc="Visualizing Predictions")):
            if processed_samples >= num_samples:
                break
                
            images = data['pixel_values'].to(device)
            masks = data['labels'].to(device)
            
            # 추론
            outputs = model(pixel_values=images)
            logits = outputs.logits
            
            upsampled_logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            predictions = torch.argmax(upsampled_logits, dim=1)
            
            # 배치 내의 각 샘플에 대해 시각화
            for j in range(images.size(0)):
                if processed_samples >= num_samples:
                    break

                img = images[j].cpu().numpy().transpose(1, 2, 0)
                gt_mask = masks[j].cpu().numpy()
                pred_mask = predictions[j].cpu().numpy()
                
                # 이미지 정규화 해제 (0-1 범위로)
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = (img - img_min) / (img_max - img_min)
            
                # 서브플롯 생성
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                axes[0].imshow(img)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                axes[1].imshow(gt_mask, cmap=colormap, vmin=0, vmax=len(CLASS_TO_IDX)-1)
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                
                axes[2].imshow(pred_mask, cmap=colormap, vmin=0, vmax=len(CLASS_TO_IDX)-1)
                axes[2].set_title('Prediction')
                axes[2].axis('off')
                
                plt.tight_layout()
                
                save_path = os.path.join(save_dir, f"result_sample_{processed_samples + 1}.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                processed_samples += 1

    print(f"\n총 {processed_samples}개의 샘플 시각화를 완료하고 '{save_dir}'에 저장했습니다.")

def main():
    parser = argparse.ArgumentParser(description="SegFormer 모델 추론 및 시각화 스크립트")
    parser.add_argument('--model_path', type=str, default = "ckpts/ddp_seg_model_epoch_100.pth", required=True, help='학습된 모델 가중치(.pth) 파일 경로')
    parser.add_argument('--save_dir', type=str, default='inference_results', help='시각화 결과 저장 디렉토리')
    parser.add_argument('--num_samples', type=int, default=10, help='시각화할 샘플 수')
    parser.add_argument('--device', type=str, default='cuda:4', help='추론에 사용할 장치 (e.g., "cuda:4", "cpu")')
    args = parser.parse_args()

    # 장치 설정
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    # 클래스 정보 설정
    id2label = {int(idx): label for label, idx in CLASS_TO_IDX.items()}
    label2id = CLASS_TO_IDX
    num_labels = len(id2label)

    # 모델 생성
    print("모델을 생성합니다...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    
    # 학습된 가중치 불러오기
    print(f"'{args.model_path}' 에서 모델 가중치를 불러옵니다...")
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except FileNotFoundError:
        print(f"오류: 모델 파일 '{args.model_path}'를 찾을 수 없습니다.")
        sys.exit(1)

    model.to(device)
    
    # 데이터 로더 생성
    print("검증 데이터 로더를 생성합니다...")
    valid_loader = create_validation_loader()
    
    # 시각화 실행
    visualize_and_save(model, valid_loader, device, args.save_dir, args.num_samples)

if __name__ == "__main__":
    main()