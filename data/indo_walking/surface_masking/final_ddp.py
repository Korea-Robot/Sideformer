# train_ddp.py - 수정된 DDP 버전
import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"  # 중복 제거
os.environ["NCCL_P2P_DISABLE"] = '1'  # GPU P2P 통신 끔

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import SegformerForSemanticSegmentation
import wandb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2

from surface_dataset import SurfaceSegmentationDataset


def setup_ddp(rank, world_size):
    """DDP 환경 설정"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12359'
    
    # 중요: rank가 실제 GPU 인덱스와 일치하도록 수정
    # CUDA_VISIBLE_DEVICES="3,4,5"로 설정했으므로 
    # PyTorch에서는 0,1,2로 매핑됨
    local_gpu = rank
    
    # CUDA 디바이스 설정
    torch.cuda.set_device(local_gpu)
    
    # 프로세스 그룹 초기화
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    return local_gpu

def cleanup():
    """DDP 환경 정리"""
    if dist.is_initialized():  # 초기화 확인 후 정리
        dist.destroy_process_group()

def create_colormap(num_classes):
    """클래스별 고유 색상 생성"""
    colors = []
    for i in range(num_classes):
        # HSV 색공간에서 균등하게 분포된 색상 생성
        hue = i / num_classes
        saturation = 0.8
        value = 0.9
        
        # HSV를 RGB로 변환
        c = value * saturation
        x = c * (1 - abs((hue * 6) % 2 - 1))
        m = value - c
        
        if hue < 1/6:
            r, g, b = c, x, 0
        elif hue < 2/6:
            r, g, b = x, c, 0
        elif hue < 3/6:
            r, g, b = 0, c, x
        elif hue < 4/6:
            r, g, b = 0, x, c
        elif hue < 5/6:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
            
        colors.append([r + m, g + m, b + m])
    
    # 배경은 검은색으로
    colors[0] = [0, 0, 0]
    
    return ListedColormap(colors)

def visualize_predictions(model, valid_loader, device, epoch, save_dir="validation_results", num_samples=4, rank=0):
    """검증 결과를 시각화하고 저장 (rank 0에서만 실행)"""
    if rank != 0:
        return
        
    model.eval()
    
    os.makedirs(save_dir, exist_ok=True)
    
    colormap = create_colormap(7)  # num_labels 하드코딩
    
    with torch.no_grad():
        for batch_idx, data in enumerate(valid_loader):
            if batch_idx >= num_samples:  # 지정된 수만큼만 시각화
                break
                
            images = data['pixel_values'].to(device)
            masks = data['labels'].to(device)
            
            # 추론 - SegFormer 모델로 예측
            inputs = {'pixel_values': images}
            outputs = model(**inputs)
            logits = outputs.logits
            
            upsampled_logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            
            predictions = torch.argmax(upsampled_logits, dim=1)
            
            # 첫 번째 이미지만 시각화
            img = images[0].cpu().numpy().transpose(1, 2, 0)
            gt_mask = masks[0].cpu().numpy()
            pred_mask = predictions[0].cpu().numpy()
            
            # 이미지 정규화 해제 (0-1 범위로)
            img = (img - img.min()) / (img.max() - img.min())
            
            # 서브플롯 생성
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 원본 이미지
            axes[0].imshow(img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Ground Truth
            axes[1].imshow(gt_mask, cmap=colormap, vmin=0, vmax=6)  # 클래스 수에 맞게 수정
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            # Prediction
            axes[2].imshow(pred_mask, cmap=colormap, vmin=0, vmax=6)  # 클래스 수에 맞게 수정
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # 저장
            save_path = os.path.join(save_dir, f"loss_epoch_{epoch+1}_sample_{batch_idx+1}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Validation visualization saved: {save_path}")

def validate_model(model, valid_loader, device):
    """검증 함수 - SegFormer 내장 loss 사용"""
    model.eval()
    val_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    
    with torch.no_grad():
        for val_data in valid_loader:
            val_imgs = val_data['pixel_values'].to(device)
            val_masks = val_data['labels'].to(device)
            
            # SegFormer 모델 직접 사용
            val_inputs = {
                'pixel_values': val_imgs,
                'labels': val_masks
            }
            val_outputs = model(**val_inputs)
            
            loss = val_outputs.loss
            val_loss += loss.item()
            
            # 예측 결과 계산
            val_logits = val_outputs.logits
            val_upsampled = F.interpolate(
                val_logits,
                size=val_masks.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            
            # 정확도 계산
            preds = torch.argmax(val_upsampled, dim=1)
            correct_pixels += (preds == val_masks).sum().item()
            total_pixels += val_masks.numel()
    
    avg_val_loss = val_loss / len(valid_loader)
    pixel_accuracy = correct_pixels / total_pixels
    
    return avg_val_loss, pixel_accuracy

def main(rank, world_size):
    """메인 학습 함수"""
    
    try:
        # DDP 설정
        local_gpu = setup_ddp(rank, world_size)
        device = torch.device(f'cuda:{local_gpu}')
        
        # 상수 정의
        metadata_path = "/home/work/data/indo_walking/surface_masking/processed_dataset/metadata.json"

        # 클래스 매핑    
        class_to_idx = {
            'background': 0, 'caution_zone': 1, 'bike_lane': 2, 'alley': 3,
            'roadway': 4, 'braille_guide_blocks': 5, 'sidewalk': 6
        }
        id2label = {int(idx): label for label, idx in class_to_idx.items()}
        label2id = class_to_idx
        num_labels = len(id2label)
        
        if rank == 0:
            print(f"클래스 수: {num_labels}")
            print(f"클래스 레이블: {id2label}")
        
        # --- 데이터 로드 ---
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        data_base_path = os.path.dirname(os.path.dirname(metadata_path))

        # 하이퍼파라미터 설정 (상단에서 정의)
        batch_size = 64  # GPU 메모리에 맞게 조정 (64는 너무 클 수 있음)
        target_size = (512, 512)
        num_workers = 4  # 프로세스당 worker 수 감소
        epochs = 100
        lr = 5e-5
        weight_decay = 1e-4

        # 데이터셋 인스턴스 생성
        train_dataset = SurfaceSegmentationDataset(
            metadata['train_data'], 
            target_size, 
            is_train=True, 
            data_base_path=data_base_path
        )
        valid_dataset = SurfaceSegmentationDataset(
            metadata['valid_data'], 
            target_size, 
            is_train=False, 
            data_base_path=data_base_path
        )

        # DDP용 샘플러 생성
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=True  # 명시적으로 shuffle 설정
        )
        valid_sampler = DistributedSampler(
            valid_dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=False  # 검증에서는 shuffle하지 않음
        )
        
        # 사용자 정의 collate 함수 (필요한 경우)
        def custom_collate_fn(batch):
            """사용자 정의 collate 함수"""
            pixel_values = torch.stack([item['pixel_values'] for item in batch])
            labels = torch.stack([item['labels'] for item in batch])
            return {
                'pixel_values': pixel_values,
                'labels': labels
            }
        
        # 데이터로더 생성 - num_workers=0으로 설정해서 멀티프로세싱 비활성화
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=train_sampler,
            num_workers=4,  # pickle 에러 방지를 위해 0으로 설정
            pin_memory=True,
            drop_last=True,  # 마지막 배치 크기가 다를 때 DDP 문제 방지
            collate_fn=custom_collate_fn  # 사용자 정의 collate 함수
        )
        
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=batch_size, 
            sampler=valid_sampler,
            num_workers=4,  # pickle 에러 방지를 위해 0으로 설정
            pin_memory=True,
            drop_last=False,  # 검증에서는 모든 데이터 사용
            collate_fn=custom_collate_fn  # 사용자 정의 collate 함수
        )
        
        # 모델 생성
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            id2label=id2label,  # 레이블 매핑 추가
            label2id=label2id
        )
        model.to(device)
        
        # DDP 래핑 - find_unused_parameters=True 추가 (필요시)
        model = DDP(
            model, 
            device_ids=[local_gpu], 
            output_device=local_gpu,
            find_unused_parameters=False  # 성능상 False가 좋지만 문제시 True로 변경
        )
        
        # 옵티마이저
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # 학습률 스케줄러
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # WandB 초기화 (rank 0에서만)
        if rank == 0:
            wandb.init(
                project="segmentation_project", 
                name=f"ddp_run_0709-surface_masking",
                config={
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "epochs": epochs,
                    "num_gpus": world_size,
                    "model": "nvidia/mit-b0",
                    "target_size": target_size
                }
            )
        
        # 학습 루프
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # 샘플러 epoch 설정 (중요!)
            train_sampler.set_epoch(epoch)
            
            model.train()
            running_loss = 0.0
            num_batches = 0
            
            for i, data in enumerate(train_loader):
                images = data['pixel_values'].to(device, non_blocking=True)
                masks = data['labels'].to(device, non_blocking=True)
                
                # SegFormer 모델 입력 구성
                inputs = {
                    'pixel_values': images,
                    'labels': masks
                }
                
                # 순전파
                outputs = model(**inputs)
                
                # SegFormer 내장 loss 사용
                loss = outputs.loss
                
                # 역전파
                optimizer.zero_grad()
                loss.backward()
                
                # 그래디언트 클리핑 (선택사항)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                running_loss += loss.item()
                num_batches += 1
                
                # 로그 출력 (rank 0에서만)
                if (i + 1) % 50 == 0 and rank == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], "
                          f"Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
            
            # 검증 (모든 GPU에서 실행)
            val_loss, pixel_acc = validate_model(model, valid_loader, device)
            
            # 결과 수집 (all_reduce) - 평균 계산을 위해
            val_loss_tensor = torch.tensor(val_loss, device=device)
            pixel_acc_tensor = torch.tensor(pixel_acc, device=device)
            
            # 모든 GPU의 결과를 합산
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(pixel_acc_tensor, op=dist.ReduceOp.SUM)
            
            # 평균 계산
            val_loss = val_loss_tensor.item() / world_size
            pixel_acc = pixel_acc_tensor.item() / world_size
            
            # 학습률 스케줄러 업데이트
            scheduler.step()
            
            # 평균 손실 계산
            avg_train_loss = running_loss / num_batches
            
            # 모든 GPU의 훈련 손실도 동기화
            train_loss_tensor = torch.tensor(avg_train_loss, device=device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss = train_loss_tensor.item() / world_size
            
            # 검증 결과 시각화 (rank 0에서만)
            if rank == 0:
                visualize_predictions(
                    model.module, valid_loader, device, epoch, 
                    save_dir="validation_results", num_samples=4, rank=rank
                )
                
                # WandB 로깅
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "pixel_accuracy": pixel_acc,
                    "learning_rate": scheduler.get_last_lr()[0]
                })
                
                print(f"Epoch [{epoch+1}/{epochs}] - "
                      f"Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Pixel Acc: {pixel_acc:.4f}")
            
            # 모델 저장 (rank 0에서만)
            if rank == 0:
                # 체크포인트 디렉토리 생성
                os.makedirs("ckpts", exist_ok=True)
                
                # 모델 저장
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.module.state_dict(), f"ckpts/best_seg_model.pth")
                    print(f"Best 모델 저장! Val Loss: {val_loss:.4f}")
                
                # 정기적으로 체크포인트 저장
                if (epoch + 1) % 10 == 0:
                    torch.save(model.module.state_dict(), f"ckpts/seg_model_epoch_{epoch+1}.pth")
        
        if rank == 0:
            print("학습 완료!")
            wandb.finish()
        
    except Exception as e:
        print(f"Rank {rank}에서 에러 발생: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # DDP 환경 정리
        cleanup()

if __name__ == "__main__":
    # 사용할 GPU 수 설정 (3,4,5번 GPU)
    world_size = 3
    
    # 멀티프로세싱으로 DDP 실행
    import torch.multiprocessing as mp
    
    # 멀티프로세싱 시작 방법 설정
    mp.set_start_method('spawn', force=True)
    
    try:
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    except KeyboardInterrupt:
        print("학습이 중단되었습니다.")
    except Exception as e:
        print(f"DDP 실행 중 에러 발생: {e}")
        import traceback
        traceback.print_exc()