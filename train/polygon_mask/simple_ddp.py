# simple_ddp_train.py - 간단한 DDP 버전

import os
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

# 이전 단계에서 작성한 PolygonSegmentationDataset 클래스를 임포트합니다.
from polygon_dataset import PolygonSegmentationDataset

def setup_ddp(rank, world_size):
    """DDP 환경 설정"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1'
    
    # CUDA 디바이스 설정
    torch.cuda.set_device(rank)
    
    # 프로세스 그룹 초기화
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    return rank

def cleanup():
    """DDP 환경 정리"""
    if dist.is_initialized():
        dist.destroy_process_group()

def validate_model(model, valid_loader, device, rank, world_size):
    """검증 함수"""
    model.eval()
    val_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    
    with torch.no_grad():
        for val_data in valid_loader:
            val_imgs = val_data['pixel_values'].to(device)
            val_masks = val_data['labels'].to(device)
            
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
            
            preds = torch.argmax(val_upsampled, dim=1)
            correct_pixels += (preds == val_masks).sum().item()
            total_pixels += val_masks.numel()
    
    # 분산 처리
    val_loss_tensor = torch.tensor(val_loss, device=device)
    correct_pixels_tensor = torch.tensor(correct_pixels, device=device)
    total_pixels_tensor = torch.tensor(total_pixels, device=device)
    
    if world_size > 1:
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_pixels_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_pixels_tensor, op=dist.ReduceOp.SUM)
    
    avg_val_loss = val_loss_tensor.item() / (len(valid_loader) * world_size)
    pixel_accuracy = correct_pixels_tensor.item() / total_pixels_tensor.item()
    
    return avg_val_loss, pixel_accuracy

def main(rank, world_size):
    """메인 학습 함수"""
    
    print(f"Starting process {rank}/{world_size}")
    
    # DDP 설정
    try:
        local_gpu = setup_ddp(rank, world_size)
        device = torch.device(f'cuda:{local_gpu}')
        print(f"Process {rank} using device: {device}")
    except Exception as e:
        print(f"DDP setup failed for rank {rank}: {e}")
        return
    
    # 데이터 경로 설정
    ROOT_DIRECTORY = "/home/work/data/indo_walking/polygon_segmentation"
    CLASS_MAPPING_FILE = os.path.join(ROOT_DIRECTORY, 'class_mapping.json')
    
    # 클래스 매핑 (하드코딩)
    class_to_idx = {
        'background': 0, 'barricade': 1, 'bench': 2, 'bicycle': 3, 'bollard': 4,
        'bus': 5, 'car': 6, 'carrier': 7, 'cat': 8, 'chair': 9, 'dog': 10,
        'fire_hydrant': 11, 'kiosk': 12, 'motorcycle': 13, 'movable_signage': 14,
        'parking_meter': 15, 'person': 16, 'pole': 17, 'potted_plant': 18,
        'power_controller': 19, 'scooter': 20, 'stop': 21, 'stroller': 22,
        'table': 23, 'traffic_light': 24, 'traffic_light_controller': 25,
        'traffic_sign': 26, 'tree_trunk': 27, 'truck': 28, 'wheelchair': 29
    }
    
    num_labels = len(class_to_idx)
    
    if rank == 0:
        print(f"클래스 수: {num_labels}")
    
    # 데이터셋 생성
    try:
        print(f"Rank {rank}: Loading datasets...")
        train_dataset = PolygonSegmentationDataset(root_dir=ROOT_DIRECTORY, is_train=True)
        valid_dataset = PolygonSegmentationDataset(root_dir=ROOT_DIRECTORY, is_train=False)
        
        if rank == 0:
            print(f"Train dataset: {len(train_dataset)}")
            print(f"Valid dataset: {len(valid_dataset)}")
            
    except Exception as e:
        print(f"Rank {rank}: Dataset loading failed: {e}")
        cleanup()
        return
    
    # 샘플러 생성
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank)
    
    # 데이터로더 생성
    batch_size = 64  # 메모리 절약을 위해 작은 배치 사이즈
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=2,  # 워커 수 줄임
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        sampler=valid_sampler,
        num_workers=2,
        pin_memory=True
    )
    
    if rank == 0:
        print(f"Train batches: {len(train_loader)}")
        print(f"Valid batches: {len(valid_loader)}")
    
    # 모델 생성
    try:
        print(f"Rank {rank}: Loading model...")
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        model.to(device)
        print(f"Rank {rank}: Model loaded successfully")
        
    except Exception as e:
        print(f"Rank {rank}: Model loading failed: {e}")
        cleanup()
        return
    
    # DDP 래핑
    model = DDP(model, device_ids=[local_gpu], output_device=local_gpu)
    
    # 하이퍼파라미터
    epochs = 50  # 테스트를 위해 에폭 수 줄임
    lr = 5e-5
    weight_decay = 1e-4
    
    # 옵티마이저 및 스케줄러
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    
    # WandB 초기화 (rank 0에서만)
    if rank == 0:
        wandb.init(project="segmentation_ddp", name="simple_ddp_test")
    
    print(f"Rank {rank}: Starting training...")
    
    # 학습 루프
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        
        model.train()
        running_loss = 0.0
        
        for i, data in enumerate(train_loader):
            try:
                images = data['pixel_values'].to(device)
                masks = data['labels'].to(device)
                
                inputs = {
                    'pixel_values': images,
                    'labels': masks
                }
                
                outputs = model(**inputs)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                
                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                running_loss += loss.item()
                
                if (i + 1) % 20 == 0 and rank == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                    
            except Exception as e:
                print(f"Rank {rank}: Training step error: {e}")
                continue
        
        # 검증
        try:
            val_loss, pixel_acc = validate_model(model, valid_loader, device, rank, world_size)
            
            scheduler.step()
            
            avg_train_loss = running_loss / len(train_loader)
            
            if rank == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Pixel Acc: {pixel_acc:.4f}")
                
                # WandB 로깅
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "pixel_accuracy": pixel_acc,
                    "learning_rate": scheduler.get_last_lr()[0]
                })
                
                # 최고 성능 모델 저장
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    os.makedirs("simple_ckpts", exist_ok=True)
                    torch.save(model.module.state_dict(), "simple_ckpts/best_model.pth")
                    print(f"Best model saved! Val Loss: {val_loss:.4f}")
                    
        except Exception as e:
            print(f"Rank {rank}: Validation error: {e}")
            continue
    
    if rank == 0:
        print("Training completed!")
        wandb.finish()
    
    cleanup()

if __name__ == "__main__":
    # 환경 변수 설정
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    
    world_size = 3
    
    # 멀티프로세싱으로 DDP 실행
    import torch.multiprocessing as mp
    
    print(f"Starting DDP training with {world_size} processes...")
    
    try:
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    except Exception as e:
        print(f"Training failed: {e}")
    finally:
        print("Training process finished.")