# train_ddp_hyperopt_fixed.py - 수정된 하이퍼파라미터 최적화 + DDP 학습
import os 
import os

# CUDA가 GPU를 인식하는 순서를 PCI 버스 ID 순서로 설정
# 시스템에 여러 개의 GPU가 장착된 경우, 운영체제나 드라이버가 GPU를 인식하는 순서는 매번 달라짐
# 이 설정을 통해 GPU의 물리적 위치(PCI 버스)를 기준으로 순서를 고정하여,
# 'CUDA_VISIBLE_DEVICES'에서 특정 GPU를 일관되게 선택할 수 있도록 함
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# 프로그램이 사용할 수 있는 GPU를 제한합니다.
# 이렇게 설정하면, 해당 파이썬 스크립트는 시스템에 설치된 다른 GPU는 인식하지 못하고
# 오직 3, 4, 5번 GPU만 접근
# 코드 내에서는 이 GPU들이 각각 0번, 1번, 2번으로 매핑되어 사용
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"

# NCCL의 P2P(Peer-to-Peer) 통신을 비활성화합니다.
# P2P는 GPU 간 직접 데이터 전송을 가능하게 하여 분산 훈련 성능을 높여주지만,
# 하드웨어나 드라이버 호환성 문제로 오류가 발생할 수 있습니다.
# 이 설정을 '1'로 지정하면 P2P 기능을 끄고, 대신 PCIe를 통한 데이터 전송을 강제하여 안정성을 확보
os.environ["NCCL_P2P_DISABLE"] = '1'

# NCCL의 InfiniBand (IB) 통신을 비활성화
# InfiniBand는 고속 네트워킹 기술로, 다중 노드(서버) 환경에서 GPU 간 통신 속도를 높이기 위해 사용
# IB 관련 하드웨어나 설정에 문제가 있을 경우, 이 옵션을 '1'로 설정하여 IB 사용을 중지하고
# 일반적인 TCP/IP 소켓 통신으로 대체할 수 있습니다. 이는 통신 속도가 느려질 수 있지만 안정성을 높임
os.environ["NCCL_IB_DISABLE"] = '1'


import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import SegformerForSemanticSegmentation
import wandb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2
import pickle
import sys
import traceback
from typing import Dict, Any, List
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import time
import signal
import shutil
import torch.multiprocessing as mp

# 글로벌 변수로 클래스 매핑 정의
CLASS_TO_IDX = {
    'background': 0, 'caution_zone': 1, 'bike_lane': 2, 'alley': 3,
    'roadway': 4, 'braille_guide_blocks': 5, 'sidewalk': 6
}

def custom_collate_fn(batch):
    """pickle 에러 방지를 위한 글로벌 collate 함수"""
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

def init_worker(worker_id):
    """워커 초기화 함수"""
    np.random.seed(torch.initial_seed() % 2**32)

# 메인 프로세스에서만 import
try:
    from surface_dataset import SurfaceSegmentationDataset
except ImportError as e:
    print(f"surface_dataset import 실패: {e}")
    sys.exit(1)

def setup_ddp(rank, world_size):
    """DDP 환경 설정
    DDP (Distributed Data Parallel) 환경을 설정하는 함수입니다.
    여러 GPU에서 모델을 병렬로 학습시키기 위한 초기화 작업을 수행합니다.

    Args:
        rank (int): 현재 프로세스의 고유 순위(ID). 0번이 마스터 프로세스입니다.
        world_size (int): 학습에 참여하는 총 프로세스(GPU)의 수.
    """
    # 1. 분산 통신을 위한 마스터 프로세스의 주소와 포트를 환경 변수로 설정
    # 모든 프로세스가 동일한 주소와 포트를 바라보게 하여 서로를 찾음
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12359'
    
    local_gpu = rank
    torch.cuda.set_device(local_gpu)
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=torch.distributed.default_pg_timeout * 2
    )
    
    return local_gpu

def cleanup():
    """DDP 환경 정리
    학습이 완료된 후, 할당된 리소스를 해제하고 프로세스 그룹을 파괴
    """
    if dist.is_initialized():
        dist.destroy_process_group()

def create_data_loaders(rank, world_size, batch_size=32, include_valid_in_train=False):
    """데이터 로더 생성 함수"""
    metadata_path = "/home/work/data/indo_walking/surface_masking/processed_dataset/metadata.json"
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    data_base_path = os.path.dirname(os.path.dirname(metadata_path))
    target_size = (512, 512)
    
    # 하이퍼파라미터 최적화 시에는 train/valid 분리
    # 최종 학습 시에는 train + valid 합쳐서 사용
    if include_valid_in_train:
        # 전체 데이터를 train으로 사용
        all_data = metadata['train_data'] + metadata['valid_data']
        train_dataset = SurfaceSegmentationDataset(
            all_data, target_size, is_train=True, data_base_path=data_base_path
        )
        valid_dataset = None
    else:
        # 하이퍼파라미터 최적화용 - train/valid 분리
        train_dataset = SurfaceSegmentationDataset(
            metadata['train_data'], target_size, is_train=True, data_base_path=data_base_path
        )
        valid_dataset = SurfaceSegmentationDataset(
            metadata['valid_data'], target_size, is_train=False, data_base_path=data_base_path
        )

    # 샘플러 생성
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank,
        shuffle=True, drop_last=True
    )
    
    valid_sampler = None
    if valid_dataset is not None:
        valid_sampler = DistributedSampler(
            valid_dataset, num_replicas=world_size, rank=rank,
            shuffle=False, drop_last=False
        )

    num_workers = min(4, os.cpu_count() // world_size)
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=custom_collate_fn, persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else 2,
        worker_init_fn=init_worker if num_workers > 0 else None
    )
    
    valid_loader = None
    if valid_dataset is not None:
        valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=True, drop_last=False,
            collate_fn=custom_collate_fn, persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else 2,
            worker_init_fn=init_worker if num_workers > 0 else None
        )
    
    return train_loader, valid_loader, train_sampler, valid_sampler

def validate_model(model, valid_loader, device):
    """검증 함수"""
    if valid_loader is None:
        return float('inf'), 0.0
        
    model.eval()
    val_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    
    with torch.no_grad():
        for val_data in valid_loader:
            val_imgs = val_data['pixel_values'].to(device, non_blocking=True)
            val_masks = val_data['labels'].to(device, non_blocking=True)
            
            val_inputs = {
                'pixel_values': val_imgs,
                'labels': val_masks
            }
            val_outputs = model(**val_inputs)
            
            loss = val_outputs.loss
            val_loss += loss.item()
            
            # 정확도 계산
            val_logits = val_outputs.logits
            val_upsampled = F.interpolate(
                val_logits, size=val_masks.shape[-2:],
                mode='bilinear', align_corners=False
            )
            
            preds = torch.argmax(val_upsampled, dim=1)
            correct_pixels += (preds == val_masks).sum().item()
            total_pixels += val_masks.numel()
    
    avg_val_loss = val_loss / len(valid_loader)
    pixel_accuracy = correct_pixels / total_pixels
    
    return avg_val_loss, pixel_accuracy

def train_model_with_hyperparams(rank, world_size, hyperparams, max_epochs=30, is_hyperopt=True):
    """하이퍼파라미터로 모델 학습"""
    try:
        local_gpu = setup_ddp(rank, world_size)
        device = torch.device(f'cuda:{local_gpu}')
        
        # 클래스 설정
        class_to_idx = CLASS_TO_IDX
        id2label = {int(idx): label for label, idx in class_to_idx.items()}
        label2id = class_to_idx
        num_labels = len(id2label)
        
        # 데이터 로더 생성
        include_valid_in_train = not is_hyperopt  # 하이퍼파라미터 최적화가 아닐 때만 valid 포함
        train_loader, valid_loader, train_sampler, valid_sampler = create_data_loaders(
            rank, world_size, hyperparams['batch_size'], include_valid_in_train
        )
        
        # 모델 생성
        model = SegformerForSemanticSegmentation.from_pretrained(
            hyperparams['model_name'],
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            id2label=id2label,
            label2id=label2id
        )
        model.to(device)
        
        # DDP 래핑
        model = DDP(
            model, device_ids=[local_gpu], output_device=local_gpu,
            find_unused_parameters=False, broadcast_buffers=False,
            gradient_as_bucket_view=True
        )
        
        # 옵티마이저 설정
        if hyperparams['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=hyperparams['learning_rate'], 
                weight_decay=hyperparams['weight_decay']
            )
        else:  # sgd
            optimizer = optim.SGD(
                model.parameters(), 
                lr=hyperparams['learning_rate'], 
                weight_decay=hyperparams['weight_decay'],
                momentum=0.9
            )
        
        # 스케줄러 설정
        if hyperparams['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        elif hyperparams['scheduler'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        else:  # none
            scheduler = None
        
        # WandB 초기화 (rank 0에서만)
        if rank == 0:
            run_name = f"hyperopt_{int(time.time())}" if is_hyperopt else f"final_training_{int(time.time())}"
            wandb.init(
                project="segmentation_hyperopt",
                name=run_name,
                config=hyperparams
            )
        
        # 학습 루프
        best_val_loss = float('inf')
        patience = 10 if is_hyperopt else 20
        patience_counter = 0
        
        for epoch in range(max_epochs):
            train_sampler.set_epoch(epoch)
            
            model.train()
            running_loss = 0.0
            num_batches = 0
            
            for i, data in enumerate(train_loader):
                try:
                    images = data['pixel_values'].to(device, non_blocking=True)
                    masks = data['labels'].to(device, non_blocking=True)
                    
                    inputs = {
                        'pixel_values': images,
                        'labels': masks
                    }
                    
                    outputs = model(**inputs)
                    loss = outputs.loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # 그래디언트 클리핑
                    if hyperparams.get('grad_clip', 0) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=hyperparams['grad_clip'])
                    
                    optimizer.step()
                    
                    running_loss += loss.item()
                    num_batches += 1
                    
                    # 하이퍼파라미터 최적화 시 빠른 피드백을 위해 배치별 조기 종료
                    if is_hyperopt and i > 200:  # 200 배치만 학습
                        break
                
                except Exception as e:
                    print(f"Rank {rank}, Batch {i}: 배치 처리 중 에러 - {e}")
                    continue
            
            # 검증
            val_loss, pixel_acc = validate_model(model, valid_loader, device)
            
            # 결과 동기화
            if valid_loader is not None:
                val_loss_tensor = torch.tensor(val_loss, device=device)
                pixel_acc_tensor = torch.tensor(pixel_acc, device=device)
                
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(pixel_acc_tensor, op=dist.ReduceOp.SUM)
                
                val_loss = val_loss_tensor.item() / world_size
                pixel_acc = pixel_acc_tensor.item() / world_size
            
            if scheduler:
                scheduler.step()
            
            # 평균 훈련 손실 계산
            if num_batches > 0:
                avg_train_loss = running_loss / num_batches
                train_loss_tensor = torch.tensor(avg_train_loss, device=device)
                dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
                avg_train_loss = train_loss_tensor.item() / world_size
            else:
                avg_train_loss = float('inf')
            
            # 로깅 (rank 0에서만)
            if rank == 0:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "pixel_accuracy": pixel_acc,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
                
                print(f"Epoch [{epoch+1}/{max_epochs}] - "
                      f"Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Pixel Acc: {pixel_acc:.4f}")
            
            # 조기 종료 체크
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # 모델 저장 (rank 0에서만)
                if rank == 0:
                    os.makedirs("temp_ckpts", exist_ok=True)
                    if is_hyperopt:
                        torch.save(model.module.state_dict(), "temp_ckpts/best_hyperopt_model.pth")
                    else:
                        torch.save(model.module.state_dict(), "temp_ckpts/best_final_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if rank == 0:
                        print(f"조기 종료: {patience} 에폭 동안 개선 없음")
                    break
        
        if rank == 0:
            wandb.finish()
        
        return best_val_loss
        
    except Exception as e:
        print(f"Rank {rank}에서 에러 발생: {e}")
        traceback.print_exc()
        return float('inf')
    
    finally:
        cleanup()

# 전역 함수로 분리하여 pickle 가능하게 만듦
def hyperopt_wrapper(rank, world_size, hyperparams, result_dict):
    """하이퍼파라미터 최적화용 래퍼 함수"""
    try:
        val_loss = train_model_with_hyperparams(
            rank, world_size, hyperparams, max_epochs=20, is_hyperopt=True
        )
        if rank == 0:
            result_dict['val_loss'] = val_loss
    except Exception as e:
        print(f"Hyperopt wrapper error in rank {rank}: {e}")
        if rank == 0:
            result_dict['val_loss'] = float('inf')

def final_train_wrapper(rank, world_size, hyperparams):
    """최종 학습용 래퍼 함수"""
    try:
        val_loss = train_model_with_hyperparams(
            rank, world_size, hyperparams, 
            max_epochs=100, is_hyperopt=False
        )
        if rank == 0:
            print(f"최종 학습 완료! 최종 검증 손실: {val_loss:.4f}")
    except Exception as e:
        print(f"최종 학습 중 에러 발생 (rank {rank}): {e}")
        traceback.print_exc()

def objective(trial):
    """Optuna 최적화 목적 함수"""
    # 하이퍼파라미터 샘플링
    hyperparams = {
        'batch_size': trial.suggest_categorical('batch_size', [16, 32,64]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-4, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        'model_name': trial.suggest_categorical('model_name', [
            'nvidia/mit-b0'# , 'nvidia/mit-b1', 'nvidia/mit-b2'
        ]),
        'optimizer': trial.suggest_categorical('optimizer', ['adamw', 'sgd']),
        'scheduler': trial.suggest_categorical('scheduler', ['cosine', 'step', 'none']),
        'grad_clip': trial.suggest_float('grad_clip', 0.5, 2.0),
    }
    
    print(f"Trial {trial.number}: {hyperparams}")
    
    world_size = 3
    
    try:
        # 공유 메모리를 통한 결과 전달
        manager = mp.Manager()
        result_dict = manager.dict()
        
        # 프로세스 실행
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=hyperopt_wrapper, args=(rank, world_size, hyperparams, result_dict))
            p.start()
            processes.append(p)
        
        # 모든 프로세스 완료 대기
        for p in processes:
            p.join()
        
        # 결과 확인
        val_loss = result_dict.get('val_loss', float('inf'))
        
        print(f"Trial {trial.number} 완료: Val Loss = {val_loss:.4f}")
        
        return val_loss
        
    except Exception as e:
        print(f"Trial {trial.number} 실패: {e}")
        return float('inf')

def run_hyperparameter_optimization():
    """하이퍼파라미터 최적화 실행"""
    print("하이퍼파라미터 최적화 시작...")
    
    # Optuna 스터디 생성
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # 최적화 실행
    n_trials = 20  # 시행 횟수
    study.optimize(objective, n_trials=n_trials, timeout=144000)  # 2시간 제한
    
    # 최적 하이퍼파라미터 출력
    print("\n=== 하이퍼파라미터 최적화 결과 ===")
    print(f"최적 검증 손실: {study.best_value:.4f}")
    print(f"최적 하이퍼파라미터: {study.best_params}")
    
    # 결과 저장
    with open('best_hyperparams.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    return study.best_params

def run_final_training(best_hyperparams):
    """최적 하이퍼파라미터로 최종 학습"""
    print("\n=== 최종 학습 시작 ===")
    print(f"사용할 하이퍼파라미터: {best_hyperparams}")
    
    world_size = 3
    
    # 최종 학습 실행
    try:
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=final_train_wrapper, args=(rank, world_size, best_hyperparams))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        # 최종 모델을 메인 디렉토리로 이동
        if os.path.exists("temp_ckpts/best_final_model.pth"):
            os.makedirs("final_ckpts", exist_ok=True)
            shutil.move("temp_ckpts/best_final_model.pth", "final_ckpts/best_model.pth")
            print("최종 모델이 final_ckpts/best_model.pth에 저장되었습니다.")
        
        print("최종 학습 완료!")
        
    except Exception as e:
        print(f"최종 학습 실행 중 에러: {e}")
        traceback.print_exc()

def main():
    """메인 함수"""
    # multiprocessing 설정
    mp.set_start_method('spawn', force=True)
    mp.set_sharing_strategy('file_system')
    
    parser = argparse.ArgumentParser(description='하이퍼파라미터 최적화 및 학습')
    parser.add_argument('--skip-hyperopt', action='store_true', 
                       help='하이퍼파라미터 최적화 건너뛰기')
    parser.add_argument('--hyperparams-file', type=str, default='best_hyperparams.json',
                       help='하이퍼파라미터 파일 경로')
    
    args = parser.parse_args()
    
    # 임시 디렉토리 생성
    os.makedirs("temp_ckpts", exist_ok=True)
    
    try:
        if args.skip_hyperopt:
            # 하이퍼파라미터 파일 로드
            if os.path.exists(args.hyperparams_file):
                with open(args.hyperparams_file, 'r') as f:
                    best_hyperparams = json.load(f)
                print(f"기존 하이퍼파라미터 로드: {best_hyperparams}")
            else:
                print("하이퍼파라미터 파일이 없습니다. 기본값을 사용합니다.")
                best_hyperparams = {
                    'batch_size': 64,
                    'learning_rate': 5e-5,
                    'weight_decay': 1e-4,
                    'model_name': 'nvidia/mit-b0',
                    'optimizer': 'adamw',
                    'scheduler': 'cosine',
                    'grad_clip': 1.0
                }
        else:
            # 하이퍼파라미터 최적화 실행
            best_hyperparams = run_hyperparameter_optimization()
        
        # 최종 학습 실행
        run_final_training(best_hyperparams)
        
    except KeyboardInterrupt:
        print("사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"메인 실행 중 에러: {e}")
        traceback.print_exc()
    finally:
        # 임시 파일 정리
        if os.path.exists("temp_ckpts"):
            shutil.rmtree("temp_ckpts")

if __name__ == "__main__":
    main()