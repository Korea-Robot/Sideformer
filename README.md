

# SideFormer
## 보행로 세그멘테이션 모델

이 프로젝트는 `Segformer` 모델을 사용하여 주행가능영역과 객체 (도로,보도, 자전거 도로, 점자블록 등) 다양한 보행 환경 요소를 분할(Segmentation)하는 딥러닝 모델을 학습하고 최적화합니다. 분산 데이터 병렬 처리(DDP)를 활용하여 다중 GPU 환경에서 효율적으로 학습하며, `Optuna`를 이용해 최적의 하이퍼파라미터를 탐색합니다.

-----

## 🌟 주요 기능

  - **분산 학습 (DDP)**: `torch.nn.parallel.DistributedDataParallel`을 사용하여 여러 GPU에서 빠르고 효율적으로 모델을 학습합니다.
  - **하이퍼파라미터 최적화**: `Optuna` 라이브러리를 통해 최적의 학습률, 배치 크기, 옵티마이저 등을 자동으로 탐색합니다.
  - **유연한 학습 제어**: 커맨드 라인 인자를 통해 하이퍼파라미터 최적화를 건너뛰고, 저장된 최적의 설정으로 최종 학습을 바로 시작할 수 있습니다.
  - **학습 과정 모니터링**: `wandb` (Weights & Biases)와 연동하여 학습 손실, 검증 손실, 정확도 등의 지표를 실시간으로 시각화하고 추적합니다.
  - **조기 종료 (Early Stopping)**: 검증 손실이 일정 기간 개선되지 않으면 학습을 자동으로 중단하여 불필요한 시간과 리소스 낭비를 방지합니다.

-----

## 🛠️ 실행 환경 설정

### 1\. 필요 라이브러리 설치

이 스크립트를 실행하기 위해 필요한 주요 라이브러리는 다음과 같습니다. `requirements.txt` 파일을 통해 한번에 설치할 수 있습니다.

```bash
pip install torch transformers datasets wandb optuna numpy matplotlib opencv-python accelerate
```

### 2\. 데이터셋 준비

스크립트가 예상하는 데이터셋 구조는 다음과 같습니다. `metadata.json` 파일에는 학습 및 검증에 사용할 이미지와 마스크 파일의 경로 정보가 포함되어야 합니다.

```
/home/work/data/indo_walking/surface_masking/
├── processed_dataset/
│   ├── metadata.json
│   ├── train/
│   │   ├── images/
│   │   │   └── *.jpg
│   │   └── masks/
│   │       └── *.png
│   └── valid/
│       ├── images/
│       │   └── *.jpg
│       └── masks/
│           └── *.png
└── ...
```

  - **`metadata.json`**: 학습, 검증 데이터의 파일 경로 목록을 포함하는 JSON 파일입니다.
  - **데이터 경로**: 스크립트 내 `create_data_loaders` 함수에서 데이터셋의 기본 경로 (`/home/work/data/indo_walking/surface_masking/processed_dataset/metadata.json`)를 설정하고 있으므로, 실제 환경에 맞게 경로를 수정해야 합니다.

### 3\. GPU 환경 변수 설정

스크립트 상단에서 사용할 GPU 및 분산 통신 관련 환경 변수를 설정합니다.

```python
# 사용할 GPU 번호 지정 (예: 3, 4, 5번 GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"

# NCCL P2P 및 InfiniBand 통신 비활성화 (안정성 확보)
os.environ["NCCL_P2P_DISABLE"] = '1'
os.environ["NCCL_IB_DISABLE"] = '1'
```

  - `CUDA_VISIBLE_DEVICES`: 사용하고자 하는 GPU의 ID를 쉼표로 구분하여 지정합니다. 코드 내부에서는 이 GPU들이 0, 1, 2번으로 자동 매핑됩니다.
  - `world_size`: 사용할 총 GPU 수와 일치해야 합니다. (예: "3,4,5"로 설정 시 `world_size`는 3)

-----

## 🚀 사용 방법

스크립트는 두 가지 주요 모드로 실행할 수 있습니다.

1.  **하이퍼파라미터 최적화 후 최종 학습 진행 (기본값)**

      - `Optuna`를 사용하여 최적의 하이퍼파라미터를 찾고, 그 결과를 `best_hyperparams.json` 파일에 저장합니다.
      - 저장된 최적의 파라미터로 전체 데이터셋(학습+검증)을 사용하여 최종 모델을 학습합니다.
      - 최종 학습된 모델은 `final_ckpts/best_model.pth` 경로에 저장됩니다.

    <!-- end list -->

    ```bash
    python train_ddp_hyperopt_fixed.py
    ```

2.  **하이퍼파라미터 최적화 건너뛰고 최종 학습만 진행**

      - `--skip-hyperopt` 플래그를 사용하여 최적화 단계를 건너뛸 수 있습니다.
      - 이 경우, `--hyperparams-file` 인자로 지정된 경로의 JSON 파일을 읽어와 최종 학습을 진행합니다. (기본값: `best_hyperparams.json`)
      - 기존에 찾아둔 최적의 파라미터가 있다면 이 방법을 사용하여 학습을 재현할 수 있습니다.

    <!-- end list -->

    ```bash
    # best_hyperparams.json 파일을 사용하여 최종 학습 실행
    python train_ddp_hyperopt_fixed.py --skip-hyperopt

    # 다른 이름의 하이퍼파라미터 파일을 사용하려면
    python train_ddp_hyperopt_fixed.py --skip-hyperopt --hyperparams-file my_best_params.json
    ```

-----

## 📜 코드 주요 구성 요소

  - **`main()`**: 프로그램의 진입점으로, 커맨드 라인 인자를 파싱하여 하이퍼파라미터 최적화 또는 최종 학습을 제어합니다.

  - **`run_hyperparameter_optimization()`**: `Optuna` 스터디를 생성하고 `objective` 함수를 실행하여 최적의 하이퍼파라미터 조합을 탐색합니다.

  - **`objective(trial)`**: `Optuna`의 각 `trial`(시도)에 대해 하이퍼파라미터 값을 샘플링하고, `hyperopt_wrapper`를 호출하여 모델 학습 및 검증 손실을 반환합니다.

  - **`run_final_training(best_hyperparams)`**: 최적화된 하이퍼파라미터를 입력받아 `final_train_wrapper`를 호출하고 최종 모델을 학습합니다.

  - **`train_model_with_hyperparams(...)`**: 실제 모델 학습, 검증, 로깅, 조기 종료 등 핵심 로직을 수행하는 함수입니다. `is_hyperopt` 플래그를 통해 최적화용 학습(일부 데이터만 사용)과 최종 학습(전체 데이터 사용)을 구분합니다.

  - **`hyperopt_wrapper`, `final_train_wrapper`**: `multiprocessing`을 사용하여 각 GPU에서 병렬로 학습 프로세스를 실행하기 위한 래퍼(wrapper) 함수입니다.

  - **`setup_ddp(rank, world_size)`**: 분산 학습 환경을 초기화합니다. 각 프로세스(GPU)의 순위(`rank`)와 총 프로세스 수(`world_size`)를 기반으로 통신 그룹을 설정합니다.

  - **`create_data_loaders(...)`**: 학습 및 검증용 데이터셋과 `DataLoader`를 생성합니다. `DistributedSampler`를 사용하여 각 GPU에 데이터를 분배합니다.

-----

## 📁 결과물

  - **`best_hyperparams.json`**: 하이퍼파라미터 최적화 후 가장 성능이 좋았던 파라미터 조합이 저장되는 JSON 파일입니다.
  - **`final_ckpts/best_model.pth`**: 최종 학습 완료 후 가장 우수한 검증 성능을 보인 모델의 가중치(state\_dict)가 저장되는 파일입니다.
  - **`temp_ckpts/`**: 학습 중간에 생성되는 임시 체크포인트 파일들이 저장되는 디렉토리이며, 스크립트 종료 시 자동으로 삭제됩니다.