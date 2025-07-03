from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from transformers import SegformerFeatureExtractor
import torch 
import numpy as np

from PIL import Image
import requests

#  b4 1024 - 1024  
# processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b4-finetuned-cityscapes-1024-1024")
# model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-cityscapes-1024-1024")

#  b0 640 - 1280 conda 
processor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-640-1280")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-640-1280")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

import time

inference_times = []

for i in range(1, 7):
    image_path = f"images/{i}.jpg"
    image = Image.open(image_path)

    inputs = processor(images=image, return_tensors="pt").to(device)

    # Inference time 측정 시작
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()

    with torch.no_grad():
        outputs = model(**inputs)

    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    inference_time = end_time - start_time
    inference_times.append(inference_time)
    print(f"[{i}] Inference time: {inference_time:.4f} seconds")

    # 후처리 및 저장(생략, 기존 코드와 동일)
    predicted_semantic_map = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]
    cityscapes_palette = np.array([
        [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
        [20, 20, 20], [111, 74, 0], [81, 0, 81], [128, 64, 128],
        [244, 35, 232], [250, 170, 160], [230, 150, 140], [70, 70, 70],
        [102, 102, 156], [190, 153, 153], [180, 165, 180], [150, 100, 100],
        [150, 120, 90], [153, 153, 153], [153, 153, 153], [250, 170, 30],
        [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180],
        [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
        [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32],
        [0, 0, 142], [0, 0, 142], [0, 0, 142], [0, 0, 142]
    ], dtype=np.uint8)

    mask = predicted_semantic_map.cpu().numpy().astype(np.uint8)
    color_mask = cityscapes_palette[mask]
    image_np = np.array(image)
    alpha = 0.5
    blended = (image_np * (1 - alpha) + color_mask * alpha).astype(np.uint8)
    blended_img = Image.fromarray(blended)
    blended_img.save(f'outputs/segformer_overlay_{i}.png')

# 전체 평균 추론 시간 출력
print(f"\nAverage inference time: {np.mean(inference_times):.4f} seconds")

