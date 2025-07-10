import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


# load Mask2Former fine-tuned on Cityscapes semantic segmentation

# swin - tiny 
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-cityscapes-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-cityscapes-semantic")

# # swin - large
# processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
# model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")


# panoptic swin - large
model_name  = "facebook/mask2former-swin-large-cityscapes-panoptic"
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-panoptic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-panoptic")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

import time

inference_times = []


for i in range(1,7):
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
    
    # model predicts class_queries_logits of shape `(batch_size, num_queries)`
    # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits

    # you can pass them to processor for postprocessing
    predicted_semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    # we refer to the demo notebooks for visualization (see "Resources" section in the Mask2Former docs)


    # breakpoint()
    import numpy as np
    from PIL import Image

    # Cityscapes color palette (35 classes, RGB, values in 0-255)
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

    # Convert predicted_semantic_map (PyTorch tensor) to numpy array
    mask = predicted_semantic_map.cpu().numpy().astype(np.uint8)

    # Map each class to its RGB color
    color_mask = cityscapes_palette[mask]

    # Convert PIL image to numpy array
    image_np = np.array(image)

    # Blend the original image and the color mask
    alpha = 0.5
    blended = (image_np * (1 - alpha) + color_mask * alpha).astype(np.uint8)

    # Convert back to PIL and save
    blended_img = Image.fromarray(blended)
    blended_img.save(f'outputs/maskforemr_{i}.png')

print(f"\nAverage inference time: {np.mean(inference_times):.4f} seconds")
