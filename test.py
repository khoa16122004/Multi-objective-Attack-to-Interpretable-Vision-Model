from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image
from util import get_torchvision_model
from explain_method import simple_gradient_map, integrated_gradients, get_gradcam_map
from torchvision.utils import save_image
import matplotlib.pyplot as plt

model, sptransform, normtransform = get_torchvision_model(
    "resnet50", dataset_name="imagenet", pretrained=True
)
model.eval()

img = Image.open("test_imgs/dog_human.png").convert("RGB")

img_tensor_oris = sptransform(img).unsqueeze(0).repeat(5, 1, 1, 1)
if normtransform:
    img_tensor = torch.stack([normtransform(img_tensor_ori) for img_tensor_ori in img_tensor_oris])
else:
    img_tensor = img_tensor_ori


# saliency_map, model_outputs  = simple_gradient_map(model, img_tensor)
# saliency_map, model_outputs = integrated_gradients(model, img_tensor)
saliency_map, model_outputs = get_gradcam_map(model, 'resnet50', img_tensor)
print(saliency_map.shape, model_outputs.shape)

sal = saliency_map[0]
print(sal.shape)
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img_tensor_oris[0].squeeze().permute(1, 2, 0).cpu())
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(sal, cmap="hot")
plt.title("Saliency")
plt.axis("off")

plt.tight_layout()
plt.savefig("gradcam_saliency_side_by_side.png", bbox_inches="tight", pad_inches=0)


# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
# import numpy as np
# from PIL import Image
# from util import get_torchvision_model
# import matplotlib.pyplot as plt


# def vit_reshape_transform(tensor, height=7, width=7):
#     tensor = tensor[:, 1:, :]
#     B, N, C = tensor.shape
#     tensor = tensor.reshape(B, height, width, C)
#     tensor = tensor.permute(0, 3, 1, 2)
#     return tensor


# model, sptransform, normtransform = get_torchvision_model(
#     "vit_b_32", dataset_name="imagenet", pretrained=True
# )

# # ViT target layer
# target_layers = [model.encoder.layers[-1].ln_1]

# img = Image.open("test_imgs/dog_human.png").convert("RGB")

# img_tensor_ori = sptransform(img).unsqueeze(0)

# if normtransform:
#     img_tensor = normtransform(img_tensor_ori)
# else:
#     img_tensor = img_tensor_ori

# targets = None

# rgb_img = img_tensor_ori.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.float32)
# rgb_img = np.clip(rgb_img, 0.0, 1.0)

# with GradCAM(
#         model=model,
#         target_layers=target_layers,
#         reshape_transform=vit_reshape_transform
# ) as cam:

#     grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
#     grayscale_cam = grayscale_cam[0]

#     visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

#     model_outputs = cam.outputs
#     print(visualization.shape)
#     print(model_outputs.argmax(dim=1))

# plt.imshow(visualization)
# plt.axis("off")
# plt.show()


