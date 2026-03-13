from __future__ import annotations

from typing import Optional, Tuple

import torchvision.models as tv_models
from torchvision.models import get_model_weights
import torchvision.transforms as T


_DATASET_NUM_CLASSES = {
    "imagenet": 1000,
    "imagenet1k": 1000,
    "cifar10": 10,
    "cifar100": 100,
    "mnist": 10,
    "fashionmnist": 10,
    "svhn": 10,
    "caltech101": 101,
    "caltech256": 256,
}


def split_transform_from_weights(weights):

    resize = weights.transforms().resize_size
    crop = weights.transforms().crop_size
    mean = weights.transforms().mean
    std = weights.transforms().std

    spatial = T.Compose([
        T.Resize(resize),
        T.CenterCrop(crop),
        T.ToTensor()
    ])

    normalize = T.Normalize(mean=mean, std=std)

    return spatial, normalize

def get_torchvision_model(
    model_name,
    dataset_name=None,
    pretrained=True,
    num_classes=None,
):

    if not hasattr(tv_models, model_name):
        raise ValueError(f"Unknown model {model_name}")

    model_fn = getattr(tv_models, model_name)

    if pretrained:
        weights_enum = get_model_weights(model_name).DEFAULT
        model = model_fn(weights=weights_enum)

        spatial, normalize = split_transform_from_weights(weights_enum)

        return model, spatial, normalize

    kwargs = {}
    if num_classes is not None:
        kwargs["num_classes"] = num_classes

    model = model_fn(weights=None, **kwargs)

    return model, None, None