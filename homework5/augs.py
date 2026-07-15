import random
from PIL import ImageFilter, ImageEnhance
from typing import Callable


class RandomBlur:
    def __init__(self, blur_radius=5, p=0.5):
        self.blur_radius = blur_radius
        self.p = p

    def __call__(self, image):
        if random.random() > self.p:
            return image

        radius = random.uniform(0, self.blur_radius)
        return image.filter(ImageFilter.GaussianBlur(radius))


class RandomSharpness:
    def __init__(self, sharpness_factor=2.0, p=0.5):
        self.sharpness_factor = sharpness_factor
        self.p = p

    def __call__(self, image):
        if random.random() > self.p:
            return image

        enhancer = ImageFilter.UnsharpMask(radius=2, percent=int(self.sharpness_factor * 100))
        return image.filter(enhancer)


class RandomContrast:
    def __init__(self, contrast_factor=1.0, p=0.5):
        self.contrast_factor = contrast_factor
        self.p = p

    def __call__(self, image):
        if random.random() > self.p:
            return image

        factor = random.uniform(0, self.contrast_factor)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)


class AugmentationPipeline:
    def __init__(self, augmentations: dict[str, Callable]):
        self.augmentations = augmentations

    def add_augmentation(self, name: str, augmentation: Callable):
        self.augmentations[name] = augmentation

    def remove_augmentation(self, name: str):
        if name in self.augmentations:
            del self.augmentations[name]

    def apply(self, image):
        for name, augmentation in self.augmentations.items():
            image = augmentation(image)
        return image

    def get_augmentations(self):
        return self.augmentations
