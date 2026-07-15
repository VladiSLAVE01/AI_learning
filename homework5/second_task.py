import random
from augs import RandomBlur, RandomSharpness, RandomContrast
from augmentations_basics.datasets import CustomImageDataset
from torchvision import transforms
from augmentations_basics.utils import show_multiple_augmentations

if __name__ == '__main__':
    root = 'data/train'

    dataset = CustomImageDataset(root)

    augmentations = transforms.Compose([
        RandomBlur(blur_radius=5, p=1),
        RandomSharpness(sharpness_factor=2.0, p=1),
        RandomContrast(contrast_factor=1.5, p=1),
    ])

    for i in range(5):
        idx = random.randint(0, len(dataset) - 1)
        original_img, label = dataset[idx]
        class_names = dataset.get_class_names()
        print(f"Оригинальное изображение, класс: {class_names[label]}")

        augmented_imgs = []
        titles = []

        for aug in augmentations.transforms:
            augmentation = transforms.Compose([
                aug,
                transforms.ToTensor()
            ])
            augmented_img = augmentation(original_img)
            augmented_imgs.append(augmented_img)
            titles.append(augmentation.transforms[0].__class__.__name__)

        show_multiple_augmentations(original_img, augmented_imgs, titles)
