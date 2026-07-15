import random
from augmentations_basics.datasets import CustomImageDataset
from augmentations_basics.utils import show_multiple_augmentations
from augs import AugmentationPipeline, RandomContrast
from torchvision import transforms

if __name__ == '__main__':
    root = 'data/train'

    dataset = CustomImageDataset(root)

    light_conf = AugmentationPipeline({
        'RandomHorizontalFlip': transforms.RandomHorizontalFlip()
    })

    medium_conf = AugmentationPipeline({
        **light_conf.get_augmentations(),
        'RandomRotation': transforms.RandomRotation(degrees=15),
        'RandomContrast': RandomContrast(p=1.0),
    })

    heavy_conf = AugmentationPipeline({
        **medium_conf.get_augmentations(),
        'RandomPerspective': transforms.RandomPerspective(p=1),
        'GaussianBlur': transforms.GaussianBlur(5),
    })

    image, label = random.choice(dataset)

    configurations = {
        'Light Configuration': light_conf,
        'Medium Configuration': medium_conf,
        'Heavy Configuration': heavy_conf
    }

    results = []

    for name, config in configurations.items():
        to_tensor = transforms.ToTensor()
        augmented_image = config.apply(image)

        results.append(to_tensor(augmented_image))

    show_multiple_augmentations(image, results, configurations.keys())
