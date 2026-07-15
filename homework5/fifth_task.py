import random
import tracemalloc
import time
import matplotlib.pyplot as plt
from augmentations_basics.datasets import CustomImageDataset
from augs import AugmentationPipeline
from torchvision import transforms


def show_size_dependent(sizes, time_list, memory_usage_list):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(sizes, time_list, marker='o')
    plt.title('Время выполнения аугментаций')
    plt.xlabel('Размер изображения (px)')
    plt.ylabel('Время (сек)')

    plt.subplot(1, 2, 2)
    plt.plot(sizes, memory_usage_list, marker='o', color='orange')
    plt.title('Пиковое использование памяти')
    plt.xlabel('Размер изображения (px)')
    plt.ylabel('Память (МБ)')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    root = 'data/train'

    sizes = [64, 128, 224, 512]

    dataset64x64 = CustomImageDataset(root, target_size=(sizes[0], sizes[0]))
    dataset128x128 = CustomImageDataset(root, target_size=(sizes[1], sizes[1]))
    dataset224x224 = CustomImageDataset(root, target_size=(sizes[2], sizes[2]))
    dataset512x512 = CustomImageDataset(root, target_size=(sizes[3], sizes[3]))

    augmentations = AugmentationPipeline({
        'RandomHorizontalFlip': transforms.RandomHorizontalFlip(p=0.5),
        'RandomRotation': transforms.RandomRotation(degrees=15),
        'ColorJitter': transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        'GaussianBlur': transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
    })

    time_list = []
    memory_usage_list = []

    for dataset in [dataset64x64, dataset128x128, dataset224x224, dataset512x512]:
        tracemalloc.start()
        start_time = time.time()

        for i in range(100):
            idx = random.randint(0, len(dataset) - 1)
            original_img, label = dataset[idx]
            class_names = dataset.get_class_names()

            _ = augmentations.apply(original_img)

        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        time_list.append(end_time - start_time)
        memory_usage_list.append(peak / 1024 / 1024)

    show_size_dependent(sizes, time_list, memory_usage_list)
