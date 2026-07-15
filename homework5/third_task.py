from augmentations_basics.datasets import CustomImageDataset
import numpy as np

if __name__ == '__main__':
    root_dir = 'data/train'

    dataset = CustomImageDataset(root_dir)

    for class_name, class_idx in dataset.class_to_idx.items():
        print(f"Класс: {class_name}")
        print(f"Количество изображений: {dataset.labels.count(class_idx)}")

    sizes = [dataset[i][0].size for i in range(len(dataset))]

    width, height = zip(*sizes)

    print(f"Минимальная ширина: {min(width)}, высота: {min(height)}")
    print(f"Максимальная ширина: {max(width)}, высота: {max(height)}")
    print(f"Средняя ширина: {np.mean(width):.1f}, высота: {np.mean(height):.1f}")
