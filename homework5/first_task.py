import random
from torchvision import transforms
from augmentations_basics.datasets import CustomImageDataset
from augmentations_basics.utils import show_multiple_augmentations

if __name__ == "__main__":
    root = 'data/train'

    dataset = CustomImageDataset(root, transform=transforms.ToTensor())

    # Инициализация аугментаций
    augmentations_dict = transforms.Compose({
        'RandomHorizontalFlip': transforms.RandomHorizontalFlip(p=1),
        'RandomCrop': transforms.RandomCrop(200, padding=10),
        'ColorJitter': transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        'RandomRotation': transforms.RandomRotation(30),
        'RandomGrayscale': transforms.RandomGrayscale(p=1),
    })

    # Получение списка аугментаций
    augmentations = transforms.Compose(augmentations_dict.transforms.values())

    for i in range(5):
        idx = random.randint(0, len(dataset) - 1)
        original_img, label = dataset[idx]
        class_names = dataset.get_class_names()
        print(f"Оригинальное изображение, класс: {class_names[label]}")

        augmented_imgs = []
        titles = []

        for (name, augmentation) in augmentations_dict.transforms.items():
            augmented_img = augmentation(original_img)
            augmented_imgs.append(augmented_img)
            titles.append(name)

        augmented_img = augmentations(original_img)

        # Добавление последнего аугментированного изображения
        augmented_imgs.append(augmented_img)
        titles.append("Все аугментации")

        show_multiple_augmentations(original_img, augmented_imgs, titles)
