# Подготовка датасета для урока 5

## Скачивание датасета

1. Перейдите по ссылке: [https://drive.google.com/drive/folders/1BzUQeJgICv2ZDXA9HEWcfiAXBRi4Ib2a?usp=sharing](https://drive.google.com/drive/folders/1BzUQeJgICv2ZDXA9HEWcfiAXBRi4Ib2a?usp=sharing)
2. Скачайте все файлы из папки
3. Распакуйте архив в папку `data/` в корне проекта

## Очистка датасета

**ВАЖНО**: После распаковки необходимо удалить лишние папки:

```bash
# Удалите папки labels и images из каждой подпапки
rm -rf data/train/labels
rm -rf data/train/images
rm -rf data/val/labels
rm -rf data/val/images
rm -rf data/test/labels
rm -rf data/test/images
```

## Проверка структуры

После очистки структура должна выглядеть так:

```
data/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.png
│   │   └── ...
│   ├── class2/
│   │   └── ...
│   └── ...
├── val/
│   ├── class1/
│   │   └── ...
│   └── ...
└── test/
    ├── class1/
    │   └── ...
    └── ...
```

## Проверка работоспособности

Запустите пример для проверки:

```bash
cd lesson5_augmentations/augmentations_basics
python example.py
```

Если все настроено правильно, вы увидите визуализацию аугментаций.

## Возможные проблемы

1. **Ошибка "No such file or directory"**: Проверьте, что датасет скачан и распакован в правильную папку
2. **Ошибка с PIL**: Установите Pillow: `pip install Pillow`
3. **Ошибка с matplotlib**: Установите matplotlib: `pip install matplotlib` 