# Домашнее задание к уроку 3: Полносвязные сети

## Задание 1: Эксперименты с глубиной сети

Модели с различной глубиной обучены на датасете MNIST. Ниже приведены результаты экспериментов.

Модель с 1 слоем - время обучения: 97 секунд
![depth_1_layers.png](results/depth_experiments/depth_1_layers.png)
![depth_1_layers_extended.png](results/depth_experiments/depth_1_layers_extended.png)
Модель с 2 слоями - время обучения: 107 секунд
![depth_2_layers.png](results/depth_experiments/depth_2_layers.png)
![depth_2_layers_extended.png](results/depth_experiments/depth_2_layers_extended.png)
Модель с 3 слоями - время обучения: 110 секунд
![depth_3_layers.png](results/depth_experiments/depth_3_layers.png)
![depth_3_layers_extended.png](results/depth_experiments/depth_3_layers_extended.png)
Модель с 5 слоями - время обучения: 115 секунд
![depth_5_layers.png](results/depth_experiments/depth_5_layers.png)
![depth_5_layers_extended.png](results/depth_experiments/depth_5_layers_extended.png)
Модель с 7 слоями - время обучения: 122 секунды
![depth_7_layers.png](results/depth_experiments/depth_7_layers.png)
![depth_7_layers_extended.png](results/depth_experiments/depth_7_layers_extended.png)

Можно сделать вывод, что увеличение глубины сети приводит к увеличению времени обучения, но не всегда к улучшению
качества.

При применении регуляризации и нормализации данных качество моделей увеличилось при небольшом увеличении времени,
тестовые результаты лучше чем тренировочные.

### 2.1 Сравнение моделей разной ширины

Узкие слои - время обучения: 109 секунд
![width_1.png](results/width_experiments/width_1.png)

Средние слои - время обучения: 123 секунды
![width_2.png](results/width_experiments/width_2.png)

Широкие слои - время обучения: 116 секунд

![width_3.png](results/width_experiments/width_3.png)

Очень широкие слои - время обучения: 128 секунд

![width_4.png](results/width_experiments/width_4.png)

С увеличением ширины слоев время обучения увеличивается, но качество не всегда улучшается.

## Задание 3: Эксперименты с регуляризацией

Без регуляризации - время обучения: 121 секунда
![regularization_1.png](results/regularization_experiments/regularization_1.png)

Dropout - время обучения: 130 секунд
![regularization_1.png](results/regularization_experiments/regularization_2.png)

Batch Normalization - время обучения: 127 секунд
![regularization_2.png](results/regularization_experiments/regularization_3.png)

Dropout + Batch Normalization - время обучения: 130 секунд
![regularization_3.png](results/regularization_experiments/regularization_4.png)

L2-регуляризация - время обучения: 126 секунд
![regularization_4.png](results/regularization_experiments/regularization_5.png)

Dropout снижает переобучение, test accuracy стабильно высокая, но train accuracy не доходит до максимума.
Batch Normalization улучшает качество, но не всегда снижает переобучение.
Dropout в сочетании с Batch Normalization не дает значительного прироста.
L2-регуляризация также помогает снизить переобучение, но не всегда улучшает качество.