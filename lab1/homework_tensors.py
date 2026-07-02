### Домашняя работа 1
##Задание 1.1

import torch
# - Тензор размером 3x4, заполненный случайными числами от 0 до 1
A = torch.rand(3, 4)
print(f"A = {A}")

# - Тензор размером 2x3x4, заполненный нулями
B = torch.zeros(2, 3, 4)
print(f"B = {B}")

# - Тензор размером 5x5, заполненный единицами
C = torch.ones(5, 5)
print(f"C = {C}")

# - Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)
D = torch.arange(16).reshape(4, 4) # В параметры arange порог указываем с + 1
print(f"D = {D}")

##Задание 1.2

A2 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

B2 = torch.tensor([[10, 20], [30, 40]], dtype=torch.float32)

# - Транспонирование тензора A
transposed_A2 = A2.transpose(0, 1)
print(f"transposed_A2 = {transposed_A2}")

# - Матричное умножение A и B
print(f'A2 @ B2 = {A2 @ B2}')

# - Поэлементное умножение A и транспонированного B
transposed_B2 = B2.transpose(0, 1)
print(f"transposed_B2 = {transposed_B2}")
print(f'transposed_B2 * A2 = {transposed_B2 * A2}')

# - Вычислите сумму всех элементов тензора A
sum_A2 = A2.sum()
print(f'sum_A2 = {sum_A2}')

### Задание 1.3
E = torch.arange(125).view(5, 5, 5)
print(f"E = {E}")

# Извлеките:
# - Первую строку
first_row = E[0, :, :]
print(f"first_row = {first_row}")

# - Последний столбец
last_column = E[:,:, -1]
print(f"last_column = {last_column}")

# - Подматрицу размером 2x2 из центра тензора
center_submatrix = E[1:3, 1:3]
print(f"center_submatrix = {center_submatrix}")

# - Все элементы с четными индексами
even_elements = E[::2, ::2, ::2] # все 3 индекса четные
print(f"even_elements = {even_elements}")

## Задание 1.4
t = torch.arange(24)
print(f"t = {t}")

t_2x12 = t.view(2, 12)
t_3x8 = t.view(3, 8)
t_4x6 = t.view(4, 6)
t_2x3x4 = t.view(2, 3, 4)
t_2x2x2x3 = t.view(2, 2, 2, 3)

print("2x12:\n", t_2x12, "\n")
print("3x8:\n", t_3x8, "\n")
print("4x6:\n", t_4x6, "\n")
print("2x3x4:\n", t_2x3x4, "\n")
print("2x2x2x3:\n", t_2x2x2x3, "\n")
