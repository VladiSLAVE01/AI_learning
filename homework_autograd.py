import torch

## Задание 2.1
# 1. Создаём тензоры x, y, z с requires_grad=True
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = torch.tensor(4.0, requires_grad=True)

# 2. Вычисляем функцию f(x, y, z) = x^2 + y^2 + z^2 + 2*x*y*z
f = x**2 + y**2 + z**2 + 2 * x * y * z
print(f"Значение функции f = {f.item()}")

# 3. Вычисляем градиенты с помощью backward()
f.backward()

grad_x = x.grad.item()
grad_y = y.grad.item()
grad_z = z.grad.item()

print(f"Градиент по x (автоматически): {grad_x}")
print(f"Градиент по y (автоматически): {grad_y}")
print(f"Градиент по z (автоматически): {grad_z}")

x_val, y_val, z_val = x.item(), y.item(), z.item()

analytical_dx = 2 * x_val + 2 * y_val * z_val
analytical_dy = 2 * y_val + 2 * x_val * z_val
analytical_dz = 2 * z_val + 2 * x_val * y_val

print(f"\nГрадиент по x (аналитически): {analytical_dx}")
print(f"Градиент по y (аналитически): {analytical_dy}")
print(f"Градиент по z (аналитически): {analytical_dz}\n")

# 5. Проверка совпадения
assert abs(grad_x - analytical_dx) < 1e-6, "Градиенты по x не совпадают!"
assert abs(grad_y - analytical_dy) < 1e-6, "Градиенты по y не совпадают!"
assert abs(grad_z - analytical_dz) < 1e-6, "Градиенты по z не совпадают!"


#Задание 2.2
import torch

# 1. Создадим пример данных
torch.manual_seed(0)
n = 10
x = torch.randn(n)          # входные данные
y_true = torch.randn(n)     # истинные значения

# 2. Параметры модели (требуют вычисления градиентов)
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)

# 3. Предсказания: y_pred = w * x + b
y_pred = w * x + b

# 4. Вычисление MSE
mse = torch.mean((y_pred - y_true) ** 2)
print(f"MSE = {mse.item():.4f}")

# 5. Обратное распространение для вычисления градиентов
mse.backward()

grad_w = w.grad.item()
grad_b = b.grad.item()
print(f"Градиент по w: {grad_w:.4f}")
print(f"Градиент по b: {grad_b:.4f}\n")


### 2.3 Цепное правило
x = torch.tensor([2.0], requires_grad=True)
f = torch.sin(x**2 + 1)
df_dx = torch.autograd.grad(outputs=f, inputs=x, create_graph=False)[0]

print(f"x = {x.item()}")
print(f"f(x) = {f.item():.6f}")
print(f"df/dx = {df_dx.item():.6f}")

#Проверка на совпадение
analytical_df_dx = torch.cos(x**2 + 1) * (2 * x)
assert abs(df_dx.item() - analytical_df_dx.item()) < 1e-6, "Градиенты не совпадают!"