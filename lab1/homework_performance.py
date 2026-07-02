import torch
import time

### 3.1 Подготовка данных
torch.manual_seed(0)  # для воспроизводимости

# 1. Матрица 64 x 1024 x 1024
A = torch.randn(64, 1024, 1024)
print("A shape:", A.shape, "| size (elements):", A.numel(), "| memory (MB):", A.element_size() * A.numel() / (1024**2))

# 2. Матрица 128 x 512 x 512
B = torch.randn(128, 512, 512)
print("B shape:", B.shape, "| size (elements):", B.numel(), "| memory (MB):", B.element_size() * B.numel() / (1024**2))

# 3. Матрица 256 x 256 x 256
C = torch.randn(256, 256, 256)
print("C shape:", C.shape, "| size (elements):", C.numel(), "| memory (MB):", C.element_size() * C.numel() / (1024**2))

### 3.2 Функция измерения времени
def measure_time_cpu(fn, *args, **kwargs):
    """
    Измеряет время выполнения функции на CPU с помощью time.time().
    Возвращает (результат функции, время в секундах).
    """
    start = time.time()
    result = fn(*args, **kwargs)
    end = time.time()
    return result, end - start

def measure_time_gpu(fn, device="cuda", *args, **kwargs):
    """
    Измеряет время выполнения функции на GPU с помощью torch.cuda.Event().
    Функция fn должна работать с тензорами на указанном device.
    Возвращает (результат функции, время в миллисекундах).
    """
    if device != "cuda":
        raise ValueError("torch.cuda.Event() работает только на CUDA-устройствах")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Синхронизируем перед стартом, чтобы не мерить «чужие» операции
    torch.cuda.synchronize(device)
    start_event.record()

    result = fn(*args, **kwargs)

    end_event.record()
    torch.cuda.synchronize(device)  # ждём завершения всех операций

    elapsed_ms = start_event.elapsed_time(end_event)  # в миллисекундах
    return result, elapsed_ms

# 1. CPU-операция: создание большого тензора и сумма
def cpu_op(n):
    x = torch.randn(n, n)
    return x.sum()

_, cpu_time = measure_time_cpu(cpu_op, 2048)
print(f"CPU: сумма по матрице 2048x2048 заняла {cpu_time:.4f} сек")


# 2. GPU-операция
if torch.cuda.is_available():
    device = "cuda"

    def gpu_op(n, dev):
        x = torch.randn(n, n, device=dev)
        return x.sum()

    _, gpu_time_ms = measure_time_gpu(gpu_op, device, 4096, device)
    print(f"GPU: сумма по матрице 4096x4096 заняла {gpu_time_ms:.2f} мс")
else:
    print("CUDA недоступна — пропуск GPU-теста")



