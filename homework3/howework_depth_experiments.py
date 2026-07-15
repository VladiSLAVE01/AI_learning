import torch
import logging
from utils.model_utils import make_layer_config
from models import FullyConnectedModel
from trainer import train_model
from datasets import get_mnist_loaders
from utils.visualization_utils import plot_training_history


def experiment(hidden_layers, train_loader, test_loader, device='cpu', use_batch_norm=False, dropout_p=0.5,
               l2_weight_decay=0):
    layers_config = make_layer_config(hidden_layers, use_batch_norm=use_batch_norm, dropout_p=dropout_p)

    model = FullyConnectedModel(
        input_size=784,
        num_classes=10,
        layers=layers_config
    ).to(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record(torch.cuda.current_stream())
    history = train_model(model, train_loader, test_loader, device=device, weight_decay=l2_weight_decay)
    end.record(torch.cuda.current_stream())
    torch.cuda.synchronize()

    return history, start.elapsed_time(end)


def main():
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_mnist_loaders()
    hidden_layers_list = [
        [],
        [128],
        [128, 128],
        [128, 128, 128, 128],
        [128, 128, 128, 128, 128, 128],
    ]

    for hidden_layers in hidden_layers_list:
        logging.info(f"Обучение модели. Количество слоев: {len(hidden_layers) + 1}")
        history, train_time = experiment(hidden_layers, train_loader, test_loader, device)
        plot_training_history(history, f'results/depth_experiments/depth_{len(hidden_layers) + 1}_layers_extended.png')
        logging.info(f'Время обучения: {train_time:.2f}')
        logging.info(
            f'Финальная train accuracy: {history["train_accuracies"][-1]:.4f}, test accuracy: {history["test_accuracies"][-1]:.4f}')


if __name__ == '__main__':
    main()
