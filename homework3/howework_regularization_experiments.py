import torch
import logging
from datasets import get_mnist_loaders
from utils.visualization_utils import plot_training_history
from howework_depth_experiments import experiment


def main():
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_mnist_loaders()
    hidden_layers = [256, 128, 64]
    config = [
        [False, 0, 0],
        [False, 0.5, 0],
        [True, 0, 0],
        [True, 0.5, 0],
        [False, 0, 1e-4]
    ]

    for i, (use_batch_norm, dropout_p, l2_weight_decay) in enumerate(config):
        history, train_time = experiment(hidden_layers, train_loader, test_loader, device, use_batch_norm, dropout_p, l2_weight_decay)
        plot_training_history(history, f'results/regularization_experiments/regularization_{i + 1}.png')
        logging.info(f'Время обучения: {train_time:.2f}')
        logging.info(
            f'Финальная train accuracy: {history["train_accuracies"][-1]:.4f}, test accuracy: {history["test_accuracies"][-1]:.4f}')


if __name__ == '__main__':
    main()
