import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()

    ax2.plot(history['train_accuracies'], label='Train Accuracy')
    ax2.plot(history['test_accuracies'], label='Test Accuracy')
    ax2.set_title('Accuracy')
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
