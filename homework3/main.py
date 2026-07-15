import torch
from datasets import get_mnist_loaders
from models import FullyConnectedModel
from trainer import train_model
from utils.visualization_utils import plot_training_history

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loaders, test_loaders = get_mnist_loaders()

model = FullyConnectedModel(
    input_size=784,
    num_classes=10,
    layers=[
        {'type': 'linear', 'size': 512},
        {'type': 'batch_norm'},
        {'type': 'relu'},
        {'type': 'dropout', 'rate': 0.2},
        {'type': 'linear', 'size': 256},
        {'type': 'relu'},
        {'type': 'dropout', 'rate': 0.1},
        {'type': 'linear', 'size': 128},
        {'type': 'relu'},
    ]
).to(device)

history = train_model(model, train_loaders, test_loaders, device=device)

plot_training_history(history)
