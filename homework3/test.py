# download_datasets.py
import torchvision
import torchvision.transforms as transforms


def download_datasets():
	"""Скачивает датасеты MNIST и CIFAR-10"""
	print("Downloading MNIST...")
	try:
		# Скачиваем MNIST
		mnist_train = torchvision.datasets.MNIST(
			root='./data', train=True, download=True,
			transform=transforms.ToTensor()
		)
		print("MNIST downloaded successfully!")
	except Exception as e:
		print(f"Error downloading MNIST: {e}")

	print("\nDownloading CIFAR-10...")
	try:
		# Скачиваем CIFAR-10
		cifar_train = torchvision.datasets.CIFAR10(
			root='./data', train=True, download=True,
			transform=transforms.ToTensor()
		)
		print("CIFAR-10 downloaded successfully!")
	except Exception as e:
		print(f"Error downloading CIFAR-10: {e}")


if __name__ == "__main__":
	download_datasets()