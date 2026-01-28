import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from pathlib import Path


class MNISTDataset(Dataset):
    def __init__(
        self,
        train: bool = True,
        classes: tuple = None,
        output_size: int = 14,
        flatten: bool = True,
        transform: callable = None,
        root: str = None,
    ):
        """
        MNIST dataset with optional class filtering and downscaling.

        Args:
            train (bool, optional): If True, load training set; else test set. Defaults to True.
            classes (tuple, optional): Tuple of classes to keep (e.g., (0, 1) for binary).
                If None, keep all 10 classes. Defaults to None.
            output_size (int, optional): Output image size (e.g., 14 for 14x14).
                Original MNIST is 28x28. Defaults to 14.
            flatten (bool, optional): If True, flatten images to 1D vectors. Defaults to True.
            transform (callable, optional): Optional transform to apply to samples. Defaults to None.
            root (str, optional): Root directory for MNIST data. If None, uses src/data/real.
                Defaults to None.
        """

        super().__init__()

        self.transform = transform
        self.output_size = output_size
        self.flatten = flatten
        self.classes = classes
        self.__features = []
        self.__labels = []

        # Set root directory for MNIST data
        if root is None:
            # Default to src/data/real relative to this file
            root = Path(__file__).parent

        # Download and load MNIST dataset
        torch_transform = transforms.ToTensor()
        mnist_dataset = datasets.MNIST(
            root=str(root), train=train, download=True, transform=torch_transform
        )

        # Filter by classes if specified
        if classes is not None:
            indices = [
                i
                for i in range(len(mnist_dataset))
                if mnist_dataset.targets[i] in classes
            ]
        else:
            indices = list(range(len(mnist_dataset)))

        # Process and store all samples
        for idx in indices:
            img, label = mnist_dataset[idx]

            # Downscale the image if needed
            processed_img = self._downscale_image(img)

            # Flatten to vector if requested
            if flatten:
                features = processed_img.flatten().numpy().astype(np.float32)
            else:
                features = processed_img.squeeze().numpy().astype(np.float32)

            # Remap labels if filtering classes
            if classes is not None:
                # Map to 0, 1, 2, ... based on position in classes tuple
                remapped_label = classes.index(label)
            else:
                remapped_label = label

            self.__features.append(features)
            self.__labels.append(remapped_label)

    def _downscale_image(self, img_tensor):
        """
        Downscale image from 28x28 to output_size x output_size.

        Args:
            img_tensor: tensor of shape (1, 28, 28)

        Returns:
            downscaled_tensor: tensor of shape (1, output_size, output_size)
        """
        if self.output_size == 28:
            return img_tensor

        # Check if 28 is evenly divisible by output_size
        if 28 % self.output_size == 0:
            # Use average pooling with exact kernel size
            kernel_size = 28 // self.output_size
            downscaled = F.avg_pool2d(
                img_tensor, kernel_size=kernel_size, stride=kernel_size
            )
        else:
            # Use adaptive average pooling for non-divisible sizes
            downscaled = F.adaptive_avg_pool2d(
                img_tensor, output_size=(self.output_size, self.output_size)
            )

        return downscaled

    def __getitem__(self, index: int):
        """
        Get a sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.
        Returns:
            tuple: (features, label) of the sample.
        """

        sample = (self.__features[index].copy(), self.__labels[index])

        # Apply transform if provided
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """

        return len(self.__labels)
