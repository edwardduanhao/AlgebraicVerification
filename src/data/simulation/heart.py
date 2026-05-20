import numpy as np
from torch.utils.data.dataset import Dataset


class HeartDataset(Dataset):
    def __init__(
        self,
        size: int = 1000,
        seed: int = 42,
        transform: callable = None,
    ):
        """
        Heart dataset generator with 2 classes.

        Args:
            size (int, optional): Number of samples to generate. Defaults to 1000.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
            transform (callable, optional): Optional transform to apply to samples. Defaults to None.
        """

        super().__init__()

        # Set numpy random seed for reproducibility
        self.rng = np.random.RandomState(seed)
        self.transform = transform
        self.__features = []
        self.__labels = []

        for i in range(size):
            # Keep num of class instances balanced by using rejection sampling
            goal_class = self.rng.randint(2)
            x, c = self.get_sample(goal=goal_class)
            val = np.array(x, dtype=np.float32)

            self.__features.append(val)
            self.__labels.append(c)

    def get_sample(self, goal: int = None):
        """
        Sample a single data point from the Heart dataset.

        Args:
            goal (int, optional): Desired class label for the sample. If None, any class is accepted.
        Returns:
            tuple: (x, class_label)
        """

        # Sample until goal is satisfied
        found_sample_yet = False

        while not found_sample_yet:
            # Sample uniformly in [-1.5, 1.5] cube
            x = self.rng.uniform(low=-1.5, high=1.5, size=3)

            # Determine class based on angular position
            c = self.which_class(x)

            if goal is None or c == goal:
                found_sample_yet = True
                break

        return x, c

    def which_class(self, x: np.ndarray) -> int:
        """
        Determine the class of a point based on its polar coordinates.

        Args:
            x (np.ndarray): Point coordinates.

        Returns:
            int: Class label of the point (0 or 1).
        """

        c = (
            (x[0] ** 2 + 20 * x[1] ** 2 + x[2] ** 2 - 1) ** 3
            - x[0] ** 2 * x[2] ** 3
            - x[1] ** 2 * x[2] ** 3
        ) > 0

        return int(c)

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
