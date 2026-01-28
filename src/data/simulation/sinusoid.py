import numpy as np
from torch.utils.data.dataset import Dataset


class SinusoidDataset(Dataset):
    def __init__(
        self,
        size: int = 1000,
        x_range: tuple[float, float] = (-np.pi, np.pi),
        y_range: tuple[float, float] = (-2, 2),
        seed: int = 42,
        transform: callable = None,
    ):
        """
        A dataset generator based on a sinusoidal decision boundary.

        Args:
            size (int, optional): Number of samples to generate. Defaults to 1000.
            x_range (tuple[float, float], optional): Range for x values. Defaults to (-pi, pi).
            y_range (tuple[float, float], optional): Range for y values. Defaults to (-2, 2).
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
            transform (callable, optional): Optional transform to apply to samples. Defaults to None.
        """

        super().__init__()

        # Set numpy random seed for reproducibility
        self.rng = np.random.RandomState(seed)
        self.transform = transform
        self.x_range = x_range
        self.y_range = y_range
        self.__features = []
        self.__labels = []

        for i in range(size):
            # Keep num of class instances balanced by using rejection sampling
            goal_class = self.rng.randint(2)
            x, y, c = self.get_sample(goal=goal_class)
            val = np.array([x, y], dtype=np.float32)

            self.__features.append(val)
            self.__labels.append(c)

    def get_sample(self, goal: int = None):
        """
        Sample a single data point from the sinusoid dataset.

        Args:
            goal (int, optional): Desired class label for the sample. If None, any class is accepted.

        Returns:
            tuple: (x, y, class_label)
        """

        # Sample until goal is satisfied
        found_sample_yet = False

        while not found_sample_yet:
            # Sample x coordinate
            x = self.rng.rand() * (self.x_range[1] - self.x_range[0]) + self.x_range[0]

            # Sample y coordinate
            y = self.rng.rand() * (self.y_range[1] - self.y_range[0]) + self.y_range[0]

            # Check if they have the same class as the goal for this sample
            c = self.which_class(x, y)

            if goal is None or c == goal:
                found_sample_yet = True
                break

        return x, y, c

    def which_class(self, x: float, y: float) -> int:
        """
        Determine the class of a point (x, y) in the sinusoid dataset.

        Args:
            x (float): X-coordinate of the point.
            y (float): Y-coordinate of the point.

        Returns:
            int: Class label of the point (0 or 1).
        """

        return int(y > np.sin(x))

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
