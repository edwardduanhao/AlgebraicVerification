import numpy as np
from torch.utils.data.dataset import Dataset


class XORDataset(Dataset):
    def __init__(
        self,
        size: int = 1000,
        xy_range: float = 1.0,
        margin: float = 0.1,
        seed: int = 42,
        transform: callable = None,
    ):
        """
        A dataset generator for 2D XOR data with a margin around the axes.

        Points are drawn from [-xy_range, xy_range]^2, excluding a band of
        width `margin` around each axis. Quadrants 1 and 3 (x*y > 0) are
        class 0; quadrants 2 and 4 (x*y < 0) are class 1.

        Args:
            size (int, optional): Number of samples to generate. Defaults to 1000.
            xy_range (float, optional): Half-width of the sampling square. Defaults to 1.0.
            margin (float, optional): Exclusion band half-width around each axis. Defaults to 0.1.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
            transform (callable, optional): Optional transform to apply to samples. Defaults to None.
        """

        super().__init__()

        self.rng = np.random.RandomState(seed)
        self.transform = transform
        self.xy_range = xy_range
        self.margin = margin
        self.__features = []
        self.__labels = []

        for _ in range(size):
            goal_class = self.rng.randint(2)
            x, y, c = self.get_sample(goal=goal_class)
            self.__features.append(np.array([x, y], dtype=np.float32))
            self.__labels.append(c)

    def get_sample(self, goal: int = None):
        """
        Sample a single data point from the XOR dataset.

        Args:
            goal (int, optional): Desired class label for the sample. If None, any class is accepted.

        Returns:
            tuple: (x, y, class_label)
        """

        while True:
            x = self.rng.uniform(-self.xy_range, self.xy_range)
            y = self.rng.uniform(-self.xy_range, self.xy_range)
            c = self.which_class(x, y)
            if c is not None and (goal is None or c == goal):
                return x, y, c

    def which_class(self, x: float, y: float):
        """
        Determine the class of a point (x, y) in the XOR dataset.

        Points within the margin band around either axis return None (invalid).
        Quadrants 1 & 3 (x*y > 0) → class 0.
        Quadrants 2 & 4 (x*y < 0) → class 1.

        Args:
            x (float): X-coordinate of the point.
            y (float): Y-coordinate of the point.

        Returns:
            int or None: Class label (0 or 1), or None if the point is in the margin zone.
        """

        if abs(x) < self.margin or abs(y) < self.margin:
            return None
        return int(x * y < 0)

    def __getitem__(self, index: int):
        """
        Get a sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: (features, label) of the sample.
        """

        sample = (self.__features[index].copy(), self.__labels[index])
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
