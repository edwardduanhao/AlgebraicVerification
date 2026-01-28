import numpy as np
from torch.utils.data.dataset import Dataset


class FanDataset(Dataset):
    def __init__(
        self,
        size: int = 1000,
        alpha: float = 3.0,
        phi: float = np.pi / 6.0,
        seed: int = 42,
        transform: callable = None,
    ):
        """
        Fan dataset generator with 3 classes.

        Args:
            size (int, optional): Number of samples to generate. Defaults to 1000.
            alpha (float, optional): Curvature parameter for spiral effect. Defaults to 3.0.
            phi (float, optional): Global rotation angle. Defaults to pi/6.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
            transform (callable, optional): Optional transform to apply to samples. Defaults to None.
        """

        super().__init__()

        # Set numpy random seed for reproducibility
        self.rng = np.random.RandomState(seed)
        self.transform = transform
        self.alpha = alpha
        self.phi = phi
        self.__features = []
        self.__labels = []

        for i in range(size):
            # Keep num of class instances balanced by using rejection sampling
            goal_class = self.rng.randint(3)
            x, y, c = self.get_sample(goal=goal_class)
            val = np.array([x, y], dtype=np.float32)

            self.__features.append(val)
            self.__labels.append(c)

    def get_sample(self, goal: int = None):
        """
        Sample a single data point from the fan dataset.

        Args:
            goal (int, optional): Desired class label for the sample. If None, any class is accepted.

        Returns:
            tuple: (x, y, class_label)
        """

        # Sample until goal is satisfied
        found_sample_yet = False

        while not found_sample_yet:
            # Sample uniformly in unit disk via polar coordinates
            r = np.sqrt(self.rng.rand())  # radius
            theta = self.rng.uniform(-np.pi, np.pi)  # base angle

            # Determine class based on angular position
            c = self.which_class(r, theta)

            if goal is None or c == goal:
                found_sample_yet = True
                break

        # Warp angles to get curved boundaries (spiral-like)
        theta = theta + self.alpha * (r**2)

        # Map to Cartesian coordinates
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        return x, y, c

    def which_class(self, r: float, theta: float) -> int:
        """
        Determine the class of a point based on its polar coordinates.

        Args:
            r (float): Radius coordinate.
            theta0 (float): Base angle coordinate.

        Returns:
            int: Class label of the point (0, 1, or 2).
        """

        # 3 equal angular sectors, with a shift so boundaries aren't on axes
        theta_shifted = (theta - self.phi + 2 * np.pi) % (2 * np.pi)
        c = int(3 * theta_shifted / (2 * np.pi))  # 0, 1, 2

        return c

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
