import numpy as np
from torch.utils.data.dataset import Dataset


class YinYangDataset(Dataset):
    def __init__(
        self,
        size: int = 1000,
        r_small: float = 0.1,
        r_big: float = 0.5,
        seed: int = 42,
        transform: callable = None,
    ):
        """
        Yin-Yang dataset generator.

        Args:
            size (int, optional): Number of samples to generate. Defaults to 1000.
            r_small (float, optional): Radius of small inner circles. Defaults to 0.1.
            r_big (float, optional): Radius of the big outer circle. Defaults to 0.5.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
            transform (callable, optional): Optional transform to apply to samples. Defaults to None.
        """

        super().__init__()

        # Set numpy random seed for reproducibility
        self.rng = np.random.RandomState(seed)
        self.transform = transform
        self.r_small = r_small
        self.r_big = r_big
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
        Sample a single data point from the Yin-Yang dataset.

        Args:
            goal (int, optional): Desired class label for the sample. If None, any class is accepted.

        Returns:
            tuple: (x, y, class_label)
        """

        # Sample until goal is satisfied
        found_sample_yet = False

        while not found_sample_yet:
            # Sample x,y coordinates
            x, y = self.rng.rand(2) * 2.0 * self.r_big

            # Check if within yin-yang circle
            if np.sqrt((x - self.r_big) ** 2 + (y - self.r_big) ** 2) > self.r_big:
                continue

            # Check if they have the same class as the goal for this sample
            c = self.which_class(x, y)

            if goal is None or c == goal:
                found_sample_yet = True
                break

        return x, y, c

    def which_class(self, x: float, y: float) -> int:
        """
        Determine the class of a point (x, y) in the Yin-Yang dataset.
        https://link.springer.com/content/pdf/10.1007/11564126_19.pdf

        Args:
            x (float): X-coordinate of the point.
            y (float): Y-coordinate of the point.

        Returns:
            int: Class label of the point.
        """

        d_right = self.dist_to_right_dot(x, y)
        d_left = self.dist_to_left_dot(x, y)

        criterion1 = d_right <= self.r_small
        criterion2 = d_left > self.r_small and d_left <= 0.5 * self.r_big
        criterion3 = y > self.r_big and d_right > 0.5 * self.r_big

        is_yin = criterion1 or criterion2 or criterion3
        is_circles = d_right < self.r_small or d_left < self.r_small

        if is_circles:
            return 2

        return int(is_yin)

    def dist_to_right_dot(self, x: float, y: float) -> float:
        """
        Calculate distance from point (x, y) to the center of the right small circle.

        Args:
            x (float): X-coordinate of the point.
            y (float): Y-coordinate of the point.

        Returns:
            float: Distance to the center of the right small circle.
        """

        return np.sqrt((x - 1.5 * self.r_big) ** 2 + (y - self.r_big) ** 2)

    def dist_to_left_dot(self, x: float, y: float) -> float:
        """
        Calculate distance from point (x, y) to the center of the left small circle.

        Args:
            x (float): X-coordinate of the point.
            y (float): Y-coordinate of the point.

        Returns:
            float: Distance to the center of the left small circle.
        """

        return np.sqrt((x - 0.5 * self.r_big) ** 2 + (y - self.r_big) ** 2)

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
