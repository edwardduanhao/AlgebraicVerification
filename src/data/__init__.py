from .simulation.yinyang import YinYangDataset
from .simulation.sinusoid import SinusoidDataset
from .simulation.fan import FanDataset
from .simulation.steinerroman import SteinerRomanDataset
from .simulation.heart import HeartDataset
from .simulation.xor import XORDataset
from .real.mnist import MNISTDataset

__all__ = [
    "YinYangDataset",
    "SinusoidDataset",
    "FanDataset",
    "SteinerRomanDataset",
    "HeartDataset",
    "XORDataset",
    "MNISTDataset",
]
