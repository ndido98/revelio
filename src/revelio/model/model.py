from abc import ABC, abstractmethod

from torch.utils.data import DataLoader

from revelio.config.config import Config


class Model(ABC):

    batched: bool

    def __init__(
        self,
        *,
        config: Config,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
    ):
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    @abstractmethod
    def fit(self) -> None:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def evaluate(self) -> None:
        raise NotImplementedError  # pragma: no cover
