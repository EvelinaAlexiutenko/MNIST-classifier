from abc import ABC, abstractmethod
import numpy as np

class DigitClassificationInterface(ABC):
    @abstractmethod
    def predict(self, image: np.ndarray) -> int:
        pass

    def train(self):
        raise NotImplementedError("Training is not implemented for this model.")
