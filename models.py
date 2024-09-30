# digit_classifier/models.py
import numpy as np
from interface import DigitClassificationInterface
from sklearn.ensemble import RandomForestClassifier

class CNNModel(DigitClassificationInterface):
    def __init__(self):
        pass

    def predict(self, image: np.ndarray) -> int:
        # Simulate prediction (e.g., sum the image pixels mod 10)
        return int(np.sum(image) % 10)

class RandomForestModel(DigitClassificationInterface):
    def __init__(self):
        self.model = RandomForestClassifier()

    def predict(self, image: np.ndarray) -> int:
        # Flatten the image to a 1D array
        image_flat = image.flatten().reshape(1, -1)
        return int(np.sum(image_flat) % 10)

class RandomModel(DigitClassificationInterface):
    def __init__(self):
        pass

    def predict(self, image: np.ndarray) -> int:
        # Crop the image to 10x10 from the center
        center_crop = image[9:19, 9:19]
        # Return a random value between 0 and 9
        return np.random.randint(0, 10)
