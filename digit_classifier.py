import numpy as np
from cfg import MODEL_MAP

class DigitClassifier:
    def __init__(self, algorithm: str):
        if algorithm in MODEL_MAP:
            self.model = MODEL_MAP[algorithm]()
        else:
            raise ValueError("Unsupported algorithm. Use 'cnn', 'rf', or 'rand'.")

    def preprocess(self, image: np.ndarray, algorithm: str) -> np.ndarray:
        """
        Ensures that the input image has the correct dimensions.
        """
        if image.shape != (28, 28, 1):
            raise ValueError(f"Expected shape (28, 28, 1) for CNN, got {image.shape}.")

        return image

    def predict(self, image: np.ndarray) -> int:
        """
        Predict an output according to the current model.
        """
        preprocessed_image = self.preprocess(
            image, self.model.__class__.__name__.lower()
        )
        return self.model.predict(preprocessed_image)
