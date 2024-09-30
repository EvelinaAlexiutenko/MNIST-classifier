
from models import CNNModel, RandomForestModel, RandomModel


MODEL_MAP = {
    "cnn": CNNModel,
    "rf": RandomForestModel,
    "rand": RandomModel
}
