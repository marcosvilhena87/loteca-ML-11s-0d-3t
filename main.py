from scripts.common import configure_logging
from scripts.preprocess_data import preprocess
from scripts.train_model import train
from scripts.predict_results import generate_predictions


def main() -> None:
    configure_logging()
    preprocess()
    train()
    generate_predictions()


if __name__ == "__main__":
    main()
