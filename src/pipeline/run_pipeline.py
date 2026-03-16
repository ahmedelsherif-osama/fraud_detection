from src.data.split import split_dataset
from src.data.ingest import ingest_data
from src.features.build_features import build_features
from src.models.train_model import train_model
from src.evaluation.evaluate_model import evaluate_model
from src.models.final_train import train_final


def run_pipeline():
    ingest_data()
    split_dataset()
    build_features()
    train_model()
    train_final()
    evaluate_model()
    # save_model()
    # serve_model (API)


if __name__ == "__main__":
    run_pipeline()
