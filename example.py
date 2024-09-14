import pandas as pd 
import numpy as np 
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow 
from mlflow.models.signature import infer_signature
import logging
import sys 




logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.FileHandler(filename=f"logs/{__name__}.log", mode="w")
handler.setFormatter(
    logging.Formatter("File[%(name)s.py] %(asctime)s %(levelname)s %(message)s")
)

logger.addHandler(handler)
logger.info(f"Start logging in file {__name__}.py")


def eval_metrics(target, pred):
    rmse = np.sqrt(mean_squared_error(target, pred))
    mae = mean_absolute_error(target, pred)
    r2 = r2_score(target, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    np.random.seed(40)
    data_path = "data/winequality-red.csv"

    try:
        data = pd.read_csv(data_path)
    except Exception as e:
        logger.exception("Dataset can't be downloaded", exc_info=True)
    
    features, targets = data.iloc[:, :-1], data.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.25, random_state=42, stratify=targets)

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5 
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5 

    with mlflow.start_run():
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(x_train, y_train)

        predicted_qualities = model.predict(x_test)

        rmse, mae, r2 = eval_metrics(y_test, predicted_qualities)
        print(f"ElasticNet => alpha: {alpha} | l1_ratio: {l1_ratio} ")
        print(f"Metrics => rmse: {rmse}, mae: {mae}, r2-score: {r2}")

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2-score", r2)

        predictions = model.predict(x_train)
        signature = infer_signature(x_train, predictions)

        url_type = urlparse(mlflow.get_tracking_uri()).scheme 

        if url_type != "file":
            mlflow.sklearn.log_model(
                model, "model", registred_model_name="ElasticNetModel", signature=signature
            )
        else:
            mlflow.sklearn.log_model(
                model, "model", signature=signature
            )







