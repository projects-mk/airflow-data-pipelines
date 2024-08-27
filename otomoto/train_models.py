from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import mlflow
import mlflow.sklearn
import datetime
from warnings import filterwarnings
import os
from pandas import DataFrame

filterwarnings("ignore")


class ModelTrainer:
    def __init__(
        self,
        project_name: str,
        x_train: DataFrame,
        y_train: DataFrame,
        x_test: DataFrame,
        y_test: DataFrame,
    ):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.project_name = project_name

        self.mlflow_uri = os.getenv("MLFLOW_URI")
        self.models = [
            ("XGBRegressor", XGBRegressor()),
            ("LGBMRegressor", LGBMRegressor()),
        ]
        self.parameters = {"n_estimators": [200, 500, 700], "max_depth": [6]}
        self.encoder = OneHotEncoder(handle_unknown="ignore")

    def one_hot_encoder(self):
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(self.project_name)

        encoder = OneHotEncoder(handle_unknown="ignore")
        encoder.fit(
            self.x_train[self.x_train.select_dtypes(include=["object"]).columns]
        )

        self.x_train = encoder.transform(
            self.x_train[self.x_train.select_dtypes(include=["object"]).columns]
        )
        self.x_test = encoder.transform(
            self.x_test[self.x_test.select_dtypes(include=["object"]).columns]
        )

        self.y_train = self.y_train.values
        self.y_test = self.y_test.values

        with mlflow.start_run() as encoder_run:
            run_name = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            mlflow.set_tag("mlflow.runName", run_name + "_encoder")
            mlflow.sklearn.log_model(encoder, f"{self.project_name}_data_encoder")
            mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/{self.project_name}_data_encoder",
                f"{self.project_name}_data_encoder",
            )
        mlflow.end_run()

    def train_models(self):
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(self.project_name)

        for model_name, model in self.models:
            with mlflow.start_run() as run:
                run_name = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                mlflow.set_tag("mlflow.runName", run_name)

                cv = KFold(n_splits=5, random_state=42, shuffle=True)
                grid_cv = GridSearchCV(model, self.parameters, cv=cv)
                model_name = model_name + f"_{self.project_name}_price_predictor"

                grid_cv.fit(self.x_train, self.y_train)

                for name, dataset, y_true in [
                    ("Train", self.x_train, self.y_train),
                    ("Validation", self.x_test, self.y_test),
                    ("Test", self.x_test, self.y_test),
                ]:
                    pred = grid_cv.predict(dataset)
                    r2 = round(r2_score(y_true, pred), 2)
                    mae = round(mean_absolute_error(y_true, pred), 2)
                    rmse = round(np.sqrt(mean_squared_error(y_true, pred)), 2)
                    mlflow.log_metric(f"{name} R2", r2)
                    mlflow.log_metric(f"{name} MAE", mae)
                    mlflow.log_metric(f"{name} RMSE", rmse)

                params = {
                    f"{model_name}_{k}": v for k, v in grid_cv.best_params_.items()
                }
                mlflow.log_params(params)

                model_info = mlflow.sklearn.log_model(grid_cv.best_estimator_, "model")
                model_uri = "runs:/" + run.info.run_id + "/" + model_info.artifact_path
                mlflow.register_model(model_uri, f"{model_name}")

    def __call__(self):
        self.one_hot_encoder()
        self.train_models()
