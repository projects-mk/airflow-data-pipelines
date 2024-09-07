from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import mlflow
import mlflow.sklearn
import datetime
from warnings import filterwarnings
from pandas import DataFrame
import pandas as pd
from otomoto.utils import generate_conn_string, get_mlflow_uri

filterwarnings("ignore")


class OtodomModelTrainer:
    def __init__(self, project_name: str):
        self.x_train = DataFrame()
        self.x_test = DataFrame()
        self.y_train = DataFrame()
        self.y_test = DataFrame()

        self.project_name = project_name

        self.mlflow_uri = get_mlflow_uri()
        self.models = [
            ("xgboost", XGBRegressor()),
        ]
        self.parameters = {"n_estimators": [200, 500, 700], "max_depth": [6]}
        self.encoder = OneHotEncoder(handle_unknown="ignore")

    def load_data(self):
        conn_str = generate_conn_string("projects")

        self.x_train = pd.read_sql_table(
            "x_train",
            con=conn_str,
            schema=self.project_name.replace("_price_predictor", ""),
        ).drop(columns=["index"], errors="ignore")
        self.x_test = pd.read_sql_table(
            "x_test",
            con=conn_str,
            schema=self.project_name.replace("_price_predictor", ""),
        ).drop(columns=["index"], errors="ignore")
        self.y_train = pd.read_sql_table(
            "y_train",
            con=conn_str,
            schema=self.project_name.replace("_price_predictor", ""),
        ).drop(columns=["index"], errors="ignore")
        self.y_test = pd.read_sql_table(
            "y_test",
            con=conn_str,
            schema=self.project_name.replace("_price_predictor", ""),
        ).drop(columns=["index"], errors="ignore")

    def one_hot_encoder(self):
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(self.project_name)

        categorical_cols = self.x_train.select_dtypes(include=["object"]).columns
        non_categorical_cols = self.x_train.select_dtypes(exclude=["object"]).columns

        encoder = OneHotEncoder(handle_unknown="ignore")
        encoder.fit(self.x_train[categorical_cols])

        x_train_encoded = encoder.transform(self.x_train[categorical_cols])
        x_test_encoded = encoder.transform(self.x_test[categorical_cols])

        x_train_encoded_df = pd.DataFrame(x_train_encoded.toarray(), index=self.x_train.index)
        x_test_encoded_df = pd.DataFrame(x_test_encoded.toarray(), index=self.x_test.index)

        self.x_train = pd.concat([self.x_train[non_categorical_cols], x_train_encoded_df], axis=1)
        self.x_test = pd.concat([self.x_test[non_categorical_cols], x_test_encoded_df], axis=1)

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
                    ("Test", self.x_test, self.y_test),
                ]:
                    pred = grid_cv.predict(dataset)
                    r2 = round(r2_score(y_true, pred), 2)
                    mae = round(mean_absolute_error(y_true, pred), 2)
                    rmse = round(np.sqrt(mean_squared_error(y_true, pred)), 2)
                    mlflow.log_metric(f"{name} R2", r2)
                    mlflow.log_metric(f"{name} MAE", mae)
                    mlflow.log_metric(f"{name} RMSE", rmse)

                params = {f"{model_name}_{k}": v for k, v in grid_cv.best_params_.items()}
                mlflow.log_params(params)

                model_info = mlflow.sklearn.log_model(grid_cv.best_estimator_, "model")
                model_uri = "runs:/" + run.info.run_id + "/" + model_info.artifact_path
                mlflow.register_model(model_uri, f"{model_name}")

    def __call__(self):
        self.load_data()
        self.one_hot_encoder()
        self.train_models()
