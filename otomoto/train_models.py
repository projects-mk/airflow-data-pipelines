import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import datetime
from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class OtomotoModelTrainer:
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.x_train = DataFrame()
        self.x_test = DataFrame()
        self.y_train = DataFrame()
        self.y_test = DataFrame()
        self.mlflow_uri = get_mlflow_uri()
        self.models = [("xgboost", XGBRegressor())]
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

    def one_hot_encode(self):
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(self.project_name)

        categorical_cols = self.x_train.select_dtypes(include=["object"]).columns
        numerical_cols = self.x_train.select_dtypes(exclude=["object"]).columns

        self.encoder.fit(self.x_train[categorical_cols])

        x_train_encoded = self.encoder.transform(self.x_train[categorical_cols])
        x_test_encoded = self.encoder.transform(self.x_test[categorical_cols])

        x_train_encoded_df = pd.DataFrame(x_train_encoded.toarray(), index=self.x_train.index)
        x_test_encoded_df = pd.DataFrame(x_test_encoded.toarray(), index=self.x_test.index)

        self.x_train = pd.concat([self.x_train[numerical_cols], x_train_encoded_df], axis=1)
        self.x_test = pd.concat([self.x_test[numerical_cols], x_test_encoded_df], axis=1)

        self.y_train = self.y_train.values.ravel()
        self.y_test = self.y_test.values.ravel()

    def evaluate_on_train_and_test(self):
        """Trains models on training data and logs evaluation metrics on train & test sets."""
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(self.project_name)

        for model_name, model in self.models:
            with mlflow.start_run() as run:
                run_name = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                mlflow.set_tag("mlflow.runName", run_name)

                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                grid_cv = GridSearchCV(estimator=model, param_grid=self.parameters, cv=cv)

                grid_cv.fit(self.x_train, self.y_train)

                for dataset_name, X, y_true in [
                    ("Train", self.x_train, self.y_train),
                    ("Test", self.x_test, self.y_test),
                ]:
                    y_pred = grid_cv.predict(X)
                    r2 = r2_score(y_true, y_pred)
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

                    mlflow.log_metric(f"{dataset_name}_R2", round(r2, 2))
                    mlflow.log_metric(f"{dataset_name}_MAE", round(mae, 2))
                    mlflow.log_metric(f"{dataset_name}_RMSE", round(rmse, 2))

                mlflow.log_params(grid_cv.best_params_)

        mlflow.end_run()

    def retrain_on_full_data(self):
        """Retrains encoder and model on the full dataset (train + test) and uploads these to MLflow."""
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(self.project_name)

        full_x_data = pd.concat([self.x_train, self.x_test], axis=0)
        full_y_data = np.concatenate((self.y_train, self.y_test), axis=0)

        categorical_cols = full_x_data.select_dtypes(include=["object"]).columns
        numerical_cols = full_x_data.select_dtypes(exclude=["object"]).columns

        self.encoder.fit(full_x_data[categorical_cols])

        full_x_encoded = self.encoder.transform(full_x_data[categorical_cols])

        full_x_encoded_df = pd.DataFrame(full_x_encoded.toarray(), index=full_x_data.index)

        full_x_data = pd.concat([full_x_data[numerical_cols], full_x_encoded_df], axis=1)

        with mlflow.start_run() as full_encoder_run:
            run_name = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            mlflow.set_tag("mlflow.runName", run_name + "_full_dataset_encoder")

            encoder_name = f"{self.project_name}_data_encoder"
            mlflow.sklearn.log_model(self.encoder, artifact_path=encoder_name)
            mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/{encoder_name}", encoder_name)

        for model_name, model in self.models:
            with mlflow.start_run() as full_model_run:
                run_name = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                mlflow.set_tag("mlflow.runName", run_name + "_full_dataset_model")

                model.fit(full_x_data, full_y_data)

                y_pred = model.predict(full_x_data)
                r2 = r2_score(full_y_data, y_pred)
                mae = mean_absolute_error(full_y_data, y_pred)
                rmse = np.sqrt(mean_squared_error(full_y_data, y_pred))

                mlflow.log_metric(f"Full_Data_R2", round(r2, 2))
                mlflow.log_metric(f"Full_Data_MAE", round(mae, 2))
                mlflow.log_metric(f"Full_Data_RMSE", round(rmse, 2))

                model_full_name = f"{model_name}_full_{self.project_name}_price_predictor"
                model_info = mlflow.sklearn.log_model(model, artifact_path=model_full_name)
                model_uri = f"runs:/{full_model_run.info.run_id}/{model_info.artifact_path}"

                mlflow.register_model(model_uri, model_full_name)

        mlflow.end_run()

    def __call__(self):
        self.load_data()
        self.one_hot_encode()
        self.evaluate_on_train_and_test()
        self.retrain_on_full_data()
