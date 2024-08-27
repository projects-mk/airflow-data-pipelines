import requests
import os
from pandas import DataFrame
from datetime import datetime, timedelta


def generate_conn_string(db: str) -> str:

    url = os.environ["VAULT_URL"]
    token = os.environ["VAULT_TOKEN"]

    resp = requests.get(url, headers={"X-Vault-Token": token}).json()
    if not os.getenv("IS_TEST_ENV"):
        return resp["data"]["data"]["postgres"] + db

    return resp["data"]["data"]["postgres"] + "test_db"


def get_mlflow_uri() -> str:

    url = os.environ["VAULT_URL"]
    token = os.environ["VAULT_TOKEN"]

    resp = requests.get(url, headers={"X-Vault-Token": token}).json()

    return resp["data"]["data"]["mlflow_uri"]


def clean_df(df: DataFrame, y_col: str, fillna_dict: dict) -> DataFrame:
    df = df.drop_duplicates()
    df = df.dropna(how="all")
    df = df.drop(columns=["index", "generacja", "wersja"], errors="ignore")

    df.reset_index(inplace=True, drop=True)

    idxs_to_drop = []
    for i in range(0, len(df)):
        row = df.iloc[i]
        cols_with_nans = []
        for col in df.columns:
            if not row[col]:
                cols_with_nans.append(col)

        if len(cols_with_nans) / len(df.columns) > 0.45 or not row[y_col]:
            idxs_to_drop.append(i)

    df = df.drop(idxs_to_drop)

    for key, val in fillna_dict.items():
        df[key] = df[key].fillna(val)

    return df


def convert_to(s: str, dtype: int | float):

    if isinstance(s, str):
        s = s.lower().replace("km", "")
        s = s.lower().replace("l/100km", "")
        s = s.lower().replace("l/100", "")

        s = s.lower().replace("cm3", "")
        s = s.replace(" ", "")
        s = s.replace(",", ".")
        s = s.strip()

    if not s:
        return dtype(0)

    return dtype(s)
