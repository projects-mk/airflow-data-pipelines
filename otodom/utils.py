import requests
import os
import pandas as pd


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


def clean_df(
    df: pd.DataFrame, y_col: str, fillna_dict: dict = {}, replace_dict: dict = {}
) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna(how="all")
    df = df.drop(
        columns=["index", "czynsz", "dojazd", "ogrodzenie", "okolica", "opis"],
        errors="ignore",
    )

    df.reset_index(inplace=True, drop=True)

    idxs_to_drop = []
    for i in range(0, len(df)):
        row = df.iloc[i]
        cols_with_nans = []
        for col in df.columns:
            if not row[col]:
                cols_with_nans.append(col)

        if len(cols_with_nans) / len(df.columns) > 0.4 or not row[y_col]:
            idxs_to_drop.append(i)

    df = df.drop(idxs_to_drop)

    for key, val in fillna_dict.items():
        df[key] = df[key].fillna(val)

    for key, val in replace_dict.items():
        df[key] = df[key].apply(lambda x: x.replace(val[0], val[1]) if x else "inne")

    return df


import numpy as np


def group_other_attributes(df: pd.DataFrame, attributes: dict) -> pd.DataFrame:
    for column_name, core_attributes in attributes.items():
        fill = core_attributes[1] if len(core_attributes) > 1 else "inne"
        df[column_name] = df[column_name].apply(
            lambda x: x if x in core_attributes[0] else fill
        )

    return df


def fix_rok_budowy(s: str | int) -> int:

    if isinstance(s, str):
        s = s[:4]
        if len(s) == 4 and s.isdigit():
            return round(int(s))

    elif isinstance(s, int):
        if len(str(round(s))) == 4:
            return round(int(s))

    return None


def fix_pietra(s: str) -> str:
    return s[0]


def convert_to(s: str, dtype: int | float):
    if isinstance(s, str):
        s = s.lower().replace("km", "")
        s = s.lower().replace("l/100km", "")
        s = s.lower().replace("l/100", "")
        s = s.lower().replace("zł", "")
        s = s.lower().replace("cm3", "")
        s = s.replace(" ", "")
        s = s.replace(",", ".")
        s = s.strip()

    if not s or "usd" in str(s).lower() or "eur" in str(s).lower():
        return dtype(0)

    try:
        return dtype(s)
    except ValueError:
        if dtype == int:
            return 0
        elif dtype == float:
            return 0
