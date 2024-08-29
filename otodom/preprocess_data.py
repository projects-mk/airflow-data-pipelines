from typing import Any
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split

load_dotenv()
from otodom.utils import (
    generate_conn_string,
    clean_df,
    convert_to,
    fix_pietra,
    fix_rok_budowy,
    group_other_attributes,
    fix_miasto,
    fix_dzielnica,
    fix_pokoje,
)

replace_dict = {"zabudowa": ["dom wolnostojący", "wolnostojący"]}

attributes = {
    "polozenie": [["miasto", "pod miastem", "wieś"]],
    "ogrzewanie": [
        [
            "gazowe",
            "pompa ciepła",
            "węglowe",
            "kominkowe",
            "elektryczne",
            "biomasa",
            "piece kaflowe",
            "miejskie",
            "olejowe",
        ]
    ],
    "material_budynku": [["cegła", "pustak", "beton komórkowy"]],
    "pietra": [
        ["1 piętro", "2 piętra", "parterowy", "3 piętra lub więcej"],
        "parterowy",
    ],
    "okna": [["plastikowe", "drewniane", "aluminiowe"]],
    "dach": [["płaski", "skośny"]],
    "pokrycie_dachu": [["blacha", "dachówka", "papa"]],
    "poddasze": [["użytkowe", "nieużytkowe"], "brak"],
}


class OtodomDomyPreprocessor:
    project_name = "otodom_domy"

    def __init__(self) -> None:
        self.conn_str = generate_conn_string("projects")
        self.df = pd.read_sql_table("otodom_domy_raw", con=self.conn_str)

    def _clean_values(self):
        self.df.replace({"brak informacji": None}, inplace=True)
        self.df = clean_df(self.df, y_col="cena", replace_dict=replace_dict)
        self.df = group_other_attributes(self.df, attributes)

        self.df["pietra"] = self.df["pietra"].apply(fix_pietra)
        self.df["rok_budowy"] = self.df["rok_budowy"].apply(fix_rok_budowy)

    def _convert_dtypes(self):
        for col in ["cena", "powierzchnia_dzialki"]:
            self.df[col] = self.df[col].apply(convert_to, dtype=float)

        for col in ["rok_budowy"]:
            self.df[col] = self.df[col].apply(convert_to, dtype=int)

    def _save_to_db(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
    ):

        self.df.to_sql(
            "preprocessed",
            con=self.conn_str,
            schema=self.project_name,
            if_exists="replace",
        )

        X_test.to_sql(
            "x_test", con=self.conn_str, schema=self.project_name, if_exists="replace"
        )
        y_test.to_sql(
            "y_test", con=self.conn_str, schema=self.project_name, if_exists="replace"
        )

        X_train.to_sql(
            "x_train", con=self.conn_str, schema=self.project_name, if_exists="replace"
        )
        y_train.to_sql(
            "y_train", con=self.conn_str, schema=self.project_name, if_exists="replace"
        )

    def _split_to_datasets(self) -> list[pd.DataFrame]:
        y = self.df[["cena"]]
        X = self.df[[col for col in self.df.columns if col != "cena"]]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42
        )

        return X_train, X_test, y_train, y_test

    def __call__(self) -> Any:
        self._clean_values()
        self._convert_dtypes()
        X_train, X_test, y_train, y_test = self._split_to_datasets()
        self._save_to_db(X_train, X_test, y_train, y_test)


mieszkania_attributes = {
    "winda": [["tak", "nie"], "brak"],
    "rodzaj_zabudowy": [
        [
            "blok",
            "apartamentowiec",
            "plastikowe",
            "kamienica",
            "cegła",
            "wielka płyta",
            "szeregowiec",
        ],
        "inny",
    ],
}


class OtodomMieszkaniaPreprocessor:
    project_name = "otodom_mieszkania"

    def __init__(self) -> None:
        self.conn_str = generate_conn_string("projects")
        self.df = pd.read_sql_table("otodom_mieszkania_raw", con=self.conn_str)

    def _clean_values(self):
        self.df.replace({"brak informacji": None}, inplace=True)
        self.df = clean_df(self.df, y_col="cena", replace_dict={})
        self.df = group_other_attributes(self.df, mieszkania_attributes)

        self.df = self.df.drop(columns=["okna", "material_budynku"], errors="ignore")

        self.df["pokoje"] = self.df["pokoje"].apply(fix_pokoje)
        self.df["pietro"] = self.df["pietro"].apply(fix_pietra)

        self.df["miasto"] = self.df["adres"].apply(fix_miasto)
        self.df = self.df.dropna(subset=["miasto"])

        self.df["dzielnica"] = self.df["adres"].apply(fix_dzielnica)
        self.df = self.df.drop(columns=["adres"])

    def _convert_dtypes(self):
        for col in ["cena", "m2"]:
            self.df[col] = self.df[col].apply(convert_to, dtype=float)

    def _save_to_db(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
    ):

        self.df.to_sql(
            "preprocessed",
            con=self.conn_str,
            schema=self.project_name,
            if_exists="replace",
        )

        X_test.to_sql(
            "x_test", con=self.conn_str, schema=self.project_name, if_exists="replace"
        )
        y_test.to_sql(
            "y_test", con=self.conn_str, schema=self.project_name, if_exists="replace"
        )

        X_train.to_sql(
            "x_train", con=self.conn_str, schema=self.project_name, if_exists="replace"
        )
        y_train.to_sql(
            "y_train", con=self.conn_str, schema=self.project_name, if_exists="replace"
        )

    def _split_to_datasets(self) -> list[pd.DataFrame]:
        y = self.df[["cena"]]
        X = self.df[[col for col in self.df.columns if col != "cena"]]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42
        )

        return X_train, X_test, y_train, y_test

    def __call__(self) -> Any:
        self._clean_values()
        self._convert_dtypes()
        X_train, X_test, y_train, y_test = self._split_to_datasets()
        self._save_to_db(X_train, X_test, y_train, y_test)
