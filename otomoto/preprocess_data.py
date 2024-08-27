from typing import Any
import pandas as pd
from otomoto.utils import generate_conn_string, clean_df, convert_to
from sklearn.model_selection import train_test_split


d = {'serwisowany_w_aso':'Nie',
     'bezwypadkowy':'Nie',
     'stan':'Używany'}

class OtomotoPreprocessor():
    def __init__(self) -> None:
        self.conn_str = generate_conn_string('projects')
        self.df = pd.read_sql_table('otomoto_raw', con=self.conn_str)

    
    def _remove_nulls(self):
        self.df = clean_df(self.df, y_col='cena', fillna_dict=d)

    def _convert_dtypes(self):

        for col in ['cena','przebieg','pojemosc_skokowa','spalanie_w_miescie']:
            self.df[col]= self.df[col].apply(convert_to, dtype=float)

        for col in ['rok_produkcji','moc','liczba_drzwi','liczba_miejsc']:
            self.df[col]= self.df[col].apply(convert_to, dtype=int)

        self.df['stan'] = self.df['stan'].apply(lambda x: x.replace('Używane','Używany').replace('Nowe','Nowy'))


    def _split_to_datasets(self) -> list[pd.DataFrame]:
        y = self.df[['cena']]
        X = self.df[[col for col in self.df.columns if col != 'cena']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

        return X_train, X_test, y_train, y_test

    def _save_to_db(self,X_train: pd.DataFrame, X_test: pd.DataFrame, y_train:pd.DataFrame, y_test:pd.DataFrame):
        
        self.df.to_sql('preprocessed',con=self.conn_str, schema='otomoto')

        X_test.to_sql('x_test',con=self.conn_str, schema='otomoto')
        y_test.to_sql('y_test',con=self.conn_str, schema='otomoto')

        X_train.to_sql('x_train',con=self.conn_str, schema='otomoto')
        y_train.to_sql('y_train',con=self.conn_str, schema='otomoto')

    def __call__(self) -> Any:
        self._remove_nulls()
        self._convert_dtypes()
        X_train, X_test, y_train, y_test = self._split_to_datasets()
        self._save_to_db(X_train, X_test, y_train, y_test)