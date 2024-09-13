import polars as pl

from preprocessing import Preprocess
from utils import TRAIN_PATH


def main():
    df_train_raw = pl.read_csv(TRAIN_PATH)
    preprocess = Preprocess()
    df = preprocess.fit_transform(df_train_raw)

    print(df.head())
