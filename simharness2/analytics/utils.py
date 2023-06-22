from typing import List, Dict
from pandas import DataFrame


def reset_df(df: DataFrame, df_cols: List[str], df_dtypes: Dict[str, str], df_index: str):
    """Resets the input dataframe to its initial state and then returns it."""
    if df is not None:
        # FIXME convert to usage of df.iat, if possible
        df = df.iloc[0:0]
    else:
        df = DataFrame(columns=df_cols).astype(df_dtypes).set_index(df_index)

    return df
