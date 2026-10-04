from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


col_trans = make_column_transformer(
    (
        OneHotEncoder(
            sparse_output=False,
            handle_unknown='ignore'
        ),
        ['location']
    ),
    remainder='passthrough'
)


preprocessing_pipeline = make_pipeline(
    col_trans,
    StandardScaler()
)