import pandas as pd
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import roc_auc_score


def train_catboost(train_df: pd.DataFrame,
               y_train: pd.Series,
               valid_df: pd.DataFrame,
               y_valid: pd.Series):


    model = CatBoostClassifier(loss_function="Logloss",
                               eval_metric="AUC",
                               task_type="GPU",
                               learning_rate=0.01,
                               iterations=70000,
                               l2_leaf_reg=50,
                               random_seed=42,
                               od_type="Iter",
                               depth=5,
                               early_stopping_rounds=15000,
                               border_count=64
                               )

    _train = Pool(train_df, label=y_train)
    _valid = Pool(valid_df, label=y_valid)

    clf = model.fit(_train,
                    eval_set=_valid,
                    use_best_model=True,
                    verbose=5000)

    pred = clf.predict_proba(_valid)[:, 1]
    score = roc_auc_score(y_valid, pred)

    return clf, pred, score