import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss


def fit_all(Xtr, ytr, Xva, yva, seed=42):
    models = {}

    m = lgb.LGBMClassifier(
        n_estimators=3000, learning_rate=0.02, num_leaves=31, min_child_samples=200,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
        random_state=seed, verbose=-1,
    )
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)])
    models["lightgbm"] = m

    m = CatBoostClassifier(
        iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5,
        random_seed=seed, verbose=0, thread_count=-1,
    )
    m.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=100)
    models["catboost"] = m

    m = XGBClassifier(
        n_estimators=3000, learning_rate=0.02, max_depth=5, min_child_weight=50,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, tree_method="hist",
        eval_metric="logloss", early_stopping_rounds=100, random_state=seed, n_jobs=-1,
    )
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    models["xgboost"] = m
    return models


def predict_all(models, X):
    return {k: m.predict_proba(X)[:, 1] for k, m in models.items()}


def ensemble(probs):
    return np.mean(list(probs.values()), axis=0)


def score(y, p):
    return {
        "auc": roc_auc_score(y, p),
        "logloss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
    }
