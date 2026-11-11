"""Build model pipelines and run a small, fair comparison."""

from __future__ import annotations


from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC

from src.config import CV_SPLITS, RANDOM_STATE
from src.preprocess import Densifier, build_preprocessor, build_selector

# Shared feature-selection grid kept small so the full experiment
# finishes in a reasonable time on a laptop CPU.
PERCENTILE_GRID = [5, 10]
LINEAR_C_GRID = [0.5, 2.0]


def _base_steps():
    return [
        ("preprocess", build_preprocessor()),
        ("select", build_selector(percentile=10)),
        ("dense", Densifier()),
    ]


def make_pipeline(clf, sampler=None) -> ImbPipeline:
    """Preprocessor -> chi2 percentile filter -> optional oversampler -> model.

    The sampler sits *after* the transformer steps inside an imblearn
    Pipeline, so resampling happens only on each training fold, never
    on the validation fold or the held-out test set.
    """
    steps = _base_steps()
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", clf))
    return ImbPipeline(steps=steps)


def experiment_specs() -> list[dict]:
    """Seven experiments: RF baseline, class weights, oversampling, linear models."""
    rf_kw = dict(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE)
    logreg_kw = dict(max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE)
    svc_kw = dict(max_iter=4000, dual="auto", random_state=RANDOM_STATE)

    def ros():
        return RandomOverSampler(random_state=RANDOM_STATE)

    def smote():
        # k_neighbors=1 because the smallest class has only a handful of
        # training rows; the default of 5 would raise.
        return SMOTE(k_neighbors=1, random_state=RANDOM_STATE)

    return [
        {
            "name": "rf_baseline",
            "model": "RandomForest",
            "handling": "none (course-style baseline)",
            "pipeline": make_pipeline(RandomForestClassifier(**rf_kw)),
            "param_grid": {"select__percentile": PERCENTILE_GRID},
        },
        {
            "name": "rf_balanced",
            "model": "RandomForest",
            "handling": "class_weight='balanced'",
            "pipeline": make_pipeline(
                RandomForestClassifier(**rf_kw, class_weight="balanced")
            ),
            "param_grid": {"select__percentile": PERCENTILE_GRID},
        },
        {
            "name": "rf_ros",
            "model": "RandomForest",
            "handling": "RandomOverSampler (train folds only)",
            "pipeline": make_pipeline(RandomForestClassifier(**rf_kw), sampler=ros()),
            "param_grid": {"select__percentile": PERCENTILE_GRID},
        },
        {
            "name": "logreg_balanced",
            "model": "LogisticRegression",
            "handling": "class_weight='balanced'",
            "pipeline": make_pipeline(LogisticRegression(**logreg_kw, class_weight="balanced")),
            "param_grid": {
                "select__percentile": PERCENTILE_GRID,
                "model__C": LINEAR_C_GRID,
            },
        },
        {
            "name": "logreg_smote",
            "model": "LogisticRegression",
            "handling": "SMOTE (k=1, train folds only)",
            "pipeline": make_pipeline(LogisticRegression(**logreg_kw), sampler=smote()),
            "param_grid": {
                "select__percentile": PERCENTILE_GRID,
                "model__C": LINEAR_C_GRID,
            },
        },
        {
            "name": "linearsvc_balanced",
            "model": "LinearSVC",
            "handling": "class_weight='balanced'",
            "pipeline": make_pipeline(LinearSVC(**svc_kw, class_weight="balanced")),
            "param_grid": {
                "select__percentile": PERCENTILE_GRID,
                "model__C": LINEAR_C_GRID,
            },
        },
        {
            "name": "linearsvc_ros",
            "model": "LinearSVC",
            "handling": "RandomOverSampler (train folds only)",
            "pipeline": make_pipeline(LinearSVC(**svc_kw), sampler=ros()),
            "param_grid": {
                "select__percentile": PERCENTILE_GRID,
                "model__C": LINEAR_C_GRID,
            },
        },
    ]


def fit_experiment(spec: dict, x_train, y_train) -> GridSearchCV:
    """Small GridSearchCV, scored by macro-F1 (better than accuracy under imbalance)."""
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=spec["pipeline"],
        param_grid=spec["param_grid"],
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(x_train, y_train)
    return search
