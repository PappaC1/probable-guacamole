"""
South African Horse Racing Prediction Tool
For Hollywoodbets Punters' Challenge research.

Outputs, for every race:
1. Probable Winner - highest model win probability
2. Value Bet       - best positive edge versus market odds
3. Danger Horse    - strongest remaining challenger

The best selection is highlighted first, followed by the second choice.

IMPORTANT:
This is a decision-support tool, not a guarantee of profit.
Horse racing is highly uncertain; never stake money you cannot afford to lose.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


# -----------------------------
# Configuration
# -----------------------------

HISTORICAL_RESULTS_CSV = "sa_historical_results.csv"
TODAYS_RUNNERS_CSV = "todays_runners.csv"
OUTPUT_CSV = "punters_challenge_selections.csv"
MODEL_TYPE = "lightgbm"  # "lightgbm" or "logistic"

# A runner is considered a Value Bet when:
# model_win_probability - implied_probability_from_odds >= VALUE_EDGE_THRESHOLD
VALUE_EDGE_THRESHOLD = 0.03

# Minimum decimal odds for a Value Bet; avoids recommending very short-priced horses
MIN_VALUE_BET_ODDS = 2.0

RANDOM_STATE = 42


# -----------------------------
# Expected input format
# -----------------------------
# Both historical_results.csv and todays_runners.csv should use these columns.
#
# race_id              Unique race identifier, e.g. "2026-11-20_TUR_03"
# race_date            YYYY-MM-DD
# course               e.g. Turffontein, Greyville, Kenilworth
# race_number          1, 2, 3...
# distance_m           Race distance in metres
# going                e.g. Good, Good to Soft, Soft
# class                Race class, e.g. 1, 2, 3, Maiden, handicap
# race_type            e.g. Handicap, Maiden, Conditions, Stakes
# horse_name
# horse_id             Stable identifier if available; otherwise horse_name
# trainer_name
# jockey_name
# draw                 Barrier/stall number
# weight_kg            Carried weight in kilograms
# age                  Horse age
# sex                  e.g. C, F, G, H, M
# days_since_last_run  Days since previous run; blank if unknown
# runs_last_90_days
# wins_last_12_months
# places_last_12_months
# starts_last_12_months
# last_3_positions     Comma-separated, newest first, e.g. "1,4,2"
# last_win_days_ago
# official_rating      Optional
# market_odds          Decimal odds, e.g. 4.5; use your best available pre-race price
# finished_position    Historical results only; 1 = won, 2 = second, etc.
# 
# For today's runners, leave finished_position blank.


REQUIRED_COLUMNS = [
    "race_id", "race_date", "course", "race_number", "distance_m", "going",
    "class", "race_type", "horse_name", "horse_id", "trainer_name",
    "jockey_name", "draw", "weight_kg", "age", "sex",
    "days_since_last_run", "runs_last_90_days", "wins_last_12_months",
    "places_last_12_months", "starts_last_12_months", "last_3_positions",
    "last_win_days_ago", "official_rating", "market_odds",
    "finished_position"
]


# -----------------------------
# Data helpers
# -----------------------------

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Create it using the column structure in the script comments."
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df


def parse_last_3(value) -> list[int]:
    """Convert '1,4,2' into [1, 4, 2]. Missing or DNF values become 99."""
    if pd.isna(value) or str(value).strip() == "":
        return [99, 99, 99]
    parts = str(value).replace(" ", "").split(",")
    result = []
    for p in parts[:3]:
        try:
            result.append(int(float(p)))
        except ValueError:
            result.append(99)
    while len(result) < 3:
        result.append(99)
    return result


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["distance_m"] = pd.to_numeric(df["distance_m"], errors="coerce")
    df["draw"] = pd.to_numeric(df["draw"], errors="coerce")
    df["weight_kg"] = pd.to_numeric(df["weight_kg"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["market_odds"] = pd.to_numeric(df["market_odds"], errors="coerce")

    for col in [
        "days_since_last_run", "runs_last_90_days", "wins_last_12_months",
        "places_last_12_months", "starts_last_12_months",
        "last_win_days_ago", "official_rating"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    positions = df["last_3_positions"].apply(parse_last_3)
    df["last_finish"] = [p[0] for p in positions]
    df["second_last_finish"] = [p[1] for p in positions]
    df["third_last_finish"] = [p[2] for p in positions]

    df["recent_form_score"] = (
        (4 - df["last_finish"].clip(upper=4)) * 3
        + (4 - df["second_last_finish"].clip(upper=4)) * 2
        + (4 - df["third_last_finish"].clip(upper=4))
    )

    df["win_rate_12m"] = df["wins_last_12_months"] / df[
        "starts_last_12_months"
    ].replace(0, np.nan)
    df["place_rate_12m"] = df["places_last_12_months"] / df[
        "starts_last_12_months"
    ].replace(0, np.nan)

    df["implied_probability"] = 1.0 / df["market_odds"].replace(0, np.nan)
    df["log_odds"] = np.log(df["market_odds"].clip(lower=1.01))

    # Race-level market rank: 1 = favourite
    df["market_rank"] = df.groupby("race_id")["market_odds"].rank(
        method="min"
    )
    df["field_size"] = df.groupby("race_id")["horse_name"].transform("count")

    # Course/distance specialisation, calculated only from historical rows
    hist = df[df["finished_position"].notna()].copy()
    hist["won"] = (hist["finished_position"] == 1).astype(int)

    course_distance_win = (
        hist.groupby(["course", "distance_m"])["won"].mean()
        .rename("course_distance_win_rate")
    )
    course_win = (
        hist.groupby("course")["won"].mean().rename("course_win_rate")
    )

    df = df.merge(course_distance_win, on=["course", "distance_m"], how="left")
    df = df.merge(course_win, on="course", how="left")

    global_win_rate = hist["won"].mean() if len(hist) else 0.12
    df["course_distance_win_rate"] = df["course_distance_win_rate"].fillna(
        global_win_rate
    )
    df["course_win_rate"] = df["course_win_rate"].fillna(global_win_rate)

    # Avoid lookahead: trainer/jockey strike rates should ideally be calculated
    # using only races before each race_date in a full production system.
    trainer_win = (
        hist.groupby("trainer_name")["won"].mean()
        .rename("trainer_win_rate")
    )
    jockey_win = (
        hist.groupby("jockey_name")["won"].mean().rename("jockey_win_rate")
    )

    df = df.merge(trainer_win, on="trainer_name", how="left")
    df = df.merge(jockey_win, on="jockey_name", how="left")

    df["trainer_win_rate"] = df["trainer_win_rate"].fillna(global_win_rate)
    df["jockey_win_rate"] = df["jockey_win_rate"].fillna(global_win_rate)

    return df


NUMERIC_FEATURES = [
    "distance_m", "draw", "weight_kg", "age", "days_since_last_run",
    "runs_last_90_days", "wins_last_12_months", "places_last_12_months",
    "starts_last_12_months", "last_finish", "second_last_finish",
    "third_last_finish", "recent_form_score", "win_rate_12m",
    "place_rate_12m", "log_odds", "implied_probability", "market_rank",
    "field_size", "course_distance_win_rate", "course_win_rate",
    "trainer_win_rate", "jockey_win_rate", "last_win_days_ago",
    "official_rating"
]

CATEGORICAL_FEATURES = [
    "course", "going", "class", "race_type", "sex", "trainer_name",
    "jockey_name"
]


# -----------------------------
# Model
# -----------------------------

def make_model(model_type: str):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", "passthrough"),
    ])

    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])

    if model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
        classifier = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=30,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            verbose=-1,
        )
    else:
        classifier = LogisticRegression(
            max_iter=2000,
            C=0.5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    return Pipeline([
        ("features", preprocessor),
        ("model", classifier),
    ])


def train_and_predict(
    history: pd.DataFrame,
    today: pd.DataFrame,
    model_type: str,
) -> tuple[pd.DataFrame, dict]:
    history = history[history["finished_position"].notna()].copy()
    history["target"] = (history["finished_position"] == 1).astype(int)

    if history["target"].nunique() < 2:
        raise ValueError(
            "Historical data must contain both winners and non-winners."
        )

    X_history = history[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_history = history["target"]

    groups = history["race_id"]
    model = make_model(model_type)

    # Time-ordered validation would be preferable for final deployment.
    # GroupKFold prevents horses/runners from the same race appearing in both
    # training and validation folds.
    gkf = GroupKFold(n_splits=5)
    oof_probs = np.zeros(len(history))

    for train_idx, valid_idx in gkf.split(X_history, y_history, groups):
        fold_model = make_model(model_type)
        fold_model.fit(
            X_history.iloc[train_idx], y_history.iloc[train_idx]
        )
        oof_probs[valid_idx] = fold_model.predict_proba(
            X_history.iloc[valid_idx]
        )[:, 1]

    metrics = {
        "historical_rows": int(len(history)),
        "winner_rate": float(y_history.mean()),
        "out_of_fold_log_loss": float(log_loss(y_history, oof_probs)),
        "out_of_fold_brier_score": float(
            brier_score_loss(y_history, oof_probs)
        ),
    }

    model.fit(X_history, y_history)

    X_today = today[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    today["model_win_probability"] = model.predict_proba(X_today)[:, 1]

    # Normalise within each race so probabilities sum to 1.
    today["model_win_probability"] = today.groupby("race_id")[
        "model_win_probability"
    ].transform(lambda s: s / s.sum())

    today["model_edge"] = (
        today["model_win_probability"] - today["implied_probability"]
    )

    return today, metrics


# -----------------------------
# Selection logic
# -----------------------------

@dataclass
class Selection:
    race_id: str
    race_number: int
    course: str
    selection_type: str
    horse_name: str
    model_win_probability: float
    market_odds: float
    model_edge: float
    rank: int
    is_best_selection: bool


def choose_selections(df: pd.DataFrame) -> list[Selection]:
    selections: list[Selection] = []

    for race_id, race in df.groupby("race_id", sort=False):
        race = race.sort_values(
            ["model_win_probability", "market_odds"],
            ascending=[False, True],
        ).reset_index(drop=True)

        if race.empty:
            continue

        probable_winner = race.iloc[0]

        value_candidates = race[
            (race["model_edge"] >= VALUE_EDGE_THRESHOLD)
            & (race["market_odds"] >= MIN_VALUE_BET_ODDS)
        ].sort_values("model_edge", ascending=False)

        value_bet = value_candidates.iloc[0] if len(value_candidates) else None

        # Danger horse: second strongest model runner, unless it is the same
        # horse as the value bet; then choose the next best available runner.
        danger_candidates = race.iloc[1:].copy()

        if value_bet is not None:
            danger_candidates = danger_candidates[
                danger_candidates["horse_name"] != value_bet["horse_name"]
            ]

        danger_horse = (
            danger_candidates.iloc[0]
            if len(danger_candidates)
            else race.iloc[-1]
        )

        race_selections = [
            Selection(
                race_id=race_id,
                race_number=int(race["race_number"].iloc[0]),
                course=race["course"].iloc[0],
                selection_type="Probable Winner",
                horse_name=probable_winner["horse_name"],
                model_win_probability=float(
                    probable_winner["model_win_probability"]
                ),
                market_odds=float(probable_winner["market_odds"]),
                model_edge=float(probable_winner["model_edge"]),
                rank=1,
                is_best_selection=True,
            )
        ]

        if value_bet is not None:
            race_selections.append(
                Selection(
                    race_id=race_id,
                    race_number=int(race["race_number"].iloc[0]),
                    course=race["course"].iloc[0],
                    selection_type="Value Bet",
                    horse_name=value_bet["horse_name"],
                    model_win_probability=float(
                        value_bet["model_win_probability"]
                    ),
                    market_odds=float(value_bet["market_odds"]),
                    model_edge=float(value_bet["model_edge"]),
                    rank=2,
                    is_best_selection=False,
                )
            )

        race_selections.append(
            Selection(
                race_id=race_id,
                race_number=int(race["race_number"].iloc[0]),
                course=race["course"].iloc[0],
                selection_type="Danger Horse",
                horse_name=danger_horse["horse_name"],
                model_win_probability=float(
                    danger_horse["model_win_probability"]
                ),
                market_odds=float(danger_horse["market_odds"]),
                model_edge=float(danger_horse["model_edge"]),
                rank=3,
                is_best_selection=False,
            )
        )

        selections.extend(race_selections)

    return selections


def print_race_card(selections: list[Selection]) -> None:
    current_race = None

    for sel in selections:
        if sel.race_id != current_race:
            current_race = sel.race_id
            print("\n" + "=" * 72)
            print(
                f"RACE {sel.race_number} | {sel.course} | {sel.race_id}"
            )
            print("=" * 72)

        marker = "★ BEST SELECTION" if sel.is_best_selection else "  "
        print(
            f"{marker} | {sel.selection_type:<16} | "
            f"{sel.horse_name:<25} | "
            f"Win prob: {sel.model_win_probability:.1%} | "
            f"Odds: {sel.market_odds:.2f} | "
            f"Edge: {sel.model_edge:+.1%}"
        )


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    history = load_csv(HISTORICAL_RESULTS_CSV)
    today = load_csv(TODAYS_RUNNERS_CSV)

    history = build_features(history)
    today = build_features(today)

    predictions, metrics = train_and_predict(
        history, today, MODEL_TYPE
    )

    selections = choose_selections(predictions)

    output = pd.DataFrame([s.__dict__ for s in selections])
    output.to_csv(OUTPUT_CSV, index=False)

    print("\nMODEL VALIDATION")
    print(json.dumps(metrics, indent=2))

    print("\nPUNTERS' CHALLENGE SELECTIONS")
    print_race_card(selections)
    print(f"\nSaved selections to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
