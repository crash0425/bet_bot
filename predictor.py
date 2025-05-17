# predictor.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load
import os
from flask import Flask

app = Flask(__name__)

# --- Dummy MLB Data Generator ---
def generate_mlb_data():
    np.random.seed(42)
    data = pd.DataFrame({
        'home_team_win_pct': np.random.rand(100),
        'away_team_win_pct': np.random.rand(100),
        'home_pitcher_era': np.random.rand(100) * 5,
        'away_pitcher_era': np.random.rand(100) * 5,
        'home_team_win': np.random.randint(0, 2, 100)
    })
    return data

# --- Dummy NFL Data Generator ---
def generate_nfl_data():
    np.random.seed(43)
    data = pd.DataFrame({
        'home_team_win_pct': np.random.rand(100),
        'away_team_win_pct': np.random.rand(100),
        'home_qb_rating': np.random.rand(100) * 100,
        'away_qb_rating': np.random.rand(100) * 100,
        'home_team_win': np.random.randint(0, 2, 100)
    })
    return data

# --- Train MLB Model ---
def train_mlb_model():
    data = generate_mlb_data()
    X = data.drop('home_team_win', axis=1)
    y = data['home_team_win']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("MLB Model Accuracy:", accuracy_score(y_test, y_pred))

    dump(model, 'mlb_model.pkl')

# --- Train NFL Model ---
def train_nfl_model():
    data = generate_nfl_data()
    X = data.drop('home_team_win', axis=1)
    y = data['home_team_win']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=43)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("NFL Model Accuracy:", accuracy_score(y_test, y_pred))

    dump(model, 'nfl_model.pkl')

# --- Predict Today's MLB Games ---
def predict_mlb_games():
    if not os.path.exists('mlb_model.pkl'):
        print("MLB model not found. Training now.")
        train_mlb_model()

    model = load('mlb_model.pkl')

    today_games = pd.DataFrame({
        'home_team_win_pct': [0.6, 0.4],
        'away_team_win_pct': [0.5, 0.7],
        'home_pitcher_era': [3.2, 4.1],
        'away_pitcher_era': [4.0, 3.5]
    })

    predictions = model.predict(today_games)
    result = f"MLB Predictions (1 = Home win, 0 = Away win): {predictions.tolist()}"
    print(result)
    return result

# --- Predict Today's NFL Games ---
def predict_nfl_games():
    if not os.path.exists('nfl_model.pkl'):
        print("NFL model not found. Training now.")
        train_nfl_model()

    model = load('nfl_model.pkl')

    today_games = pd.DataFrame({
        'home_team_win_pct': [0.7, 0.5],
        'away_team_win_pct': [0.4, 0.6],
        'home_qb_rating': [95.2, 88.5],
        'away_qb_rating': [89.1, 91.0]
    })

    predictions = model.predict(today_games)
    result = f"NFL Predictions (1 = Home win, 0 = Away win): {predictions.tolist()}"
    print(result)
    return result

@app.route("/")
def index():
    mlb_result = predict_mlb_games()
    nfl_result = predict_nfl_games()
    return f"<h1>Daily Sports Predictions</h1><p>{mlb_result}</p><p>{nfl_result}</p>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
