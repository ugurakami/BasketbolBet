"""
Basketball Betting Analytics System - V4.2 (Optimal & Final)
Author: AI Assistant
Description: Optimal ML system combining Regressor-based Limit Line prediction and Calibrated Classifiers for Confidence-Based Staking.
Version: 4.2 - Optimal combination of all fixes and CalibratedCV for best probability estimates.
"""

import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV # Entegre kalibrasyon
import warnings
warnings.filterwarnings('ignore')

# =================================================================
#                          CONFIGURATION
# =================================================================

# Data Settings
FILE_NAME = "BasketbolFikstür - Sayfa1.tsv"

# Model Settings
MIN_PROBABILITY_THRESHOLD = 0.60  # Minimum confidence threshold
OVER_UNDER_MARGIN_THRESHOLD = 5   # Expected score margin for Over/Under (points)
RECENT_MATCHES_COUNT = 8          # Matches used for Feature Engineering
MIN_ACCURACY_THRESHOLD = 0.55

# Risk Management (Dynamic Percentage Assignment instead of Kelly)
MAX_BANKROLL_PERCENTAGE = 2.0     # Maximum risk on a single bet
STAKE_SIZES = {                   # Bet percentage assigned by probability
    (0.80, 1.00): MAX_BANKROLL_PERCENTAGE,
    (0.70, 0.80): 1.5,
    (0.60, 0.70): 1.0,
    (0.00, 0.60): 0.0
}

# Telegram Settings
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# =================================================================
#                          UTILITY FUNCTIONS
# =================================================================

def calculate_stake_percentage(probability):
    """Apply confidence-based fixed percentage stake size"""
    for (low, high), stake in STAKE_SIZES.items():
        if low <= probability < high:
            return stake
    return 0.0

def validate_model_quality(side_accuracy, over_accuracy, score_r2):
    """Model kalitesini kontrol et"""
    if side_accuracy < MIN_ACCURACY_THRESHOLD or over_accuracy < MIN_ACCURACY_THRESHOLD or score_r2 < 0.0:
        print(f"⚠️  Model accuracy ({side_accuracy:.3f}, {over_accuracy:.3f}, R2: {score_r2:.2f}) below threshold. Proceed with caution.")
        return False
    print("✅ Model accuracies are above threshold.")
    return True

def send_telegram_message(df_bets, analysis_date):
    """Send analysis results via Telegram"""
    print(f"📱 Preparing Telegram message for {analysis_date}...")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Telegram credentials not configured.")
        return

    if df_bets.empty:
        message = f"🏀 *Basketball Betting Analysis - {analysis_date}*\n\n"
        message += "No valuable bets found today based on Confidence Threshold 🚫"
    else:
        message = f"💰 *Basketball Betting Recommendations - {analysis_date}* 🏀\n\n"

        matches = {}
        for _, bet in df_bets.iterrows():
            match_name = bet['Match']
            if match_name not in matches:
                matches[match_name] = []
            matches[match_name].append(bet)

        for match_name, bets in matches.items():
            limit_line = bets[0]['Limit_Line']
            limit_line_int = int(round(limit_line / 2.5) * 2.5) # Yuvarlanmış Limit Line

            message += f"• *{match_name}*\n"
            message += f"  _Limit: {limit_line_int} (Model Avg Score)_\n"

            for bet in bets:
                bet_type = "🎯" if bet['Bet_Type'] == 'Side' else "📊"
                message += f"  {bet_type} {bet['Selection']}\n"
                message += f"    - Model: {bet['Model_Probability']:.1%}\n"
                message += f"    - *STAKE: {bet['Stake_Percentage']:.1f}%*\n"
                
            message += "  ────────\n"

        total_bets = len(df_bets)
        high_confidence = len(df_bets[df_bets['Model_Probability'] >= 0.80])
        message += f"\n*Total Recommendations: {total_bets}*"
        message += f"\n*High Confidence (>=80%): {high_confidence}*"

    telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'
    }

    try:
        requests.post(telegram_url, data=payload, timeout=30)
        print("✅ Telegram message sent successfully!")
    except requests.exceptions.RequestException as e:
        print(f"❌ Telegram connection error: {e}")

def safe_label_encode(train_series, test_series):
    """Safe label encoding with unknown value handling"""
    le = LabelEncoder()
    train_encoded = le.fit_transform(train_series.astype(str))

    test_encoded = []
    for val in test_series.astype(str):
        if val in le.classes_:
            test_encoded.append(le.transform([val])[0])
        else:
            test_encoded.append(-1)

    return train_encoded, np.array(test_encoded)

def calibrate_limit_line(predicted_scores):
    """Limit line'ı 2.5'in katlarına yuvarla ve sınırla"""
    calibrated = np.round(predicted_scores / 2.5) * 2.5
    return np.clip(calibrated, 150.0, 230.0)

# =================================================================
#                          FEATURE ENGINEERING (FH Ratio Eklendi)
# =================================================================

def calculate_team_offensive_rating(team, date, historical_data, n_matches):
    """Calculate team offensive and defensive ratings"""
    home_matches = historical_data[
        (historical_data['Home_Team'] == team) &
        (historical_data['Date'] < date)
    ].tail(n_matches)

    away_matches = historical_data[
        (historical_data['Away_Team'] == team) &
        (historical_data['Date'] < date)
    ].tail(n_matches)

    total_points = home_matches['Home_Score'].sum() + away_matches['Away_Score'].sum()
    total_matches = len(home_matches) + len(away_matches)

    conceded_points = home_matches['Away_Score'].sum() + away_matches['Home_Score'].sum()

    offensive_rating = total_points / total_matches if total_matches > 0 else 90.0
    defensive_rating = conceded_points / total_matches if total_matches > 0 else 90.0

    return offensive_rating, defensive_rating

def calculate_team_form(team, date, historical_data, n_matches):
    """Calculate team form (win rate) and First Half Ratio (Yeni)"""
    home_matches = historical_data[
        (historical_data['Home_Team'] == team) &
        (historical_data['Date'] < date)
    ].tail(n_matches)

    away_matches = historical_data[
        (historical_data['Away_Team'] == team) &
        (historical_data['Date'] < date)
    ].tail(n_matches)

    home_wins = (home_matches['Winning_Side'] == 1).sum()
    away_wins = (away_matches['Winning_Side'] == 0).sum()
    total_wins = home_wins + away_wins
    total_matches = len(home_matches) + len(away_matches)

    # Yeni: İlk Yarı Oranı
    total_score = home_matches['Home_Score'].sum() + away_matches['Away_Score'].sum()
    total_fh_score = home_matches['Home_FH_Score'].sum() + away_matches['Away_FH_Score'].sum()
    fh_score_ratio = total_fh_score / total_score if total_score > 0 else 0.5 # Default 0.5

    return total_wins / total_matches if total_matches > 0 else 0.5, fh_score_ratio

def create_features(df, historical_data, n_matches):
    """Create comprehensive features for model training - FIXED FEATURE ASSIGNMENT"""
    df_features = df.copy()
    df_features = df_features.reset_index(drop=True)

    # 1. Rest Days Calculation
    all_matches = pd.concat([
        df_features[['Date', 'Home_Team']].rename(columns={'Home_Team': 'Team'}),
        df_features[['Date', 'Away_Team']].rename(columns={'Away_Team': 'Team'})
    ]).sort_values('Date').reset_index(drop=True)

    all_matches['Previous_Date'] = all_matches.groupby('Team')['Date'].shift(1)
    all_matches['Rest_Days'] = (all_matches['Date'] - all_matches['Previous_Date']).dt.days

    df_features = pd.merge(df_features, all_matches[['Date', 'Team', 'Rest_Days']].rename(
        columns={'Team': 'Home_Team', 'Rest_Days': 'Home_Rest_Days'}),
        on=['Date', 'Home_Team'], how='left')
    df_features = pd.merge(df_features, all_matches[['Date', 'Team', 'Rest_Days']].rename(
        columns={'Team': 'Away_Team', 'Rest_Days': 'Away_Rest_Days'}),
        on=['Date', 'Away_Team'], how='left')

    df_features['Home_Rest_Days'] = df_features['Home_Rest_Days'].fillna(7)
    df_features['Away_Rest_Days'] = df_features['Away_Rest_Days'].fillna(7)
    df_features['Rest_Days_Diff'] = df_features['Home_Rest_Days'] - df_features['Away_Rest_Days']

    # 2. Advanced Team Ratings - FIX: Row-wise application to correctly use historical_data
    ratings_data = []
    for index, row in df_features.iterrows():
        h_off, h_def = calculate_team_offensive_rating(row['Home_Team'], row['Date'], historical_data, n_matches)
        a_off, a_def = calculate_team_offensive_rating(row['Away_Team'], row['Date'], historical_data, n_matches)
        
        # 3. Team Form and First Half Ratio
        h_form, h_fh_ratio = calculate_team_form(row['Home_Team'], row['Date'], historical_data, n_matches)
        a_form, a_fh_ratio = calculate_team_form(row['Away_Team'], row['Date'], historical_data, n_matches)
        
        ratings_data.append([h_off, h_def, a_off, a_def, h_form, a_form, h_fh_ratio, a_fh_ratio])

    ratings_df = pd.DataFrame(
        ratings_data,
        columns=[
            'Home_Offensive_Rating', 'Home_Defensive_Rating', 
            'Away_Offensive_Rating', 'Away_Defensive_Rating',
            'Home_Team_Form', 'Away_Team_Form',
            'Home_FH_Ratio', 'Away_FH_Ratio'
        ],
        index=df_features.index
    )
    
    df_features = pd.concat([df_features.drop(columns=['Home_Stats', 'Away_Stats']), ratings_df], axis=1) # Home_Stats/Away_Stats çıkarıldı

    # 4. Expected Total Score and Strength Differences
    df_features['Expected_Total_Score'] = (
        df_features['Home_Offensive_Rating'] + df_features['Away_Offensive_Rating'])

    df_features['Offensive_Strength_Diff'] = (
        df_features['Home_Offensive_Rating'] - df_features['Away_Offensive_Rating'])

    df_features['Defensive_Strength_Diff'] = (
        df_features['Home_Defensive_Rating'] - df_features['Away_Defensive_Rating'])

    return df_features

# =================================================================
#                          MAIN PIPELINE FUNCTIONS
# =================================================================

def load_and_clean_data(file_path):
    """Load and clean basketball data, extract First Half scores (FH Score eklendi)"""
    try:
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    except:
        try:
            df = pd.read_csv(file_path, sep='\t', encoding='latin-1')
        except:
            print(f"❌ Error: Could not read file with any encoding")
            return None

    df = df.rename(columns={
        'MS(Ev)': 'Home_Score', 'MS(Dep)': 'Away_Score',
        'İY(Ev)': 'Home_FH_Score_Str', 'İY(Dep)': 'Away_FH_Score_Str',
        'Ev Sahibi': 'Home_Team', 'Deplasman': 'Away_Team',
        'Tarih': 'Date', 'Lig': 'League'
    })

    df['Home_Score'] = pd.to_numeric(df['Home_Score'], errors='coerce')
    df['Away_Score'] = pd.to_numeric(df['Away_Score'], errors='coerce')

    # Yeni: İlk Yarı skorlarını temizle ve dönüştür
    def safe_fh_score(score_str):
        try:
            return int(str(score_str).split(' ')[0])
        except:
            return np.nan

    df['Home_FH_Score'] = df['Home_FH_Score_Str'].apply(safe_fh_score)
    df['Away_FH_Score'] = df['Away_FH_Score_Str'].apply(safe_fh_score)

    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
    df = df[df['Date'].notna()].sort_values('Date').copy()

    # Maç tamamlanma maskesi: Final skoru, İlk Yarı skoru ve Tarih var olmalı.
    mask_completed = (
        df['Home_Score'].notna() & df['Away_Score'].notna() &
        df['Home_FH_Score'].notna() & df['Away_FH_Score'].notna()
    )

    df_valid = df[mask_completed].copy()
    df_future = df[~mask_completed].copy() # Gelecek maçlar veya eksik skorlu maçlar

    # Eğitim verisi için hedefleri oluştur
    df_valid['Total_Score'] = df_valid['Home_Score'] + df_valid['Away_Score']
    df_valid['Winning_Side'] = (df_valid['Home_Score'] > df_valid['Away_Score']).astype(int)

    overall_limit = df_valid['Total_Score'].median() if not df_valid.empty else 180
    df_valid['Over_Line'] = (df_valid['Total_Score'] > overall_limit).astype(int)

    print(f"✅ Training data: {len(df_valid)} records")
    print(f"✅ Future matches: {len(df_future)} records")

    return pd.concat([df_valid, df_future], ignore_index=True)


def prepare_model_data(df, n_matches):
    """Prepare features and targets for model training"""

    historical_data = df[df['Winning_Side'].notna()].copy()
    future_data = df[df['Winning_Side'].isna()].copy()

    if historical_data.empty:
        return None, None, None, None, None, None, None, None # Hata durumunda 8 değer döndürülür

    print("🔄 Creating features for historical data...")
    historical_features = create_features(historical_data, historical_data, n_matches)

    print("🔄 Creating features for future data...")
    future_features = create_features(future_data, historical_data, n_matches) if not future_data.empty else pd.DataFrame()

    # Regressor'ın tahminini Limit Line olarak kullanmak için, gelecekteki maçlar için Limit Line'ı burada doldurmayız.

    # Define feature columns
    feature_columns = [
        'Rest_Days_Diff', 'Home_Team_Form', 'Away_Team_Form', 'Home_FH_Ratio', 'Away_FH_Ratio',
        'Home_Offensive_Rating', 'Away_Offensive_Rating',
        'Home_Defensive_Rating', 'Away_Defensive_Rating',
        'Expected_Total_Score', 'Offensive_Strength_Diff', 'Defensive_Strength_Diff',
        'Home_Team', 'Away_Team', 'League'
    ]

    X_train_full = historical_features[feature_columns].copy().dropna()
    X_predict_full = future_features[feature_columns].copy() if not future_features.empty else pd.DataFrame()

    # Encoding
    for col in ['Home_Team', 'Away_Team', 'League']:
        if not X_train_full.empty and col in X_train_full.columns:
            if not X_predict_full.empty and col in X_predict_full.columns:
                X_train_full[col], X_predict_full[col] = safe_label_encode(X_train_full[col], X_predict_full[col])
            else:
                le = LabelEncoder()
                X_train_full[col] = le.fit_transform(X_train_full[col].astype(str))

    # Regressor için sadece sayısal kolonlar
    numeric_features = [col for col in feature_columns if col not in ['Home_Team', 'Away_Team', 'League']]
    X_train_num = X_train_full[numeric_features].copy()
    X_predict_num = X_predict_full[numeric_features].copy() if not X_predict_full.empty else pd.DataFrame()


    y_side = historical_features.loc[X_train_full.index, 'Winning_Side']
    y_over = historical_features.loc[X_train_full.index, 'Over_Line']
    y_total_score = historical_features.loc[X_train_full.index, 'Total_Score']

    # Return değerleri senkronize edildi (8 değer)
    return X_train_full, X_predict_full, X_train_num, X_predict_num, y_side, y_over, y_total_score, future_data


def train_and_evaluate_models(X_train_full, X_train_num, y_side, y_over, y_total_score):
    """Train and evaluate machine learning models (Classification + Regression)"""
    print("🤖 Training models...")

    # 1. Side Model (Calibrated for probability)
    side_model_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    side_model = CalibratedClassifierCV(side_model_base, cv=3, method='isotonic')

    # 2. Over/Under Model (Calibrated for probability)
    over_model_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    over_model = CalibratedClassifierCV(over_model_base, cv=3, method='isotonic')

    # 3. Total Score Model (Regression - Limit Line Tahmini)
    score_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    tscv = TimeSeriesSplit(n_splits=5)

    # Evaluation
    side_scores = cross_val_score(side_model, X_train_full, y_side, cv=tscv, scoring='accuracy')
    over_scores = cross_val_score(over_model, X_train_full, y_over, cv=tscv, scoring='accuracy')
    score_scores = cross_val_score(score_model, X_train_num, y_total_score, cv=tscv, scoring='r2')

    side_accuracy = side_scores.mean()
    over_accuracy = over_scores.mean()
    score_r2 = score_scores.mean()

    print(f"✅ Side Model Accuracy: {side_accuracy:.3f}")
    print(f"✅ Over/Under Model Accuracy: {over_accuracy:.3f}")
    print(f"✅ Score Regressor R2: {score_r2:.3f}")

    # Train final models
    side_model.fit(X_train_full, y_side)
    over_model.fit(X_train_full, y_over)
    score_model.fit(X_train_num, y_total_score)

    return (side_model, over_model, score_model), (side_accuracy, over_accuracy, score_r2)


def find_valuable_bets(predictions_df, min_probability, margin_threshold):
    """Identify valuable bets using Confidence-Based Staking and Margin Check"""
    valuable_bets = []

    for _, match in predictions_df.iterrows():
        match_name = f"{match['Home_Team']} vs {match['Away_Team']}"
        limit_line = match['Limit_Line']
        expected_score = match['Expected_Total_Score']

        # Side Bets
        for side, prob, selection in [
            ('Home', match['P_Home'], f"{match['Home_Team']} to Win"),
            ('Away', match['P_Away'], f"{match['Away_Team']} to Win")
        ]:
            if prob >= min_probability:
                stake = calculate_stake_percentage(prob)
                if stake > 0.0:
                    valuable_bets.append({
                        'Match': match_name,
                        'Bet_Type': 'Side',
                        'Selection': selection,
                        'Model_Probability': prob,
                        'Stake_Percentage': stake,
                        'Limit_Line': limit_line
                    })

        # Over/Under Bets (Confidence + Margin Check)

        # Over Check: High probability AND Expected score above limit by margin
        if match['P_Over'] >= min_probability and (expected_score - limit_line) >= margin_threshold:
            stake = calculate_stake_percentage(match['P_Over'])
            if stake > 0.0:
                valuable_bets.append({
                    'Match': match_name,
                    'Bet_Type': 'Points Line',
                    'Selection': f"Over {int(limit_line)}",
                    'Model_Probability': match['P_Over'],
                    'Stake_Percentage': stake,
                    'Limit_Line': limit_line
                })

        # Under Check: High probability AND Expected score below limit by margin
        if match['P_Under'] >= min_probability and (limit_line - expected_score) >= margin_threshold:
            stake = calculate_stake_percentage(match['P_Under'])
            if stake > 0.0:
                valuable_bets.append({
                    'Match': match_name,
                    'Bet_Type': 'Points Line',
                    'Selection': f"Under {int(limit_line)}",
                    'Model_Probability': match['P_Under'],
                    'Stake_Percentage': stake,
                    'Limit_Line': limit_line
                })

    return pd.DataFrame(valuable_bets)


def main():
    """Main execution function"""
    print("🏀 Basketball Betting Analytics System - V4.2 (Optimal & Final)")
    print("=" * 60)

    # Load data
    df = load_and_clean_data(FILE_NAME)
    if df is None:
        return

    # Prepare model data
    model_data = prepare_model_data(df, RECENT_MATCHES_COUNT)

    if model_data[0] is None:
        return

    X_train_full, X_predict_full, X_train_num, X_predict_num, y_side, y_over, y_total_score, future_data = model_data

    if X_predict_full.empty:
        print("ℹ️  No upcoming matches to predict")
        return

    # Get this week's matches only
    current_date = datetime.now().date()
    next_week = current_date + timedelta(days=7)
    this_weeks_matches_idx = X_predict_full.index.intersection(
        future_data[
            (future_data['Date'].dt.date >= current_date) &
            (future_data['Date'].dt.date <= next_week)
        ].index
    )

    X_predict_this_week_full = X_predict_full.loc[this_weeks_matches_idx]
    X_predict_this_week_num = X_predict_num.loc[this_weeks_matches_idx]
    this_weeks_matches = future_data.loc[this_weeks_matches_idx].copy()

    if this_weeks_matches.empty:
        print("❌ No matches found for this week")
        return

    print(f"📅 This week's matches: {len(this_weeks_matches)}")

    # Train models
    models, accuracies = train_and_evaluate_models(
        X_train_full, X_train_num, y_side, y_over, y_total_score
    )
    side_model, over_model, score_model = models
    side_accuracy, over_accuracy, score_r2 = accuracies

    # Model Kalite Kontrolü
    validate_model_quality(side_accuracy, over_accuracy, score_r2)

    # Make predictions
    print("🔮 Making predictions for this week...")

    # 1. Limit Line Tahmini (Regresyon Modeli)
    predicted_limit_lines_raw = score_model.predict(X_predict_this_week_num)
    predicted_limit_lines = calibrate_limit_line(predicted_limit_lines_raw)

    # 2. Sınıflandırma ve Kalibrasyon
    side_proba = side_model.predict_proba(X_predict_this_week_full)
    over_proba = over_model.predict_proba(X_predict_this_week_full)

    # Prepare predictions dataframe
    this_weeks_matches['Limit_Line'] = predicted_limit_lines
    this_weeks_matches['P_Home'] = side_proba[:, 1]
    this_weeks_matches['P_Away'] = side_proba[:, 0]
    this_weeks_matches['P_Over'] = over_proba[:, 1]
    this_weeks_matches['P_Under'] = over_proba[:, 0]

    # Analyze each day of the week
    analysis_results = []
    for date in sorted(this_weeks_matches['Date'].dt.date.unique()):
        days_matches = this_weeks_matches[this_weeks_matches['Date'].dt.date == date].copy()

        print(f"\n📅 Analysis Date: {date}")

        # Find valuable bets (Risk ve Marj Kontrolü)
        valuable_bets = find_valuable_bets(days_matches, MIN_PROBABILITY_THRESHOLD, OVER_UNDER_MARGIN_THRESHOLD)

        if valuable_bets.empty:
            print("❌ No valuable bets found")
        else:
            print(f"✅ Found {len(valuable_bets)} valuable bet(s)")
            analysis_results.append((date, valuable_bets))

    # Send Telegram messages
    for date, valuable_bets in analysis_results:
        if not valuable_bets.empty:
            send_telegram_message(valuable_bets, date)

    print("\n🎯 Analysis complete!")

if __name__ == "__main__":
    main()
