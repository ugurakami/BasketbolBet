"""
Basketball Betting Analytics System
Author: AI Assistant
Description: Machine learning system for basketball match predictions and betting value detection
Version: 1.0
"""

import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# =================================================================
#                         CONFIGURATION
# =================================================================

# Data Settings
FILE_NAME = "basketball_data.tsv"

# Model Settings
DEFAULT_ODDS = 1.90
MIN_PROBABILITY_THRESHOLD = 0.55
RECENT_MATCHES_COUNT = 5
MIN_ACCURACY_THRESHOLD = 0.55

# Risk Management
KELLY_FRACTION = 0.25
MAX_BANKROLL_PERCENTAGE = 1.0

# Telegram Settings (Set via GitHub Secrets)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

# =================================================================
#                         UTILITY FUNCTIONS
# =================================================================

def calculate_kelly_criterion(probability, odds):
    """Calculate Kelly Criterion bet size"""
    net_odds = odds - 1
    q = 1 - probability
    
    if (net_odds * probability - q) <= 0:
        return 0.0
    return (net_odds * probability - q) / net_odds

def fractional_kelly_bet_size(full_kelly_percentage, fraction=KELLY_FRACTION, max_percent=MAX_BANKROLL_PERCENTAGE):
    """Apply fractional Kelly and bankroll limits"""
    fractional = full_kelly_percentage * fraction
    return min(fractional, max_percent)

def send_telegram_message(df_bets, analysis_date):
    """Send analysis results via Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials not configured. Skipping message.")
        return

    if df_bets.empty:
        message = f"🏀 *Basketball Betting Analysis - {analysis_date}*\n\n"
        message += "No valuable bets found today based on Kelly Criterion. 🚫"
    else:
        message = f"💰 *Basketball Betting Recommendations - {analysis_date}*\n\n"
        
        for _, row in df_bets.iterrows():
            message += f"• *Match:* {row['Match']}\n"
            message += f"  - *Bet:* {row['Selection']}\n"
            message += f"  - *Model Probability:* {row['Model_Probability']:.1%}\n"
            message += f"  - *Bet Size:* {row['Kelly_Percentage']:.1f}% of bankroll\n"
            message += f"  - *Odds:* {row['Odds']:.2f}\n"
            message += "────────────────────\n"

    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'
    }
    
    try:
        response = requests.post(TELEGRAM_API_URL, data=payload, timeout=10)
        response.raise_for_status()
        print("✅ Telegram message sent successfully")
    except requests.exceptions.RequestException as e:
        print(f"❌ Telegram error: {e}")

# =================================================================
#                         FEATURE ENGINEERING
# =================================================================

def safe_label_encode(train_series, test_series):
    """Safe label encoding with unknown value handling"""
    le = LabelEncoder()
    train_encoded = le.fit_transform(train_series.astype(str))
    
    test_encoded = []
    for val in test_series.astype(str):
        if val in le.classes_:
            test_encoded.append(le.transform([val])[0])
        else:
            test_encoded.append(-1)  # Unknown value
            
    return train_encoded, np.array(test_encoded)

def calculate_team_form(team, date, is_home, historical_data, n_matches):
    """Calculate team form based on recent matches"""
    if is_home:
        matches = historical_data[
            (historical_data['Home_Team'] == team) & 
            (historical_data['Date'] < date)
        ].tail(n_matches)
        wins = (matches['Winning_Side'] == 1).sum()
    else:
        matches = historical_data[
            (historical_data['Away_Team'] == team) & 
            (historical_data['Date'] < date)
        ].tail(n_matches)
        wins = (matches['Winning_Side'] == 0).sum()
    
    return wins / len(matches) if len(matches) > 0 else 0.5

def calculate_h2h_record(home_team, away_team, date, historical_data, n_matches):
    """Calculate head-to-head record between teams"""
    h2h_matches = historical_data[
        ((historical_data['Home_Team'] == home_team) & (historical_data['Away_Team'] == away_team)) |
        ((historical_data['Home_Team'] == away_team) & (historical_data['Away_Team'] == home_team))
    ].query('Date < @date').tail(n_matches)
    
    if len(h2h_matches) == 0:
        return 0.5
    
    home_wins = h2h_matches.apply(
        lambda row: 1 if (
            (row['Home_Team'] == home_team and row['Winning_Side'] == 1) or
            (row['Away_Team'] == home_team and row['Winning_Side'] == 0)
        ) else 0, axis=1
    ).sum()
    
    return home_wins / len(h2h_matches)

def calculate_rest_days(df):
    """Calculate rest days between matches for each team"""
    home_matches = df[['Date', 'Home_Team']].rename(columns={'Home_Team': 'Team'})
    away_matches = df[['Date', 'Away_Team']].rename(columns={'Away_Team': 'Team'})
    
    all_matches = pd.concat([home_matches, away_matches]).sort_values('Date').reset_index(drop=True)
    all_matches['Previous_Date'] = all_matches.groupby('Team')['Date'].shift(1)
    all_matches['Rest_Days'] = (all_matches['Date'] - all_matches['Previous_Date']).dt.days
    
    return all_matches

def create_features(df, historical_data, n_matches):
    """Create comprehensive features for model training"""
    df_features = df.copy()
    
    # Rest days calculation
    rest_days_data = calculate_rest_days(df_features)
    
    home_rest = rest_days_data[['Date', 'Team', 'Rest_Days']].rename(
        columns={'Team': 'Home_Team', 'Rest_Days': 'Home_Rest_Days'})
    away_rest = rest_days_data[['Date', 'Team', 'Rest_Days']].rename(
        columns={'Team': 'Away_Team', 'Rest_Days': 'Away_Rest_Days'})
    
    df_features = pd.merge(df_features, home_rest, on=['Date', 'Home_Team'], how='left')
    df_features = pd.merge(df_features, away_rest, on=['Date', 'Away_Team'], how='left')
    
    df_features['Home_Rest_Days'] = df_features['Home_Rest_Days'].fillna(7)
    df_features['Away_Rest_Days'] = df_features['Away_Rest_Days'].fillna(7)
    df_features['Rest_Days_Diff'] = df_features['Home_Rest_Days'] - df_features['Away_Rest_Days']
    
    # Team form calculations
    df_features['Home_Team_Home_Form'] = df_features.apply(
        lambda row: calculate_team_form(row['Home_Team'], row['Date'], True, historical_data, n_matches), axis=1)
    
    df_features['Away_Team_Away_Form'] = df_features.apply(
        lambda row: calculate_team_form(row['Away_Team'], row['Date'], False, historical_data, n_matches), axis=1)
    
    # Head-to-head record
    df_features['H2H_Record'] = df_features.apply(
        lambda row: calculate_h2h_record(row['Home_Team'], row['Away_Team'], row['Date'], historical_data, n_matches), axis=1)
    
    return df_features

# =================================================================
#                         BET ANALYSIS
# =================================================================

def find_valuable_bets(predictions_df, odds, min_probability):
    """Identify valuable bets using Kelly Criterion"""
    valuable_bets = []
    net_odds = odds - 1
    
    for _, match in predictions_df.iterrows():
        match_name = f"{match['Home_Team']} vs {match['Away_Team']}"
        limit_line = int(match['Limit_Line'])
        
        # Home win bet
        if match['P_Home'] > min_probability:
            kelly_full = calculate_kelly_criterion(match['P_Home'], odds)
            kelly_fractional = fractional_kelly_bet_size(kelly_full)
            
            if kelly_fractional > 0.01:  # Minimum bet size threshold
                valuable_bets.append({
                    'Match': match_name,
                    'Bet_Type': 'Side',
                    'Selection': f"{match['Home_Team']} to Win",
                    'Model_Probability': match['P_Home'],
                    'Odds': odds,
                    'Kelly_Percentage': kelly_fractional * 100
                })
        
        # Away win bet
        if match['P_Away'] > min_probability:
            kelly_full = calculate_kelly_criterion(match['P_Away'], odds)
            kelly_fractional = fractional_kelly_bet_size(kelly_full)
            
            if kelly_fractional > 0.01:
                valuable_bets.append({
                    'Match': match_name,
                    'Bet_Type': 'Side',
                    'Selection': f"{match['Away_Team']} to Win",
                    'Model_Probability': match['P_Away'],
                    'Odds': odds,
                    'Kelly_Percentage': kelly_fractional * 100
                })
        
        # Over bet
        if match['P_Over'] > min_probability:
            kelly_full = calculate_kelly_criterion(match['P_Over'], odds)
            kelly_fractional = fractional_kelly_bet_size(kelly_full)
            
            if kelly_fractional > 0.01:
                valuable_bets.append({
                    'Match': match_name,
                    'Bet_Type': 'Points Line',
                    'Selection': f"Over {limit_line}",
                    'Model_Probability': match['P_Over'],
                    'Odds': odds,
                    'Kelly_Percentage': kelly_fractional * 100
                })
        
        # Under bet
        if match['P_Under'] > min_probability:
            kelly_full = calculate_kelly_criterion(match['P_Under'], odds)
            kelly_fractional = fractional_kelly_bet_size(kelly_full)
            
            if kelly_fractional > 0.01:
                valuable_bets.append({
                    'Match': match_name,
                    'Bet_Type': 'Points Line',
                    'Selection': f"Under {limit_line}",
                    'Model_Probability': match['P_Under'],
                    'Odds': odds,
                    'Kelly_Percentage': kelly_fractional * 100
                })
    
    return pd.DataFrame(valuable_bets)

# =================================================================
#                         MAIN PIPELINE
# =================================================================

def load_and_clean_data(file_path):
    """Load and clean basketball data"""
    try:
        df = pd.read_csv(file_path, sep='\t')
        print(f"✅ Data loaded successfully: {len(df)} records")
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found")
        return None
    
    # Column renaming and cleaning
    df = df.rename(columns={
        'MS(Ev)': 'Home_Score',
        'MS(Dep)': 'Away_Score', 
        'İY(Ev)': 'Home_Stats',
        'İY(Dep)': 'Away_Stats',
        'Ev Sahibi': 'Home_Team',
        'Deplasman': 'Away_Team',
        'Tarih': 'Date'
    })
    
    # Data type conversions
    df['Home_Score'] = pd.to_numeric(df['Home_Score'], errors='coerce')
    df['Away_Score'] = pd.to_numeric(df['Away_Score'], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Create target variables
    df['Total_Score'] = df['Home_Score'] + df['Away_Score']
    df['Winning_Side'] = (df['Home_Score'] > df['Away_Score']).astype('float')
    df['Winning_Side'] = df['Winning_Side'].where(df['Home_Score'].notna(), -1)
    
    return df

def prepare_model_data(df, n_matches):
    """Prepare features and targets for model training"""
    # Split data into historical and future matches
    historical_data = df[df['Home_Score'].notna()].copy()
    future_data = df[df['Home_Score'].isna()].copy()
    
    if historical_data.empty:
        print("❌ No historical data for training")
        return None, None, None, None, None, None
    
    # Calculate limit line from historical data only
    limit_line = historical_data['Total_Score'].median()
    historical_data['Over_Line'] = (historical_data['Total_Score'] > limit_line).astype('float')
    
    # Create features
    print("🔄 Creating features...")
    historical_features = create_features(historical_data, historical_data, n_matches)
    
    if not future_data.empty:
        future_features = create_features(future_data, historical_data, n_matches)
    else:
        future_features = pd.DataFrame()
    
    # Define feature columns
    feature_columns = [
        'Rest_Days_Diff', 'Home_Team_Home_Form', 'Away_Team_Away_Form', 
        'H2H_Record', 'Home_Team', 'Away_Team', 'League'
    ]
    
    # Prepare training data
    X_train = historical_features[feature_columns].copy()
    X_train = X_train.dropna()
    
    if X_train.empty:
        print("❌ No valid training data after preprocessing")
        return None, None, None, None, None, None
    
    # Prepare prediction data
    if not future_features.empty:
        X_predict = future_features[feature_columns].copy()
        X_predict = X_predict.dropna()
    else:
        X_predict = pd.DataFrame()
    
    # Encode categorical variables
    for col in ['Home_Team', 'Away_Team', 'League']:
        if not X_train.empty and col in X_train.columns:
            if not X_predict.empty and col in X_predict.columns:
                X_train[col], X_predict[col] = safe_label_encode(X_train[col], X_predict[col])
            else:
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col].astype(str))
    
    y_side = historical_features.loc[X_train.index, 'Winning_Side']
    y_over = historical_features.loc[X_train.index, 'Over_Line']
    
    return X_train, X_predict, y_side, y_over, limit_line, future_features

def train_and_evaluate_models(X_train, y_side, y_over):
    """Train and evaluate machine learning models"""
    print("🤖 Training models...")
    
    # Initialize models
    side_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    over_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Time-series cross validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Evaluate side model
    side_scores = cross_val_score(side_model, X_train, y_side, cv=tscv, scoring='accuracy')
    side_accuracy = side_scores.mean()
    print(f"✅ Side Model Accuracy: {side_accuracy:.3f} (+/- {side_scores.std() * 2:.3f})")
    
    # Evaluate over/under model  
    over_scores = cross_val_score(over_model, X_train, y_over, cv=tscv, scoring='accuracy')
    over_accuracy = over_scores.mean()
    print(f"✅ Over/Under Model Accuracy: {over_accuracy:.3f} (+/- {over_scores.std() * 2:.3f})")
    
    # Check model quality
    if side_accuracy < MIN_ACCURACY_THRESHOLD or over_accuracy < MIN_ACCURACY_THRESHOLD:
        print("⚠️  Model accuracy below threshold. Proceed with caution.")
    
    # Train final models
    side_model.fit(X_train, y_side)
    over_model.fit(X_train, y_over)
    
    return side_model, over_model, side_accuracy, over_accuracy

def main():
    """Main execution function"""
    print("🏀 Basketball Betting Analytics System")
    print("=" * 50)
    
    # Load data
    df = load_and_clean_data(FILE_NAME)
    if df is None:
        return
    
    # Prepare model data
    model_data = prepare_model_data(df, RECENT_MATCHES_COUNT)
    if model_data[0] is None:
        return
        
    X_train, X_predict, y_side, y_over, limit_line, future_data = model_data
    
    # Check if we have data for prediction
    if X_predict.empty:
        print("ℹ️  No upcoming matches to predict")
        return
    
    # Train models
    models = train_and_evaluate_models(X_train, y_side, y_over)
    side_model, over_model, side_accuracy, over_accuracy = models
    
    # Make predictions
    print("🔮 Making predictions...")
    side_proba = side_model.predict_proba(X_predict)
    over_proba = over_model.predict_proba(X_predict)
    
    # Prepare predictions dataframe
    future_data = future_data.loc[X_predict.index].copy()
    future_data['P_Home'] = side_proba[:, 1]  # Home win probability
    future_data['P_Away'] = side_proba[:, 0]  # Away win probability  
    future_data['P_Over'] = over_proba[:, 1]  # Over probability
    future_data['P_Under'] = over_proba[:, 0]  # Under probability
    future_data['Limit_Line'] = limit_line
    
    # Find today's matches
    today = future_data['Date'].min()
    todays_matches = future_data[future_data['Date'] == today].copy()
    
    print(f"\n📅 Analysis Date: {today.strftime('%Y-%m-%d')}")
    print(f"📊 Total Score Limit Line: {limit_line:.1f}")
    print(f"🎯 Minimum Probability Threshold: {MIN_PROBABILITY_THRESHOLD}")
    
    # Find valuable bets
    valuable_bets = find_valuable_bets(todays_matches, DEFAULT_ODDS, MIN_PROBABILITY_THRESHOLD)
    
    if valuable_bets.empty:
        print("❌ No valuable bets found today")
    else:
        print(f"✅ Found {len(valuable_bets)} valuable bet(s)")
        print("\n" + valuable_bets.to_string(index=False))
    
    # Send Telegram notification
    send_telegram_message(valuable_bets, today.strftime('%Y-%m-%d'))
    
    print("\n🎯 Analysis complete!")

if __name__ == "__main__":
    main()
