"""
Basketball Betting Analytics System
Author: AI Assistant
Description: Machine learning system for basketball match predictions and betting value detection
Version: 3.1 - Fixed Telegram Issues
"""

import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# =================================================================
#                         CONFIGURATION
# =================================================================

# Data Settings
FILE_NAME = "BasketbolFikstür - Sayfa1.tsv"

# Model Settings
DEFAULT_ODDS = 1.90
MIN_PROBABILITY_THRESHOLD = 0.60
RECENT_MATCHES_COUNT = 8
MIN_ACCURACY_THRESHOLD = 0.55

# Risk Management
KELLY_FRACTION = 0.15
MAX_BANKROLL_PERCENTAGE = 2.0

# Telegram Settings
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

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

def send_telegram_message(df_bets, analysis_date, limit_line):
    """Send analysis results via Telegram - FIXED VERSION"""
    print(f"📱 Preparing Telegram message for {analysis_date}...")
    
    # Debug information
    print(f"🔍 Telegram Bot Token: {'Set' if TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_TOKEN != '' else 'Not Set'}")
    print(f"🔍 Telegram Chat ID: {'Set' if TELEGRAM_CHAT_ID and TELEGRAM_CHAT_ID != '' else 'Not Set'}")
    
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "":
        print("❌ TELEGRAM_BOT_TOKEN not configured")
        print("💡 Please set TELEGRAM_BOT_TOKEN in GitHub Secrets")
        return
        
    if not TELEGRAM_CHAT_ID or TELEGRAM_CHAT_ID == "":
        print("❌ TELEGRAM_CHAT_ID not configured")
        print("💡 Please set TELEGRAM_CHAT_ID in GitHub Secrets")
        return

    print("✅ Telegram credentials found, preparing message...")

    if df_bets.empty:
        message = f"🏀 *Basketball Betting Analysis - {analysis_date}*\n\n"
        message += "No valuable bets found today based on Kelly Criterion 🚫\n"
        message += f"*Limit Line:* {limit_line:.1f}"
    else:
        message = f"💰 *Basketball Betting Recommendations - {analysis_date}*\n\n"
        message += f"*Limit Line:* {limit_line:.1f}\n\n"
        
        # Bahisleri türlerine göre grupla
        side_bets = df_bets[df_bets['Bet_Type'] == 'Side']
        over_under_bets = df_bets[df_bets['Bet_Type'] == 'Points Line']
        
        if not side_bets.empty:
            message += "*🎯 SIDE BETS:*\n"
            for _, row in side_bets.iterrows():
                message += f"• {row['Match']}\n"
                message += f"  - {row['Selection']}\n"
                message += f"  - Model: {row['Model_Probability']:.1%}\n"
                message += f"  - Bet Size: {row['Kelly_Percentage']:.1f}%\n"
                message += f"  - Odds: {row['Odds']:.2f}\n"
                message += "  ────────\n"
        
        if not over_under_bets.empty:
            message += "\n*📊 OVER/UNDER BETS:*\n"
            for _, row in over_under_bets.iterrows():
                message += f"• {row['Match']}\n"
                message += f"  - {row['Selection']}\n"
                message += f"  - Model: {row['Model_Probability']:.1%}\n"
                message += f"  - Bet Size: {row['Kelly_Percentage']:.1f}%\n"
                message += f"  - Odds: {row['Odds']:.2f}\n"
                message += "  ────────\n"
        
        message += f"\n*Total Bets:* {len(df_bets)}"

    # Telegram API URL - FIXED: Do not use f-string with token
    telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'
    }
    
    try:
        print("🔄 Sending request to Telegram API...")
        print(f"🔍 URL: https://api.telegram.org/bot***/sendMessage")
        print(f"🔍 Chat ID: {TELEGRAM_CHAT_ID}")
        print(f"🔍 Message length: {len(message)} characters")
        
        response = requests.post(telegram_url, data=payload, timeout=30)
        
        print(f"🔍 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Telegram message sent successfully!")
            result = response.json()
            if result.get('ok'):
                print("✅ Message delivered to Telegram")
            else:
                print(f"❌ Telegram API error: {result.get('description', 'Unknown error')}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"❌ Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Telegram connection error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"❌ Response text: {e.response.text}")

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

def calculate_team_offensive_rating(team, date, historical_data, n_matches):
    """Calculate team offensive rating based on recent matches"""
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
    
    return total_points / total_matches if total_matches > 0 else 85.0

def calculate_team_defensive_rating(team, date, historical_data, n_matches):
    """Calculate team defensive rating based on recent matches"""
    home_matches = historical_data[
        (historical_data['Home_Team'] == team) & 
        (historical_data['Date'] < date)
    ].tail(n_matches)
    
    away_matches = historical_data[
        (historical_data['Away_Team'] == team) & 
        (historical_data['Date'] < date)
    ].tail(n_matches)
    
    conceded_points = home_matches['Away_Score'].sum() + away_matches['Home_Score'].sum()
    total_matches = len(home_matches) + len(away_matches)
    
    return conceded_points / total_matches if total_matches > 0 else 85.0

def calculate_team_form(team, date, historical_data, n_matches):
    """Calculate team form based on recent matches (home + away)"""
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
    
    return total_wins / total_matches if total_matches > 0 else 0.5

def calculate_league_limit_line(league, date, historical_data):
    """Calculate dynamic limit line for each league"""
    league_matches = historical_data[
        (historical_data['League'] == league) & 
        (historical_data['Date'] < date)
    ].tail(50)
    
    if len(league_matches) > 10:
        return league_matches['Total_Score'].median()
    else:
        return historical_data['Total_Score'].median()

def calculate_matchup_limit_line(home_team, away_team, date, historical_data):
    """Calculate limit line based on specific team matchup"""
    matchup_matches = historical_data[
        (((historical_data['Home_Team'] == home_team) & (historical_data['Away_Team'] == away_team)) |
         ((historical_data['Home_Team'] == away_team) & (historical_data['Away_Team'] == home_team))) &
        (historical_data['Date'] < date)
    ].tail(10)
    
    if len(matchup_matches) > 3:
        return matchup_matches['Total_Score'].median()
    else:
        return None

def create_features(df, historical_data, n_matches):
    """Create comprehensive features for model training"""
    df_features = df.copy()
    
    # Rest days calculation
    home_matches = df_features[['Date', 'Home_Team']].rename(columns={'Home_Team': 'Team'})
    away_matches = df_features[['Date', 'Away_Team']].rename(columns={'Away_Team': 'Team'})
    
    all_matches = pd.concat([home_matches, away_matches]).sort_values('Date').reset_index(drop=True)
    all_matches['Previous_Date'] = all_matches.groupby('Team')['Date'].shift(1)
    all_matches['Rest_Days'] = (all_matches['Date'] - all_matches['Previous_Date']).dt.days
    
    home_rest = all_matches[['Date', 'Team', 'Rest_Days']].rename(
        columns={'Team': 'Home_Team', 'Rest_Days': 'Home_Rest_Days'})
    away_rest = all_matches[['Date', 'Team', 'Rest_Days']].rename(
        columns={'Team': 'Away_Team', 'Rest_Days': 'Away_Rest_Days'})
    
    df_features = pd.merge(df_features, home_rest, on=['Date', 'Home_Team'], how='left')
    df_features = pd.merge(df_features, away_rest, on=['Date', 'Away_Team'], how='left')
    
    df_features['Home_Rest_Days'] = df_features['Home_Rest_Days'].fillna(7)
    df_features['Away_Rest_Days'] = df_features['Away_Rest_Days'].fillna(7)
    df_features['Rest_Days_Diff'] = df_features['Home_Rest_Days'] - df_features['Away_Rest_Days']
    
    # Advanced team ratings
    df_features['Home_Offensive_Rating'] = df_features.apply(
        lambda row: calculate_team_offensive_rating(row['Home_Team'], row['Date'], historical_data, n_matches), axis=1)
    
    df_features['Away_Offensive_Rating'] = df_features.apply(
        lambda row: calculate_team_offensive_rating(row['Away_Team'], row['Date'], historical_data, n_matches), axis=1)
    
    df_features['Home_Defensive_Rating'] = df_features.apply(
        lambda row: calculate_team_defensive_rating(row['Home_Team'], row['Date'], historical_data, n_matches), axis=1)
    
    df_features['Away_Defensive_Rating'] = df_features.apply(
        lambda row: calculate_team_defensive_rating(row['Away_Team'], row['Date'], historical_data, n_matches), axis=1)
    
    # Team form (combined home + away)
    df_features['Home_Team_Form'] = df_features.apply(
        lambda row: calculate_team_form(row['Home_Team'], row['Date'], historical_data, n_matches), axis=1)
    
    df_features['Away_Team_Form'] = df_features.apply(
        lambda row: calculate_team_form(row['Away_Team'], row['Date'], historical_data, n_matches), axis=1)
    
    # Expected total score
    df_features['Expected_Total_Score'] = (
        df_features['Home_Offensive_Rating'] + df_features['Away_Offensive_Rating']
    ) / 2
    
    # Strength difference
    df_features['Offensive_Strength_Diff'] = (
        df_features['Home_Offensive_Rating'] - df_features['Away_Offensive_Rating']
    )
    
    df_features['Defensive_Strength_Diff'] = (
        df_features['Home_Defensive_Rating'] - df_features['Away_Defensive_Rating']
    )
    
    return df_features

# =================================================================
#                         BET ANALYSIS
# =================================================================

def find_valuable_bets(predictions_df, min_probability):
    """Identify valuable bets using Kelly Criterion with dynamic odds"""
    valuable_bets = []
    
    for _, match in predictions_df.iterrows():
        match_name = f"{match['Home_Team']} vs {match['Away_Team']}"
        limit_line = int(match['Limit_Line'])
        
        # Dynamic odds based on probability
        def get_dynamic_odds(probability):
            if probability > 0.75:
                return 1.60
            elif probability > 0.65:
                return 1.75
            elif probability > 0.55:
                return 1.85
            else:
                return 1.95
        
        # Home win bet
        if match['P_Home'] > min_probability:
            odds = get_dynamic_odds(match['P_Home'])
            kelly_full = calculate_kelly_criterion(match['P_Home'], odds)
            kelly_fractional = fractional_kelly_bet_size(kelly_full)
            
            if kelly_fractional > 0.005:
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
            odds = get_dynamic_odds(match['P_Away'])
            kelly_full = calculate_kelly_criterion(match['P_Away'], odds)
            kelly_fractional = fractional_kelly_bet_size(kelly_full)
            
            if kelly_fractional > 0.005:
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
            odds = get_dynamic_odds(match['P_Over'])
            kelly_full = calculate_kelly_criterion(match['P_Over'], odds)
            kelly_fractional = fractional_kelly_bet_size(kelly_full)
            
            if kelly_fractional > 0.005:
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
            odds = get_dynamic_odds(match['P_Under'])
            kelly_full = calculate_kelly_criterion(match['P_Under'], odds)
            kelly_fractional = fractional_kelly_bet_size(kelly_full)
            
            if kelly_fractional > 0.005:
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
        'Tarih': 'Date',
        'Lig': 'League'
    })
    
    # Data type conversions
    df['Home_Score'] = pd.to_numeric(df['Home_Score'], errors='coerce')
    df['Away_Score'] = pd.to_numeric(df['Away_Score'], errors='coerce')
    
    # Date parsing
    print("🔄 Converting dates...")
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
    
    # Filter invalid dates
    invalid_dates = df['Date'].isna().sum()
    if invalid_dates > 0:
        print(f"⚠️  Filtered out {invalid_dates} invalid dates")
    
    df = df[df['Date'].notna()].copy()
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Create target variables
    df['Total_Score'] = df['Home_Score'] + df['Away_Score']
    df['Winning_Side'] = (df['Home_Score'] > df['Away_Score']).astype('float')
    df['Winning_Side'] = df['Winning_Side'].where(df['Home_Score'].notna(), -1)
    
    # Create Over/Under target
    overall_limit = df[df['Home_Score'].notna()]['Total_Score'].median()
    df['Over_Line'] = (df['Total_Score'] > overall_limit).astype('float')
    df['Over_Line'] = df['Over_Line'].where(df['Home_Score'].notna(), -1)
    
    print(f"✅ Final dataset: {len(df)} valid records")
    print(f"📊 Overall Limit Line: {overall_limit:.1f}")
    return df

def prepare_model_data(df, n_matches):
    """Prepare features and targets for model training"""
    # Split data into historical and future matches
    historical_data = df[df['Home_Score'].notna()].copy()
    future_data = df[df['Home_Score'].isna()].copy()
    
    if historical_data.empty:
        print("❌ No historical data for training")
        return None, None, None, None, None
    
    # Create features
    print("🔄 Creating features...")
    historical_features = create_features(historical_data, historical_data, n_matches)
    
    if not future_data.empty:
        future_features = create_features(future_data, historical_data, n_matches)
        
        # Calculate dynamic limit lines for each match
        future_features['Limit_Line'] = future_features.apply(
            lambda row: calculate_matchup_limit_line(
                row['Home_Team'], row['Away_Team'], row['Date'], historical_data
            ) or calculate_league_limit_line(row['League'], row['Date'], historical_data), axis=1)
    else:
        future_features = pd.DataFrame()
    
    # Define feature columns
    feature_columns = [
        'Rest_Days_Diff', 'Home_Team_Form', 'Away_Team_Form', 
        'Home_Offensive_Rating', 'Away_Offensive_Rating',
        'Home_Defensive_Rating', 'Away_Defensive_Rating',
        'Expected_Total_Score', 'Offensive_Strength_Diff', 'Defensive_Strength_Diff',
        'Home_Team', 'Away_Team', 'League'
    ]
    
    # Prepare training data
    X_train = historical_features[feature_columns].copy()
    X_train = X_train.dropna()
    
    if X_train.empty:
        print("❌ No valid training data after preprocessing")
        return None, None, None, None, None
    
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
    
    return X_train, X_predict, y_side, y_over, future_features

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
    print("🏀 Basketball Betting Analytics System - Fixed Telegram")
    print("=" * 60)
    
    # Debug Telegram settings
    print(f"🔍 Telegram Bot Token: {'***' + TELEGRAM_BOT_TOKEN[-4:] if TELEGRAM_BOT_TOKEN else 'Not Set'}")
    print(f"🔍 Telegram Chat ID: {TELEGRAM_CHAT_ID if TELEGRAM_CHAT_ID else 'Not Set'}")
    
    # Load data
    df = load_and_clean_data(FILE_NAME)
    if df is None:
        return
    
    # Prepare model data
    model_data = prepare_model_data(df, RECENT_MATCHES_COUNT)
    if model_data[0] is None:
        return
        
    X_train, X_predict, y_side, y_over, future_data = model_data
    
    # Check if we have data for prediction
    if X_predict.empty:
        print("ℹ️  No upcoming matches to predict")
        return
    
    # Get this week's matches only
    current_date = datetime.now().date()
    next_week = current_date + timedelta(days=7)
    this_weeks_matches = future_data[
        (future_data['Date'].dt.date >= current_date) & 
        (future_data['Date'].dt.date <= next_week)
    ].copy()
    
    if this_weeks_matches.empty:
        print("❌ No matches found for this week")
        return
    
    print(f"📅 This week's matches: {len(this_weeks_matches)}")
    
    # Train models
    models = train_and_evaluate_models(X_train, y_side, y_over)
    side_model, over_model, side_accuracy, over_accuracy = models
    
    # Make predictions only for this week's matches
    print("🔮 Making predictions for this week...")
    X_predict_this_week = X_predict.loc[this_weeks_matches.index]
    side_proba = side_model.predict_proba(X_predict_this_week)
    over_proba = over_model.predict_proba(X_predict_this_week)
    
    # Prepare predictions dataframe
    this_weeks_matches = this_weeks_matches.copy()
    this_weeks_matches['P_Home'] = side_proba[:, 1]
    this_weeks_matches['P_Away'] = side_proba[:, 0]
    this_weeks_matches['P_Over'] = over_proba[:, 1]
    this_weeks_matches['P_Under'] = over_proba[:, 0]
    
    # Analyze each day of the week
    analysis_results = []
    for date in sorted(this_weeks_matches['Date'].dt.date.unique()):
        days_matches = this_weeks_matches[this_weeks_matches['Date'].dt.date == date].copy()
        
        print(f"\n📅 Analysis Date: {date}")
        print(f"📊 Matches to analyze: {len(days_matches)}")
        
        # Calculate average limit line for the day
        avg_limit_line = days_matches['Limit_Line'].mean()
        print(f"📊 Average Limit Line: {avg_limit_line:.1f}")
        
        # Find valuable bets
        valuable_bets = find_valuable_bets(days_matches, MIN_PROBABILITY_THRESHOLD)
        
        if valuable_bets.empty:
            print("❌ No valuable bets found")
        else:
            print(f"✅ Found {len(valuable_bets)} valuable bet(s)")
            analysis_results.append((date, valuable_bets, avg_limit_line))
    
    # Send Telegram messages for each day with valuable bets
    for date, valuable_bets, limit_line in analysis_results:
        if not valuable_bets.empty:
            send_telegram_message(valuable_bets, date, limit_line)
        else:
            print(f"📅 No bets to send for {date}")
    
    print("\n🎯 Analysis complete!")

if __name__ == "__main__":
    main()
