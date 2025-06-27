import pandas as pd
import numpy as np
from scipy.stats import poisson

# Load datasets
home = pd.read_csv("data/palmeiras.csv")
away = pd.read_csv("data/botafogo.csv")
matchup = pd.read_csv("data/matchup.csv")

# Extract home/away team names
home_team = matchup["Home"].iloc[0] 
away_team = matchup["Away"].iloc[0]  

# Filter team stats
home_df = home[home["Squad"] == home_team].iloc[0]
away_df = away[away["Squad"] == away_team].iloc[0]

# Display stats
stats = ['Squad', 'MP', 'W', 'L', 'GF', 'GA', 'GD', 'xG', 'xGA']
col_width = 6  # Adjust this if you need more/less space

print("Home Team Stats ---")
# Header (right-aligned, all uppercase for consistency)
print(" ".join([f"{stat.upper():>{col_width}}" for stat in stats]))
# Values
print(" ".join([
    f"{str(home_df['Squad']):>{col_width}}" if stat == 'Squad'
    else f"{home_df[stat]:>{col_width}.1f}" if isinstance(home_df[stat], float)
    else f"{home_df[stat]:>{col_width}d}" if stat in ['MP', 'W', 'L', 'GF', 'GA', 'GD']
    else f"{home_df[stat]:>{col_width}}"
    for stat in stats
]))

print("\n--- Away Team Stats ---")
print(" ".join([f"{stat.upper():>{col_width}}" for stat in stats]))
print(" ".join([
    f"{str(away_df['Squad']):>{col_width}}" if stat == 'Squad'
    else f"{away_df[stat]:>{col_width}.1f}" if isinstance(away_df[stat], float)
    else f"{away_df[stat]:>{col_width}d}" if stat in ['MP', 'W', 'L', 'GF', 'GA', 'GD']
    else f"{away_df[stat]:>{col_width}}"
    for stat in stats
]))

def american_to_decimal(odds):
    """Convert American odds to decimal odds"""
    if odds.startswith('+'):
        return int(odds[1:])/100 + 1
    elif odds.startswith('-'):
        return 100/int(odds[1:]) + 1
    else:
        return float(odds)

def calculate_betting_probabilities(home_odds, away_odds):
    """Convert betting odds to probabilities"""
    # Convert American odds to decimal
    home_dec = american_to_decimal(home_odds)
    away_dec = american_to_decimal(away_odds)
    
    # Calculate implied probabilities
    home_prob = 1 / home_dec
    away_prob = 1 / away_dec
    
    # Normalize to account for bookmaker margin
    total = home_prob + away_prob
    return home_prob/total, away_prob/total

def calculate_model_probabilities(home, away):
    """Calculate probabilities based on team stats"""
    home_xG = home['xG']
    away_xG = away['xG']
    home_xGA = home['xGA']
    away_xGA = away['xGA']
    
    # Calculate adjusted expected goals
    home_exp = (home_xG + away_xGA) / 2
    away_exp = (away_xG + home_xGA) / 2
    
    # Poisson-based probabilities
    home_win, away_win = 0, 0
    max_goals = 6
    
    for i in range(max_goals):
        for j in range(max_goals):
            p = poisson.pmf(i, home_exp) * poisson.pmf(j, away_exp)
            if i > j:
                home_win += p
            elif i < j:
                away_win += p
                
    total = home_win + away_win
    return home_win/total, away_win/total, home_exp, away_exp

def hybrid_prediction(model_probs, betting_probs, model_weight=0.6):
    """Combine model and betting probabilities"""
    m_home, m_away = model_probs
    b_home, b_away = betting_probs
    
    hybrid_home = m_home * model_weight + b_home * (1 - model_weight)
    hybrid_away = m_away * model_weight + b_away * (1 - model_weight)
    
    # Normalize
    total = hybrid_home + hybrid_away
    return hybrid_home/total, hybrid_away/total

# Get betting odds from the data
home_odds = str(home_df['HomeOdds'])
away_odds = str(home_df['AwayOdds'])

# Calculate probabilities
model_home, model_away, home_exp, away_exp = calculate_model_probabilities(home_df, away_df)
bet_home, bet_away = calculate_betting_probabilities(home_odds, away_odds)
final_home, final_away = hybrid_prediction(
    (model_home, model_away),
    (bet_home, bet_away)
)

# Get most likely scores
def get_most_likely_scores(home_exp, away_exp, n=5):
    scores = []
    for i in range(6):
        for j in range(6):
            p = poisson.pmf(i, home_exp) * poisson.pmf(j, away_exp)
            scores.append(((i, j), p))
    scores.sort(key=lambda x: -x[1])
    return scores[:n]

likely_scores = get_most_likely_scores(home_exp, away_exp)

# Display results
print("\n--- Probability Breakdown ---")
print(f"{'Source':<10}{home_team:<10}{away_team}")
print(f"{'Model':<10}{model_home*100:>6.1f}% {model_away*100:>6.1f}%")
print(f"{'Betting':<10}{bet_home*100:>6.1f}% {bet_away*100:>6.1f}%")
print(f"{'Final':<10}{final_home*100:>6.1f}% {final_away*100:>6.1f}%")

print("\n--- Most Likely Scorelines ---")
for score, prob in likely_scores:
    print(f"{score[0]}-{score[1]}: {prob*100:.1f}%")

# Final prediction
max_prob = max(final_home, final_away)
if max_prob == final_home:
    prediction = f"{home_team} win"
else:
    prediction = f"{away_team} win"

print(f"\nFinal Prediction: {prediction} ({max_prob*100:.1f}% probability)")