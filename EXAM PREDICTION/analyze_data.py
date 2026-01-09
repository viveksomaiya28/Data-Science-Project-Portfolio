
import pandas as pd
import numpy as np

# Load Data
try:
    df = pd.read_csv("Exam_Score_Prediction.csv")
    print(f"Data Loaded. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: CSV not found.")
    exit()

# 1. Feature Engineering (Re-creating logic)
quality_map = {'poor': 1, 'average': 2, 'good': 3}
difficulty_map = {'easy': 1, 'moderate': 2, 'hard': 3}
rating_map = {'low': 1, 'medium': 2, 'high': 3}

df['sleep_quality_ord'] = df['sleep_quality'].map(quality_map)
df['exam_difficulty_ord'] = df['exam_difficulty'].map(difficulty_map)
df['facility_rating_ord'] = df['facility_rating'].map(rating_map)
df['effective_effort'] = df['study_hours'] * df['class_attendance']
df['log_study_hours'] = np.log1p(df['study_hours'])

# 2. Check Correlations
print("\n--- Correlations with Exam Score ---")
numeric_df = df.select_dtypes(include=[np.number])
correlations = numeric_df.corr()['exam_score'].sort_values(ascending=False)
print(correlations)

# 3. Check Signal Strength
print("\n--- Top 5 Features Signal ---")
print(correlations.head(6)) # Top 5 + itself

# 4. Check Target Distribution
print("\n--- Target Stats ---")
print(f"Skewness: {df['exam_score'].skew():.4f}")
print(df['exam_score'].describe())

# 5. Check if 'course' matters (One-Way ANOVA equivalent logic visually)
print("\n--- Mean Score by Course ---")
print(df.groupby('course')['exam_score'].mean().sort_values())
