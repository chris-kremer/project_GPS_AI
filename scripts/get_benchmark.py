import os
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output (optional)
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# 1. Data Loading and Cleaning
# ---------------------------


# File paths
file_path = '/Users/chris/Documents/project_GPS_AI/benchmarks/individual_new.dta'
file_path_run = '/Users/chris/Documents/project_GPS_AI/data/processed/cleaned_data.csv'

# Load the .dta file (without encoding argument)
try:
    benchmark = pd.read_stata(file_path)
except Exception as e:
    print(f"Error reading Stata file: {e}")
    raise

benchmark['category'] = 'Benchmark'
columns_to_drop = ['isocode', 'region', 'date', 'id_gallup']
benchmark = benchmark.drop(columns=columns_to_drop)

# Load the run data
try:
    run = pd.read_csv(file_path_run)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    raise

columns_to_drop_run = ['Participant Hash', 'Question']
run = run.drop(columns=columns_to_drop_run)
run.columns = run.columns.str.lower()

# Remove numbers from 'short title'
run['short title'] = run['short title'].str.replace(r'\d+', '', regex=True)

# Mapping short titles
title_mapping = {
    "Will do revenge": "revenge_q",
    "Will return favor": "favour_q",
    "Willingness to delay consumption": "patience_q",
    "Willingness to donate": "donate_q",
    "Willingness to take risk": "risk_q",
    "Retribution on others' behalf": "retribution_oth_q",
    "Good at math": "subj_math_skills",
    "Reciprocation ": "recip_perc",
    "Donation .": "donate_perc",
    "Delay .": "patience_e",
    "Risk ": "risk_e",
    "People have best intentions": "trust"
}

# Replace short titles
run['short title'] = run['short title'].replace(title_mapping)

# Pivot the dataframe
pivoted_df = run.pivot_table(
    index=['participant id', 'age', 'gender', 'country'],
    columns='short title',
    values='answer'
).reset_index()

# Create composite variables
pivoted_df['patience'] = 0.7115185 * pivoted_df['patience_e'] + 0.2884815 * pivoted_df['patience_q']
pivoted_df['risktaking'] = 0.4729985 * pivoted_df['risk_e'] + 0.5270015 * pivoted_df['risk_q']
pivoted_df['posrecip'] = 0.4847038 * pivoted_df['favour_q'] + 0.5152962 * pivoted_df['recip_perc']
pivoted_df['negrecip'] = (
    0.3130969 * pivoted_df['revenge_q'] +
    0.3130969 * pivoted_df['retribution_oth_q'] +
    0.3738062 * pivoted_df['revenge_q']
)
pivoted_df['altruism'] = 0.6350048 * pivoted_df['donate_q'] + 0.3649952 * pivoted_df['donate_perc']

# Flatten column index and add category
pivoted_df.columns.name = None
pivoted_df['category'] = 'run'

# Replace gender strings with integers explicitly using map and astype
pivoted_df['gender'] = pivoted_df['gender'].map({'male': 0, 'female': 1}).astype(int)

# Concatenate run and benchmark dataframes
combined_df = pd.concat([pivoted_df, benchmark], axis=0)

# Replace any remaining spaces in column names with underscores
combined_df.columns = combined_df.columns.str.replace(' ', '_', regex=True)

# Ensure all 'gender' entries are numeric
if combined_df['gender'].dtype == object:
    combined_df['gender'] = combined_df['gender'].map({'male': 0, 'female': 1}).astype(int)

# Convert 'country' into dummy variables
combined_df = pd.get_dummies(combined_df, columns=['country'], drop_first=True)

# Verify that dummy variables are created
country_dummies = [col for col in combined_df.columns if col.startswith('country_')]

# Ensure no spaces in dummy variable names (already handled, but double-check)
combined_df.columns = combined_df.columns.str.replace(' ', '_', regex=True)

# Update the list of country dummy variables after renaming
country_dummies = [col for col in combined_df.columns if col.startswith('country_')]

# Convert 'category' to binary: 1 for 'run', 0 for 'Benchmark'
combined_df['category_binary'] = combined_df['category'].apply(
    lambda x: 1 if x.lower() == 'run' else 0
)

# Drop rows with missing values in relevant columns
combined_df = combined_df.dropna(subset=['altruism', 'age', 'gender', 'category_binary'])

# ---------------------------
# 5. Save Processed Data to Desktop
# ---------------------------

# Define the path to save the processed data
desktop_path = os.path.expanduser('~/Desktop/')
output_file = os.path.join(desktop_path, 'processed_data.csv')

# Save the processed data to Desktop
combined_df.to_csv(output_file, index=False)
print(f"\nProcessed data saved to {output_file}")

# ---------------------------
# 2. Verify Column Names and Formula
# ---------------------------

# Print all column names to verify
print("\nColumns in combined_df:")
print(combined_df.columns.tolist())

# Define the regression formula
# Including main effects and interaction terms between category_binary and each predictor
# Using parentheses to group main effects and interactions
formula = 'altruism ~ (age + gender + ' + ' + '.join(country_dummies) + ') * category_binary'
print("\nRegression Formula:")
print(formula)

# ---------------------------
# 3. Regression Analysis
# ---------------------------

# Fit the OLS regression model
try:
    model = smf.ols(formula=formula, data=combined_df).fit()
except Exception as e:
    print(f"\nError fitting the regression model: {e}")
    # Identify missing variables
    import re
    # Extract variable names from the formula using regex
    variables_in_formula = re.findall(r'[\w_]+', formula)
    # Remove duplicates and operators
    variables_in_formula = list(set(variables_in_formula))
    # Remove 'altruism' as it's the dependent variable
    if 'altruism' in variables_in_formula:
        variables_in_formula.remove('altruism')
    # Check which variables are missing
    missing_vars = [var for var in variables_in_formula if var not in combined_df.columns]
    if missing_vars:
        print(f"\nThe following variables are missing from combined_df: {missing_vars}")
    else:
        print("\nNo missing variables detected.")
    raise

# Display the regression summary
print("\n--- Regression Summary ---\n")
print(model.summary())

# ---------------------------
# 4. F-Test: Comparing Models
# ---------------------------

# Define the restricted model (without interaction terms)
restricted_formula = 'altruism ~ age + gender + ' + ' + '.join(country_dummies) + ' + category_binary'

# Fit the restricted model
try:
    restricted_model = smf.ols(formula=restricted_formula, data=combined_df).fit()
except Exception as e:
    print(f"\nError fitting the restricted regression model: {e}")
    raise

# Perform the F-test
f_test = model.compare_f_test(restricted_model)
print("\n--- F-Test Results ---\n")
print(f"F-test statistic: {f_test[0]:.4f}")
print(f"P-value: {f_test[1]:.4f}")
print(f"Degrees of freedom difference: {int(f_test[2])}")

# Interpretation
if f_test[1] < 0.05:
    print("\nThe interaction terms significantly improve the model. This suggests that the effects of age, gender, or country on altruism differ between 'run' and 'Benchmark' categories.")
else:
    print("\nThe interaction terms do not significantly improve the model. This suggests that the effects of age, gender, and country on altruism are similar across 'run' and 'Benchmark' categories.")

# ---------------------------
# 6. Additional Diagnostics (Optional)
# ---------------------------

# Check for Multicollinearity using Variance Inflation Factor (VIF)
print("\n--- Variance Inflation Factor (VIF) ---\n")
# Select predictor variables including interaction terms
predictors = ['age', 'gender'] + country_dummies + ['category_binary']
# Add interaction terms between category_binary and each predictor
for var in ['age', 'gender'] + country_dummies:
    interaction_term = f'{var}:category_binary'
    predictors.append(interaction_term)

# Create a DataFrame for VIF calculation
X = combined_df[predictors]
X = sm.add_constant(X)

# Calculate VIF for each predictor
vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)

# Residual Analysis
print("\n--- Residual Analysis ---\n")

# Residuals vs Fitted
sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, line_kws={'color': 'red'})
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.show()

# Q-Q Plot for residuals
sm.qqplot(model.resid, line='45')
plt.title('Q-Q Plot of Residuals')
plt.show()

# Histogram of residuals
sns.histplot(model.resid, kde=True)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# ---------------------------
# End of Script
# ---------------------------