# filter_and_regress.py

import pandas as pd
from pathlib import Path
import logging
import statsmodels.api as sm
import yaml
import numpy as np
import argparse

class GPSDataRegressor:
    def __init__(self, input_csv: Path, config_file: Path, results_folder: Path):
        self.input_csv = input_csv
        self.config_file = config_file
        self.results_folder = results_folder
        self.regression_configs = self.load_regression_configs()

    def load_regression_configs(self) -> list:
        """
        Load regression configurations from a YAML file.
        """
        try:
            with open(self.config_file, 'r') as file:
                config = yaml.safe_load(file)
            return config.get('regressions', [])
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {self.config_file}")
            return []
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML file: {e}")
            return []

    def filter_risk_questions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the DataFrame for rows where 'Short Title' starts with 'Risk'.
        """
        if 'Short Title' not in df.columns:
            logging.error("'Short Title' column is missing from the data.")
            return pd.DataFrame()

        risk_df = df[df['Short Title'].str.startswith('Risk', na=False)]
        logging.info(f"Filtered down to {len(risk_df)} rows where 'Short Title' starts with 'Risk'.")
        return risk_df

    def perform_regression(self, df: pd.DataFrame, regression_setup: dict):
        """
        Performs a regression based on the provided setup.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the data.
            regression_setup (dict): Dictionary containing regression parameters.

        Returns:
            None
        """
        y_col = regression_setup['y']
        x_cols_config = regression_setup['x']  # List of dicts with 'name' and 'type'
        include_country_dummies = regression_setup.get('include_country_dummies', False)

        # Extract x column names and types
        x_cols = [col['name'] for col in x_cols_config]
        x_types = {col['name']: col['type'] for col in x_cols_config}

        # Ensure required columns exist
        required_columns = [y_col] + x_cols + ['Country']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"Required columns missing for {regression_setup['name']}: {missing_columns}")
            return

        # Drop rows with missing data in the relevant columns
        initial_count = len(df)
        df = df.dropna(subset=[y_col] + x_cols + ['Country'])
        dropped_count = initial_count - len(df)
        if dropped_count > 0:
            logging.info(f"Dropped {dropped_count} rows due to missing data.")

        # Strip whitespace from string columns to prevent conversion issues
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in object_cols:
            df[col] = df[col].str.strip()

        # Convert y to numeric if it's not already
        if not pd.api.types.is_numeric_dtype(df[y_col]):
            # Attempt to map categorical to numeric if applicable
            unique_answers = df[y_col].unique()
            logging.info(f"Unique '{y_col}' values before mapping: {unique_answers}")

            # Example for binary categorical data
            binary_mapping = {'No': 0, 'Yes': 1}
            if set(unique_answers).issubset(binary_mapping.keys()):
                df[y_col] = df[y_col].map(binary_mapping)
                logging.info(f"Mapped '{y_col}' categories to numeric codes: {binary_mapping}")
                logging.debug(f"Unique '{y_col}' values after mapping: {df[y_col].unique()}")
            else:
                # Attempt to convert to numeric, coercing errors to NaN
                df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
                logging.info(f"Converted '{y_col}' to numeric, coerced errors to NaN.")

            # Check for unmapped or invalid entries
            nan_count = df[y_col].isna().sum()
            if nan_count > 0:
                logging.warning(f"{nan_count} rows have non-numeric '{y_col}' values and will be dropped.")
                df = df.dropna(subset=[y_col])

        # After potential mapping, log unique 'Answer' values
        logging.info(f"Unique '{y_col}' values after mapping: {df[y_col].unique()}")

        # Initialize list for final x columns after encoding
        final_x_cols = []

        # Process each independent variable based on its type
        for col in x_cols:
            if x_types[col] == "numeric":
                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isnull().any():
                    logging.warning(f"Some values in '{col}' could not be converted to numeric and are set to NaN.")
                final_x_cols.append(col)
            elif x_types[col] == "categorical":
                # Create dummy variables, drop the first category
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                dummies = dummies.astype(int)  # Convert bool to int
                df = pd.concat([df, dummies], axis=1)
                dummy_cols = dummies.columns.tolist()
                final_x_cols.extend(dummy_cols)
                logging.info(f"Created dummy variables for categorical column: {col}")
            else:
                logging.error(f"Unsupported type for column '{col}': {x_types[col]}")
                return

        # Handle country dummies if needed
        if include_country_dummies:
            if 'Country' not in df.columns:
                logging.error("'Country' column is missing from the data.")
                return
            country_dummies = pd.get_dummies(df['Country'], prefix='Country', drop_first=True)
            country_dummies = country_dummies.astype(int)  # Convert bool to int
            df = pd.concat([df, country_dummies], axis=1)
            final_x_cols.extend(country_dummies.columns.tolist())
            logging.info("Created dummy variables for 'Country'.")

        # Drop rows where any of the x columns are NaN after conversion/encoding
        df = df.dropna(subset=final_x_cols)
        if df.empty:
            logging.error(f"No valid data for independent variables after preprocessing for {regression_setup['name']}.")
            return

        y = df[y_col]
        X = df[final_x_cols]

        # Convert all X columns to numeric explicitly, especially bools to ints
        X = X.astype(int)

        # **Save the processed data to CSV for inspection**
        processed_data_file = self.results_folder / f"{regression_setup['name'].replace(' ', '_')}_processed_data.csv"
        df.to_csv(processed_data_file, index=False)
        logging.info(f"Processed data saved to: {processed_data_file}")

        # **Data Type Checks**
        if not pd.api.types.is_numeric_dtype(y):
            logging.error(f"Dependent variable '{y_col}' is not numeric after preprocessing.")
            return

        non_numeric_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if non_numeric_cols:
            logging.error(f"Independent variables contain non-numeric columns: {non_numeric_cols}")
            return

        # Check for infinite values
        if np.isinf(X).any().any():
            logging.error("Infinite values found in independent variables (X).")
            return

        if np.isinf(y).any():
            logging.error("Infinite values found in dependent variable (y).")
            return

        # Check for NaN values
        if X.isnull().values.any():
            logging.error("NaN values found in independent variables (X) after preprocessing.")
            return

        if y.isnull().values.any():
            logging.error("NaN values found in dependent variable (y) after preprocessing.")
            return

        # Add a constant term to the regression
        X = sm.add_constant(X, has_constant='add')

        # Final validation
        if X.empty or y.empty:
            logging.error(f"No valid data for regression: {regression_setup['name']}")
            return

        logging.info(
            f"Running regression for {regression_setup['name']} with y: {y_col} and X columns: {X.columns.tolist()}"
        )

        try:
            # Fit the linear regression model
            model = sm.OLS(y, X).fit()

            # Save the summary of the regression results to a text file
            results_file = self.results_folder / f"{regression_setup['name'].replace(' ', '_')}_results.txt"
            with open(results_file, 'w') as f:
                f.write(model.summary().as_text())

            logging.info(f"Regression results saved to: {results_file}")
        except Exception as e:
            logging.error(f"Error performing regression {regression_setup['name']}: {e}")

    def run(self):
        """
        Execute the filtering and regression analysis.
        """
        try:
            df = pd.read_csv(self.input_csv)
            logging.info(f"Loaded data from {self.input_csv} with {len(df)} rows.")
        except FileNotFoundError:
            logging.error(f"Input CSV file not found: {self.input_csv}")
            return
        except Exception as e:
            logging.error(f"Error reading input CSV file: {e}")
            return

        if df.empty:
            logging.error("Input CSV is empty. Exiting.")
            return

        # Filter for risk questions
        risk_df = self.filter_risk_questions(df)

        if risk_df.empty:
            logging.error("No risk questions found after filtering. Exiting.")
            return

        # Perform multiple regressions as per the configurations
        for regression_setup in self.regression_configs:
            self.perform_regression(risk_df, regression_setup)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Filter GPS data for risk questions and perform regressions.")
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the concatenated and cleaned CSV file.')
    parser.add_argument('--config_file', type=str, required=True, help='Path to the regression config YAML file.')
    parser.add_argument('--results_folder', type=str, required=True, help='Path to the folder to save regression results.')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).')
    args = parser.parse_args()

    # Configure logging with timestamp and dynamic level
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.log_level}')
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=numeric_level,  # Set based on argument
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),  # Log to console
            logging.FileHandler("filter_and_regress.log")  # Log to a file
        ]
    )

    # Define paths
    input_csv = Path(args.input_csv)
    config_file = Path(args.config_file)
    results_folder = Path(args.results_folder)

    # Create results folder if it doesn't exist
    if not results_folder.exists():
        results_folder.mkdir(parents=True)
        logging.info(f"Created results folder: {results_folder}")

    # Create an instance of GPSDataRegressor and run
    regressor = GPSDataRegressor(input_csv, config_file, results_folder)
    regressor.run()