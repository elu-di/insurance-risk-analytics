import pandas as pd
import numpy as np
from scipy import stats

def chi_squared_test(data: pd.DataFrame, col1: str, col2: str):
    """Run a Chi-Squared test between two categorical columns."""
    contingency_table = pd.crosstab(data[col1], data[col2])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    return p_value

def t_test_independent(data1: pd.Series, data2: pd.Series):
    """Run an independent T-test between two numerical series."""
    t_stat, p_value = stats.ttest_ind(data1.dropna(), data2.dropna(), equal_var=False)
    return p_value

def calculate_margin(df: pd.DataFrame):
    """Calculate the margin (TotalPremium - TotalClaims)."""
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    return df

def define_claim_frequency(df: pd.DataFrame):
    """Create a binary column indicating if a claim occurred."""
    df['Claim_Occurred'] = (df['TotalClaims'] > 0).astype(int)
    return df
