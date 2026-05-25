"""EDA utilities for insurance data analysis."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class InsuranceEDA:
    """Exploratory Data Analysis utilities for insurance data."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize EDA with DataFrame.
        
        Args:
            df: Insurance data DataFrame
        """
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
    def calculate_portfolio_metrics(self) -> dict:
        """
        Calculate overall portfolio metrics.
        
        Returns:
            Dictionary with portfolio metrics
        """
        if 'LossRatio' not in self.df.columns:
            self.df['LossRatio'] = self.df['TotalClaims'] / self.df['TotalPremium']
        if 'Margin' not in self.df.columns:
            self.df['Margin'] = self.df['TotalPremium'] - self.df['TotalClaims']
        
        metrics = {
            'total_premium': self.df['TotalPremium'].sum(),
            'total_claims': self.df['TotalClaims'].sum(),
            'overall_loss_ratio': self.df['LossRatio'].mean(),
            'avg_margin': self.df['Margin'].mean(),
            'total_policies': len(self.df),
            'policies_with_claims': (self.df['TotalClaims'] > 0).sum(),
            'claim_frequency': (self.df['TotalClaims'] > 0).mean()
        }
        return metrics
    
    def plot_loss_ratio_by_category(self, category_col: str, figsize: Tuple[int, int] = (12, 6)):
        """
        Plot loss ratio by categorical variable.
        
        Args:
            category_col: Categorical column to group by
            figsize: Figure size
        """
        if 'LossRatio' not in self.df.columns:
            self.df['LossRatio'] = self.df['TotalClaims'] / self.df['TotalPremium']
        
        grouped = self.df.groupby(category_col)['LossRatio'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x_pos = range(len(grouped))
        ax.bar(x_pos, grouped['mean'], alpha=0.7, edgecolor='black')
        ax.errorbar(x_pos, grouped['mean'], yerr=grouped['std'], fmt='none', color='red', capsize=5)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(grouped.index, rotation=45, ha='right')
        ax.set_xlabel(category_col)
        ax.set_ylabel('Average Loss Ratio')
        ax.set_title(f'Loss Ratio by {category_col}')
        
        for i, (idx, row) in enumerate(grouped.iterrows()):
            ax.annotate(f'n={row["count"]}', (i, row['mean']), 
                       xytext=(0, 5), textcoords='offset points', ha='center', fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        return grouped
    
    def plot_premium_vs_claims_scatter(self, color_by: Optional[str] = None, figsize: Tuple[int, int] = (10, 8)):
        """
        Create scatter plot of TotalPremium vs TotalClaims.
        
        Args:
            color_by: Column to color points by
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if color_by and color_by in self.df.columns:
            scatter = ax.scatter(self.df['TotalPremium'], self.df['TotalClaims'], 
                               c=pd.Categorical(self.df[color_by]).codes, alpha=0.6, s=20)
        else:
            ax.scatter(self.df['TotalPremium'], self.df['TotalClaims'], alpha=0.5, s=20)
        
        max_val = max(self.df['TotalPremium'].max(), self.df['TotalClaims'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Premium = Claims')
        
        ax.set_xlabel('Total Premium')
        ax.set_ylabel('Total Claims')
        ax.set_title('Premium vs Claims Scatter Plot')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_temporal_trends(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot temporal trends over time.
        
        Args:
            figsize: Figure size
        """
        if 'TransactionMonth' not in self.df.columns:
            logger.warning("TransactionMonth column not found")
            return
        
        # Extract date components
        self.df['YearMonth'] = self.df['TransactionMonth'].dt.to_period('M')
        
        # Aggregate by month
        monthly_metrics = self.df.groupby('YearMonth').agg({
            'TotalPremium': 'sum',
            'TotalClaims': 'sum',
            'LossRatio': 'mean',
            'Margin': 'mean'
        }).reset_index()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Total Premium over time
        axes[0,0].plot(range(len(monthly_metrics)), monthly_metrics['TotalPremium'], marker='o', linewidth=2)
        axes[0,0].set_title('Total Premium Over Time')
        axes[0,0].set_ylabel('Total Premium (R)')
        
        # Total Claims over time
        axes[0,1].plot(range(len(monthly_metrics)), monthly_metrics['TotalClaims'], marker='o', linewidth=2, color='orange')
        axes[0,1].set_title('Total Claims Over Time')
        axes[0,1].set_ylabel('Total Claims (R)')
        
        # Loss Ratio trend
        axes[1,0].plot(range(len(monthly_metrics)), monthly_metrics['LossRatio'], marker='o', linewidth=2, color='green')
        axes[1,0].axhline(y=monthly_metrics['LossRatio'].mean(), color='r', linestyle='--', label='Average')
        axes[1,0].set_title('Loss Ratio Trend')
        axes[1,0].set_ylabel('Loss Ratio')
        axes[1,0].legend()
        
        # Margin trend
        axes[1,1].plot(range(len(monthly_metrics)), monthly_metrics['Margin'], marker='o', linewidth=2, color='purple')
        axes[1,1].axhline(y=monthly_metrics['Margin'].mean(), color='r', linestyle='--', label='Average')
        axes[1,1].set_title('Average Margin Trend')
        axes[1,1].set_ylabel('Margin (R)')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.show()
        
        return monthly_metrics
    
    def detect_outliers(self, col: str, figsize: Tuple[int, int] = (10, 6)):
        """
        Detect and visualize outliers in a numeric column.
        
        Args:
            col: Column name
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Box plot
        ax1.boxplot(self.df[col].dropna())
        ax1.set_title(f'Box Plot of {col}')
        ax1.set_ylabel(col)
        
        # Histogram
        ax2.hist(self.df[col].dropna(), bins=50, alpha=0.7, edgecolor='black')
        ax2.set_title(f'Distribution of {col}')
        ax2.set_xlabel(col)
        ax2.set_ylabel('Frequency')
        
        # IQR outlier detection
        Q1 = self.df[col].quantile(0.25)
        Q3 = self.df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
        
        ax1.axhline(y=lower_bound, color='r', linestyle='--', label=f'Lower bound: {lower_bound:.2f}')
        ax1.axhline(y=upper_bound, color='g', linestyle='--', label=f'Upper bound: {upper_bound:.2f}')
        ax1.legend()
        
        plt.tight_layout()
        plt.show()
        
        print(f"Found {len(outliers)} outliers ({len(outliers)/len(self.df)*100:.2f}% of data)")
        return outliers