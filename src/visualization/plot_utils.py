"""
Visualization utilities for creating publication-quality charts
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Color schemes
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#06A77D',
    'warning': '#E63946',
    'neutral': '#6C757D'
}

PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#E63946', '#6C757D']


def set_plot_style(style: str = 'professional'):
    """
    Set consistent plot styling
    
    Args:
        style: 'professional', 'minimal', or 'presentation'
    """
    if style == 'professional':
        plt.rcParams.update({
            'figure.figsize': (12, 6),
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'axes.grid': True,
            'grid.alpha': 0.3
        })
    elif style == 'minimal':
        sns.set_style('white')
        plt.rcParams['figure.figsize'] = (10, 5)
    elif style == 'presentation':
        plt.rcParams.update({
            'figure.figsize': (14, 7),
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'figure.titlesize': 20
        })


def plot_time_series(df: pd.DataFrame, date_col: str, value_col: str, 
                     title: str = "Time Series Analysis", 
                     save_path: Optional[str] = None):
    """
    Create professional time series plot
    
    Args:
        df: Input dataframe
        date_col: Date column name
        value_col: Value column name
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df[date_col], df[value_col], linewidth=2.5, color=COLORS['primary'], label=value_col)
    ax.fill_between(df[date_col], df[value_col], alpha=0.3, color=COLORS['primary'])
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(date_col.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(value_col.replace('_', ' ').title(), fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved plot to {save_path}")
    
    plt.show()


def plot_distribution(data: pd.Series, title: str = "Distribution Analysis",
                      bins: int = 30, save_path: Optional[str] = None):
    """
    Create distribution plot with histogram and KDE
    
    Args:
        data: Data series to plot
        title: Plot title
        bins: Number of histogram bins
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram with KDE
    ax1.hist(data, bins=bins, alpha=0.7, color=COLORS['primary'], edgecolor='black')
    ax1.set_title(f"{title} - Histogram", fontweight='bold')
    ax1.set_xlabel(data.name)
    ax1.set_ylabel("Frequency")
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    bp = ax2.boxplot(data, vert=True, patch_artist=True,
                     boxprops=dict(facecolor=COLORS['primary'], alpha=0.7),
                     medianprops=dict(color='red', linewidth=2))
    ax2.set_title(f"{title} - Box Plot", fontweight='bold')
    ax2.set_ylabel(data.name)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved plot to {save_path}")
    
    plt.show()


def plot_categorical(df: pd.DataFrame, column: str, top_n: int = 10,
                     title: Optional[str] = None, save_path: Optional[str] = None):
    """
    Create bar plot for categorical data
    
    Args:
        df: Input dataframe
        column: Column to plot
        top_n: Show top N categories
        title: Plot title
        save_path: Optional path to save figure
    """
    value_counts = df[column].value_counts().head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.barh(range(len(value_counts)), value_counts.values, color=PALETTE[:len(value_counts)])
    ax.set_yticks(range(len(value_counts)))
    ax.set_yticklabels(value_counts.index)
    ax.set_xlabel("Count", fontsize=12)
    ax.set_title(title or f"Top {top_n} {column.replace('_', ' ').title()}", 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, value_counts.values)):
        ax.text(value, i, f' {value:,}', va='center', fontsize=10)
    
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved plot to {save_path}")
    
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Matrix",
                             save_path: Optional[str] = None):
    """
    Create correlation heatmap
    
    Args:
        df: Input dataframe
        title: Plot title
        save_path: Optional path to save figure
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved plot to {save_path}")
    
    plt.show()


def plot_geographic_comparison(df: pd.DataFrame, region_col: str, value_col: str,
                               top_n: int = 15, title: Optional[str] = None,
                               save_path: Optional[str] = None):
    """
    Create geographic comparison chart
    
    Args:
        df: Input dataframe
        region_col: Region/state column
        value_col: Value column to plot
        top_n: Show top N regions
        title: Plot title
        save_path: Optional path to save figure
    """
    region_data = df.groupby(region_col)[value_col].sum().sort_values(ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.bar(range(len(region_data)), region_data.values, color=PALETTE * (top_n // len(PALETTE) + 1))
    ax.set_xticks(range(len(region_data)))
    ax.set_xticklabels(region_data.index, rotation=45, ha='right')
    ax.set_ylabel(value_col.replace('_', ' ').title(), fontsize=12)
    ax.set_title(title or f"Top {top_n} {region_col.replace('_', ' ').title()} by {value_col.replace('_', ' ').title()}", 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels
    for bar, value in zip(bars, region_data.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:,.0f}', ha='center', va='bottom', fontsize=9)
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved plot to {save_path}")
    
    plt.show()


def create_dashboard(plots_config: List[dict], title: str = "Data Dashboard",
                     save_path: Optional[str] = None):
    """
    Create a multi-panel dashboard
    
    Args:
        plots_config: List of plot configurations
        title: Dashboard title
        save_path: Optional path to save figure
    """
    n_plots = len(plots_config)
    ncols = 2
    nrows = (n_plots + 1) // 2
    
    fig = plt.figure(figsize=(16, 6 * nrows))
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.995)
    
    for idx, config in enumerate(plots_config, 1):
        ax = plt.subplot(nrows, ncols, idx)
        # Custom plotting logic based on config
        # This is a template - implement specific plot types as needed
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Saved dashboard to {save_path}")
    
    plt.show()


# Initialize style on import
set_plot_style('professional')


if __name__ == "__main__":
    print("Visualization utilities loaded successfully!")
    print("\nAvailable functions:")
    print("  - set_plot_style()")
    print("  - plot_time_series()")
    print("  - plot_distribution()")
    print("  - plot_categorical()")
    print("  - plot_correlation_heatmap()")
    print("  - plot_geographic_comparison()")
    print("  - create_dashboard()")
