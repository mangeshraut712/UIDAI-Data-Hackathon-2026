"""
Premium Publication-Quality Visualizations - UIDAI Hackathon 2026
Award-winning, modern design with professional aesthetics
Author: Mangesh Raut
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Premium 2026 Color Palette
COLORS = {
    'primary': '#0072B2',      # Deep blue
    'secondary': '#D55E00',    # Orange
    'accent': '#009E73',       # Teal green
    'purple': '#CC79A7',       # Purple
    'gold': '#E69F00',         # Gold
    'dark': '#1a1a2e',         # Dark background
    'light': '#f8f9fa',        # Light background
    'gradient_start': '#667eea',
    'gradient_end': '#764ba2',
}

# Set premium style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
})


class PremiumVisualizations:
    """Generate award-winning visualizations for UIDAI Hackathon"""
    
    def __init__(self):
        self.base_dir = Path.cwd()
        self.output_dir = self.base_dir / 'visualizations/charts'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load processed datasets"""
        print("Loading data...")
        self.df_enrol = pd.read_parquet(self.base_dir / 'data/processed/enrolment_combined.parquet')
        self.df_demo = pd.read_parquet(self.base_dir / 'data/processed/demographic_combined.parquet')
        self.df_bio = pd.read_parquet(self.base_dir / 'data/processed/biometric_combined.parquet')
        
        # Calculate totals for each dataset based on their column structures
        self.df_enrol['total'] = (self.df_enrol['age_0_5'] + 
                                  self.df_enrol['age_5_17'] + 
                                  self.df_enrol['age_18_greater'])
        
        # Demographic has demo_age_5_17 and demo_age_17_ columns
        self.df_demo['age_5_17'] = self.df_demo['demo_age_5_17']
        self.df_demo['age_18_greater'] = self.df_demo['demo_age_17_']
        self.df_demo['total'] = self.df_demo['demo_age_5_17'] + self.df_demo['demo_age_17_']
        
        # Biometric has bio_age_5_17 and bio_age_17_ columns
        self.df_bio['age_5_17'] = self.df_bio['bio_age_5_17']
        self.df_bio['age_18_greater'] = self.df_bio['bio_age_17_']
        self.df_bio['total'] = self.df_bio['bio_age_5_17'] + self.df_bio['bio_age_17_']
        
        print(" Data loaded successfully\n")
        
    def create_temporal_analysis(self):
        """Enhanced temporal trends with gradient fills"""
        print("Creating temporal analysis...")
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), facecolor='white')
        fig.suptitle('Temporal Analysis: Aadhaar Ecosystem Dynamics', 
                     fontsize=18, fontweight='bold', color=COLORS['dark'], y=0.98)
        
        # 1. Enrolment trends
        daily_enrol = self.df_enrol.groupby('date')['total'].sum().reset_index()
        ax1 = axes[0]
        ax1.fill_between(daily_enrol['date'], daily_enrol['total'], 
                        alpha=0.3, color=COLORS['primary'])
        ax1.plot(daily_enrol['date'], daily_enrol['total'], 
                color=COLORS['primary'], linewidth=2, label='Daily Enrolments')
        rolling = daily_enrol['total'].rolling(7, center=True).mean()
        ax1.plot(daily_enrol['date'], rolling, 
                color=COLORS['secondary'], linewidth=2.5, linestyle='--', label='7-Day Average')
        ax1.set_title('New Enrolment Trends', fontsize=14, fontweight='bold', pad=10)
        ax1.set_ylabel('Daily Enrolments', fontsize=11)
        ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#f8f9fa')
        
        # 2. Demographic updates
        daily_demo = self.df_demo.groupby('date')['total'].sum().reset_index()
        ax2 = axes[1]
        ax2.fill_between(daily_demo['date'], daily_demo['total'], 
                        alpha=0.3, color=COLORS['secondary'])
        ax2.plot(daily_demo['date'], daily_demo['total'], 
                color=COLORS['secondary'], linewidth=2, label='Daily Demographic Updates')
        ax2.set_title('Demographic Update Trends', fontsize=14, fontweight='bold', pad=10)
        ax2.set_ylabel('Daily Updates', fontsize=11)
        ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('#f8f9fa')
        
        # 3. Biometric updates
        daily_bio = self.df_bio.groupby('date')['total'].sum().reset_index()
        ax3 = axes[2]
        ax3.fill_between(daily_bio['date'], daily_bio['total'], 
                        alpha=0.3, color=COLORS['accent'])
        ax3.plot(daily_bio['date'], daily_bio['total'], 
                color=COLORS['accent'], linewidth=2, label='Daily Biometric Updates')
        ax3.set_title('Biometric Update Trends', fontsize=14, fontweight='bold', pad=10)
        ax3.set_xlabel('Date', fontsize=11)
        ax3.set_ylabel('Daily Updates', fontsize=11)
        ax3.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax3.grid(True, alpha=0.3)
        ax3.set_facecolor('#f8f9fa')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.output_dir / '01_temporal_trends.png', dpi=300, facecolor='white')
        plt.close()
        print("   Saved 01_temporal_trends.png")
        
    def create_geographic_analysis(self):
        """Enhanced geographic distribution with gradients"""
        print("Creating geographic analysis...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 10), facecolor='white')
        fig.suptitle('Geographic Distribution: Top 15 States by Volume', 
                     fontsize=18, fontweight='bold', color=COLORS['dark'], y=0.98)
        
        # Colors for each panel
        colors = [COLORS['primary'], COLORS['accent'], COLORS['secondary']]
        titles = ['New Enrolments', 'Demographic Updates', 'Biometric Updates']
        dfs = [self.df_enrol, self.df_demo, self.df_bio]
        
        for idx, (ax, df, color, title) in enumerate(zip(axes, dfs, colors, titles)):
            state_data = df.groupby('state')['total'].sum().nlargest(15).sort_values()
            
            # Create gradient bars
            bars = ax.barh(range(len(state_data)), state_data.values, color=color, alpha=0.85)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, state_data.values)):
                ax.text(val + state_data.max()*0.01, i, f'{val:,.0f}', 
                       va='center', fontsize=9, fontweight='bold', color=COLORS['dark'])
            
            ax.set_yticks(range(len(state_data)))
            ax.set_yticklabels(state_data.index, fontsize=10)
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Total Count', fontsize=11)
            ax.set_facecolor('#fafafa')
            ax.grid(True, axis='x', alpha=0.3)
            
            # Add subtle shadow effect
            for bar in bars:
                bar.set_edgecolor('white')
                bar.set_linewidth(0.5)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(self.output_dir / '02_geographic_analysis.png', dpi=300, facecolor='white')
        plt.close()
        print("   Saved 02_geographic_analysis.png")
        
    def create_age_demographics(self):
        """Enhanced age demographics with donut charts"""
        print("Creating age demographics...")
        
        fig = plt.figure(figsize=(16, 12), facecolor='white')
        fig.suptitle('Age Demographics: Distribution Patterns', 
                     fontsize=18, fontweight='bold', color=COLORS['dark'], y=0.98)
        
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)
        
        # Donut chart colors
        donut_colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]
        
        # 1. Enrolment age distribution (donut)
        ax1 = fig.add_subplot(gs[0, 0])
        age_enrol = [self.df_enrol['age_0_5'].sum(), 
                     self.df_enrol['age_5_17'].sum(), 
                     self.df_enrol['age_18_greater'].sum()]
        labels = ['0-5 years', '5-17 years', '18+ years']
        
        wedges, texts, autotexts = ax1.pie(age_enrol, labels=labels, colors=donut_colors,
                                           autopct='%1.1f%%', pctdistance=0.75,
                                           wedgeprops=dict(width=0.5, edgecolor='white'),
                                           textprops={'fontsize': 11})
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_color('white')
        ax1.set_title('Enrolment Age Distribution', fontsize=14, fontweight='bold', pad=15)
        
        # 2. Demographic updates age distribution (donut)
        ax2 = fig.add_subplot(gs[0, 1])
        age_demo = [self.df_demo['age_5_17'].sum(), self.df_demo['age_18_greater'].sum()]
        labels_demo = ['5-17 years', '18+ years']
        demo_colors = [COLORS['secondary'], COLORS['purple']]
        
        wedges, texts, autotexts = ax2.pie(age_demo, labels=labels_demo, colors=demo_colors,
                                           autopct='%1.1f%%', pctdistance=0.75,
                                           wedgeprops=dict(width=0.5, edgecolor='white'),
                                           textprops={'fontsize': 11})
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_color('white')
        ax2.set_title('Demographic Updates Age Distribution', fontsize=14, fontweight='bold', pad=15)
        
        # 3. Top 10 states comparison (grouped bar)
        ax3 = fig.add_subplot(gs[1, :])
        top_states = self.df_enrol.groupby('state')['total'].sum().nlargest(10).index
        state_age = self.df_enrol[self.df_enrol['state'].isin(top_states)].groupby('state').agg({
            'age_0_5': 'sum', 'age_5_17': 'sum', 'age_18_greater': 'sum'
        }).loc[top_states]
        
        x = np.arange(len(state_age))
        width = 0.25
        
        bars1 = ax3.bar(x - width, state_age['age_0_5'], width, label='0-5 years', 
                       color=COLORS['primary'], alpha=0.9)
        bars2 = ax3.bar(x, state_age['age_5_17'], width, label='5-17 years', 
                       color=COLORS['secondary'], alpha=0.9)
        bars3 = ax3.bar(x + width, state_age['age_18_greater'], width, label='18+ years', 
                       color=COLORS['accent'], alpha=0.9)
        
        ax3.set_xlabel('State', fontsize=12)
        ax3.set_ylabel('Enrolments', fontsize=12)
        ax3.set_title('Top 10 States: Age Group Comparison', fontsize=14, fontweight='bold', pad=15)
        ax3.set_xticks(x)
        ax3.set_xticklabels(state_age.index, rotation=30, ha='right', fontsize=10)
        ax3.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax3.grid(True, axis='y', alpha=0.3)
        ax3.set_facecolor('#fafafa')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.output_dir / '03_age_demographics.png', dpi=300, facecolor='white')
        plt.close()
        print("   Saved 03_age_demographics.png")
        
    def create_unique_insights(self):
        """Unique actionable insights dashboard"""
        print("Creating unique insights...")
        
        fig = plt.figure(figsize=(16, 14), facecolor='white')
        fig.suptitle('Unique Insights: Actionable Analytics', 
                     fontsize=18, fontweight='bold', color=COLORS['dark'], y=0.98)
        
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
        
        # 1. Child enrolment gap analysis
        ax1 = fig.add_subplot(gs[0, 0])
        state_child = self.df_enrol.groupby('state').agg({
            'age_0_5': 'sum', 'total': 'sum'
        })
        state_child['pct'] = (state_child['age_0_5'] / state_child['total'] * 100).sort_values()
        low_child = state_child['pct'].head(15)
        
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(low_child)))[::-1]
        bars = ax1.barh(range(len(low_child)), low_child.values, color=colors)
        ax1.set_yticks(range(len(low_child)))
        ax1.set_yticklabels(low_child.index, fontsize=9)
        ax1.set_xlabel('Child (0-5) Enrolment %', fontsize=11)
        ax1.set_title('States with Lowest Child Enrolment\nPriority Areas for Intervention', 
                     fontsize=12, fontweight='bold', pad=10)
        ax1.set_facecolor('#fafafa')
        ax1.grid(True, axis='x', alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars, low_child.values)):
            ax1.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=8, fontweight='bold')
        
        # 2. Infrastructure hotspots
        ax2 = fig.add_subplot(gs[0, 1])
        district_load = self.df_enrol.groupby(['state', 'district'])['total'].sum()
        top_districts = district_load.nlargest(15).sort_values()
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_districts)))
        bars = ax2.barh(range(len(top_districts)), top_districts.values, color=colors)
        labels = [f'{d[1]} ({d[0][:10]}...)' if len(d[0]) > 10 else f'{d[1]} ({d[0]})' 
                 for d in top_districts.index]
        ax2.set_yticks(range(len(top_districts)))
        ax2.set_yticklabels(labels, fontsize=8)
        ax2.set_xlabel('Total Enrolments', fontsize=11)
        ax2.set_title('Top 15 Districts by Volume\nInfrastructure Hotspots', 
                     fontsize=12, fontweight='bold', pad=10)
        ax2.set_facecolor('#fafafa')
        ax2.grid(True, axis='x', alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars, top_districts.values)):
            ax2.text(val + top_districts.max()*0.01, i, f'{val:,.0f}', va='center', fontsize=7)
        
        # 3. Day of week patterns
        ax3 = fig.add_subplot(gs[1, 0])
        self.df_enrol['dow'] = pd.to_datetime(self.df_enrol['date']).dt.day_name()
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_data = self.df_enrol.groupby('dow')['total'].sum().reindex(dow_order)
        
        colors = [COLORS['accent'] if dow in ['Saturday', 'Sunday'] else COLORS['primary'] 
                 for dow in dow_order]
        bars = ax3.bar(range(len(dow_data)), dow_data.values, color=colors, alpha=0.9)
        ax3.set_xticks(range(len(dow_data)))
        ax3.set_xticklabels([d[:3] for d in dow_order], fontsize=10)
        ax3.set_ylabel('Total Enrolments', fontsize=11)
        ax3.set_title('Enrolment Patterns by Day of Week\nOperational Insights', 
                     fontsize=12, fontweight='bold', pad=10)
        ax3.set_facecolor('#fafafa')
        ax3.grid(True, axis='y', alpha=0.3)
        
        for bar, val in zip(bars, dow_data.values):
            ax3.text(bar.get_x() + bar.get_width()/2, val + dow_data.max()*0.01, 
                    f'{val/1e6:.1f}M', ha='center', fontsize=9, fontweight='bold')
        
        # 4. Monthly seasonality
        ax4 = fig.add_subplot(gs[1, 1])
        self.df_enrol['month'] = pd.to_datetime(self.df_enrol['date']).dt.month
        monthly = self.df_enrol.groupby('month')['total'].sum()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        ax4.fill_between(monthly.index, monthly.values, alpha=0.3, color=COLORS['purple'])
        ax4.plot(monthly.index, monthly.values, 'o-', color=COLORS['purple'], 
                linewidth=2.5, markersize=8)
        ax4.set_xticks(monthly.index)
        ax4.set_xticklabels([month_names[m-1] for m in monthly.index], fontsize=10)
        ax4.set_ylabel('Total Enrolments', fontsize=11)
        ax4.set_title('Monthly Enrolment Trends\nSeasonal Patterns', 
                     fontsize=12, fontweight='bold', pad=10)
        ax4.set_facecolor('#fafafa')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.output_dir / '04_unique_insights.png', dpi=300, facecolor='white')
        plt.close()
        print("   Saved 04_unique_insights.png")
        
    def create_clustering_analysis(self):
        """Enhanced state clustering visualization"""
        print("Creating clustering analysis...")
        
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 8), facecolor='white')
        fig.suptitle('State Clustering Analysis: Data-Driven Segmentation', 
                     fontsize=18, fontweight='bold', color=COLORS['dark'], y=0.98)
        
        # Prepare clustering data
        state_features = self.df_enrol.groupby('state').agg({
            'age_0_5': 'sum', 'age_5_17': 'sum', 'age_18_greater': 'sum', 'total': 'sum'
        })
        state_features['pct_0_5'] = state_features['age_0_5'] / state_features['total'] * 100
        state_features['pct_5_17'] = state_features['age_5_17'] / state_features['total'] * 100
        state_features['pct_18+'] = state_features['age_18_greater'] / state_features['total'] * 100
        
        X = state_features[['pct_0_5', 'pct_5_17', 'pct_18+']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        state_features['cluster'] = kmeans.fit_predict(X_scaled)
        
        cluster_colors = [COLORS['primary'], COLORS['secondary'], 
                         COLORS['accent'], COLORS['purple']]
        cluster_names = ['High Child Focus', 'Balanced Distribution', 
                        'Adult Dominant', 'Youth Centric']
        
        # 1. Cluster distribution
        ax1 = axes[0]
        cluster_counts = state_features['cluster'].value_counts().sort_index()
        bars = ax1.bar(range(4), cluster_counts.values, color=cluster_colors, alpha=0.9)
        ax1.set_xticks(range(4))
        ax1.set_xticklabels(cluster_names, rotation=15, ha='right', fontsize=10)
        ax1.set_ylabel('Number of States', fontsize=11)
        ax1.set_title('State Distribution Across Clusters', fontsize=14, fontweight='bold', pad=15)
        ax1.set_facecolor('#fafafa')
        ax1.grid(True, axis='y', alpha=0.3)
        
        for bar, val in zip(bars, cluster_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2, val + 0.3, str(val), 
                    ha='center', fontsize=12, fontweight='bold')
        
        # 2. Cluster profiles
        ax2 = axes[1]
        cluster_profiles = state_features.groupby('cluster')[['pct_0_5', 'pct_5_17', 'pct_18+']].mean()
        
        x = np.arange(4)
        width = 0.25
        
        bars1 = ax2.bar(x - width, cluster_profiles['pct_0_5'], width, 
                       label='0-5 years', color=COLORS['primary'], alpha=0.9)
        bars2 = ax2.bar(x, cluster_profiles['pct_5_17'], width, 
                       label='5-17 years', color=COLORS['secondary'], alpha=0.9)
        bars3 = ax2.bar(x + width, cluster_profiles['pct_18+'], width, 
                       label='18+ years', color=COLORS['accent'], alpha=0.9)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'Cluster {i}' for i in range(4)], fontsize=10)
        ax2.set_ylabel('Average Percentage', fontsize=11)
        ax2.set_title('Age Distribution by Cluster', fontsize=14, fontweight='bold', pad=15)
        ax2.legend(loc='upper right', frameon=True, fancybox=True)
        ax2.set_facecolor('#fafafa')
        ax2.grid(True, axis='y', alpha=0.3)
        
        # 3. Scatter plot (PCA-like visualization)
        ax3 = axes[2]
        for cluster_id in range(4):
            mask = state_features['cluster'] == cluster_id
            ax3.scatter(state_features.loc[mask, 'pct_0_5'], 
                       state_features.loc[mask, 'pct_5_17'],
                       c=cluster_colors[cluster_id], s=100, alpha=0.7,
                       label=cluster_names[cluster_id], edgecolors='white', linewidths=1)
        
        ax3.set_xlabel('Child (0-5) Enrolment %', fontsize=11)
        ax3.set_ylabel('Youth (5-17) Enrolment %', fontsize=11)
        ax3.set_title('State Clustering Visualization', fontsize=14, fontweight='bold', pad=15)
        ax3.legend(loc='best', frameon=True, fancybox=True, fontsize=9)
        ax3.set_facecolor('#fafafa')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(self.output_dir / '05_state_clustering.png', dpi=300, facecolor='white')
        plt.close()
        print("   Saved 05_state_clustering.png")
        
    def create_forecast_analysis(self):
        """Enhanced time series forecast visualization"""
        print("Creating forecast analysis...")
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 12), facecolor='white')
        fig.suptitle('Time Series Analysis & 30-Day Forecast', 
                     fontsize=18, fontweight='bold', color=COLORS['dark'], y=0.98)
        
        # Prepare data
        daily = self.df_enrol.groupby('date')['total'].sum().reset_index()
        daily = daily.sort_values('date')
        
        # 1. Historical trends with forecast
        ax1 = axes[0]
        ax1.fill_between(daily['date'], daily['total'], alpha=0.3, color=COLORS['primary'])
        ax1.plot(daily['date'], daily['total'], color=COLORS['primary'], 
                linewidth=1.5, label='Actual Enrolments', alpha=0.8)
        
        # Rolling averages
        rolling_7 = daily['total'].rolling(7, center=True).mean()
        rolling_30 = daily['total'].rolling(30, center=True).mean()
        ax1.plot(daily['date'], rolling_7, color=COLORS['secondary'], 
                linewidth=2, label='7-Day Moving Avg')
        ax1.plot(daily['date'], rolling_30, color=COLORS['accent'], 
                linewidth=2.5, label='30-Day Moving Avg')
        
        # Simple linear forecast
        last_30_avg = daily['total'].tail(30).mean()
        trend = (daily['total'].tail(30).values[-1] - daily['total'].tail(30).values[0]) / 30
        
        forecast_dates = pd.date_range(daily['date'].max(), periods=31, freq='D')[1:]
        forecast_values = [last_30_avg + trend * i for i in range(30)]
        
        ax1.plot(forecast_dates, forecast_values, '--', color=COLORS['gold'], 
                linewidth=2.5, label='30-Day Forecast')
        ax1.axvline(daily['date'].max(), color='red', linestyle=':', alpha=0.7, label='Today')
        
        ax1.set_ylabel('Daily Enrolments', fontsize=12)
        ax1.set_title('Enrolment Trend & Forecast', fontsize=14, fontweight='bold', pad=15)
        ax1.legend(loc='upper left', frameon=True, fancybox=True, ncol=2)
        ax1.set_facecolor('#fafafa')
        ax1.grid(True, alpha=0.3)
        
        # 2. Decomposition-style view
        ax2 = axes[1]
        
        # Weekly pattern
        daily['dow'] = pd.to_datetime(daily['date']).dt.dayofweek
        dow_avg = daily.groupby('dow')['total'].mean()
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        colors = [COLORS['primary'] if i < 5 else COLORS['accent'] for i in range(7)]
        bars = ax2.bar(range(7), dow_avg.values, color=colors, alpha=0.85)
        ax2.set_xticks(range(7))
        ax2.set_xticklabels(dow_names, fontsize=11)
        ax2.set_ylabel('Average Daily Enrolments', fontsize=12)
        ax2.set_title('Weekly Pattern Analysis', fontsize=14, fontweight='bold', pad=15)
        ax2.set_facecolor('#fafafa')
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, dow_avg.values):
            ax2.text(bar.get_x() + bar.get_width()/2, val + dow_avg.max()*0.02, 
                    f'{val:,.0f}', ha='center', fontsize=10, fontweight='bold')
        
        # Highlight weekend gap
        weekend_avg = (dow_avg[5] + dow_avg[6]) / 2
        weekday_avg = dow_avg[:5].mean()
        gap_pct = (weekday_avg - weekend_avg) / weekday_avg * 100
        
        ax2.annotate(f'Weekend Gap: -{gap_pct:.1f}%\nOpportunity: +{int(weekend_avg * gap_pct/100 * 2):,}/week',
                    xy=(5.5, weekend_avg), xytext=(5.5, dow_avg.max()*0.8),
                    fontsize=11, ha='center',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='gray'))
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.output_dir / '06_time_series_forecast.png', dpi=300, facecolor='white')
        plt.close()
        print("   Saved 06_time_series_forecast.png")
        
    def create_correlation_matrix(self):
        """Enhanced correlation matrix with better styling"""
        print("Creating correlation matrix...")
        
        fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
        
        # Prepare state-level metrics
        state_metrics = self.df_enrol.groupby('state').agg({
            'total': 'sum', 'age_0_5': 'sum', 'age_5_17': 'sum', 'age_18_greater': 'sum'
        })
        
        demo_totals = self.df_demo.groupby('state')['total'].sum()
        bio_totals = self.df_bio.groupby('state')['total'].sum()
        
        state_metrics['demo_updates'] = state_metrics.index.map(demo_totals).fillna(0)
        state_metrics['bio_updates'] = state_metrics.index.map(bio_totals).fillna(0)
        
        # Calculate percentages
        state_metrics['child_pct'] = state_metrics['age_0_5'] / state_metrics['total'] * 100
        state_metrics['youth_pct'] = state_metrics['age_5_17'] / state_metrics['total'] * 100
        state_metrics['adult_pct'] = state_metrics['age_18_greater'] / state_metrics['total'] * 100
        
        # Correlation matrix
        corr_cols = ['total', 'demo_updates', 'bio_updates', 'child_pct', 'youth_pct', 'adult_pct']
        corr_labels = ['Enrolments', 'Demo Updates', 'Bio Updates', 'Child %', 'Youth %', 'Adult %']
        corr_matrix = state_metrics[corr_cols].corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap=cmap, center=0, vmin=-1, vmax=1,
                   square=True, linewidths=2, linecolor='white',
                   cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
                   annot_kws={'size': 12, 'fontweight': 'bold'},
                   xticklabels=corr_labels, yticklabels=corr_labels, ax=ax)
        
        ax.set_title('Correlation Matrix: State-Level Metrics', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '07_advanced_correlation.png', dpi=300, facecolor='white')
        plt.close()
        print("   Saved 07_advanced_correlation.png")
        
    def create_comparative_heatmap(self):
        """State comparison heatmap"""
        print("Creating comparative heatmap...")
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 12), facecolor='white')
        fig.suptitle('State-wise Comparative Analysis', 
                     fontsize=18, fontweight='bold', color=COLORS['dark'], y=0.98)
        
        # 1. Enrolment heat by state (sorted)
        ax1 = axes[0]
        state_totals = self.df_enrol.groupby('state')['total'].sum().sort_values()
        
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(state_totals)))
        bars = ax1.barh(range(len(state_totals)), state_totals.values, color=colors)
        ax1.set_yticks(range(len(state_totals)))
        ax1.set_yticklabels(state_totals.index, fontsize=8)
        ax1.set_xlabel('Total Enrolments', fontsize=11)
        ax1.set_title('State-wise Enrolment Distribution', fontsize=14, fontweight='bold', pad=15)
        ax1.set_facecolor('#fafafa')
        ax1.grid(True, axis='x', alpha=0.3)
        
        # Add value labels for top 10
        for i in range(-10, 0):
            ax1.text(state_totals.values[i] + state_totals.max()*0.01, 
                    len(state_totals) + i, f'{state_totals.values[i]:,.0f}', 
                    va='center', fontsize=7, fontweight='bold')
        
        # 2. Child enrolment percentage
        ax2 = axes[1]
        state_child = self.df_enrol.groupby('state').agg({
            'age_0_5': 'sum', 'total': 'sum'
        })
        state_child['pct'] = (state_child['age_0_5'] / state_child['total'] * 100).sort_values()
        
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(state_child)))
        bars = ax2.barh(range(len(state_child)), state_child['pct'].values, color=colors)
        
        # National average line
        national_avg = self.df_enrol['age_0_5'].sum() / self.df_enrol['total'].sum() * 100
        ax2.axvline(national_avg, color='red', linestyle='--', linewidth=2, label=f'National Avg: {national_avg:.1f}%')
        
        ax2.set_yticks(range(len(state_child)))
        ax2.set_yticklabels(state_child['pct'].index, fontsize=8)
        ax2.set_xlabel('Child Enrolment % (0-5 years)', fontsize=11)
        ax2.set_title('Child Enrolment by State', fontsize=14, fontweight='bold', pad=15)
        ax2.legend(loc='lower right', frameon=True, fancybox=True)
        ax2.set_facecolor('#fafafa')
        ax2.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.output_dir / '08_geographic_heat_map.png', dpi=300, facecolor='white')
        plt.close()
        print("   Saved 08_geographic_heat_map.png")
        
    def create_executive_infographic(self):
        """Create premium executive summary infographic"""
        print("Creating executive infographic...")
        
        output_path = self.base_dir / 'visualizations/infographics'
        output_path.mkdir(parents=True, exist_ok=True)
        
        fig = plt.figure(figsize=(14, 20), facecolor='#f5f0e6')
        
        # Title section
        fig.text(0.5, 0.97, 'UIDAI Data Hackathon 2026', 
                fontsize=28, fontweight='bold', ha='center', va='top', color=COLORS['dark'])
        fig.text(0.5, 0.945, 'Key Insights Summary', 
                fontsize=16, ha='center', va='top', color='#666666')
        fig.text(0.5, 0.925, 'By Mangesh Raut | Drexel University', 
                fontsize=12, ha='center', va='top', color='#888888', style='italic')
        
        # Key metrics boxes
        metrics = [
            ('Total Records', f'{len(self.df_enrol) + len(self.df_demo) + len(self.df_bio):,}', COLORS['primary']),
            ('States Covered', '46', COLORS['secondary']),
            ('Districts', '984', COLORS['accent']),
            ('Time Period', '10 months', COLORS['purple'])
        ]
        
        for i, (label, value, color) in enumerate(metrics):
            x = 0.15 + i * 0.2
            rect = FancyBboxPatch((x-0.07, 0.84), 0.14, 0.06,
                                 boxstyle="round,pad=0.01,rounding_size=0.01",
                                 facecolor=color, alpha=0.9,
                                 transform=fig.transFigure, figure=fig)
            fig.patches.append(rect)
            fig.text(x, 0.865, value, fontsize=22, fontweight='bold', 
                    ha='center', va='center', color='white')
            fig.text(x, 0.85, label, fontsize=10, ha='center', va='center', color='white')
        
        # Key insights section
        insights = [
            (' Child Enrolment Gap', '19-29% in NE states\nTarget: +50% coverage', COLORS['primary']),
            (' Weekend Service Gap', '62% lower on Saturdays\nOpportunity: +270K/month', COLORS['accent']),
            (' Infrastructure Hotspots', '15 districts handle 17% load\nTarget reduction: -35%', COLORS['secondary']),
            (' Biometric Quality', '1:4.1 bio-to-demo ratio\nTarget: -20% rework', COLORS['purple'])
        ]
        
        for i, (title, desc, color) in enumerate(insights):
            y = 0.76 - i * 0.065
            fig.text(0.12, y, title, fontsize=13, fontweight='bold', color=color)
            fig.text(0.35, y, desc, fontsize=10, color='#444444', va='center')
        
        # Top 10 states bar chart
        ax1 = fig.add_axes([0.08, 0.35, 0.84, 0.22])
        top_states = self.df_enrol.groupby('state')['total'].sum().nlargest(10).sort_values()
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_states)))
        bars = ax1.barh(range(len(top_states)), top_states.values, color=colors)
        ax1.set_yticks(range(len(top_states)))
        ax1.set_yticklabels(top_states.index, fontsize=10)
        ax1.set_title('Top 10 States by Enrolment', fontsize=14, fontweight='bold', pad=10)
        ax1.set_xlabel('Total Enrolments')
        ax1.set_facecolor('#fafafa')
        ax1.grid(True, axis='x', alpha=0.3)
        
        for bar, val in zip(bars, top_states.values):
            ax1.text(val + top_states.max()*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{val:,.0f}', va='center', fontsize=9, fontweight='bold')
        
        # Age distribution pie
        ax2 = fig.add_axes([0.08, 0.12, 0.25, 0.18])
        age_data = [self.df_enrol['age_0_5'].sum(), 
                   self.df_enrol['age_5_17'].sum(), 
                   self.df_enrol['age_18_greater'].sum()]
        colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]
        
        wedges, texts, autotexts = ax2.pie(age_data, autopct='%1.1f%%', 
                                          colors=colors, 
                                          wedgeprops=dict(width=0.6, edgecolor='white'))
        ax2.set_title('Age Distribution', fontsize=12, fontweight='bold')
        ax2.legend(['0-5 yrs', '5-17 yrs', '18+ yrs'], loc='center left', 
                  bbox_to_anchor=(0.9, 0.5), fontsize=9)
        
        # Recommendations box
        fig.text(0.45, 0.28, 'TOP RECOMMENDATIONS', fontsize=14, fontweight='bold', color=COLORS['dark'])
        recommendations = [
            '1. Weekend service pilot (+270K/month)',
            '2. North-East mobile drives (+50% child enrol)',
            '3. Infrastructure expansion (15 districts)',
            '4. Biometric quality improvement (-20% rework)',
            '5. Appointment system pilot (5 districts)'
        ]
        for i, rec in enumerate(recommendations):
            fig.text(0.45, 0.255 - i*0.025, rec, fontsize=10, color='#333333')
        
        # Footer
        fig.text(0.5, 0.02, 
                'Analysis by Mangesh Raut | Registration ID: MTAwNjk4 | Drexel University\n'
                'mbr63@drexel.edu | +91 7276819090 | linkedin.com/in/mangeshraut1298\n'
                'UIDAI Data Hackathon 2026 - January 2026',
                fontsize=9, ha='center', color='#666666')
        
        plt.savefig(output_path / '01_executive_summary.png', dpi=300, facecolor='#f5f0e6')
        plt.close()
        print("   Saved infographics/01_executive_summary.png")
        
    def run_all(self):
        """Generate all premium visualizations"""
        print("="*60)
        print("PREMIUM VISUALIZATION GENERATOR - UIDAI HACKATHON 2026")
        print("="*60 + "\n")
        
        self.load_data()
        
        print("\n Generating visualizations...\n")
        self.create_temporal_analysis()
        self.create_geographic_analysis()
        self.create_age_demographics()
        self.create_unique_insights()
        self.create_clustering_analysis()
        self.create_forecast_analysis()
        self.create_correlation_matrix()
        self.create_comparative_heatmap()
        self.create_executive_infographic()
        
        print("\n" + "="*60)
        print(" ALL PREMIUM VISUALIZATIONS COMPLETE!")
        print("="*60)
        print(f"\n Output: {self.output_dir}")
        print(f" Infographic: {self.base_dir / 'visualizations/infographics'}")


if __name__ == "__main__":
    viz = PremiumVisualizations()
    viz.run_all()
