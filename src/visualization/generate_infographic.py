"""
Executive Summary Infographic - Premium Award-Winning Design
UIDAI Data Hackathon 2026
Author: Mangesh Raut
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as path_effects
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Premium Color Palette - Modern 2026 Executive Series
COLORS = {
    'primary': '#0F172A',       # Slate 900
    'secondary': '#3B82F6',     # Blue 500
    'accent': '#E11D48',        # Rose 600
    'success': '#059669',       # Emerald 600
    'warning': '#D97706',       # Amber 600
    'gold': '#B45309',          # Gold/Bronze
    'purple': '#7C3AED',        # Violet 600
    'bg_surface': '#F8FAFC',    # Slate 50
    'bg_gradient_start': '#FFFFFF',
    'bg_gradient_end': '#F1F5F9',
    'text_main': '#1E293B',     # Slate 800
    'text_muted': '#64748B',    # Slate 500
    'white': '#FFFFFF',
}


def create_premium_infographic():
    """Create award-winning executive summary infographic"""
    
    # Load data
    base_dir = Path.cwd()
    df_enrol = pd.read_parquet(base_dir / 'data/processed/enrolment_combined.parquet')
    df_demo = pd.read_parquet(base_dir / 'data/processed/demographic_combined.parquet')
    df_bio = pd.read_parquet(base_dir / 'data/processed/biometric_combined.parquet')
    
    # Calculate totals
    df_enrol['total'] = df_enrol['age_0_5'] + df_enrol['age_5_17'] + df_enrol['age_18_greater']
    df_demo['total'] = df_demo['demo_age_5_17'] + df_demo['demo_age_17_']
    df_bio['total'] = df_bio['bio_age_5_17'] + df_bio['bio_age_17_']
    
    # Calculate key metrics
    total_enrol = df_enrol['total'].sum()
    total_demo = df_demo['total'].sum()
    total_bio = df_bio['total'].sum()
    total_records = len(df_enrol) + len(df_demo) + len(df_bio)
    num_states = df_enrol['state'].nunique()
    num_districts = df_enrol['district'].nunique()
    
    # Age distribution
    age_0_5 = df_enrol['age_0_5'].sum()
    age_5_17 = df_enrol['age_5_17'].sum()
    age_18_plus = df_enrol['age_18_greater'].sum()
    
    # Top 10 states
    top_states = df_enrol.groupby('state')['total'].sum().nlargest(10)
    
    # Create figure with high DPI
    fig = plt.figure(figsize=(16, 22), facecolor=COLORS['bg_surface'])
    
    # ========== HEADER SECTION ==========
    # Main badge
    rect_header = FancyBboxPatch((0.05, 0.90), 0.90, 0.08,
                               boxstyle="round,pad=0,rounding_size=0.02",
                               facecolor=COLORS['primary'], transform=fig.transFigure)
    fig.patches.append(rect_header)

    fig.text(0.5, 0.955, 'UIDAI DATA HACKATHON 2026', 
            fontsize=36, fontweight='bold', ha='center', va='top',
            color=COLORS['white'], fontfamily='sans-serif')
    
    fig.text(0.5, 0.93, 'Strategic Analysis & Future Roadmap: Unlocking Societal Potential', 
            fontsize=16, ha='center', va='top', color=COLORS['secondary'],
            fontweight='600', fontfamily='sans-serif')
    
    fig.text(0.5, 0.91, 'Mangesh Bharat Raut | Senior Research Lead | Team ID: UIDAI_4879', 
            fontsize=12, ha='center', va='top', color=COLORS['white'], alpha=0.8)
    
    # ========== KEY METRICS ROW ==========
    metrics = [
        ('Records', f'{total_records:,}', 'Total Records', COLORS['secondary']),
        ('States', str(num_states), 'States/UTs', COLORS['success']),
        ('Districts', str(num_districts), 'Districts', COLORS['warning']),
        ('Months', '10', 'Months', COLORS['purple']),
    ]
    
    for i, (icon, value, label, color) in enumerate(metrics):
        x = 0.14 + i * 0.2
        
        # Card background
        rect = FancyBboxPatch((x - 0.08, 0.835), 0.16, 0.055,
                             boxstyle="round,pad=0.008,rounding_size=0.015",
                             facecolor=color, alpha=0.95, edgecolor='white', linewidth=2,
                             transform=fig.transFigure, figure=fig)
        fig.patches.append(rect)
        
        # Icon and value
        fig.text(x, 0.875, icon, fontsize=18, ha='center', va='center')
        fig.text(x, 0.858, value, fontsize=22, fontweight='bold', 
                ha='center', va='center', color=COLORS['white'])
        fig.text(x, 0.843, label, fontsize=10, ha='center', va='center', 
                color=COLORS['white'], alpha=0.9)
    
    # ========== DATASET ECO-SYSTEM ==========
    fig.text(0.1, 0.81, 'DATASET ECO-SYSTEM', 
            fontsize=14, ha='left', va='top', color=COLORS['primary'], fontweight='900')
    
    datasets = [
        ('New Enrolments', f'{total_enrol/1e6:.1f}M', 'Acquisition Root', COLORS['secondary']),
        ('Demographic Updates', f'{total_demo/1e6:.1f}M', 'Lifecycle Mgmt', COLORS['success']),
        ('Biometric Updates', f'{total_bio/1e6:.1f}M', 'Identity Assurance', COLORS['warning']),
    ]
    
    for i, (title, value, subtitle, color) in enumerate(datasets):
        x = 0.1 + i * 0.28
        
        # Card
        rect = FancyBboxPatch((x, 0.74), 0.24, 0.055,
                             boxstyle="round,pad=0,rounding_size=0.01",
                             facecolor=COLORS['white'], edgecolor='#E2E8F0', linewidth=1,
                             transform=fig.transFigure)
        fig.patches.append(rect)
        
        # Side accent
        rect2 = Rectangle((x, 0.74), 0.005, 0.055, facecolor=color, transform=fig.transFigure)
        fig.patches.append(rect2)
        
        fig.text(x + 0.02, 0.782, title, fontsize=11, fontweight='bold', color=COLORS['text_muted'])
        fig.text(x + 0.02, 0.762, value, fontsize=20, fontweight='900', color=color)
        fig.text(x + 0.02, 0.748, subtitle, fontsize=9, color=COLORS['text_muted'])
    
    # ========== KEY INSIGHTS SECTION ==========
    # ========== QUANTIFIED STRATEGIC INSIGHTS ==========
    fig.text(0.1, 0.71, 'QUANTIFIED STRATEGIC INSIGHTS', 
            fontsize=14, ha='left', va='top', color=COLORS['primary'], fontweight='900')
    
    insights = [
        ('01', 'Facility Utilization Gap', 'Weekend service 62% decrease', 'Opportunity: +3.2M / Year', COLORS['accent']),
        ('02', 'Targeted Demographic Pull', 'Child gaps in NE (19-29%)', 'Sig: χ²=913,965 (p<0.01)', COLORS['secondary']),
        ('03', 'Infrastructure Hotspots', '15 districts handle 17% load', 'Expand: Reduce strain by 35%', COLORS['warning']),
        ('04', 'System Evolution Maturity', 'Demographic vs Biometric ratio 4.1:1', 'Optimized identity spend', COLORS['success']),
    ]
    
    for i, (num, title, body, impact, color) in enumerate(insights):
        y = 0.67 - i * 0.05
        
        # Box
        rect = FancyBboxPatch((0.1, y), 0.8, 0.04,
                             boxstyle="round,pad=0,rounding_size=0.005",
                             facecolor=COLORS['white'], edgecolor='#E2E8F0', transform=fig.transFigure)
        fig.patches.append(rect)
        
        # Index Badge
        circle = Circle((0.12, y + 0.02), 0.012, facecolor=color, transform=fig.transFigure)
        fig.patches.append(circle)
        fig.text(0.12, y + 0.02, num, color='white', fontweight='bold', ha='center', va='center')
        
        fig.text(0.15, y + 0.025, title, fontsize=11, fontweight='900', color=COLORS['primary'])
        fig.text(0.15, y + 0.01, body, fontsize=9, color=COLORS['text_muted'])
        fig.text(0.88, y + 0.02, impact, fontsize=10, fontweight='bold', color=color, ha='right', va='center')
    
    # ========== TOP 10 STATES BAR CHART ==========
    fig.text(0.5, 0.52, '----------  TOP 10 STATES BY ENROLMENT  ----------', 
            fontsize=12, ha='center', va='top', color=COLORS['primary'], fontweight='bold')
    
    ax_bar = fig.add_axes([0.1, 0.30, 0.8, 0.20])
    
    # Create gradient colors
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_states)))[::-1]
    
    bars = ax_bar.barh(range(len(top_states)), top_states.values[::-1], 
                       color=colors, edgecolor='white', linewidth=1)
    
    # Style the chart
    ax_bar.set_yticks(range(len(top_states)))
    ax_bar.set_yticklabels(top_states.index[::-1], fontsize=11, fontweight='700', color=COLORS['primary'])
    ax_bar.set_facecolor(COLORS['white'])
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['left'].set_visible(False)
    ax_bar.tick_params(left=False)
    ax_bar.grid(True, axis='x', color='#F1F5F9', linewidth=1)
    
    # ========== AGE DISTRIBUTION DONUT ==========
    fig.text(0.22, 0.275, 'AGE DISTRIBUTION', fontsize=12, ha='center', va='top',
            color=COLORS['primary'], fontweight='bold')
    
    ax_pie = fig.add_axes([0.05, 0.12, 0.30, 0.15])
    
    age_data = [age_0_5, age_5_17, age_18_plus]
    age_labels = ['0-5 years', '5-17 years', '18+ years']
    age_colors = [COLORS['secondary'], COLORS['warning'], COLORS['success']]
    
    wedges, texts, autotexts = ax_pie.pie(age_data, labels=None, 
                                          autopct='%1.1f%%', pctdistance=0.75,
                                          colors=age_colors,
                                          wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2))
    
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    
    # Legend for pie
    for i, (label, color) in enumerate(zip(age_labels, age_colors)):
        fig.text(0.37, 0.23 - i * 0.025, 'o', fontsize=14, color=color, ha='center', va='center')
        fig.text(0.40, 0.23 - i * 0.025, label, fontsize=10, color=COLORS['text_main'], 
                ha='left', va='center')
    
    # ========== TOP RECOMMENDATIONS ==========
    fig.text(0.7, 0.275, 'TOP RECOMMENDATIONS', fontsize=12, ha='center', va='top',
            color=COLORS['primary'], fontweight='bold')
    
    recommendations = [
        ('1', 'Weekend service pilot', '+270K/month'),
        ('2', 'NE mobile drives', '+50% child enrol'),
        ('3', 'Infrastructure expansion', '15 districts'),
        ('4', 'Biometric quality program', '-20% rework'),
        ('5', 'Appointment system pilot', '5 districts'),
    ]
    
    for i, (num, rec, impact) in enumerate(recommendations):
        y = 0.245 - i * 0.028
        
        # Number circle
        circle = Circle((0.52, y), 0.012, facecolor=COLORS['secondary'], 
                        transform=fig.transFigure, figure=fig)
        fig.patches.append(circle)
        fig.text(0.52, y, num, fontsize=10, fontweight='bold', ha='center', va='center',
                color='white')
        
        fig.text(0.545, y, rec, fontsize=10, ha='left', va='center', color=COLORS['text_main'])
        fig.text(0.88, y, impact, fontsize=9, ha='right', va='center', 
                color=COLORS['success'], fontweight='bold')
    
    # ========== POLICY ROI CALCULATOR ==========
    fig.text(0.5, 0.11, 'POLICY IMPLEMENTATION ROI (ESTIMATED)', 
            fontsize=12, ha='center', va='center', color='#064E3B', fontweight='900')
    
    rect_roi = FancyBboxPatch((0.1, 0.06), 0.8, 0.065,
                            boxstyle="round,pad=0,rounding_size=0.005",
                            facecolor='#ECFDF5', edgecolor='#10B981', linewidth=1.5,
                            transform=fig.transFigure)
    fig.patches.append(rect_roi)
    
    roi_metrics = [
        ('₹47 Cr', 'Annual Investment', COLORS['primary']),
        ('₹117 Cr', 'Projected Benefit', COLORS['success']),
        ('149%', 'Aggregate ROI', COLORS['success']),
        ('5.7M', 'Addl. Enrolments', COLORS['secondary']),
    ]
    
    for i, (val, lab, col) in enumerate(roi_metrics):
        x = 0.2 + i * 0.2
        fig.text(x, 0.095, val, fontsize=18, fontweight='900', color=col, ha='center')
        fig.text(x, 0.078, lab, fontsize=9, fontweight='700', color=COLORS['text_muted'], ha='center')
    
    # ========== FOOTER ==========
    fig.text(0.5, 0.025, '------------------------------------------------------------------------------', 
            fontsize=10, ha='center', color=COLORS['text_muted'])
    
    fig.text(0.5, 0.012, 
            'Analysis by Mangesh Raut | mbr63@drexel.edu | +91 7276819090 | linkedin.com/in/mangeshraut1298',
            fontsize=9, ha='center', color=COLORS['text_muted'])
    
    fig.text(0.5, 0.0, 'UIDAI Data Hackathon 2026 | January 2026',
            fontsize=9, ha='center', color=COLORS['text_muted'], style='italic')
    
    # Save
    output_path = base_dir / 'visualizations/infographics'
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / '01_executive_summary.png', dpi=400, 
                facecolor=COLORS['bg_surface'], bbox_inches='tight', pad_inches=0.4)
    plt.close()
    
    print(" Premium Executive Summary saved!")
    print(f"    {output_path / '01_executive_summary.png'}")


if __name__ == "__main__":
    print("="*60)
    print("PREMIUM EXECUTIVE SUMMARY INFOGRAPHIC")
    print("="*60 + "\n")
    create_premium_infographic()
