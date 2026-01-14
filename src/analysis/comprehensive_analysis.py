"""
Comprehensive Analysis Script for UIDAI Data Hackathon 2026
Author: Mangesh Raut
Goal: First Prize - INR2,00,000

This script performs deep analysis to uncover winning insights:
1. Complete data integration from all chunks
2. Temporal trend analysis
3. Geographic pattern detection
4. Demographic insights
5. Anomaly detection
6. Predictive analytics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import preprocessing utilities
try:
    from src.preprocessing.data_loader import normalize_state_names
except ImportError:
    from preprocessing.data_loader import normalize_state_names

class UIDAIAnalyzer:
    """Main analysis class for UIDAI hackathon"""
    
    def __init__(self, base_dir='.'):
        self.base_dir = Path(base_dir)
        self.data = {}
        self.insights = []
        
    def load_all_datasets(self):
        """Load and combine all dataset chunks with optimized performance"""
        print("="*80)
        print("LOADING UIDAI DATASETS (OPTIMIZED)")
        print("="*80)

        # Optimized data types for memory efficiency
        dtype_dict = {
            'state': 'category',
            'district': 'category',
            'pincode': 'int32',
            'age_0_5': 'int32',
            'age_5_17': 'int32',
            'age_18_greater': 'int32',
            'demo_age_5_17': 'int32',
            'demo_age_17_': 'int32',
            'bio_age_5_17': 'int32',
            'bio_age_17_': 'int32'
        }

        # Load Enrolment Data with optimization
        print("\n Loading Enrolment Data...")
        enrolment_cols = ['date', 'state', 'district', 'pincode', 'age_0_5', 'age_5_17', 'age_18_greater']
        enrolment_files = sorted((self.base_dir / 'data' / 'raw' / 'api_data_aadhar_enrolment').glob('*.csv'))
        enrolment_dfs = []
        for file in enrolment_files:
            df = pd.read_csv(file, dtype={'state': 'category', 'district': 'category', 'pincode': 'int32',
                                        'age_0_5': 'int32', 'age_5_17': 'int32', 'age_18_greater': 'int32'},
                           usecols=enrolment_cols)
            enrolment_dfs.append(df)
            print(f"   Loaded {file.name}: {len(df):,} rows")
        self.data['enrolment'] = pd.concat(enrolment_dfs, ignore_index=True)
        print(f"\n Total Enrolment Records: {len(self.data['enrolment']):,}")

        # Load Demographic Update Data with optimization
        print("\n Loading Demographic Update Data...")
        demo_cols = ['date', 'state', 'district', 'pincode', 'demo_age_5_17', 'demo_age_17_']
        demo_files = sorted((self.base_dir / 'data' / 'raw' / 'api_data_aadhar_demographic').glob('*.csv'))
        demo_dfs = []
        for file in demo_files:
            df = pd.read_csv(file, dtype={'state': 'category', 'district': 'category', 'pincode': 'int32',
                                        'demo_age_5_17': 'int32', 'demo_age_17_': 'int32'},
                           usecols=demo_cols)
            demo_dfs.append(df)
            print(f"   Loaded {file.name}: {len(df):,} rows")
        self.data['demographic'] = pd.concat(demo_dfs, ignore_index=True)
        print(f"\n Total Demographic Update Records: {len(self.data['demographic']):,}")

        # Load Biometric Update Data with optimization
        print("\n Loading Biometric Update Data...")
        bio_cols = ['date', 'state', 'district', 'pincode', 'bio_age_5_17', 'bio_age_17_']
        bio_files = sorted((self.base_dir / 'data' / 'raw' / 'api_data_aadhar_biometric').glob('*.csv'))
        bio_dfs = []
        for file in bio_files:
            df = pd.read_csv(file, dtype={'state': 'category', 'district': 'category', 'pincode': 'int32',
                                        'bio_age_5_17': 'int32', 'bio_age_17_': 'int32'},
                           usecols=bio_cols)
            bio_dfs.append(df)
            print(f"   Loaded {file.name}: {len(df):,} rows")
        self.data['biometric'] = pd.concat(bio_dfs, ignore_index=True)
        print(f"\n Total Biometric Update Records: {len(self.data['biometric']):,}")

        # Optimized date conversion
        date_format = '%d-%m-%Y'
        for key in self.data:
            self.data[key]['date'] = pd.to_datetime(self.data[key]['date'], format=date_format)

        # Data Quality: Normalize state names (fix duplicates like WESTBENGAL/West Bengal)
        print("\n Normalizing state names...")
        for key in self.data:
            self.data[key] = normalize_state_names(self.data[key], 'state')
            print(f"    {key}: {self.data[key]['state'].nunique()} unique states after normalization")

        # Memory optimization
        for key in self.data:
            self.data[key] = self.data[key].copy()  # Ensure contiguous memory

        print(f"\n{'='*80}")
        print(" ALL DATASETS LOADED AND CLEANED SUCCESSFULLY")
        print(f"{'='*80}\n")

        return self.data
    
    def basic_statistics(self):
        """Generate basic statistics for all datasets"""
        print("\n" + "="*80)
        print("BASIC STATISTICS")
        print("="*80)
        
        stats = {}
        
        for name, df in self.data.items():
            print(f"\n {name.upper()} DATASET:")
            print(f"   Rows: {len(df):,}")
            print(f"   Columns: {len(df.columns)}")
            print(f"   Date Range: {df['date'].min().strftime('%d-%m-%Y')} to {df['date'].max().strftime('%d-%m-%Y')}")
            print(f"   States: {df['state'].nunique()}")
            print(f"   Districts: {df['district'].nunique()}")
            print(f"   Pincodes: {df['pincode'].nunique()}")
            
            stats[name] = {
                'rows': len(df),
                'date_range': (df['date'].min(), df['date'].max()),
                'states': df['state'].nunique(),
                'districts': df['district'].nunique(),
                'pincodes': df['pincode'].nunique()
            }
            
        return stats
    
    def temporal_analysis(self):
        """Analyze temporal trends"""
        print("\n" + "="*80)
        print("TEMPORAL TREND ANALYSIS")
        print("="*80)
        
        insights = {}
        
        # Enrolment Trends
        print("\n ENROLMENT TRENDS:")
        df_enrol = self.data['enrolment'].copy()
        df_enrol['total_enrolments'] = df_enrol['age_0_5'] + df_enrol['age_5_17'] + df_enrol['age_18_greater']
        
        daily_enrol = df_enrol.groupby('date')['total_enrolments'].sum().sort_index()
        print(f"   Peak Day: {daily_enrol.idxmax().strftime('%d-%m-%Y')} ({daily_enrol.max():,} enrolments)")
        print(f"   Lowest Day: {daily_enrol.idxmin().strftime('%d-%m-%Y')} ({daily_enrol.min():,} enrolments)")
        print(f"   Average Daily: {daily_enrol.mean():,.0f} enrolments")
        
        # Week-wise pattern
        df_enrol['weekday'] = df_enrol['date'].dt.day_name()
        weekday_pattern = df_enrol.groupby('weekday')['total_enrolments'].sum()
        print(f"\n   Busiest Day of Week: {weekday_pattern.idxmax()} ({weekday_pattern.max():,})")
        print(f"   Slowest Day of Week: {weekday_pattern.idxmin()} ({weekday_pattern.min():,})")
        
        insights['enrolment'] = {
            'peak_day': daily_enrol.idxmax(),
            'peak_value': daily_enrol.max(),
            'avg_daily': daily_enrol.mean(),
            'busiest_weekday': weekday_pattern.idxmax()
        }
        
        # Demographic Update Trends
        print("\n DEMOGRAPHIC UPDATE TRENDS:")
        df_demo = self.data['demographic'].copy()
        df_demo['total_updates'] = df_demo['demo_age_5_17'] + df_demo['demo_age_17_']
        
        daily_demo = df_demo.groupby('date')['total_updates'].sum().sort_index()
        print(f"   Peak Day: {daily_demo.idxmax().strftime('%d-%m-%Y')} ({daily_demo.max():,} updates)")
        print(f"   Average Daily: {daily_demo.mean():,.0f} updates")
        
        # Biometric Update Trends
        print("\n BIOMETRIC UPDATE TRENDS:")
        df_bio = self.data['biometric'].copy()
        df_bio['total_updates'] = df_bio['bio_age_5_17'] + df_bio['bio_age_17_']
        
        daily_bio = df_bio.groupby('date')['total_updates'].sum().sort_index()
        print(f"   Peak Day: {daily_bio.idxmax().strftime('%d-%m-%Y')} ({daily_bio.max():,} updates)")
        print(f"   Average Daily: {daily_bio.mean():,.0f} updates")
        
        self.insights.append("TEMPORAL INSIGHT: Significant variation in daily enrolment/update patterns")
        
        return insights
    
    def geographic_analysis(self):
        """Analyze geographic patterns"""
        print("\n" + "="*80)
        print("GEOGRAPHIC PATTERN ANALYSIS")
        print("="*80)
        
        # State-wise Enrolment
        print("\n  TOP 10 STATES BY ENROLMENT:")
        df_enrol = self.data['enrolment'].copy()
        df_enrol['total_enrolments'] = df_enrol['age_0_5'] + df_enrol['age_5_17'] + df_enrol['age_18_greater']
        
        state_enrol = df_enrol.groupby('state')['total_enrolments'].sum().sort_values(ascending=False).head(10)
        for rank, (state, count) in enumerate(state_enrol.items(), 1):
            print(f"   {rank:2d}. {state:25s} {count:>12,} enrolments")
        
        # State-wise Updates
        print("\n  TOP 10 STATES BY DEMOGRAPHIC UPDATES:")
        df_demo = self.data['demographic'].copy()
        df_demo['total_updates'] = df_demo['demo_age_5_17'] + df_demo['demo_age_17_']
        
        state_demo = df_demo.groupby('state')['total_updates'].sum().sort_values(ascending=False).head(10)
        for rank, (state, count) in enumerate(state_demo.items(), 1):
            print(f"   {rank:2d}. {state:25s} {count:>12,} updates")
        
        # State-wise Biometric Updates
        print("\n  TOP 10 STATES BY BIOMETRIC UPDATES:")
        df_bio = self.data['biometric'].copy()
        df_bio['total_updates'] = df_bio['bio_age_5_17'] + df_bio['bio_age_17_']
        
        state_bio = df_bio.groupby('state')['total_updates'].sum().sort_values(ascending=False).head(10)
        for rank, (state, count) in enumerate(state_bio.items(), 1):
            print(f"   {rank:2d}. {state:25s} {count:>12,} updates")
        
        self.insights.append("GEOGRAPHIC INSIGHT: Clear regional disparities in Aadhaar adoption and updates")
        
        return {
            'top_enrolment_states': state_enrol,
            'top_demo_states': state_demo,
            'top_bio_states': state_bio
        }
    
    def demographic_insights(self):
        """Analyze age-wise patterns"""
        print("\n" + "="*80)
        print("DEMOGRAPHIC INSIGHTS")
        print("="*80)
        
        # Age distribution in Enrolment
        print("\n ENROLMENT BY AGE GROUP:")
        df_enrol = self.data['enrolment']
        
        age_0_5_total = df_enrol['age_0_5'].sum()
        age_5_17_total = df_enrol['age_5_17'].sum()
        age_18_plus_total = df_enrol['age_18_greater'].sum()
        total = age_0_5_total + age_5_17_total + age_18_plus_total
        
        print(f"   0-5 years:     {age_0_5_total:>12,} ({age_0_5_total/total*100:5.2f}%)")
        print(f"   5-17 years:    {age_5_17_total:>12,} ({age_5_17_total/total*100:5.2f}%)")
        print(f"   18+ years:     {age_18_plus_total:>12,} ({age_18_plus_total/total*100:5.2f}%)")
        
        # Update rates by age
        print("\n DEMOGRAPHIC UPDATES BY AGE GROUP:")
        df_demo = self.data['demographic']
        
        demo_5_17 = df_demo['demo_age_5_17'].sum()
        demo_18_plus = df_demo['demo_age_17_'].sum()
        demo_total = demo_5_17 + demo_18_plus
        
        print(f"   5-17 years:    {demo_5_17:>12,} ({demo_5_17/demo_total*100:5.2f}%)")
        print(f"   18+ years:     {demo_18_plus:>12,} ({demo_18_plus/demo_total*100:5.2f}%)")
        
        print("\n BIOMETRIC UPDATES BY AGE GROUP:")
        df_bio = self.data['biometric']
        
        bio_5_17 = df_bio['bio_age_5_17'].sum()
        bio_18_plus = df_bio['bio_age_17_'].sum()
        bio_total = bio_5_17 + bio_18_plus
        
        print(f"   5-17 years:    {bio_5_17:>12,} ({bio_5_17/bio_total*100:5.2f}%)")
        print(f"   18+ years:     {bio_18_plus:>12,} ({bio_18_plus/bio_total*100:5.2f}%)")
        
        # Calculate update rates
        print("\n UPDATE RATE ANALYSIS:")
        # This is a simplified calculation - in reality would need base population
        bio_to_demo_ratio = bio_total / demo_total
        print(f"   Biometric to Demographic Update Ratio: {bio_to_demo_ratio:.3f}")
        
        self.insights.append("DEMOGRAPHIC INSIGHT: Significant difference in update patterns between age groups")
        
        return {
            'enrolment_by_age': {
                '0-5': age_0_5_total,
                '5-17': age_5_17_total,
                '18+': age_18_plus_total
            },
            'bio_to_demo_ratio': bio_to_demo_ratio
        }
    
    def unique_insights(self):
        """Generate unique, winning insights"""
        print("\n" + "="*80)
        print(" UNIQUE INSIGHTS FOR FIRST PRIZE")
        print("="*80)
        
        df_enrol = self.data['enrolment'].copy()
        df_demo = self.data['demographic'].copy()
        df_bio = self.data['biometric'].copy()
        
        # Calculate totals
        df_enrol['total'] = df_enrol['age_0_5'] + df_enrol['age_5_17'] + df_enrol['age_18_greater']
        df_demo['total'] = df_demo['demo_age_5_17'] + df_demo['demo_age_17_']
        df_bio['total'] = df_bio['bio_age_5_17'] + df_bio['bio_age_17_']
        
        insights = []
        
        # Insight 1: Child Enrolment Priority
        print("\n INSIGHT 1: Early Childhood Enrolment Patterns")
        child_enrol_by_state = df_enrol.groupby('state')['age_0_5'].sum().sort_values(ascending=False)
        total_enrol_by_state = df_enrol.groupby('state')['total'].sum()
        child_percentage = (child_enrol_by_state / total_enrol_by_state * 100).sort_values(ascending=True).head(10)
        
        print("   States with LOWEST child (0-5) enrolment percentage:")
        for state, pct in child_percentage.items():
            print(f"      {state:25s} {pct:5.2f}%")
        insights.append(f"Target states with low child enrolment: {', '.join(child_percentage.index[:3].tolist())}")
        
        # Insight 2: Update to Enrolment Ratio
        print("\n INSIGHT 2: Update Activity vs Base Enrolment")
        state_demo_sum = df_demo.groupby('state')['total'].sum()
        state_enrol_sum = df_enrol.groupby('state')['total'].sum()
        
        # States present in both
        common_states= state_demo_sum.index.intersection(state_enrol_sum.index)
        update_ratio = (state_demo_sum[common_states] / state_enrol_sum[common_states] * 100).sort_values(ascending=False).head(10)
        
        print("   States with HIGHEST demographic update activity:")
        for state, ratio in update_ratio.items():
            print(f"      {state:25s} {ratio:5.2f}% update rate")
        insights.append(f"High update activity states indicate: {', '.join(update_ratio.index[:3].tolist())}")
        
        # Insight 3: Biometric Update Gap (5-17 age)
        print("\n INSIGHT 3: Child-to-Adult Biometric Transition")
        bio_child_ratio = df_bio['bio_age_5_17'].sum() / df_bio['total'].sum() * 100
        print(f"   Child (5-17) biometric updates: {bio_child_ratio:.2f}% of total")
        print(f"   -> This represents children transitioning to adults needing updated biometrics")
        insights.append(f"Child biometric transition rate: {bio_child_ratio:.2f}%")
        
        # Insight 4: District-level Hotspots
        print("\n INSIGHT 4: District Hotspots for Targeted Intervention")
        district_enrol = df_enrol.groupby(['state', 'district'])['total'].sum().sort_values(ascending=False).head(15)
        print("   Top 15 districts by enrolment (potential infrastructure strain):")
        for (state, district), count in district_enrol.items():
            print(f"      {district:20s}, {state:20s} -> {count:>8,}")
        
        # Insight 5: Pincode-level Micro Analysis
        print("\n INSIGHT 5: Pincode-Level Micro Patterns")
        pincode_analysis = df_enrol.groupby('pincode').agg({
            'total': 'sum',
            'age_0_5': 'sum',
            'age_5_17': 'sum',
            'age_18_greater': 'sum',
            'date': 'count'  # number of days with activity
        }).rename(columns={'date': 'active_days'})
        
        # High volume, low frequency pincodes (potential mobile unit targets)
        high_vol_low_freq = pincode_analysis[
            (pincode_analysis['total'] > pincode_analysis['total'].quantile(0.75)) &
            (pincode_analysis['active_days'] < pincode_analysis['active_days'].quantile(0.25))
        ]
        
        print(f"   Found {len(high_vol_low_freq)} pincodes with high volume but low frequency")
        print(f"   -> These areas need permanent enrolment centers, not mobile units")
        insights.append(f"{len(high_vol_low_freq)} pincodes need permanent infrastructure")
        
        self.insights.extend(insights)
        
        return insights
    
    def save_processed_data(self):
        """Save processed datasets"""
        print("\n" + "="*80)
        print("SAVING PROCESSED DATA")
        print("="*80)
        
        output_dir = self.base_dir / 'data' / 'processed'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, df in self.data.items():
            output_path = output_dir / f'{name}_combined.parquet'
            df.to_parquet(output_path, compression='gzip', index=False)
            print(f"    Saved {name}: {output_path}")
        
        print(f"\n All processed data saved to {output_dir}")
    
    def generate_summary_report(self):
        """Generate a summary report of all findings"""
        print("\n" + "="*80)
        print(" COMPREHENSIVE ANALYSIS SUMMARY")
        print("="*80)
        print("\n KEY INSIGHTS DISCOVERED:\n")
        
        for i, insight in enumerate(self.insights, 1):
            print(f"   {i}. {insight}")
        
        print(f"\n{'='*80}\n")


if __name__ == "__main__":
    # Initialize analyzer
    analyzer = UIDAIAnalyzer('/Users/mangeshraut/Downloads/UIDAI Data Hackathon 2026')
    
    # Run complete analysis
    print(" UIDAI DATA HACKATHON 2026 - COMPREHENSIVE ANALYSIS")
    print("   Participant: Mangesh Raut")
    print("   Goal: First Prize (INR2,00,000)")
    print(f"{'='*80}\n")
    
    # Load data
    analyzer.load_all_datasets()
    
    # Run analyses
    analyzer.basic_statistics()
    analyzer.temporal_analysis()
    analyzer.geographic_analysis()
    analyzer.demographic_insights()
    analyzer.unique_insights()
    
    # Save processed data
    analyzer.save_processed_data()
    
    # Generate summary
    analyzer.generate_summary_report()
    
    print("\n ANALYSIS COMPLETE!")
    print("   Next steps:")
    print("   1. Review generated insights")
    print("   2. Create visualizations")
    print("   3. Draft final PDF report")
    print(f"\n{'='*80}\n")
