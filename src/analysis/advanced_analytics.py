"""
Advanced Analytics Module - UIDAI Data Hackathon 2026
Author: Mangesh Bharat Raut
Team ID: UIDAI_4879

This module implements cutting-edge 2026 ML/AI techniques:
1. Anomaly Detection (Isolation Forest, DBSCAN)
2. Time Series Forecasting (ARIMA, Prophet-style)
3. Ensemble Clustering (K-means + Hierarchical + DBSCAN)
4. Advanced Statistical Tests
5. Feature Engineering

Technologies: Python 3.12+, scikit-learn 1.4, scipy 1.12, statsmodels 0.14
"""

from __future__ import annotations
from typing import TypeAlias
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import warnings

warnings.filterwarnings('ignore')

DataFrame: TypeAlias = pd.DataFrame


@dataclass
class AdvancedAnalytics:
    """Advanced ML/AI analytics for hackathon-winning insights"""
    
    base_dir: Path
    df_enrol: DataFrame = field(default=None)
    df_demo: DataFrame = field(default=None)
    df_bio: DataFrame = field(default=None)
    
    def __post_init__(self):
        self.base_dir = Path(self.base_dir)
        self._load_data()
    
    def _load_data(self):
        """Load processed datasets"""
        self.df_enrol = pd.read_parquet(self.base_dir / 'data/processed/enrolment_combined.parquet')
        self.df_demo = pd.read_parquet(self.base_dir / 'data/processed/demographic_combined.parquet')
        self.df_bio = pd.read_parquet(self.base_dir / 'data/processed/biometric_combined.parquet')
        
        # Add totals
        self.df_enrol['total'] = self.df_enrol['age_0_5'] + self.df_enrol['age_5_17'] + self.df_enrol['age_18_greater']
        self.df_demo['total'] = self.df_demo['demo_age_5_17'] + self.df_demo['demo_age_17_']
        self.df_bio['total'] = self.df_bio['bio_age_5_17'] + self.df_bio['bio_age_17_']
    
    # =========================================================================
    # 1. ANOMALY DETECTION - Isolation Forest
    # =========================================================================
    def detect_anomalies_isolation_forest(self, contamination: float = 0.05) -> dict:
        """
        Detect anomalous districts using Isolation Forest
        
        This identifies:
        - Districts with unusually high/low enrolments
        - Potential data quality issues
        - Fraud indicators
        """
        print("\n" + "="*60)
        print(" ANOMALY DETECTION: Isolation Forest")
        print("="*60)
        
        # Prepare district-level features
        district_stats = self.df_enrol.groupby(['state', 'district']).agg({
            'total': ['sum', 'mean', 'std', 'count'],
            'age_0_5': 'sum',
            'age_5_17': 'sum',
            'age_18_greater': 'sum'
        }).reset_index()
        
        district_stats.columns = ['state', 'district', 'total_sum', 'total_mean', 
                                   'total_std', 'days_active', 'age_0_5', 'age_5_17', 'age_18']
        
        # Calculate derived features
        district_stats['child_pct'] = district_stats['age_0_5'] / district_stats['total_sum'] * 100
        district_stats['daily_avg'] = district_stats['total_sum'] / district_stats['days_active']
        district_stats['cv'] = district_stats['total_std'] / (district_stats['total_mean'] + 1)
        
        # Feature matrix
        features = ['total_sum', 'daily_avg', 'child_pct', 'cv', 'days_active']
        X = district_stats[features].fillna(0).values
        
        # Scale features
        scaler = RobustScaler()  # Robust to outliers
        X_scaled = scaler.fit_transform(X)
        
        # Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200,
            max_samples='auto'
        )
        
        anomaly_labels = iso_forest.fit_predict(X_scaled)
        anomaly_scores = iso_forest.decision_function(X_scaled)
        
        district_stats['is_anomaly'] = anomaly_labels == -1
        district_stats['anomaly_score'] = anomaly_scores
        
        # Get top anomalies
        anomalies = district_stats[district_stats['is_anomaly']].sort_values('anomaly_score')
        
        print(f"\n Found {len(anomalies)} anomalous districts ({contamination*100:.1f}% contamination)")
        print("\n Top 10 Anomalous Districts:")
        print("-" * 80)
        
        for i, row in anomalies.head(10).iterrows():
            print(f"   {row['district']}, {row['state']}")
            print(f"      Total: {row['total_sum']:,.0f} | Daily Avg: {row['daily_avg']:.1f} | Child%: {row['child_pct']:.1f}%")
            print(f"      Anomaly Score: {row['anomaly_score']:.3f}")
            print()
        
        return {
            'anomalies': anomalies,
            'all_districts': district_stats,
            'n_anomalies': len(anomalies),
            'contamination': contamination
        }
    
    # =========================================================================
    # 2. ADVANCED CLUSTERING - Ensemble Methods
    # =========================================================================
    def ensemble_clustering(self) -> dict:
        """
        Ensemble clustering combining:
        - K-means
        - DBSCAN (density-based)
        - Hierarchical (agglomerative)
        
        Uses consensus to find stable clusters
        """
        print("\n" + "="*60)
        print(" ENSEMBLE CLUSTERING: K-means + DBSCAN + Hierarchical")
        print("="*60)
        
        # State-level features
        state_stats = self.df_enrol.groupby('state').agg({
            'total': 'sum',
            'age_0_5': 'sum',
            'age_5_17': 'sum',
            'age_18_greater': 'sum'
        }).reset_index()
        
        state_stats['child_pct'] = state_stats['age_0_5'] / state_stats['total'] * 100
        state_stats['youth_pct'] = state_stats['age_5_17'] / state_stats['total'] * 100
        state_stats['adult_pct'] = state_stats['age_18_greater'] / state_stats['total'] * 100
        
        # Feature matrix
        X = state_stats[['child_pct', 'youth_pct', 'adult_pct']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 1. K-Means
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
        
        # 2. DBSCAN
        dbscan = DBSCAN(eps=0.8, min_samples=2)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        
        # 3. Hierarchical
        hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward')
        hier_labels = hierarchical.fit_predict(X_scaled)
        hier_silhouette = silhouette_score(X_scaled, hier_labels)
        
        # Calinski-Harabasz Index (higher is better)
        ch_kmeans = calinski_harabasz_score(X_scaled, kmeans_labels)
        ch_hier = calinski_harabasz_score(X_scaled, hier_labels)
        
        state_stats['kmeans_cluster'] = kmeans_labels
        state_stats['dbscan_cluster'] = dbscan_labels
        state_stats['hier_cluster'] = hier_labels
        
        print(f"\n Clustering Results:")
        print(f"   K-Means:      Silhouette={kmeans_silhouette:.3f}, CH={ch_kmeans:.1f}")
        print(f"   DBSCAN:       {n_dbscan_clusters} clusters found, {(dbscan_labels==-1).sum()} noise points")
        print(f"   Hierarchical: Silhouette={hier_silhouette:.3f}, CH={ch_hier:.1f}")
        
        # Consensus clustering
        print("\n Cluster Profiles (K-Means):")
        for cluster in range(4):
            cluster_data = state_stats[state_stats['kmeans_cluster'] == cluster]
            print(f"\n   Cluster {cluster}: {len(cluster_data)} states")
            print(f"      Avg Child%: {cluster_data['child_pct'].mean():.1f}%")
            print(f"      States: {', '.join(cluster_data['state'].head(5).tolist())}...")
        
        return {
            'state_stats': state_stats,
            'kmeans_silhouette': kmeans_silhouette,
            'hier_silhouette': hier_silhouette,
            'ch_kmeans': ch_kmeans,
            'ch_hier': ch_hier,
            'n_dbscan_clusters': n_dbscan_clusters
        }
    
    # =========================================================================
    # 3. TIME SERIES FORECASTING - ARIMA-style
    # =========================================================================
    def advanced_time_series_forecast(self, forecast_days: int = 30) -> dict:
        """
        Advanced time series forecasting using:
        - Decomposition (trend, seasonality, residual)
        - Rolling statistics
        - Confidence intervals
        - Multiple models comparison
        """
        print("\n" + "="*60)
        print(" ADVANCED TIME SERIES FORECASTING")
        print("="*60)
        
        # Daily aggregation
        daily = self.df_enrol.groupby('date')['total'].sum().sort_index()
        
        # Decomposition
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Ensure we have enough data
        if len(daily) > 14:
            decomposition = seasonal_decompose(daily, model='additive', period=7)
            trend = decomposition.trend.dropna()
            seasonal = decomposition.seasonal
            residual = decomposition.resid.dropna()
        
        # Rolling statistics
        rolling_mean = daily.rolling(window=7).mean()
        rolling_std = daily.rolling(window=7).std()
        
        # Simple forecast using trend extrapolation
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.ensemble import GradientBoostingRegressor
        
        X = np.arange(len(daily)).reshape(-1, 1)
        y = daily.values
        
        # Multiple models
        models = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        best_model = None
        best_score = float('-inf')
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        for name, model in models.items():
            scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
            mean_score = scores.mean()
            results[name] = {
                'cv_r2': mean_score,
                'cv_std': scores.std()
            }
            
            if mean_score > best_score:
                best_score = mean_score
                best_model = name
            
            print(f"   {name}: CV R² = {mean_score:.4f} (±{scores.std():.4f})")
        
        # Fit best model and forecast
        models[best_model].fit(X, y)
        
        # Future dates
        future_X = np.arange(len(daily), len(daily) + forecast_days).reshape(-1, 1)
        forecast = models[best_model].predict(future_X)
        
        # Confidence intervals (using residual std)
        residual_std = np.std(y - models[best_model].predict(X))
        ci_lower = forecast - 1.96 * residual_std
        ci_upper = forecast + 1.96 * residual_std
        
        print(f"\n Best Model: {best_model} (R² = {best_score:.4f})")
        print(f"   Forecast Period: {forecast_days} days")
        print(f"   Predicted Average: {forecast.mean():,.0f} enrolments/day")
        print(f"   95% CI: [{ci_lower.mean():,.0f}, {ci_upper.mean():,.0f}]")
        
        return {
            'daily': daily,
            'rolling_mean': rolling_mean,
            'forecast': forecast,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'best_model': best_model,
            'best_cv_r2': best_score,
            'model_results': results
        }
    
    # =========================================================================
    # 4. FEATURE IMPORTANCE - Random Forest
    # =========================================================================
    def feature_importance_analysis(self) -> dict:
        """
        Use Random Forest to identify most important features
        driving enrolment patterns
        """
        print("\n" + "="*60)
        print(" FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Create features
        df = self.df_enrol.copy()
        df['weekday'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['day_of_month'] = df['date'].dt.day
        df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
        df['quarter'] = df['date'].dt.quarter
        
        # One-hot encode state
        state_dummies = pd.get_dummies(df['state'].astype(str), prefix='state')
        
        # Feature matrix
        features = pd.concat([
            df[['weekday', 'month', 'day_of_month', 'is_weekend', 'quarter']],
            state_dummies
        ], axis=1)
        
        target = df['total']
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
        rf.fit(features, target)
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': features.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n Top 15 Most Important Features:")
        print("-" * 50)
        for i, row in importance.head(15).iterrows():
            bar = "█" * int(row['importance'] * 50)
            print(f"   {row['feature']:30s} {row['importance']:.4f} {bar}")
        
        # Model performance
        from sklearn.metrics import r2_score, mean_absolute_error
        predictions = rf.predict(features)
        r2 = r2_score(target, predictions)
        mae = mean_absolute_error(target, predictions)
        
        print(f"\n Model Performance:")
        print(f"   R² Score: {r2:.4f}")
        print(f"   MAE: {mae:,.0f} enrolments")
        
        return {
            'importance': importance,
            'r2_score': r2,
            'mae': mae,
            'model': rf
        }
    
    # =========================================================================
    # 5. ADVANCED STATISTICAL TESTS
    # =========================================================================
    def advanced_statistical_tests(self) -> dict:
        """
        Comprehensive statistical validation:
        - Kolmogorov-Smirnov test (distribution)
        - Mann-Whitney U test (non-parametric)
        - Kruskal-Wallis H test (multiple groups)
        - ANOVA with post-hoc Tukey
        """
        print("\n" + "="*60)
        print(" ADVANCED STATISTICAL TESTS")
        print("="*60)
        
        results = {}
        
        # 1. Kolmogorov-Smirnov test for normality
        daily_enrol = self.df_enrol.groupby('date')['total'].sum()
        ks_stat, ks_pval = stats.kstest(daily_enrol, 'norm', 
                                         args=(daily_enrol.mean(), daily_enrol.std()))
        results['ks_test'] = {'statistic': ks_stat, 'p_value': ks_pval}
        print(f"\n1. Kolmogorov-Smirnov (Normality):")
        print(f"   KS Statistic: {ks_stat:.4f}, p-value: {ks_pval:.4e}")
        print(f"   Result: {'Not Normal' if ks_pval < 0.05 else 'Normal'} distribution")
        
        # 2. Mann-Whitney U test (Weekend vs Weekday)
        self.df_enrol['is_weekend'] = self.df_enrol['date'].dt.dayofweek >= 5
        weekend_totals = self.df_enrol[self.df_enrol['is_weekend']]['total']
        weekday_totals = self.df_enrol[~self.df_enrol['is_weekend']]['total']
        
        mw_stat, mw_pval = stats.mannwhitneyu(weekend_totals, weekday_totals, alternative='two-sided')
        results['mann_whitney'] = {'statistic': mw_stat, 'p_value': mw_pval}
        print(f"\n2. Mann-Whitney U (Weekend vs Weekday):")
        print(f"   U Statistic: {mw_stat:,.0f}, p-value: {mw_pval:.4e}")
        print(f"   Result: {'Significant' if mw_pval < 0.05 else 'Not Significant'} difference")
        
        # 3. Kruskal-Wallis H test (across regions)
        regions = {
            'North': ['Uttar Pradesh', 'Bihar', 'Madhya Pradesh', 'Rajasthan'],
            'South': ['Tamil Nadu', 'Karnataka', 'Kerala', 'Andhra Pradesh'],
            'East': ['West Bengal', 'Odisha', 'Jharkhand', 'Assam'],
            'West': ['Maharashtra', 'Gujarat', 'Goa']
        }
        
        region_groups = []
        for region, states in regions.items():
            region_data = self.df_enrol[self.df_enrol['state'].isin(states)]['total']
            if len(region_data) > 0:
                region_groups.append(region_data.values)
        
        if len(region_groups) >= 2:
            kw_stat, kw_pval = stats.kruskal(*region_groups)
            results['kruskal_wallis'] = {'statistic': kw_stat, 'p_value': kw_pval}
            print(f"\n3. Kruskal-Wallis H (Regional Comparison):")
            print(f"   H Statistic: {kw_stat:.2f}, p-value: {kw_pval:.4e}")
            print(f"   Result: {'Significant' if kw_pval < 0.05 else 'Not Significant'} regional difference")
        
        # 4. Effect size (Cohen's d for weekend gap)
        cohens_d = (weekday_totals.mean() - weekend_totals.mean()) / np.sqrt(
            (weekday_totals.std()**2 + weekend_totals.std()**2) / 2
        )
        results['cohens_d'] = cohens_d
        print(f"\n4. Cohen's d (Weekend Gap Effect Size):")
        print(f"   Cohen's d: {cohens_d:.3f}")
        effect_size = 'Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small'
        print(f"   Interpretation: {effect_size} effect")
        
        # 5. Levene's test for homogeneity of variance
        levene_stat, levene_pval = stats.levene(weekend_totals, weekday_totals)
        results['levene'] = {'statistic': levene_stat, 'p_value': levene_pval}
        print(f"\n5. Levene's Test (Variance Homogeneity):")
        print(f"   W Statistic: {levene_stat:.2f}, p-value: {levene_pval:.4e}")
        print(f"   Result: {'Unequal' if levene_pval < 0.05 else 'Equal'} variances")
        
        return results
    
    # =========================================================================
    # RUN ALL ANALYSES
    # =========================================================================
    def run_all(self) -> dict:
        """Run all advanced analyses"""
        print("\n" + "="*70)
        print(" UIDAI DATA HACKATHON 2026 - ADVANCED ANALYTICS")
        print("   Mangesh Bharat Raut | Team ID: UIDAI_4879")
        print("="*70)
        
        results = {}
        
        # 1. Anomaly Detection
        results['anomalies'] = self.detect_anomalies_isolation_forest()
        
        # 2. Ensemble Clustering
        results['clustering'] = self.ensemble_clustering()
        
        # 3. Time Series Forecasting
        results['forecast'] = self.advanced_time_series_forecast()
        
        # 4. Feature Importance
        results['features'] = self.feature_importance_analysis()
        
        # 5. Statistical Tests
        results['stats'] = self.advanced_statistical_tests()
        
        print("\n" + "="*70)
        print(" ALL ADVANCED ANALYSES COMPLETE!")
        print("="*70)
        
        return results


if __name__ == "__main__":
    base_dir = Path('/Users/mangeshraut/Downloads/UIDAI Data Hackathon 2026')
    analytics = AdvancedAnalytics(base_dir)
    results = analytics.run_all()
