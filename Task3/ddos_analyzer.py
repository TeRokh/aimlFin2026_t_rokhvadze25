#!/usr/bin/env python3
"""
DDoS Attack Detection using Regression Analysis
This script analyzes web server logs to detect DDoS attack intervals
using statistical methods and regression analysis.
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def parse_log_line(line):
    """Parse a single log line and extract relevant information."""
    # Regular expression pattern for Apache/Nginx log format
    pattern = r'(\S+) - - \[(.*?)\] "(\S+) (\S+) (\S+)" (\d+) (\d+) "(.*?)" "(.*?)" (\d+)'
    match = re.match(pattern, line)
    
    if match:
        ip, timestamp, method, endpoint, protocol, status, size, referer, user_agent, response_time = match.groups()
        # Parse timestamp
        dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S%z')
        return {
            'ip': ip,
            'timestamp': dt,
            'method': method,
            'endpoint': endpoint,
            'status': int(status),
            'size': int(size),
            'response_time': int(response_time)
        }
    return None

def load_and_parse_logs(filepath):
    """Load and parse the log file."""
    print(f"Loading log file: {filepath}")
    logs = []
    
    with open(filepath, 'r') as f:
        for line in f:
            parsed = parse_log_line(line.strip())
            if parsed:
                logs.append(parsed)
    
    df = pd.DataFrame(logs)
    print(f"Loaded {len(df)} log entries")
    return df

def aggregate_by_minute(df):
    """Aggregate requests by minute."""
    df['minute'] = df['timestamp'].dt.floor('min')
    
    minute_stats = df.groupby('minute').agg({
        'ip': 'count',  # Request count
        'response_time': 'mean',  # Average response time
        'size': 'mean'  # Average response size
    }).rename(columns={'ip': 'request_count'})
    
    # Calculate unique IP count per minute
    unique_ips = df.groupby('minute')['ip'].nunique().rename('unique_ips')
    minute_stats = minute_stats.join(unique_ips)
    
    # Calculate requests per IP ratio
    minute_stats['requests_per_ip'] = minute_stats['request_count'] / minute_stats['unique_ips']
    
    return minute_stats

def calculate_regression_metrics(df):
    """Calculate regression-based anomaly detection metrics."""
    # Create time-based features
    df = df.copy()
    df['time_index'] = range(len(df))
    
    # Fit linear regression for request count
    X = df[['time_index']].values
    y = df['request_count'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Get predictions and residuals
    df['predicted_requests'] = model.predict(X)
    df['residual'] = df['request_count'] - df['predicted_requests']
    df['residual_abs'] = np.abs(df['residual'])
    
    # Calculate z-score for residuals
    df['residual_zscore'] = stats.zscore(df['residual'])
    
    # Calculate rolling statistics
    window = 5
    df['rolling_mean'] = df['request_count'].rolling(window=window, center=True).mean()
    df['rolling_std'] = df['request_count'].rolling(window=window, center=True).std()
    
    return df, model

def detect_ddos_intervals(df, threshold_zscore=2.0, min_duration_minutes=1, max_gap_minutes=3):
    """
    Detect DDoS attack intervals using regression residuals and statistical thresholds.
    
    Parameters:
    - threshold_zscore: Z-score threshold for anomaly detection
    - min_duration_minutes: Minimum duration to consider as a DDoS attack
    - max_gap_minutes: Maximum gap between attack peaks to consider as same attack
    """
    # Mark potential attack points with lower threshold for high traffic
    df['is_anomaly'] = df['residual_zscore'] > threshold_zscore
    df['is_high_traffic'] = df['request_count'] > df['request_count'].quantile(0.75)
    
    # Combine conditions for DDoS detection
    df['is_ddos'] = df['is_anomaly'] & df['is_high_traffic']
    
    # Find attack points and group them into intervals
    attack_indices = df[df['is_ddos']].index.tolist()
    
    if not attack_indices:
        return [], df
    
    intervals = []
    current_start = attack_indices[0]
    current_end = attack_indices[0]
    
    for i in range(1, len(attack_indices)):
        time_gap = (attack_indices[i] - current_end).total_seconds() / 60
        
        if time_gap <= max_gap_minutes:
            # Extend current interval
            current_end = attack_indices[i]
        else:
            # Save current interval and start new one
            duration = (current_end - current_start).total_seconds() / 60
            if duration >= min_duration_minutes:
                # Extend interval to include surrounding context
                mask = (df.index >= current_start) & (df.index <= current_end)
                interval_data = df[mask]
                
                intervals.append({
                    'start': current_start,
                    'end': current_end,
                    'duration_minutes': duration,
                    'avg_requests': interval_data['request_count'].mean(),
                    'max_requests': interval_data['request_count'].max(),
                    'avg_residual_zscore': interval_data['residual_zscore'].mean(),
                    'total_requests': interval_data['request_count'].sum(),
                    'unique_ips': interval_data['unique_ips'].sum()
                })
            current_start = attack_indices[i]
            current_end = attack_indices[i]
    
    # Don't forget the last interval
    duration = (current_end - current_start).total_seconds() / 60
    if duration >= min_duration_minutes:
        mask = (df.index >= current_start) & (df.index <= current_end)
        interval_data = df[mask]
        
        intervals.append({
            'start': current_start,
            'end': current_end,
            'duration_minutes': duration,
            'avg_requests': interval_data['request_count'].mean(),
            'max_requests': interval_data['request_count'].max(),
            'avg_residual_zscore': interval_data['residual_zscore'].mean(),
            'total_requests': interval_data['request_count'].sum(),
            'unique_ips': interval_data['unique_ips'].sum()
        })
    
    return intervals, df

def create_visualizations(df, intervals, output_dir='/home/trokhvadze'):
    """Create comprehensive visualizations for the analysis."""
    
    # Figure 1: Request Count Over Time with Regression Line
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: Request count and regression
    ax1 = axes[0]
    ax1.plot(df.index, df['request_count'], label='Actual Requests', alpha=0.7, linewidth=1)
    ax1.plot(df.index, df['predicted_requests'], label='Predicted (Linear Regression)', 
             color='red', linewidth=2, linestyle='--')
    ax1.fill_between(df.index, 
                      df['predicted_requests'] - 2*df['rolling_std'], 
                      df['predicted_requests'] + 2*df['rolling_std'], 
                      alpha=0.2, color='gray', label='±2σ Band')
    
    # Highlight DDoS intervals
    for interval in intervals:
        ax1.axvspan(interval['start'], interval['end'], alpha=0.3, color='red', 
                   label='DDoS Attack' if interval == intervals[0] else '')
    
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Requests per Minute', fontsize=12)
    ax1.set_title('Request Count Over Time with Linear Regression Model', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    ax2 = axes[1]
    ax2.plot(df.index, df['residual'], label='Residuals', alpha=0.7, linewidth=1, color='blue')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.axhline(y=df['residual'].std() * 2.5, color='red', linestyle='--', 
                linewidth=1, label='±2.5σ Threshold')
    ax2.axhline(y=-df['residual'].std() * 2.5, color='red', linestyle='--', linewidth=1)
    
    # Highlight DDoS intervals
    for interval in intervals:
        ax2.axvspan(interval['start'], interval['end'], alpha=0.3, color='red')
    
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Residual', fontsize=12)
    ax2.set_title('Regression Residuals (Actual - Predicted)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/regression_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/regression_analysis.png")
    plt.close()
    
    # Figure 2: Statistical Metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Z-scores
    ax1 = axes[0, 0]
    ax1.plot(df.index, df['residual_zscore'], label='Z-score of Residuals', 
             alpha=0.7, linewidth=1, color='purple')
    ax1.axhline(y=2.5, color='red', linestyle='--', linewidth=1, label='Threshold (±2.5)')
    ax1.axhline(y=-2.5, color='red', linestyle='--', linewidth=1)
    for interval in intervals:
        ax1.axvspan(interval['start'], interval['end'], alpha=0.3, color='red')
    ax1.set_ylabel('Z-score', fontsize=12)
    ax1.set_title('Standardized Residuals (Z-scores)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Requests per IP
    ax2 = axes[0, 1]
    ax2.plot(df.index, df['requests_per_ip'], label='Requests per IP', 
             alpha=0.7, linewidth=1, color='green')
    for interval in intervals:
        ax2.axvspan(interval['start'], interval['end'], alpha=0.3, color='red')
    ax2.set_ylabel('Requests per IP', fontsize=12)
    ax2.set_title('Average Requests per IP Address', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Response Time
    ax3 = axes[1, 0]
    ax3.plot(df.index, df['response_time'], label='Avg Response Time', 
             alpha=0.7, linewidth=1, color='orange')
    for interval in intervals:
        ax3.axvspan(interval['start'], interval['end'], alpha=0.3, color='red')
    ax3.set_ylabel('Response Time (ms)', fontsize=12)
    ax3.set_xlabel('Time', fontsize=12)
    ax3.set_title('Average Response Time', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Unique IPs
    ax4 = axes[1, 1]
    ax4.plot(df.index, df['unique_ips'], label='Unique IPs', 
             alpha=0.7, linewidth=1, color='brown')
    for interval in intervals:
        ax4.axvspan(interval['start'], interval['end'], alpha=0.3, color='red')
    ax4.set_ylabel('Unique IP Count', fontsize=12)
    ax4.set_xlabel('Time', fontsize=12)
    ax4.set_title('Unique IP Addresses per Minute', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/statistical_metrics.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/statistical_metrics.png")
    plt.close()
    
    # Figure 3: Residual Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(df['residual'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Mean')
    ax1.axvline(x=df['residual'].std() * 2.5, color='orange', linestyle='--', 
                linewidth=2, label='±2.5σ')
    ax1.axvline(x=-df['residual'].std() * 2.5, color='orange', linestyle='--', linewidth=2)
    ax1.set_xlabel('Residual Value', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Regression Residuals', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q-Q Plot
    ax2 = axes[1]
    stats.probplot(df['residual'], dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot of Residuals', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/residual_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/residual_distribution.png")
    plt.close()

def print_summary(df, intervals, model):
    """Print analysis summary."""
    print("\n" + "="*70)
    print("DDoS ATTACK DETECTION ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\nDataset Information:")
    print(f"  Total time span: {df.index[0]} to {df.index[-1]}")
    print(f"  Total duration: {(df.index[-1] - df.index[0]).total_seconds() / 60:.1f} minutes")
    print(f"  Total data points: {len(df)}")
    
    print(f"\nRegression Model Statistics:")
    print(f"  Slope: {model.coef_[0]:.4f}")
    print(f"  Intercept: {model.intercept_:.4f}")
    print(f"  R² Score: {model.score(df[['time_index']].values, df['request_count'].values):.4f}")
    
    print(f"\nTraffic Statistics:")
    print(f"  Mean requests/min: {df['request_count'].mean():.2f}")
    print(f"  Std requests/min: {df['request_count'].std():.2f}")
    print(f"  Max requests/min: {df['request_count'].max():.0f}")
    print(f"  Mean unique IPs/min: {df['unique_ips'].mean():.2f}")
    
    print(f"\nDDoS Attack Detection Results:")
    print(f"  Number of attacks detected: {len(intervals)}")
    
    if intervals:
        print(f"\n  Attack Intervals:")
        for i, interval in enumerate(intervals, 1):
            print(f"\n  Attack #{i}:")
            print(f"    Start: {interval['start']}")
            print(f"    End: {interval['end']}")
            print(f"    Duration: {interval['duration_minutes']:.2f} minutes")
            print(f"    Avg requests/min: {interval['avg_requests']:.2f}")
            print(f"    Max requests/min: {interval['max_requests']:.0f}")
            print(f"    Total requests: {interval['total_requests']:.0f}")
            print(f"    Avg Z-score: {interval['avg_residual_zscore']:.2f}")
    
    print("\n" + "="*70)

def main():
    """Main execution function."""
    # Configuration
    log_file = '/mnt/user-data/uploads/1770912107252_logs.log'
    output_dir = '/home/trokhvadze'
    
    # Load and parse logs
    df_raw = load_and_parse_logs(log_file)
    
    # Aggregate by minute
    df_minute = aggregate_by_minute(df_raw)
    
    # Calculate regression metrics
    df_analyzed, model = calculate_regression_metrics(df_minute)
    
    # Detect DDoS intervals
    intervals, df_final = detect_ddos_intervals(df_analyzed, threshold_zscore=2.0, min_duration_minutes=1, max_gap_minutes=3)
    
    # Create visualizations
    create_visualizations(df_final, intervals, output_dir)
    
    # Print summary
    print_summary(df_final, intervals, model)
    
    # Save results to CSV
    results_df = pd.DataFrame(intervals)
    if not results_df.empty:
        results_df.to_csv(f'{output_dir}/ddos_intervals.csv', index=False)
        print(f"\nResults saved to: {output_dir}/ddos_intervals.csv")
    
    # Save full analysis data
    df_final.to_csv(f'{output_dir}/analysis_data.csv')
    print(f"Full analysis data saved to: {output_dir}/analysis_data.csv")
    
    return intervals, df_final, model

if __name__ == "__main__":
    intervals, df, model = main()
