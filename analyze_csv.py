
import pandas as pd
import numpy as np

def analyze_new_impulse_data(file_path):
    # Load the data
    df = pd.read_csv(file_path)
    
    # 1. Basics
    total_entries = len(df)
    symbols = df['Symbol'].unique()
    
    # 2. Impulse Success Analysis
    reversal_rate = df['ReversalTriggered'].value_counts(normalize=True) * 100
    
    # 3. Magnitude Analysis
    avg_diff_points = df.groupby('WaveDirection')['DifferencePoints'].mean()
    avg_diff_percent = df.groupby('WaveDirection')['DifferencePercent'].mean()
    
    # 4. Session Analysis
    session_counts = df['CrossoverStartSession'].value_counts()
    session_magnitude = df.groupby('CrossoverStartSession')['DifferencePoints'].agg(['mean', 'max', 'count'])
    
    # 5. Time Analysis
    # Convert times
    df['CrossoverStartTime'] = pd.to_datetime(df['CrossoverStartTime'])
    df['AbsolutePeakTime'] = pd.to_datetime(df['AbsolutePeakTime'])
    df['TimeToPeakMinutes'] = (df['AbsolutePeakTime'] - df['CrossoverStartTime']).dt.total_seconds() / 60
    
    avg_time_to_peak = df.groupby('CrossoverStartSession')['TimeToPeakMinutes'].mean()
    
    # 6. Correlation (Implicit)
    # How does MA period affect move magnitude?
    ma_performance = df.groupby('MovingAveragePeriod')['DifferencePoints'].mean()

    results = {
        "total_entries": total_entries,
        "symbols": symbols.tolist(),
        "reversal_rate": reversal_rate.to_dict(),
        "avg_diff_points": avg_diff_points.to_dict(),
        "avg_diff_percent": avg_diff_percent.to_dict(),
        "session_magnitude": session_magnitude.to_dict('index'),
        "avg_time_to_peak": avg_time_to_peak.to_dict(),
        "ma_performance": ma_performance.to_dict()
    }
    
    return results

if __name__ == "__main__":
    file_path = r'D:\date-wise\17-03-2026\ma_impulse_data.csv'
    res = analyze_new_impulse_data(file_path)
    print("--- DATA ANALYSIS REPORT ---")
    print(f"Total Waves Analyzed: {res['total_entries']}")
    print(f"Symbols: {res['symbols']}")
    print("\n--- REVERSAL SUCCESS (Back to MA) ---")
    for k, v in res['reversal_rate'].items():
        print(f"  {k}: {v:.2f}%")
        
    print("\n--- MAGNITUDE BY DIRECTION ---")
    for k, v in res['avg_diff_points'].items():
        print(f"  {k}: Avg {v:.2f} points ({res['avg_diff_percent'][k]:.3f}%)")
        
    print("\n--- SESSION PERFORMANCE ---")
    for sess, data in res['session_magnitude'].items():
        print(f"  {sess}: Count {data['count']}, Avg Move {data['mean']:.2f}, Max Move {data['max']:.2f}")

    print("\n--- AVG TIME TO PEAK (MINUTES) ---")
    for sess, t in res['avg_time_to_peak'].items():
        print(f"  {sess}: {t:.2f} mins")
