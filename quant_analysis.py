
import pandas as pd
import numpy as np

def analyze_impulse_data(file_path):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Convert time columns to datetime
    df['cross_time'] = pd.to_datetime(df['cross_time'])
    df['segment_time'] = pd.to_datetime(df['segment_time'])
    
    # Unique identifier for each crossover event
    # (cross_time and cross_type should uniquely identify a sequence)
    df['event_id'] = df['symbol'] + "_" + df['cross_time'].astype(str) + "_" + df['cross_type']
    
    # 1. Exclusive Segmentation Analysis
    # Get the terminal (highest) segment for each event
    event_terminal = df.groupby('event_id').agg({
        'segment_index': 'max',
        'session': 'first',
        'cross_type': 'first',
        'cross_time': 'first'
    }).rename(columns={'segment_index': 'terminal_segment'})
    
    total_events = len(event_terminal)
    failures = event_terminal[event_terminal['terminal_segment'] == 0]
    successes = event_terminal[event_terminal['terminal_segment'] > 0]
    
    failure_rate = (len(failures) / total_events) * 100
    success_rate = (len(successes) / total_events) * 100
    
    # Frequency of terminal segments
    terminal_counts = event_terminal['terminal_segment'].value_counts().sort_index()
    
    # 2. Time Series / Velocity Analysis
    # Calculate duration (seconds) to reach each segment from the cross
    df['reach_duration_sec'] = (df['segment_time'] - df['cross_time']).dt.total_seconds()
    
    # Average speed to reach each segment (excluding index 0)
    velocity_analysis = df[df['segment_index'] > 0].groupby(['session', 'segment_index'])['reach_duration_sec'].median().reset_index()
    
    # Summary results
    stats = {
        "total_events": total_events,
        "failed_at_threshold": len(failures),
        "success_at_threshold": len(successes),
        "failure_rate_pct": failure_rate,
        "success_rate_pct": success_rate,
        "terminal_distribution": terminal_counts.to_dict()
    }
    
    return stats, velocity_analysis, event_terminal.head(20)

if __name__ == "__main__":
    file_path = r'd:\date-wise\17-03-2026\ma_impulse_data.csv'
    stats, velocity, sample_events = analyze_impulse_data(file_path)
    
    print("--- QUANTITATIVE SUMMARY ---")
    print(f"Total Crossover Events: {stats['total_events']}")
    print(f"Threshold Failure Rate: {stats['failure_rate_pct']:.2f}%")
    print(f"Threshold Success Rate: {stats['success_rate_pct']:.2f}%")
    print("\nTerminal Segment Distribution (Exclusive):")
    for seg, count in stats['terminal_distribution'].items():
        pct = (count / stats['total_events']) * 100
        print(f"  Segment {seg}: {count} events ({pct:.2f}%)")
        
    print("\n--- VELOCITY ANALYSIS (Median Seconds to Reach Segment) ---")
    print(velocity.pivot(index='segment_index', columns='session', values='reach_duration_sec').head(10))
