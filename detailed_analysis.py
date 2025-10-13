import pandas as pd
import numpy as np

# Load and analyze the data more deeply
df = pd.read_csv('electrify__applied_ai_engineer__training_data.csv')

print("=== DETAILED ANALYSIS FOR TASK APPROACH ===")

# Analyze each channel in detail
channels = df['channel_id'].unique()
channel_analysis = {}

for channel in channels:
    channel_data = df[df['channel_id'] == channel]
    
    # Performance analysis
    high_perf_threshold = channel_data['views_in_period'].quantile(0.8)
    high_performers = channel_data[channel_data['views_in_period'] >= high_perf_threshold]
    
    print(f"\n=== CHANNEL: {channel} ===")
    print(f"Total videos: {len(channel_data)}")
    print(f"Average views: {channel_data['views_in_period'].mean():.0f}")
    print(f"80th percentile threshold: {high_perf_threshold:.0f}")
    print(f"High performers ({len(high_performers)} videos):")
    
    for _, video in high_performers.sort_values('views_in_period', ascending=False).head(5).iterrows():
        print(f"  {video['views_in_period']:>6} views: {video['title']}")
    
    # Title pattern analysis
    titles = channel_data['title'].tolist()
    avg_length = np.mean([len(title.split()) for title in titles])
    question_ratio = sum([1 for title in titles if '?' in title or title.startswith(('Why', 'How', 'What'))]) / len(titles)
    numeric_ratio = sum([1 for title in titles if any(c.isdigit() for c in title)]) / len(titles)
    
    print(f"Average title length: {avg_length:.1f} words")
    print(f"Question ratio: {question_ratio:.2f}")
    print(f"Numeric ratio: {numeric_ratio:.2f}")
    
    # Store analysis
    channel_analysis[channel] = {
        'n_videos': len(channel_data),
        'avg_views': channel_data['views_in_period'].mean(),
        'high_perf_threshold': high_perf_threshold,
        'high_performers': high_performers[['title', 'views_in_period']].to_dict('records'),
        'avg_title_length': avg_length,
        'question_ratio': question_ratio,
        'numeric_ratio': numeric_ratio
    }

print(f"\n=== OVERALL DATASET PATTERNS ===")
all_titles = df['title'].tolist()
print(f"Dataset size: {len(df)} videos across {len(channels)} channels")
print(f"View range: {df['views_in_period'].min()} - {df['views_in_period'].max()}")
print(f"Overall avg title length: {np.mean([len(title.split()) for title in all_titles]):.1f} words")

# Common patterns across high performers
all_high_performers = df[df['views_in_period'] > df['views_in_period'].quantile(0.9)]
print(f"\nTop 10% performers ({len(all_high_performers)} videos) have common patterns:")
print("Most successful titles:")
for _, video in all_high_performers.sort_values('views_in_period', ascending=False).head(8).iterrows():
    print(f"  {video['views_in_period']:>6} views: {video['title']}")