import pandas as pd

# Load the data
df = pd.read_csv('electrify__applied_ai_engineer__training_data.csv')

print("=== DATASET ANALYSIS ===")
print(f"Total rows: {len(df)}")
print(f"Unique channels: {df['channel_id'].nunique()}")

print("\n=== CHANNEL BREAKDOWN ===")
for channel in df['channel_id'].unique():
    channel_data = df[df['channel_id'] == channel]
    print(f"\nChannel: {channel}")
    print(f"Videos: {len(channel_data)}")
    print(f"Avg views: {channel_data['views_in_period'].mean():.0f}")
    print(f"Top performing title: {channel_data.loc[channel_data['views_in_period'].idxmax(), 'title']}")
    print(f"Top views: {channel_data['views_in_period'].max()}")

print("\n=== HIGH PERFORMING TITLES (>10k views) ===")
high_performers = df[df['views_in_period'] > 10000].sort_values('views_in_period', ascending=False)
for _, row in high_performers.head(10).iterrows():
    print(f"{row['views_in_period']:>6} views: {row['title']}")