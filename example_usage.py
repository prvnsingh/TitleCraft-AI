"""
Example usage of the TitleCraft AI data module.
Demonstrates how to load data, perform analysis, and create channel profiles.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.data.analyzer import DataAnalyzer
from src.data.profiler import ChannelProfiler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function demonstrating data module usage."""
    
    # Path to your CSV data file
    data_file = "electrify__applied_ai_engineer__training_data.csv"
    
    logger.info("Starting TitleCraft AI data analysis...")
    
    try:
        # Step 1: Load and validate data
        logger.info("Loading data...")
        loader = DataLoader()
        data = loader.load_from_csv(data_file)
        
        # Get data summary
        summary = loader.get_dataset_summary()
        logger.info(f"Dataset loaded: {summary}")
        
        # Step 2: Validate data quality
        logger.info("Validating data quality...")
        validator = DataValidator()
        quality_report = validator.validate_dataset(data)
        
        print("\n=== DATA QUALITY REPORT ===")
        print(f"Overall Score: {quality_report.overall_score:.2f}")
        print(f"Completeness: {quality_report.completeness:.2f}%")
        print(f"Validity: {quality_report.validity:.2f}%")
        
        if quality_report.issues:
            print("\nIssues found:")
            for issue in quality_report.issues:
                print(f"- {issue}")
        
        # Step 3: Perform comprehensive data analysis
        logger.info("Performing data analysis...")
        analyzer = DataAnalyzer(data)
        
        # Channel analysis
        channel_stats = analyzer.analyze_channels()
        print(f"\n=== CHANNEL ANALYSIS ===")
        print(f"Total channels: {len(channel_stats)}")
        
        for channel_id, stats in channel_stats.items():
            print(f"\nChannel: {channel_id}")
            print(f"  Videos: {stats['video_count']}")
            print(f"  Avg Views: {stats['avg_views']:,.0f}")
            print(f"  Total Views: {stats['total_views']:,.0f}")
            print(f"  Performance: {stats['performance_category']}")
        
        # Performance analysis
        performance_analysis = analyzer.analyze_performance()
        print(f"\n=== PERFORMANCE ANALYSIS ===")
        print(f"High performers: {performance_analysis['high_performers']} videos")
        print(f"Average performers: {performance_analysis['average_performers']} videos")
        print(f"Low performers: {performance_analysis['low_performers']} videos")
        print(f"Performance threshold: {performance_analysis['high_performance_threshold']:,.0f} views")
        
        # Title analysis
        title_analysis = analyzer.analyze_titles()
        print(f"\n=== TITLE ANALYSIS ===")
        print(f"Average title length: {title_analysis['avg_length']:.1f} words")
        print(f"Question titles: {title_analysis['question_ratio']:.1%}")
        print(f"Numeric titles: {title_analysis['numeric_ratio']:.1%}")
        print(f"Superlative usage: {title_analysis['superlative_ratio']:.1%}")
        print(f"Emotional hooks: {title_analysis['emotional_ratio']:.1%}")
        
        print("\nMost common words:")
        for word, count in title_analysis['common_words'][:10]:
            print(f"  {word}: {count}")
        
        # Content analysis
        content_analysis = analyzer.analyze_content()
        print(f"\n=== CONTENT ANALYSIS ===")
        print(f"Average summary length: {content_analysis['avg_summary_length']:.0f} characters")
        print(f"Videos with summaries: {content_analysis['summary_coverage']:.1%}")
        
        print("\nTop content themes:")
        for theme, count in content_analysis['common_themes'][:5]:
            print(f"  {theme}: {count}")
        
        # Step 4: Create channel profiles
        logger.info("Creating channel profiles...")
        profiler = ChannelProfiler(data)
        profiles = profiler.create_all_channel_profiles()
        
        print(f"\n=== CHANNEL PROFILES ===")
        for channel_id, profile in profiles.items():
            print(f"\nChannel: {channel_id}")
            print(f"  Type: {profile.channel_type}")
            print(f"  Videos: {profile.stats.total_videos}")
            print(f"  Avg Views: {profile.stats.avg_views:,.0f}")
            print(f"  High Performers: {len(profile.high_performers)}")
            
            # Show title patterns
            patterns = profile.title_patterns
            print(f"  Title Patterns:")
            print(f"    - Avg length: {patterns.avg_length_words:.1f} words")
            print(f"    - Question format: {patterns.question_ratio:.1%}")
            print(f"    - Contains numbers: {patterns.numeric_ratio:.1%}")
            print(f"    - Emotional hooks: {patterns.emotional_hook_ratio:.1%}")
            
            # Show top performing video
            if profile.high_performers:
                top_video = profile.high_performers[0]
                print(f"  Top Video: {top_video.title[:50]}... ({top_video.views_in_period:,} views)")
        
        # Step 5: Export results
        logger.info("Exporting results...")
        
        # Export profiles
        profiler.export_profiles(profiles, "channel_profiles.json")
        profiler.export_profile_summary(profiles, "channel_profiles_summary.md")
        
        # Export analysis results
        analyzer.export_analysis("data_analysis_results.json")
        
        # Export processed data
        loader.export_processed_data("processed_data.csv")
        
        logger.info("Analysis complete! Check the exported files for detailed results.")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise

def analyze_specific_channel(channel_id: str):
    """
    Analyze a specific channel in detail.
    
    Args:
        channel_id: The ID of the channel to analyze
    """
    data_file = "electrify__applied_ai_engineer__training_data.csv"
    
    loader = DataLoader()
    data = loader.load_from_csv(data_file)
    
    # Filter data for specific channel
    channel_data = data[data['channel_id'] == channel_id]
    
    if channel_data.empty:
        print(f"No data found for channel: {channel_id}")
        return
    
    # Create detailed profile
    profiler = ChannelProfiler(data)
    profile = profiler.create_channel_profile(channel_id)
    
    print(f"\n=== DETAILED ANALYSIS FOR {channel_id} ===")
    print(f"Channel Type: {profile.channel_type}")
    print(f"Total Videos: {profile.stats.total_videos}")
    print(f"View Statistics:")
    print(f"  Average: {profile.stats.avg_views:,.0f}")
    print(f"  Median: {profile.stats.median_views:,.0f}")
    print(f"  Range: {profile.stats.min_views:,} - {profile.stats.max_views:,}")
    print(f"  High Performer Threshold: {profile.stats.high_performer_threshold:,.0f}")
    
    print(f"\nTitle Patterns:")
    patterns = profile.title_patterns
    print(f"  Length: {patterns.avg_length_words:.1f} ± {patterns.std_length_words:.1f} words")
    print(f"  Question format: {patterns.question_ratio:.1%}")
    print(f"  Contains numbers: {patterns.numeric_ratio:.1%}")
    print(f"  Superlatives: {patterns.superlative_ratio:.1%}")
    print(f"  Emotional hooks: {patterns.emotional_hook_ratio:.1%}")
    
    print(f"\nTop Words:")
    for word, count in patterns.common_words[:10]:
        print(f"  {word}: {count}")
    
    print(f"\nHigh Performing Videos:")
    for i, video in enumerate(profile.high_performers[:5], 1):
        print(f"  {i}. {video.title} ({video.views_in_period:,} views)")
    
    print(f"\nSuccess Factors:")
    for factor, data in profile.success_factors.items():
        if isinstance(data, dict):
            print(f"  {factor}:")
            for key, value in data.items():
                print(f"    {key}: {value}")
        else:
            print(f"  {factor}: {data}")

def quick_data_summary():
    """Quick summary of the dataset without full analysis."""
    data_file = "electrify__applied_ai_engineer__training_data.csv"
    
    loader = DataLoader()
    data = loader.load_from_csv(data_file)
    summary = loader.get_dataset_summary()
    
    print("=== QUICK DATA SUMMARY ===")
    print(f"Total videos: {summary['total_videos']}")
    print(f"Channels: {summary['unique_channels']}")
    print(f"Date range: {summary['date_range']}")
    print(f"Total views: {summary['total_views']:,}")
    print(f"Average views per video: {summary['avg_views_per_video']:,.0f}")
    
    # Show top channels by video count
    channel_counts = data.groupby('channel_id').size().sort_values(ascending=False)
    print(f"\nTop channels by video count:")
    for channel, count in channel_counts.head(3).items():
        print(f"  {channel}: {count} videos")

if __name__ == "__main__":
    # You can run different analysis functions:
    
    # Full analysis
    main()
    
    # Quick summary only
    # quick_data_summary()
    
    # Analyze specific channel (uncomment and replace with actual channel ID)
    # analyze_specific_channel("UC_CHANNEL_ID_HERE")