"""
Test Data Generator for TitleCraft AI

This module generates comprehensive test data for both Phase 1 and Phase 2 testing,
including realistic YouTube channel data, video titles, and expected outputs.
"""

import random
import csv
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class TestVideo:
    """Test video data structure"""
    channel_id: str
    video_id: str
    title: str
    summary: str
    views_in_period: int
    upload_date: str
    category: str
    tags: List[str]
    duration_minutes: int
    
    
@dataclass
class TestChannel:
    """Test channel data structure"""
    channel_id: str
    channel_name: str
    channel_type: str
    subscriber_count: int
    total_videos: int
    avg_views: int
    primary_topics: List[str]
    upload_frequency: str


class TestDataGenerator:
    """
    Generates realistic test data for TitleCraft AI testing
    
    Features:
    - Multiple channel types with realistic patterns
    - Varied performance distributions
    - Consistent title patterns per channel
    - Realistic view counts and metadata
    """
    
    def __init__(self, seed: Optional[int] = 42):
        """Initialize generator with optional seed for reproducibility"""
        if seed:
            random.seed(seed)
        
        self.channel_templates = self._initialize_channel_templates()
        self.title_patterns = self._initialize_title_patterns()
        self.performance_distributions = self._initialize_performance_distributions()
    
    def _initialize_channel_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize channel type templates with realistic characteristics"""
        return {
            'tech_review': {
                'name_patterns': ['Tech Reviews', 'Gadget Central', 'Tech Insider', 'Digital World'],
                'topics': ['smartphone', 'laptop', 'gaming', 'ai', 'software', 'hardware'],
                'avg_views_range': (5000, 50000),
                'subscriber_range': (10000, 500000),
                'upload_frequency': 'weekly',
                'title_characteristics': {
                    'avg_length': 8,
                    'question_ratio': 0.3,
                    'numeric_ratio': 0.4,
                    'superlative_ratio': 0.5,
                    'emotional_ratio': 0.2
                }
            },
            
            'educational': {
                'name_patterns': ['Learn Academy', 'Science Hub', 'Knowledge Base', 'Edu Channel'],
                'topics': ['science', 'history', 'mathematics', 'physics', 'biology', 'chemistry'],
                'avg_views_range': (8000, 30000),
                'subscriber_range': (25000, 300000),
                'upload_frequency': 'bi-weekly',
                'title_characteristics': {
                    'avg_length': 9,
                    'question_ratio': 0.4,
                    'numeric_ratio': 0.3,
                    'superlative_ratio': 0.2,
                    'emotional_ratio': 0.1
                }
            },
            
            'entertainment': {
                'name_patterns': ['Fun Zone', 'Comedy Central', 'Entertainment Hub', 'Viral Videos'],
                'topics': ['comedy', 'reaction', 'challenge', 'viral', 'meme', 'trending'],
                'avg_views_range': (10000, 100000),
                'subscriber_range': (50000, 1000000),
                'upload_frequency': 'daily',
                'title_characteristics': {
                    'avg_length': 7,
                    'question_ratio': 0.2,
                    'numeric_ratio': 0.2,
                    'superlative_ratio': 0.4,
                    'emotional_ratio': 0.6
                }
            },
            
            'gaming': {
                'name_patterns': ['Game Master', 'Gaming Pro', 'Game Reviews', 'Play Hub'],
                'topics': ['gameplay', 'review', 'tutorial', 'stream', 'walkthrough', 'tips'],
                'avg_views_range': (3000, 80000),
                'subscriber_range': (5000, 750000),
                'upload_frequency': 'daily',
                'title_characteristics': {
                    'avg_length': 8,
                    'question_ratio': 0.25,
                    'numeric_ratio': 0.35,
                    'superlative_ratio': 0.45,
                    'emotional_ratio': 0.3
                }
            },
            
            'lifestyle': {
                'name_patterns': ['Life Style', 'Daily Vlogs', 'Life Hacks', 'Wellness Guide'],
                'topics': ['fitness', 'cooking', 'travel', 'fashion', 'health', 'relationships'],
                'avg_views_range': (4000, 25000),
                'subscriber_range': (8000, 200000),
                'upload_frequency': 'weekly',
                'title_characteristics': {
                    'avg_length': 7,
                    'question_ratio': 0.35,
                    'numeric_ratio': 0.25,
                    'superlative_ratio': 0.3,
                    'emotional_ratio': 0.25
                }
            }
        }
    
    def _initialize_title_patterns(self) -> Dict[str, List[str]]:
        """Initialize title pattern templates by category"""
        return {
            'tech_review': [
                "{number} Best {product} of {year}",
                "{product} vs {product}: Which is Better?",
                "Is {product} Worth It? Honest Review",
                "{product} After {timeframe}: Still Good?",
                "Ultimate {product} Buying Guide {year}",
                "{shocking_adjective} {product} Features You Missed",
                "Why I {action} My {product}",
                "{product} Problems Nobody Talks About"
            ],
            
            'educational': [
                "How {process} Actually Works",
                "{number} {subject} Facts That Will {emotion} You",
                "The {adjective} Truth About {topic}",
                "Why {phenomenon} Happens: Scientific Explanation",
                "{topic} Explained in {timeframe}",
                "What If {scenario}? Science Answers",
                "The {adjective} History of {topic}",
                "{topic}: Everything You Need to Know"
            ],
            
            'entertainment': [
                "{adjective} {content_type} That Will Make You {emotion}",
                "Reacting to {viral_content}",
                "{number} {content_type} That {action}",
                "This {content} is {adjective}!",
                "{challenge_name} Challenge: {outcome}",
                "Why Everyone is Talking About {topic}",
                "{adjective} {content} Compilation",
                "You Won't Believe This {content}!"
            ],
            
            'gaming': [
                "{game_title} {content_type}: {outcome}",
                "{number} {adjective} {game_feature} in {game}",
                "How to {action} in {game_title}",
                "{game} vs {game}: Which is Better?",
                "{adjective} {game} Moments",
                "New {game} Update: {reaction}",
                "{game} {content_type} Guide",
                "Why {game} is {adjective}"
            ],
            
            'lifestyle': [
                "{number} {category} Tips for {goal}",
                "My {timeframe} {journey_type} Journey",
                "{adjective} {category} Routine That Works",
                "How I {achievement} in {timeframe}",
                "{category} Mistakes Everyone Makes",
                "{adjective} {category} Hacks You Need",
                "Why I {action} {habit}",
                "{category} Transformation: {result}"
            ]
        }
    
    def _initialize_performance_distributions(self) -> Dict[str, Dict[str, float]]:
        """Initialize performance distribution patterns"""
        return {
            'high_performer': {'probability': 0.1, 'multiplier_range': (2.0, 5.0)},
            'good_performer': {'probability': 0.2, 'multiplier_range': (1.3, 1.9)},
            'average_performer': {'probability': 0.4, 'multiplier_range': (0.8, 1.2)},
            'low_performer': {'probability': 0.3, 'multiplier_range': (0.3, 0.7)}
        }
    
    def generate_channels(self, num_channels: int = 20) -> List[TestChannel]:
        """Generate realistic test channels"""
        channels = []
        
        for i in range(num_channels):
            # Select random channel type
            channel_type = random.choice(list(self.channel_templates.keys()))
            template = self.channel_templates[channel_type]
            
            # Generate channel ID
            channel_id = f"UC{hashlib.md5(f'test_channel_{i}'.encode()).hexdigest()[:20]}"
            
            # Generate channel details
            channel_name = f"{random.choice(template['name_patterns'])} {random.randint(1, 999)}"
            subscriber_count = random.randint(*template['subscriber_range'])
            avg_views = random.randint(*template['avg_views_range'])
            total_videos = random.randint(50, 500)
            
            channel = TestChannel(
                channel_id=channel_id,
                channel_name=channel_name,
                channel_type=channel_type,
                subscriber_count=subscriber_count,
                total_videos=total_videos,
                avg_views=avg_views,
                primary_topics=template['topics'],
                upload_frequency=template['upload_frequency']
            )
            
            channels.append(channel)
        
        return channels
    
    def generate_videos_for_channel(self, channel: TestChannel, num_videos: int = 30) -> List[TestVideo]:
        """Generate realistic videos for a specific channel"""
        videos = []
        template = self.channel_templates[channel.channel_type]
        
        # Generate upload dates spanning last year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        for i in range(num_videos):
            # Generate video ID
            video_id = f"{hashlib.md5(f'{channel.channel_id}_video_{i}'.encode()).hexdigest()[:11]}"
            
            # Generate title using patterns
            title = self._generate_title(channel.channel_type, template)
            
            # Generate summary
            summary = self._generate_summary(title, channel.primary_topics)
            
            # Generate performance-based views
            performance_type = self._select_performance_type()
            base_views = channel.avg_views
            multiplier = random.uniform(*self.performance_distributions[performance_type]['multiplier_range'])
            views = int(base_views * multiplier)
            
            # Random upload date
            days_ago = random.randint(0, 365)
            upload_date = (end_date - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            
            # Generate metadata
            tags = random.sample(channel.primary_topics, k=min(3, len(channel.primary_topics)))
            duration = random.randint(2, 30)  # 2-30 minutes
            
            video = TestVideo(
                channel_id=channel.channel_id,
                video_id=video_id,
                title=title,
                summary=summary,
                views_in_period=views,
                upload_date=upload_date,
                category=channel.channel_type,
                tags=tags,
                duration_minutes=duration
            )
            
            videos.append(video)
        
        return videos
    
    def _select_performance_type(self) -> str:
        """Select performance type based on probability distribution"""
        rand = random.random()
        cumulative = 0
        
        for perf_type, config in self.performance_distributions.items():
            cumulative += config['probability']
            if rand <= cumulative:
                return perf_type
        
        return 'average_performer'
    
    def _generate_title(self, channel_type: str, template: Dict[str, Any]) -> str:
        """Generate realistic title based on channel type"""
        patterns = self.title_patterns[channel_type]
        pattern = random.choice(patterns)
        
        # Define replacement dictionaries
        replacements = {
            'number': str(random.randint(1, 20)),
            'timeframe': random.choice(['1 Month', '6 Months', '1 Year', '30 Days']),
            'year': str(random.randint(2022, 2025)),
            'adjective': random.choice(['Amazing', 'Incredible', 'Ultimate', 'Best', 'Worst', 'Hidden', 'Secret']),
            'shocking_adjective': random.choice(['Shocking', 'Amazing', 'Incredible', 'Mind-Blowing']),
            'emotion': random.choice(['Shock', 'Amaze', 'Surprise', 'Blow Your Mind']),
            'action': random.choice(['Switched From', 'Upgraded', 'Ditched', 'Chose']),
            'outcome': random.choice(['Success', 'Fail', 'Epic Win', 'Disaster', 'Incredible Result']),
            'reaction': random.choice(['Amazing', 'Terrible', 'Game-Changing', 'Disappointing'])
        }
        
        # Channel-specific replacements
        if channel_type == 'tech_review':
            replacements.update({
                'product': random.choice(['iPhone', 'MacBook', 'Android Phone', 'Laptop', 'Tablet', 'Headphones']),
            })
        elif channel_type == 'educational':
            replacements.update({
                'subject': random.choice(['Physics', 'Science', 'History', 'Math', 'Biology']),
                'topic': random.choice(['Black Holes', 'Quantum Physics', 'Evolution', 'Space', 'DNA']),
                'process': random.choice(['Gravity', 'Photosynthesis', 'Memory', 'Learning', 'Healing']),
                'phenomenon': random.choice(['Lightning', 'Earthquakes', 'Northern Lights', 'Rain']),
                'scenario': random.choice(['Gravity Stopped', 'Earth Stopped Spinning', 'Sun Disappeared'])
            })
        elif channel_type == 'entertainment':
            replacements.update({
                'content_type': random.choice(['Videos', 'Memes', 'Clips', 'Moments', 'Fails']),
                'viral_content': random.choice(['TikToks', 'Memes', 'Viral Videos', 'Trends']),
                'challenge_name': random.choice(['24 Hour', 'Impossible', 'Extreme', 'Crazy']),
                'content': random.choice(['Video', 'Meme', 'Clip', 'Challenge'])
            })
        elif channel_type == 'gaming':
            replacements.update({
                'game_title': random.choice(['Minecraft', 'Fortnite', 'COD', 'GTA', 'FIFA']),
                'game': random.choice(['Minecraft', 'Fortnite', 'Valorant', 'Apex']),
                'content_type': random.choice(['Gameplay', 'Review', 'Tutorial', 'Guide', 'Tips']),
                'game_feature': random.choice(['Tips', 'Tricks', 'Secrets', 'Hacks', 'Strategies'])
            })
        elif channel_type == 'lifestyle':
            replacements.update({
                'category': random.choice(['Fitness', 'Diet', 'Skincare', 'Fashion', 'Productivity']),
                'goal': random.choice(['Weight Loss', 'Muscle Gain', 'Better Skin', 'Success']),
                'journey_type': random.choice(['Fitness', 'Weight Loss', 'Skincare', 'Career']),
                'achievement': random.choice(['Lost 20lbs', 'Got Fit', 'Changed My Life', 'Succeeded']),
                'habit': random.choice(['Morning Routine', 'Diet', 'Exercise', 'Meditation']),
                'result': random.choice(['Amazing Results', 'Life Changed', 'Incredible Change'])
            })
        
        # Replace placeholders in pattern
        try:
            title = pattern.format(**{k: v for k, v in replacements.items() if f'{{{k}}}' in pattern})
        except KeyError:
            # Fallback to simple title if pattern matching fails
            title = f"{replacements.get('adjective', 'Amazing')} {channel_type.replace('_', ' ').title()} Content"
        
        return title
    
    def _generate_summary(self, title: str, topics: List[str]) -> str:
        """Generate realistic video summary based on title and topics"""
        topic = random.choice(topics)
        
        summary_templates = [
            f"In this video, we explore {topic} and discuss {title.lower()}",
            f"Learn everything about {topic} in this comprehensive guide",
            f"Join us as we dive deep into {topic} and uncover interesting insights",
            f"This video covers {topic} with practical tips and examples",
            f"Discover the secrets of {topic} in this detailed analysis"
        ]
        
        return random.choice(summary_templates)
    
    def generate_complete_dataset(self, 
                                 num_channels: int = 10,
                                 videos_per_channel: int = 30) -> tuple[List[TestChannel], List[TestVideo]]:
        """Generate complete test dataset with channels and videos"""
        print(f"Generating {num_channels} test channels with {videos_per_channel} videos each...")
        
        channels = self.generate_channels(num_channels)
        all_videos = []
        
        for channel in channels:
            videos = self.generate_videos_for_channel(channel, videos_per_channel)
            all_videos.extend(videos)
        
        print(f"Generated {len(channels)} channels and {len(all_videos)} videos")
        return channels, all_videos
    
    def export_to_csv(self, videos: List[TestVideo], filename: str = "test_data.csv"):
        """Export videos to CSV format compatible with TitleCraft AI"""
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['channel_id', 'video_id', 'title', 'summary', 'views_in_period']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for video in videos:
                writer.writerow({
                    'channel_id': video.channel_id,
                    'video_id': video.video_id,
                    'title': video.title,
                    'summary': video.summary,
                    'views_in_period': video.views_in_period
                })
        
        print(f"Exported {len(videos)} videos to {filename}")
    
    def export_channels_to_json(self, channels: List[TestChannel], filename: str = "test_channels.json"):
        """Export channel metadata to JSON"""
        channels_data = [asdict(channel) for channel in channels]
        
        with open(filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(channels_data, jsonfile, indent=2)
        
        print(f"Exported {len(channels)} channels to {filename}")
    
    def create_test_scenarios(self) -> Dict[str, Any]:
        """Create specific test scenarios for different testing needs"""
        scenarios = {
            'unit_tests': {
                'description': 'Small datasets for fast unit testing',
                'channels': 3,
                'videos_per_channel': 10
            },
            
            'integration_tests': {
                'description': 'Medium datasets for integration testing',
                'channels': 5,
                'videos_per_channel': 20
            },
            
            'performance_tests': {
                'description': 'Large datasets for performance testing',
                'channels': 20,
                'videos_per_channel': 100
            },
            
            'edge_cases': {
                'description': 'Special cases for edge case testing',
                'channels': 2,
                'videos_per_channel': 5
            }
        }
        
        return scenarios
    
    def generate_expected_outputs(self, channels: List[TestChannel], videos: List[TestVideo]) -> Dict[str, Any]:
        """Generate expected test outputs for validation"""
        
        # Calculate expected statistics
        total_videos = len(videos)
        total_views = sum(video.views_in_period for video in videos)
        avg_views = total_views / total_videos if total_videos > 0 else 0
        
        # Channel statistics
        channel_stats = {}
        for channel in channels:
            channel_videos = [v for v in videos if v.channel_id == channel.channel_id]
            if channel_videos:
                channel_stats[channel.channel_id] = {
                    'video_count': len(channel_videos),
                    'avg_views': sum(v.views_in_period for v in channel_videos) / len(channel_videos),
                    'total_views': sum(v.views_in_period for v in channel_videos),
                    'channel_type': channel.channel_type
                }
        
        # Title pattern analysis
        title_patterns = {
            'avg_length_words': sum(len(v.title.split()) for v in videos) / total_videos,
            'question_ratio': sum(1 for v in videos if v.title.strip().endswith('?')) / total_videos,
            'numeric_ratio': sum(1 for v in videos if any(char.isdigit() for char in v.title)) / total_videos,
            'superlative_count': sum(1 for v in videos if any(word in v.title.lower() for word in ['best', 'worst', 'top', 'ultimate']))
        }
        
        expected_outputs = {
            'dataset_summary': {
                'total_videos': total_videos,
                'unique_channels': len(channels),
                'avg_views_per_video': avg_views,
                'total_views': total_views
            },
            'channel_statistics': channel_stats,
            'title_patterns': title_patterns,
            'performance_thresholds': {
                'high_performance': avg_views * 1.5,
                'low_performance': avg_views * 0.5
            }
        }
        
        return expected_outputs


def create_test_datasets():
    """Create all test datasets needed for TitleCraft AI testing"""
    generator = TestDataGenerator(seed=42)
    scenarios = generator.create_test_scenarios()
    
    print("🧪 Creating Test Datasets for TitleCraft AI")
    print("=" * 50)
    
    for scenario_name, config in scenarios.items():
        print(f"\n📊 Creating {scenario_name}...")
        print(f"   Description: {config['description']}")
        
        # Generate data
        channels, videos = generator.generate_complete_dataset(
            num_channels=config['channels'],
            videos_per_channel=config['videos_per_channel']
        )
        
        # Export files
        csv_filename = f"test_data_{scenario_name}.csv"
        json_filename = f"test_channels_{scenario_name}.json"
        expected_filename = f"expected_outputs_{scenario_name}.json"
        
        generator.export_to_csv(videos, csv_filename)
        generator.export_channels_to_json(channels, json_filename)
        
        # Generate expected outputs
        expected_outputs = generator.generate_expected_outputs(channels, videos)
        with open(expected_filename, 'w', encoding='utf-8') as f:
            json.dump(expected_outputs, f, indent=2)
        
        print(f"   ✅ Created {len(videos)} videos across {len(channels)} channels")
        print(f"   📁 Files: {csv_filename}, {json_filename}, {expected_filename}")


if __name__ == "__main__":
    create_test_datasets()