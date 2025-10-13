"""
Enhanced Data Store for TitleCraft AI
Manages training data, channel profiles, and metadata with caching
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import pickle
import hashlib
import logging
from datetime import datetime, timedelta
import asyncio
from dataclasses import asdict

from ..config import get_config
from .models import ChannelProfile, VideoData, ChannelStats, DataQualityReport
from .loader import DataLoader
from .validator import DataValidator
from .profiler import ChannelProfiler

logger = logging.getLogger(__name__)

class DataStore:
    """
    Enhanced data store that manages all training data and metadata.
    Provides caching, validation, and efficient access to channel data.
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.data_path = Path(self.config.data.data_directory)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.validator = DataValidator()
        
        # Data cache
        self._raw_data_cache = {}
        self._processed_data_cache = {}
        self._metadata_cache = {}
        
        # Initialize data store
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize the data store structure"""
        required_dirs = [
            "raw",
            "processed", 
            "profiles",
            "embeddings",
            "cache"
        ]
        
        for dir_name in required_dirs:
            (self.data_path / dir_name).mkdir(exist_ok=True)
        
        logger.info(f"Data store initialized at {self.data_path}")
    
    def load_training_data(self, 
                          file_path: Optional[str] = None, 
                          force_reload: bool = False) -> pd.DataFrame:
        """
        Load and validate training data with caching
        
        Args:
            file_path: Path to CSV file, defaults to main training data
            force_reload: Force reload from disk ignoring cache
            
        Returns:
            Validated DataFrame with training data
        """
        if file_path is None:
            # Default to the main training data file
            file_path = "electrify__applied_ai_engineer__training_data.csv"
        
        cache_key = self._get_cache_key(file_path)
        
        # Check cache first
        if not force_reload and cache_key in self._raw_data_cache:
            logger.info(f"Loading data from cache: {file_path}")
            return self._raw_data_cache[cache_key]
        
        logger.info(f"Loading training data: {file_path}")
        
        try:
            # Load data
            loader = DataLoader(file_path)
            data = loader.load_raw_data()
            
            # Validate data quality
            quality_report = self.validate_data_quality(data)
            
            if quality_report.quality_score < 0.7:
                logger.warning(f"Data quality score low: {quality_report.quality_score:.2f}")
                if quality_report.issues:
                    logger.warning(f"Issues found: {quality_report.issues}")
            
            # Cache the data
            self._raw_data_cache[cache_key] = data
            
            # Save metadata
            self._save_data_metadata(file_path, data, quality_report)
            
            logger.info(f"Successfully loaded {len(data)} records from {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load training data from {file_path}: {e}")
            raise
    
    def validate_data_quality(self, data: pd.DataFrame) -> DataQualityReport:
        """
        Validate data quality and return comprehensive report
        
        Args:
            data: DataFrame to validate
            
        Returns:
            DataQualityReport with validation results
        """
        return self.validator.validate_dataset(data)
    
    def get_channel_data(self, 
                        channel_id: str, 
                        data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Get data for a specific channel
        
        Args:
            channel_id: YouTube channel ID
            data: Optional DataFrame, loads default if not provided
            
        Returns:
            DataFrame filtered for the specified channel
        """
        if data is None:
            data = self.load_training_data()
        
        channel_data = data[data['channel_id'] == channel_id].copy()
        
        if channel_data.empty:
            raise ValueError(f"No data found for channel: {channel_id}")
        
        return channel_data
    
    def get_channel_list(self, data: Optional[pd.DataFrame] = None) -> List[str]:
        """
        Get list of all available channels
        
        Args:
            data: Optional DataFrame, loads default if not provided
            
        Returns:
            List of channel IDs
        """
        if data is None:
            data = self.load_training_data()
        
        return data['channel_id'].unique().tolist()
    
    def get_dataset_summary(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Get comprehensive dataset summary
        
        Args:
            data: Optional DataFrame, loads default if not provided
            
        Returns:
            Dictionary with dataset statistics
        """
        if data is None:
            data = self.load_training_data()
        
        # Create basic summary
        return {
            'total_videos': len(data),
            'unique_channels': data['channel_id'].nunique(),
            'total_views': data['views_in_period'].sum(),
            'avg_views_per_video': data['views_in_period'].mean(),
            'date_range': 'N/A'
        }
    
    def export_channel_data(self, 
                           channel_id: str, 
                           output_path: Optional[str] = None) -> str:
        """
        Export data for a specific channel
        
        Args:
            channel_id: YouTube channel ID
            output_path: Optional output file path
            
        Returns:
            Path to exported file
        """
        channel_data = self.get_channel_data(channel_id)
        
        if output_path is None:
            output_path = str(self.data_path / "processed" / f"{channel_id}_data.csv")
        
        channel_data.to_csv(output_path, index=False)
        logger.info(f"Exported {len(channel_data)} records for channel {channel_id} to {output_path}")
        
        return output_path
    
    def get_high_performers(self, 
                           channel_id: Optional[str] = None,
                           threshold_percentile: float = 0.8,
                           data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Get high-performing videos for analysis
        
        Args:
            channel_id: Optional channel filter
            threshold_percentile: Performance threshold (0.8 = top 20%)
            data: Optional DataFrame
            
        Returns:
            DataFrame with high-performing videos
        """
        if data is None:
            data = self.load_training_data()
        
        if channel_id:
            data = data[data['channel_id'] == channel_id]
        
        threshold = data['views_in_period'].quantile(threshold_percentile)
        high_performers = data[data['views_in_period'] >= threshold].copy()
        
        # Sort by performance
        high_performers = high_performers.sort_values('views_in_period', ascending=False)
        
        return high_performers
    
    def search_videos_by_content(self, 
                                query: str,
                                channel_id: Optional[str] = None,
                                limit: int = 10) -> pd.DataFrame:
        """
        Search videos by title/summary content
        
        Args:
            query: Search query
            channel_id: Optional channel filter
            limit: Maximum results to return
            
        Returns:
            DataFrame with matching videos
        """
        data = self.load_training_data()
        
        if channel_id:
            data = data[data['channel_id'] == channel_id]
        
        # Simple text search (can be enhanced with fuzzy matching)
        query_lower = query.lower()
        
        matches = data[
            data['title'].str.lower().str.contains(query_lower, na=False) |
            data['summary'].str.lower().str.contains(query_lower, na=False)
        ].copy()
        
        # Sort by views and limit
        matches = matches.sort_values('views_in_period', ascending=False).head(limit)
        
        return matches
    
    def get_cached_data_info(self) -> Dict[str, Any]:
        """Get information about cached data"""
        return {
            "raw_data_cache_size": len(self._raw_data_cache),
            "processed_data_cache_size": len(self._processed_data_cache),
            "metadata_cache_size": len(self._metadata_cache),
            "cache_keys": list(self._raw_data_cache.keys())
        }
    
    def clear_cache(self):
        """Clear all cached data"""
        self._raw_data_cache.clear()
        self._processed_data_cache.clear() 
        self._metadata_cache.clear()
        logger.info("Data cache cleared")
    
    def _get_cache_key(self, file_path: str) -> str:
        """Generate cache key for file"""
        return hashlib.md5(str(file_path).encode()).hexdigest()
    
    def _save_data_metadata(self, 
                           file_path: str, 
                           data: pd.DataFrame, 
                           quality_report: DataQualityReport):
        """Save metadata about loaded data"""
        metadata = {
            "file_path": file_path,
            "load_timestamp": datetime.now().isoformat(),
            "record_count": len(data),
            "channel_count": data['channel_id'].nunique(),
            "quality_score": quality_report.quality_score,
            "data_hash": self._calculate_data_hash(data),
            "columns": list(data.columns),
            "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()}
        }
        
        cache_key = self._get_cache_key(file_path)
        self._metadata_cache[cache_key] = metadata
        
        # Also save to disk
        metadata_path = self.data_path / "cache" / f"{cache_key}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash for data versioning"""
        data_string = data.to_string()
        return hashlib.sha256(data_string.encode()).hexdigest()[:16]

class ChannelProfileManager:
    """
    Manages channel profiles with caching and automatic updates
    """
    
    def __init__(self, data_store: Optional[DataStore] = None):
        self.config = get_config()
        self.data_store = data_store or DataStore()
        self.profiles_path = Path(self.config.data.data_directory) / "profiles"
        self.profiles_path.mkdir(parents=True, exist_ok=True)
        
        # Profile cache
        self._profile_cache = {}
        self._load_existing_profiles()
    
    def _load_existing_profiles(self):
        """Load existing profiles from disk"""
        if not self.config.data.cache_profiles:
            return
        
        try:
            for profile_file in self.profiles_path.glob("*.json"):
                channel_id = profile_file.stem
                with open(profile_file, 'r') as f:
                    profile_data = json.load(f)
                
                # Convert back to ChannelProfile object
                profile = self._dict_to_profile(profile_data)
                self._profile_cache[channel_id] = profile
                
            logger.info(f"Loaded {len(self._profile_cache)} existing profiles from cache")
            
        except Exception as e:
            logger.warning(f"Failed to load existing profiles: {e}")
    
    def create_channel_profile(self, 
                              channel_id: str, 
                              force_update: bool = False) -> ChannelProfile:
        """
        Create or update channel profile
        
        Args:
            channel_id: YouTube channel ID
            force_update: Force recreation of profile
            
        Returns:
            ChannelProfile object
        """
        # Check cache first
        if not force_update and channel_id in self._profile_cache:
            cached_profile = self._profile_cache[channel_id]
            
            # Check if profile is recent enough (within 24 hours)
            if self._is_profile_recent(cached_profile):
                logger.info(f"Using cached profile for channel {channel_id}")
                return cached_profile
        
        logger.info(f"Creating new profile for channel {channel_id}")
        
        try:
            # Load channel data
            data = self.data_store.load_training_data()
            
            # Create profiler and generate profile
            profiler = ChannelProfiler(data)
            profile = profiler.create_channel_profile(channel_id)
            
            # Cache and save profile
            self._profile_cache[channel_id] = profile
            self._save_profile(profile)
            
            return profile
            
        except Exception as e:
            logger.error(f"Failed to create profile for channel {channel_id}: {e}")
            raise
    
    def get_profile_by_channel(self, channel_id: str) -> Optional[ChannelProfile]:
        """
        Get existing profile for channel
        
        Args:
            channel_id: YouTube channel ID
            
        Returns:
            ChannelProfile if exists, None otherwise
        """
        return self._profile_cache.get(channel_id)
    
    def update_profile(self, channel_id: str) -> ChannelProfile:
        """
        Force update of channel profile
        
        Args:
            channel_id: YouTube channel ID
            
        Returns:
            Updated ChannelProfile
        """
        return self.create_channel_profile(channel_id, force_update=True)
    
    def get_all_profiles(self) -> Dict[str, ChannelProfile]:
        """Get all cached profiles"""
        return self._profile_cache.copy()
    
    def create_all_profiles(self, force_update: bool = False) -> Dict[str, ChannelProfile]:
        """
        Create profiles for all channels in dataset
        
        Args:
            force_update: Force recreation of all profiles
            
        Returns:
            Dictionary of channel_id -> ChannelProfile
        """
        data = self.data_store.load_training_data()
        channel_ids = data['channel_id'].unique()
        
        logger.info(f"Creating profiles for {len(channel_ids)} channels")
        
        profiles = {}
        for channel_id in channel_ids:
            try:
                profile = self.create_channel_profile(channel_id, force_update)
                profiles[channel_id] = profile
            except Exception as e:
                logger.error(f"Failed to create profile for {channel_id}: {e}")
        
        logger.info(f"Successfully created {len(profiles)} profiles")
        return profiles
    
    def export_profiles(self, output_path: Optional[str] = None) -> str:
        """
        Export all profiles to JSON file
        
        Args:
            output_path: Optional output file path
            
        Returns:
            Path to exported file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(self.profiles_path / f"all_profiles_{timestamp}.json")
        
        export_data = {}
        for channel_id, profile in self._profile_cache.items():
            export_data[channel_id] = profile.to_dict()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(export_data)} profiles to {output_path}")
        return output_path
    
    def _is_profile_recent(self, profile: ChannelProfile, hours: int = 24) -> bool:
        """Check if profile is recent enough"""
        if not hasattr(profile, 'created_at') or profile.created_at is None:
            return False
        
        age = datetime.now() - profile.created_at
        return age < timedelta(hours=hours)
    
    def _save_profile(self, profile: ChannelProfile):
        """Save profile to disk"""
        if not self.config.data.cache_profiles:
            return
        
        profile_path = self.profiles_path / f"{profile.channel_id}.json"
        
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(profile.to_dict(), f, indent=2, ensure_ascii=False)
    
    def _dict_to_profile(self, profile_dict: Dict[str, Any]) -> ChannelProfile:
        """Convert dictionary back to ChannelProfile object"""
        # This is a simplified conversion - in a real implementation,
        # you'd want to properly reconstruct all nested objects
        
        # Convert datetime strings back to datetime objects
        if 'created_at' in profile_dict and isinstance(profile_dict['created_at'], str):
            profile_dict['created_at'] = datetime.fromisoformat(profile_dict['created_at'])
        
        # Reconstruct the profile (simplified - would need full implementation)
        # For now, return a basic reconstruction
        return ChannelProfile(**profile_dict)