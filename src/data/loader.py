"""
Data loader module for TitleCraft AI.
Handles loading, parsing, and initial validation of training data.
"""

# Standard library imports
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, List

# Third-party imports
import pandas as pd

# Local imports
from .models import VideoData, DatasetSummary, DataQualityReport
from .validator import DataValidator

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading and initial processing of YouTube video data.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize DataLoader with path to CSV file.
        
        Args:
            data_path: Path to the CSV file containing video data
        """
        self.data_path = Path(data_path)
        self.validator = DataValidator()
        self._raw_data: Optional[pd.DataFrame] = None
        self._processed_data: Optional[pd.DataFrame] = None
        
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw data from CSV file.
        
        Returns:
            Raw DataFrame from CSV
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            pd.errors.EmptyDataError: If CSV is empty
        """
        try:
            logger.info(f"Loading data from {self.data_path}")
            
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            # Load with proper data types
            dtype_dict = {
                'channel_id': 'str',
                'video_id': 'str', 
                'title': 'str',
                'summary': 'str',
                'views_in_period': 'int64'
            }
            
            self._raw_data = pd.read_csv(
                self.data_path,
                dtype=dtype_dict,
                encoding='utf-8'
            )
            
            logger.info(f"Loaded {len(self._raw_data)} records from {self.data_path.name}")
            return self._raw_data.copy()
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def validate_and_clean(self, data: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityReport]:
        """
        Validate and clean the loaded data.
        
        Args:
            data: Raw DataFrame to validate
            
        Returns:
            Tuple of (cleaned_data, quality_report)
        """
        logger.info("Validating and cleaning data...")
        
        # Run validation
        quality_report = self.validator.validate_dataset(data)
        
        # Clean data based on validation results
        cleaned_data = data.copy()
        
        # Remove duplicates
        initial_count = len(cleaned_data)
        cleaned_data = cleaned_data.drop_duplicates()
        duplicates_removed = initial_count - len(cleaned_data)
        
        # Handle missing values
        cleaned_data = cleaned_data.dropna(subset=['channel_id', 'video_id', 'title'])
        
        # Clean text fields
        cleaned_data['title'] = cleaned_data['title'].str.strip()
        cleaned_data['summary'] = cleaned_data['summary'].fillna('').str.strip()
        
        # Ensure positive views
        cleaned_data = cleaned_data[cleaned_data['views_in_period'] >= 0]
        
        # Remove empty titles
        cleaned_data = cleaned_data[cleaned_data['title'].str.len() > 0]
        
        final_count = len(cleaned_data)
        logger.info(f"Cleaned data: {initial_count} -> {final_count} records "
                   f"({initial_count - final_count} removed)")
        
        return cleaned_data, quality_report
    
    def load_and_validate(self) -> tuple[pd.DataFrame, DataQualityReport]:
        """
        Complete data loading pipeline with validation.
        
        Returns:
            Tuple of (processed_dataframe, quality_report)
        """
        # Load raw data
        raw_data = self.load_raw_data()
        
        # Validate and clean
        processed_data, quality_report = self.validate_and_clean(raw_data)
        
        self._processed_data = processed_data
        
        return processed_data, quality_report
    
    def get_dataset_summary(self) -> DatasetSummary:
        """
        Generate comprehensive dataset summary.
        
        Returns:
            DatasetSummary object with overview statistics
        """
        if self._processed_data is None:
            raise ValueError("No processed data available. Call load_and_validate() first.")
        
        data = self._processed_data
        
        # Channel distribution
        channel_dist = data['channel_id'].value_counts().to_dict()
        
        # View distribution (percentiles)
        view_percentiles = {
            'min': float(data['views_in_period'].min()),
            'p25': float(data['views_in_period'].quantile(0.25)),
            'p50': float(data['views_in_period'].quantile(0.50)),
            'p75': float(data['views_in_period'].quantile(0.75)),
            'p90': float(data['views_in_period'].quantile(0.90)),
            'p95': float(data['views_in_period'].quantile(0.95)),
            'max': float(data['views_in_period'].max()),
            'mean': float(data['views_in_period'].mean()),
            'std': float(data['views_in_period'].std())
        }
        
        # Generate insights
        insights = self._generate_dataset_insights(data)
        
        # Get quality report
        quality_report = self.validator.validate_dataset(data)
        
        return DatasetSummary(
            total_videos=len(data),
            total_channels=data['channel_id'].nunique(),
            date_range=None,  # No date info in current dataset
            channel_distribution=channel_dist,
            view_distribution=view_percentiles,
            quality_report=quality_report,
            insights=insights
        )
    
    def _generate_dataset_insights(self, data: pd.DataFrame) -> List[str]:
        """Generate insights about the dataset."""
        insights = []
        
        # Channel size analysis
        channel_sizes = data['channel_id'].value_counts()
        largest_channel = channel_sizes.index[0]
        smallest_channel = channel_sizes.index[-1]
        
        insights.append(
            f"Dataset contains {len(data)} videos across {data['channel_id'].nunique()} channels"
        )
        insights.append(
            f"Largest channel has {channel_sizes.iloc[0]} videos, "
            f"smallest has {channel_sizes.iloc[-1]} videos"
        )
        
        # Performance analysis  
        high_performers = data[data['views_in_period'] > data['views_in_period'].quantile(0.9)]
        channel_performance = high_performers['channel_id'].value_counts()
        
        if len(channel_performance) > 0:
            top_performing_channel = channel_performance.index[0]
            insights.append(
                f"Channel {top_performing_channel} has the most high-performing videos "
                f"({channel_performance.iloc[0]} in top 10%)"
            )
        
        # Title length analysis
        title_lengths = data['title'].str.len()
        avg_length = title_lengths.mean()
        insights.append(f"Average title length: {avg_length:.1f} characters")
        
        # View range analysis
        view_range = data['views_in_period'].max() - data['views_in_period'].min()
        insights.append(
            f"View performance varies widely: {data['views_in_period'].min():,} to "
            f"{data['views_in_period'].max():,} views (range: {view_range:,})"
        )
        
        return insights
    
    def get_data_hash(self) -> str:
        """
        Generate hash of processed data for versioning.
        
        Returns:
            SHA256 hash of the data
        """
        if self._processed_data is None:
            raise ValueError("No processed data available")
        
        # Create hash from data content
        data_string = self._processed_data.to_string()
        return hashlib.sha256(data_string.encode()).hexdigest()[:16]
    
    def export_processed_data(self, output_path: str):
        """
        Export processed data to CSV.
        
        Args:
            output_path: Path for output CSV file
        """
        if self._processed_data is None:
            raise ValueError("No processed data to export")
        
        self._processed_data.to_csv(output_path, index=False)
        logger.info(f"Exported processed data to {output_path}")


# Convenience function for quick data loading
def load_youtube_data(csv_path: str) -> tuple[pd.DataFrame, DataQualityReport]:
    """
    Convenience function to load and validate YouTube data.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Tuple of (dataframe, quality_report)
    """
    loader = DataLoader(csv_path)
    return loader.load_and_validate()