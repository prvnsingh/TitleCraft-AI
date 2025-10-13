"""
Data validator module for TitleCraft AI.
Handles data quality assessment and validation.
"""

# Standard library imports
import logging
import re
from typing import Dict, List, Any

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from .models import DataQualityReport

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates data quality and identifies issues in YouTube video datasets.
    """
    
    def __init__(self):
        """Initialize validator with validation rules."""
        self.required_columns = ['channel_id', 'video_id', 'title', 'summary', 'views_in_period']
        self.expected_dtypes = {
            'channel_id': 'object',
            'video_id': 'object', 
            'title': 'object',
            'summary': 'object',
            'views_in_period': ['int64', 'int32', 'float64']  # Allow multiple types
        }
        
    def validate_dataset(self, data: pd.DataFrame) -> DataQualityReport:
        """
        Perform comprehensive data quality assessment.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            DataQualityReport with detailed assessment
        """
        logger.info("Starting data quality validation...")
        
        issues = []
        recommendations = []
        
        # Basic structure validation
        structure_valid = self._validate_structure(data, issues, recommendations)
        
        # Missing values analysis
        missing_values = self._analyze_missing_values(data, issues, recommendations)
        
        # Duplicate detection
        duplicate_count = self._detect_duplicates(data, issues, recommendations)
        
        # Data type validation
        types_valid = self._validate_data_types(data, issues, recommendations)
        
        # Outlier detection
        outliers = self._detect_outliers(data, issues, recommendations)
        
        # Content validation
        self._validate_content_quality(data, issues, recommendations)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(data, issues, len(missing_values), duplicate_count)
        
        # Count valid records (no critical issues)
        valid_count = self._count_valid_records(data)
        
        return DataQualityReport(
            total_records=len(data),
            valid_records=valid_count,
            invalid_records=len(data) - valid_count,
            missing_values=missing_values,
            duplicate_records=duplicate_count,
            data_types_valid=types_valid,
            outliers=outliers,
            quality_score=quality_score,
            issues=issues,
            recommendations=recommendations
        )
    
    def _validate_structure(self, data: pd.DataFrame, issues: List[str], recommendations: List[str]) -> bool:
        """Validate basic DataFrame structure."""
        is_valid = True
        
        # Check if DataFrame is empty
        if data.empty:
            issues.append("Dataset is empty")
            recommendations.append("Provide a dataset with video data")
            return False
        
        # Check required columns
        missing_cols = set(self.required_columns) - set(data.columns)
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
            recommendations.append(f"Add missing columns: {missing_cols}")
            is_valid = False
        
        # Check for extra columns
        extra_cols = set(data.columns) - set(self.required_columns)
        if extra_cols:
            issues.append(f"Unexpected columns found: {extra_cols}")
            recommendations.append("Review if extra columns are needed")
        
        return is_valid
    
    def _analyze_missing_values(self, data: pd.DataFrame, issues: List[str], recommendations: List[str]) -> Dict[str, int]:
        """Analyze missing values in each column."""
        missing_values = {}
        
        for col in data.columns:
            missing_count = data[col].isna().sum()
            missing_values[col] = missing_count
            
            if missing_count > 0:
                missing_pct = (missing_count / len(data)) * 100
                
                if col in ['channel_id', 'video_id', 'title']:  # Critical columns
                    if missing_count > 0:
                        issues.append(f"Critical column '{col}' has {missing_count} missing values ({missing_pct:.1f}%)")
                        recommendations.append(f"Remove or fix records with missing {col}")
                else:  # Non-critical columns
                    if missing_pct > 10:
                        issues.append(f"Column '{col}' has high missing rate: {missing_pct:.1f}%")
                        recommendations.append(f"Consider imputation strategy for {col}")
        
        return missing_values
    
    def _detect_duplicates(self, data: pd.DataFrame, issues: List[str], recommendations: List[str]) -> int:
        """Detect duplicate records."""
        # Full row duplicates
        duplicate_rows = data.duplicated().sum()
        
        # Video ID duplicates (should be unique)
        if 'video_id' in data.columns:
            duplicate_videos = data['video_id'].duplicated().sum()
            if duplicate_videos > 0:
                issues.append(f"Found {duplicate_videos} duplicate video IDs")
                recommendations.append("Remove duplicate video entries")
        
        if duplicate_rows > 0:
            issues.append(f"Found {duplicate_rows} completely duplicate rows")
            recommendations.append("Remove duplicate rows")
        
        return duplicate_rows + duplicate_videos if 'video_id' in data.columns else duplicate_rows
    
    def _validate_data_types(self, data: pd.DataFrame, issues: List[str], recommendations: List[str]) -> bool:
        """Validate data types match expectations."""
        types_valid = True
        
        for col, expected_types in self.expected_dtypes.items():
            if col not in data.columns:
                continue
                
            actual_type = str(data[col].dtype)
            
            # Handle multiple acceptable types
            if isinstance(expected_types, list):
                if actual_type not in expected_types:
                    issues.append(f"Column '{col}' has type {actual_type}, expected one of {expected_types}")
                    recommendations.append(f"Convert {col} to appropriate numeric type")
                    types_valid = False
            else:
                if actual_type != expected_types:
                    issues.append(f"Column '{col}' has type {actual_type}, expected {expected_types}")
                    recommendations.append(f"Convert {col} to {expected_types}")
                    types_valid = False
        
        return types_valid
    
    def _detect_outliers(self, data: pd.DataFrame, issues: List[str], recommendations: List[str]) -> Dict[str, List[Any]]:
        """Detect outliers in numeric columns."""
        outliers = {}
        
        if 'views_in_period' in data.columns:
            views = data['views_in_period']
            
            # Negative views (invalid)
            negative_views = data[views < 0]
            if len(negative_views) > 0:
                outliers['negative_views'] = negative_views.index.tolist()
                issues.append(f"Found {len(negative_views)} records with negative views")
                recommendations.append("Remove or correct negative view counts")
            
            # Extremely high views (potential data entry errors)
            q99 = views.quantile(0.99)
            extreme_views = data[views > q99 * 10]  # 10x the 99th percentile
            if len(extreme_views) > 0:
                outliers['extreme_views'] = extreme_views.index.tolist()
                issues.append(f"Found {len(extreme_views)} records with extremely high views (>10x 99th percentile)")
                recommendations.append("Verify extreme view counts for accuracy")
        
        return outliers
    
    def _validate_content_quality(self, data: pd.DataFrame, issues: List[str], recommendations: List[str]):
        """Validate content quality of text fields."""
        
        if 'title' in data.columns:
            # Empty titles after stripping
            empty_titles = data[data['title'].str.strip() == '']
            if len(empty_titles) > 0:
                issues.append(f"Found {len(empty_titles)} records with empty titles")
                recommendations.append("Remove records with empty titles")
            
            # Very short titles (likely incomplete)
            short_titles = data[data['title'].str.len() < 10]
            if len(short_titles) > 0:
                issues.append(f"Found {len(short_titles)} records with very short titles (<10 chars)")
                recommendations.append("Review short titles for completeness")
            
            # Very long titles (potential data issues)  
            long_titles = data[data['title'].str.len() > 200]
            if len(long_titles) > 0:
                issues.append(f"Found {len(long_titles)} records with very long titles (>200 chars)")
                recommendations.append("Review long titles for data quality issues")
        
        if 'channel_id' in data.columns:
            # Invalid channel ID format (should start with UC)
            invalid_channels = data[~data['channel_id'].str.startswith('UC')]
            if len(invalid_channels) > 0:
                issues.append(f"Found {len(invalid_channels)} records with invalid channel ID format")
                recommendations.append("Verify channel ID format (should start with 'UC')")
    
    def _calculate_quality_score(self, data: pd.DataFrame, issues: List[str], 
                                missing_count: int, duplicate_count: int) -> float:
        """Calculate overall data quality score (0-1)."""
        total_records = len(data)
        if total_records == 0:
            return 0.0
        
        # Base score
        score = 1.0
        
        # Deduct for issues
        critical_issues = sum(1 for issue in issues if any(word in issue.lower() 
                             for word in ['critical', 'missing', 'negative', 'empty']))
        
        # Penalties
        score -= (critical_issues * 0.2)  # 20% per critical issue type
        score -= (missing_count / total_records) * 0.3  # Up to 30% for missing values
        score -= (duplicate_count / total_records) * 0.2  # Up to 20% for duplicates
        
        return max(0.0, min(1.0, score))
    
    def _count_valid_records(self, data: pd.DataFrame) -> int:
        """Count records without critical data quality issues."""
        if data.empty:
            return 0
        
        valid_data = data.copy()
        
        # Remove records with missing critical fields
        valid_data = valid_data.dropna(subset=['channel_id', 'video_id', 'title'])
        
        # Remove records with negative views
        if 'views_in_period' in valid_data.columns:
            valid_data = valid_data[valid_data['views_in_period'] >= 0]
        
        # Remove empty titles
        valid_data = valid_data[valid_data['title'].str.strip() != '']
        
        return len(valid_data)
    
    def get_validation_summary(self, report: DataQualityReport) -> str:
        """
        Generate human-readable validation summary.
        
        Args:
            report: DataQualityReport to summarize
            
        Returns:
            Formatted summary string
        """
        summary = f"""
Data Quality Assessment Summary
================================

Records: {report.total_records:,} total, {report.valid_records:,} valid, {report.invalid_records:,} invalid
Quality Score: {report.quality_score:.2%}
Duplicates: {report.duplicate_records:,}

Missing Values:
{chr(10).join([f"  {col}: {count:,}" for col, count in report.missing_values.items() if count > 0])}

Issues Found ({len(report.issues)}):
{chr(10).join([f"  • {issue}" for issue in report.issues[:10]])}
{f"  ... and {len(report.issues) - 10} more" if len(report.issues) > 10 else ""}

Recommendations ({len(report.recommendations)}):
{chr(10).join([f"  • {rec}" for rec in report.recommendations[:5]])}
{f"  ... and {len(report.recommendations) - 5} more" if len(report.recommendations) > 5 else ""}
        """
        return summary.strip()