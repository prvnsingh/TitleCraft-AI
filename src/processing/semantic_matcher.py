"""
Semantic Similarity Engine for TitleCraft AI

This module provides semantic matching capabilities to find contextually relevant
title examples using sentence transformers and vector similarity.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import pickle
import json
from pathlib import Path
import time

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..data.models import VideoData
from ..utils import BaseComponent


@dataclass
class SimilarityMatch:
    """Represents a semantic similarity match"""
    video: VideoData
    similarity_score: float
    match_type: str  # "semantic", "keyword", "hybrid"
    explanation: str
    relevance_factors: Dict[str, float]


@dataclass 
class EmbeddingMetadata:
    """Metadata for embeddings index"""
    model_name: str
    dimension: int
    total_vectors: int
    created_at: str
    last_updated: str
    version: str


class SemanticMatcher(BaseComponent):
    """
    Semantic similarity engine for finding contextually relevant titles.
    
    Features:
    - Sentence transformer embeddings for semantic understanding
    - FAISS vector index for fast similarity search
    - Hybrid matching (semantic + keyword)
    - Caching and persistence
    - Relevance scoring with explanations
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # Initialize models
        self.sentence_transformer = None
        self.tfidf_vectorizer = None
        
        # Vector index
        self.faiss_index = None
        self.video_lookup = {}  # Map index position to VideoData
        
        # Embeddings cache
        self.embeddings_cache = {}
        self.cache_dir = Path(self.config.data.data_directory) / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._initialize_models()
        
        # Stats
        self.stats = {
            'matches_computed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_match_time': 0.0
        }
    
    def _initialize_models(self):
        """Initialize sentence transformer and fallback models"""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                model_name = self.config.data.embeddings_model
                logger.info(f"Loading sentence transformer: {model_name}")
                self.sentence_transformer = SentenceTransformer(model_name)
                logger.info("Sentence transformer loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load sentence transformer: {e}")
                self.sentence_transformer = None
        else:
            logger.warning("sentence-transformers not available, using TF-IDF fallback")
        
        # Initialize TF-IDF as fallback
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
        
        if not (self.sentence_transformer or self.tfidf_vectorizer):
            logger.error("No similarity models available - install sentence-transformers or scikit-learn")
    
    def build_embeddings_index(self, videos: List[VideoData]) -> None:
        """Build FAISS index from video titles"""
        if not videos:
            logger.warning("No videos provided for index building")
            return
        
        self.logger.info(f"Building embeddings index from {len(videos)} videos")
        start_time = time.time()
        
        # Extract titles
        titles = [video.title for video in videos]
        
        if self.sentence_transformer:
            embeddings = self._compute_sentence_embeddings(titles)
            self._build_faiss_index(embeddings, videos)
        elif self.tfidf_vectorizer:
            embeddings = self._compute_tfidf_embeddings(titles)
            self._build_sklearn_index(embeddings, videos)
        else:
            raise RuntimeError("No embedding models available")
        
        build_time = time.time() - start_time
        self.logger.info(f"Index built in {build_time:.2f}s with {len(videos)} videos")
        
        # Save index
        self._save_index()
    
    def _compute_sentence_embeddings(self, titles: List[str]) -> np.ndarray:
        """Compute sentence transformer embeddings"""
        try:
            embeddings = self.sentence_transformer.encode(
                titles,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            logger.info(f"Computed embeddings shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to compute sentence embeddings: {e}")
            raise
    
    def _compute_tfidf_embeddings(self, titles: List[str]) -> np.ndarray:
        """Compute TF-IDF embeddings as fallback"""
        try:
            embeddings = self.tfidf_vectorizer.fit_transform(titles).toarray()
            logger.info(f"Computed TF-IDF embeddings shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to compute TF-IDF embeddings: {e}")
            raise
    
    def _build_faiss_index(self, embeddings: np.ndarray, videos: List[VideoData]) -> None:
        """Build FAISS index for fast similarity search"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("FAISS requires sentence-transformers")
            return
        
        try:
            import faiss
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Create index
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
            self.faiss_index.add(embeddings.astype(np.float32))
            
            # Build lookup table
            self.video_lookup = {i: video for i, video in enumerate(videos)}
            
            logger.info(f"FAISS index built with {self.faiss_index.ntotal} vectors, dimension {dimension}")
            
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            raise
    
    def _build_sklearn_index(self, embeddings: np.ndarray, videos: List[VideoData]) -> None:
        """Build sklearn-based index as fallback"""
        self.sklearn_embeddings = embeddings
        self.video_lookup = {i: video for i, video in enumerate(videos)}
        logger.info(f"Sklearn index built with {len(videos)} vectors")
    
    def find_similar_titles(self, 
                           idea: str, 
                           channel_id: Optional[str] = None,
                           n_matches: int = 5,
                           min_similarity: float = 0.1) -> List[SimilarityMatch]:
        """
        Find semantically similar titles to the given idea.
        
        Args:
            idea: Video idea/concept to match
            channel_id: Optional channel filter
            n_matches: Number of matches to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of SimilarityMatch objects
        """
        start_time = time.time()
        
        if not self._is_index_ready():
            logger.warning("No index available for similarity search")
            return []
        
        try:
            # Get embeddings for the idea
            if self.sentence_transformer:
                idea_embedding = self._compute_sentence_embeddings([idea])[0]
                matches = self._search_faiss_index(idea_embedding, n_matches, min_similarity)
            elif self.sklearn_embeddings is not None:
                idea_embedding = self._compute_tfidf_embeddings([idea])[0]
                matches = self._search_sklearn_index(idea_embedding, n_matches, min_similarity)
            else:
                return []
            
            # Filter by channel if specified
            if channel_id:
                matches = [m for m in matches if m.video.channel_id == channel_id]
            
            # Update stats
            match_time = time.time() - start_time
            self.stats['matches_computed'] += 1
            self.stats['avg_match_time'] = (
                (self.stats['avg_match_time'] * (self.stats['matches_computed'] - 1) + match_time) /
                self.stats['matches_computed']
            )
            
            logger.debug(f"Found {len(matches)} similar titles in {match_time:.3f}s")
            return matches
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def _search_faiss_index(self, 
                           query_embedding: np.ndarray, 
                           k: int,
                           min_similarity: float) -> List[SimilarityMatch]:
        """Search FAISS index for similar vectors"""
        try:
            # Normalize query embedding
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            import faiss
            faiss.normalize_L2(query_embedding)
            
            # Search index
            similarities, indices = self.faiss_index.search(query_embedding, k * 2)  # Get extra for filtering
            
            matches = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if similarity >= min_similarity and idx in self.video_lookup:
                    video = self.video_lookup[idx]
                    
                    match = SimilarityMatch(
                        video=video,
                        similarity_score=float(similarity),
                        match_type="semantic",
                        explanation=f"Semantic similarity score: {similarity:.3f}",
                        relevance_factors={
                            "semantic_similarity": float(similarity),
                            "embedding_model": self.config.data.embeddings_model
                        }
                    )
                    matches.append(match)
            
            return matches[:k]
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []
    
    def _search_sklearn_index(self,
                             query_embedding: np.ndarray,
                             k: int, 
                             min_similarity: float) -> List[SimilarityMatch]:
        """Search sklearn embeddings for similar vectors"""
        try:
            # Compute cosine similarities
            similarities = cosine_similarity([query_embedding], self.sklearn_embeddings)[0]
            
            # Get top k indices
            top_indices = np.argsort(similarities)[::-1][:k * 2]
            
            matches = []
            for idx in top_indices:
                similarity = similarities[idx]
                if similarity >= min_similarity and idx in self.video_lookup:
                    video = self.video_lookup[idx]
                    
                    match = SimilarityMatch(
                        video=video,
                        similarity_score=float(similarity),
                        match_type="tfidf",
                        explanation=f"TF-IDF cosine similarity: {similarity:.3f}",
                        relevance_factors={
                            "tfidf_similarity": float(similarity),
                            "method": "sklearn_cosine"
                        }
                    )
                    matches.append(match)
            
            return matches[:k]
            
        except Exception as e:
            logger.error(f"Sklearn search failed: {e}")
            return []
    
    def _is_index_ready(self) -> bool:
        """Check if similarity index is ready"""
        return (
            (self.faiss_index is not None and self.faiss_index.ntotal > 0) or
            (hasattr(self, 'sklearn_embeddings') and self.sklearn_embeddings is not None)
        ) and len(self.video_lookup) > 0
    
    def get_hybrid_matches(self,
                          idea: str,
                          keywords: List[str],
                          channel_id: Optional[str] = None,
                          n_matches: int = 5) -> List[SimilarityMatch]:
        """
        Get hybrid matches combining semantic similarity and keyword matching.
        
        Args:
            idea: Video idea for semantic matching
            keywords: Keywords for keyword matching
            channel_id: Optional channel filter
            n_matches: Number of matches to return
            
        Returns:
            Combined and ranked similarity matches
        """
        # Get semantic matches
        semantic_matches = self.find_similar_titles(idea, channel_id, n_matches * 2)
        
        # Get keyword matches
        keyword_matches = self._find_keyword_matches(keywords, channel_id, n_matches)
        
        # Combine and deduplicate
        all_matches = {}
        
        # Add semantic matches
        for match in semantic_matches:
            key = match.video.video_id
            if key not in all_matches:
                all_matches[key] = match
            else:
                # Combine scores
                existing = all_matches[key]
                combined_score = (existing.similarity_score + match.similarity_score) / 2
                existing.similarity_score = combined_score
                existing.match_type = "hybrid"
                existing.relevance_factors.update(match.relevance_factors)
        
        # Add keyword matches
        for match in keyword_matches:
            key = match.video.video_id
            if key not in all_matches:
                all_matches[key] = match
            else:
                # Boost existing matches that also have keyword relevance
                existing = all_matches[key]
                existing.similarity_score = min(1.0, existing.similarity_score * 1.2)
                existing.match_type = "hybrid"
                existing.relevance_factors.update(match.relevance_factors)
        
        # Sort by combined score and return top matches
        matches = list(all_matches.values())
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return matches[:n_matches]
    
    def _find_keyword_matches(self,
                             keywords: List[str],
                             channel_id: Optional[str] = None,
                             n_matches: int = 5) -> List[SimilarityMatch]:
        """Find matches based on keyword overlap"""
        matches = []
        
        for idx, video in self.video_lookup.items():
            if channel_id and video.channel_id != channel_id:
                continue
            
            title_lower = video.title.lower()
            keyword_matches = sum(1 for kw in keywords if kw.lower() in title_lower)
            
            if keyword_matches > 0:
                score = keyword_matches / len(keywords)
                
                match = SimilarityMatch(
                    video=video,
                    similarity_score=score,
                    match_type="keyword",
                    explanation=f"Keyword matches: {keyword_matches}/{len(keywords)}",
                    relevance_factors={
                        "keyword_overlap": score,
                        "matched_keywords": keyword_matches,
                        "total_keywords": len(keywords)
                    }
                )
                matches.append(match)
        
        # Sort and return top matches
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        return matches[:n_matches]
    
    def _save_index(self):
        """Save embeddings index and metadata to disk"""
        try:
            # Save FAISS index if available
            if self.faiss_index:
                index_path = self.cache_dir / "faiss_index.bin"
                import faiss
                faiss.write_index(self.faiss_index, str(index_path))
                logger.info(f"FAISS index saved to {index_path}")
            
            # Save sklearn embeddings if available
            if hasattr(self, 'sklearn_embeddings') and self.sklearn_embeddings is not None:
                embeddings_path = self.cache_dir / "sklearn_embeddings.pkl"
                with open(embeddings_path, 'wb') as f:
                    pickle.dump(self.sklearn_embeddings, f)
                logger.info(f"Sklearn embeddings saved to {embeddings_path}")
            
            # Save video lookup
            lookup_path = self.cache_dir / "video_lookup.json"
            serializable_lookup = {
                str(k): {
                    'channel_id': v.channel_id,
                    'video_id': v.video_id,
                    'title': v.title,
                    'summary': v.summary,
                    'views_in_period': v.views_in_period
                }
                for k, v in self.video_lookup.items()
            }
            
            with open(lookup_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_lookup, f, indent=2)
            
            # Save metadata
            metadata = EmbeddingMetadata(
                model_name=self.config.data.embeddings_model,
                dimension=self.faiss_index.d if self.faiss_index else self.sklearn_embeddings.shape[1],
                total_vectors=len(self.video_lookup),
                created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                last_updated=time.strftime("%Y-%m-%d %H:%M:%S"),
                version="1.0"
            )
            
            metadata_path = self.cache_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata.__dict__, f, indent=2)
            
            logger.info("Embeddings index and metadata saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save embeddings index: {e}")
    
    def load_index(self) -> bool:
        """Load embeddings index and metadata from disk"""
        try:
            # Check if index files exist
            faiss_path = self.cache_dir / "faiss_index.bin"
            sklearn_path = self.cache_dir / "sklearn_embeddings.pkl"
            lookup_path = self.cache_dir / "video_lookup.json"
            
            if not lookup_path.exists():
                logger.info("No saved embeddings index found")
                return False
            
            # Load video lookup
            with open(lookup_path, 'r', encoding='utf-8') as f:
                lookup_data = json.load(f)
            
            self.video_lookup = {}
            for k, v in lookup_data.items():
                video = VideoData(
                    channel_id=v['channel_id'],
                    video_id=v['video_id'],
                    title=v['title'],
                    summary=v['summary'],
                    views_in_period=v['views_in_period']
                )
                self.video_lookup[int(k)] = video
            
            # Load FAISS index if available
            if faiss_path.exists() and self.sentence_transformer:
                import faiss
                self.faiss_index = faiss.read_index(str(faiss_path))
                logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
                return True
            
            # Load sklearn embeddings if available
            if sklearn_path.exists():
                with open(sklearn_path, 'rb') as f:
                    self.sklearn_embeddings = pickle.load(f)
                logger.info(f"Loaded sklearn embeddings with shape {self.sklearn_embeddings.shape}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to load embeddings index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get similarity engine statistics"""
        index_stats = {}
        
        if self.faiss_index:
            index_stats['faiss_vectors'] = self.faiss_index.ntotal
            index_stats['faiss_dimension'] = self.faiss_index.d
        
        if hasattr(self, 'sklearn_embeddings') and self.sklearn_embeddings is not None:
            index_stats['sklearn_vectors'] = self.sklearn_embeddings.shape[0]
            index_stats['sklearn_dimension'] = self.sklearn_embeddings.shape[1]
        
        return {
            **self.stats,
            **index_stats,
            'total_videos': len(self.video_lookup),
            'models_available': {
                'sentence_transformer': self.sentence_transformer is not None,
                'tfidf': self.tfidf_vectorizer is not None
            }
        }