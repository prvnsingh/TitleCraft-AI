"""
Production Caching System for TitleCraft AI
Implements Redis-based caching with async operations for optimal performance.
"""
import asyncio
import json
import hashlib
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
import redis.asyncio as redis
import pickle
import logging
from dataclasses import asdict

from ..config import get_config

logger = logging.getLogger(__name__)

class CacheManager:
    """
    High-performance async caching manager with Redis backend.
    Provides intelligent caching for channel profiles, embeddings, and API responses.
    """
    
    def __init__(self, redis_url: str = None):
        self.config = get_config()
        self.redis_url = redis_url or self.config.get('redis_url', 'redis://localhost:6379')
        self.redis_client: Optional[redis.Redis] = None
        self.connection_pool = None
        
        # Cache TTL configurations (in seconds)
        self.ttl_config = {
            'channel_profile': 3600 * 24,      # 24 hours
            'embeddings': 3600 * 24 * 7,      # 7 days  
            'api_response': 3600,              # 1 hour
            'pattern_analysis': 3600 * 12,    # 12 hours
            'quality_scores': 3600 * 6,       # 6 hours
            'similar_titles': 3600 * 2,       # 2 hours
        }
        
        # Cache key prefixes
        self.prefixes = {
            'channel_profile': 'cp:',
            'embeddings': 'emb:',
            'api_response': 'api:',
            'pattern_analysis': 'pa:',
            'quality_scores': 'qs:',
            'similar_titles': 'st:',
        }

    async def initialize(self):
        """Initialize Redis connection pool."""
        try:
            self.connection_pool = redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=20,
                retry_on_timeout=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                decode_responses=False  # We'll handle encoding manually
            )
            
            self.redis_client = redis.Redis(connection_pool=self.connection_pool)
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            # Fallback to in-memory cache
            self.redis_client = None
            self._fallback_cache = {}

    async def close(self):
        """Close Redis connections."""
        if self.redis_client:
            await self.redis_client.close()
        if self.connection_pool:
            await self.connection_pool.disconnect()

    def _generate_cache_key(self, cache_type: str, identifier: str, **kwargs) -> str:
        """Generate cache key with optional parameters."""
        prefix = self.prefixes.get(cache_type, 'unknown:')
        
        if kwargs:
            # Include kwargs in key for parameter-specific caching
            params_str = json.dumps(sorted(kwargs.items()))
            key_data = f"{identifier}:{params_str}"
        else:
            key_data = identifier
            
        # Hash long keys to avoid Redis key length limits
        if len(key_data) > 200:
            key_data = hashlib.md5(key_data.encode()).hexdigest()
            
        return f"{prefix}{key_data}"

    async def get(self, cache_type: str, identifier: str, **kwargs) -> Optional[Any]:
        """
        Get cached data with fallback to in-memory cache.
        
        Args:
            cache_type: Type of cached data (channel_profile, embeddings, etc.)
            identifier: Unique identifier for the cached item
            **kwargs: Additional parameters for cache key generation
            
        Returns:
            Cached data or None if not found
        """
        cache_key = self._generate_cache_key(cache_type, identifier, **kwargs)
        
        try:
            if self.redis_client:
                # Try Redis first
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    try:
                        # Try JSON first, fallback to pickle
                        return json.loads(cached_data)
                    except json.JSONDecodeError:
                        return pickle.loads(cached_data)
            else:
                # Fallback to in-memory cache
                return self._fallback_cache.get(cache_key)
                
        except Exception as e:
            logger.warning(f"Cache get failed for {cache_key}: {e}")
            
        return None

    async def set(self, cache_type: str, identifier: str, data: Any, 
                  ttl: Optional[int] = None, **kwargs) -> bool:
        """
        Cache data with automatic serialization and TTL.
        
        Args:
            cache_type: Type of cached data
            identifier: Unique identifier
            data: Data to cache
            ttl: Time to live in seconds (uses default if None)
            **kwargs: Additional parameters for cache key generation
            
        Returns:
            True if successfully cached, False otherwise
        """
        cache_key = self._generate_cache_key(cache_type, identifier, **kwargs)
        ttl = ttl or self.ttl_config.get(cache_type, 3600)
        
        try:
            # Serialize data
            if isinstance(data, (dict, list, str, int, float, bool)) or data is None:
                # JSON serializable
                serialized_data = json.dumps(data, default=str)
            else:
                # Use pickle for complex objects
                serialized_data = pickle.dumps(data)
                
            if self.redis_client:
                # Store in Redis
                await self.redis_client.setex(cache_key, ttl, serialized_data)
                logger.debug(f"Cached data: {cache_key} (TTL: {ttl}s)")
                return True
            else:
                # Fallback to in-memory cache with TTL tracking
                expiry_time = datetime.now() + timedelta(seconds=ttl)
                self._fallback_cache[cache_key] = {
                    'data': data,
                    'expires': expiry_time
                }
                return True
                
        except Exception as e:
            logger.warning(f"Cache set failed for {cache_key}: {e}")
            return False

    async def delete(self, cache_type: str, identifier: str, **kwargs) -> bool:
        """Delete cached data."""
        cache_key = self._generate_cache_key(cache_type, identifier, **kwargs)
        
        try:
            if self.redis_client:
                result = await self.redis_client.delete(cache_key)
                return result > 0
            else:
                return self._fallback_cache.pop(cache_key, None) is not None
        except Exception as e:
            logger.warning(f"Cache delete failed for {cache_key}: {e}")
            return False

    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching a pattern."""
        try:
            if self.redis_client:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    return await self.redis_client.delete(*keys)
                return 0
            else:
                # For in-memory cache, match keys manually
                matching_keys = [k for k in self._fallback_cache.keys() 
                               if pattern.replace('*', '') in k]
                for key in matching_keys:
                    self._fallback_cache.pop(key, None)
                return len(matching_keys)
        except Exception as e:
            logger.warning(f"Cache pattern clear failed for {pattern}: {e}")
            return 0

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            if self.redis_client:
                info = await self.redis_client.info()
                return {
                    'cache_type': 'redis',
                    'connected_clients': info.get('connected_clients', 0),
                    'used_memory': info.get('used_memory_human', 'unknown'),
                    'keyspace_hits': info.get('keyspace_hits', 0),
                    'keyspace_misses': info.get('keyspace_misses', 0),
                    'hit_rate': self._calculate_hit_rate(info),
                }
            else:
                # Clean expired entries from fallback cache
                now = datetime.now()
                expired_keys = [k for k, v in self._fallback_cache.items() 
                              if isinstance(v, dict) and v.get('expires', now) < now]
                for key in expired_keys:
                    self._fallback_cache.pop(key, None)
                    
                return {
                    'cache_type': 'in_memory',
                    'total_keys': len(self._fallback_cache),
                    'memory_usage': 'limited',
                }
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")
            return {'cache_type': 'error', 'error': str(e)}

    def _calculate_hit_rate(self, info: dict) -> float:
        """Calculate cache hit rate percentage."""
        hits = info.get('keyspace_hits', 0)
        misses = info.get('keyspace_misses', 0)
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0

# Cached decorators for common operations
def cached_result(cache_type: str, ttl: Optional[int] = None):
    """
    Decorator for caching function results.
    
    Usage:
        @cached_result('channel_profile', ttl=3600)
        async def get_channel_profile(channel_id: str):
            # expensive operation
            return profile
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            func_name = func.__name__
            
            # Create identifier from arguments
            args_str = json.dumps([str(arg) for arg in args], sort_keys=True)
            kwargs_str = json.dumps(sorted(kwargs.items()), sort_keys=True)
            identifier = f"{func_name}:{hashlib.md5((args_str + kwargs_str).encode()).hexdigest()}"
            
            # Try to get from cache
            cache_manager = get_cache_manager()
            cached = await cache_manager.get(cache_type, identifier)
            if cached is not None:
                logger.debug(f"Cache hit for {func_name}")
                return cached
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_manager.set(cache_type, identifier, result, ttl)
            logger.debug(f"Cache set for {func_name}")
            return result
            
        return wrapper
    return decorator

# Global cache manager instance
_cache_manager: Optional[CacheManager] = None

async def get_cache_manager() -> CacheManager:
    """Get or create the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
        await _cache_manager.initialize()
    return _cache_manager

async def shutdown_cache():
    """Shutdown the global cache manager."""
    global _cache_manager
    if _cache_manager:
        await _cache_manager.close()
        _cache_manager = None