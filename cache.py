"""
LRU Cache Implementation for Text2Cypher Pipeline

Provides caching layer to reduce redundant LLM calls for repeated queries.
"""

import hashlib
import json
from typing import Dict, Any, Optional
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate,
            "total_requests": self.hits + self.misses
        }


class LRUCache:
    """
    Least Recently Used (LRU) cache implementation.
    
    Uses OrderedDict to maintain access order in O(1) time.
    Automatically evicts least recently used items when max_size is reached.
    
    Args:
        max_size: Maximum number of entries to cache
    """
    
    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self._cache: OrderedDict = OrderedDict()
        self.evictions = 0
        
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if found, None otherwise
        """
        if key in self._cache:
            # Move to end (mark as recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
        return None
    
    def put(self, key: str, value: Any):
        """
        Store value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self._cache:
            # Update existing key
            self._cache.move_to_end(key)
        else:
            # Add new key
            if len(self._cache) >= self.max_size:
                # Evict least recently used (first item)
                self._cache.popitem(last=False)
                self.evictions += 1
        
        self._cache[key] = value
    
    def clear(self):
        """Clear all cache entries"""
        self._cache.clear()
        
    def size(self) -> int:
        """Get current cache size"""
        return len(self._cache)


class QueryCache:
    """
    Query cache for Text2Cypher pipeline.
    
    Caches generated Cypher queries to avoid redundant LLM calls.
    Uses question text and schema hash as cache key.
    
    Args:
        max_size: Maximum number of cached queries (default: 256)
        enable_cache: Whether caching is enabled (default: True)
    """
    
    def __init__(self, max_size: int = 256, enable_cache: bool = True):
        self.cache = LRUCache(max_size=max_size)
        self.stats = CacheStats()
        self.enable_cache = enable_cache
        
    def _normalize_question(self, question: str) -> str:
        """Normalize question for consistent cache keys"""
        return question.lower().strip()
    
    def _hash_text(self, text: str) -> str:
        """Generate MD5 hash of text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_cache_key(self, question: str, schema: str = "") -> str:
        """
        Generate cache key for question and schema.
        
        Args:
            question: Natural language question
            schema: Database schema (optional, for schema-aware caching)
            
        Returns:
            MD5 hash to use as cache key
        """
        normalized = self._normalize_question(question)
        combined = f"{normalized}|{schema}"
        return self._hash_text(combined)
    
    def get(self, question: str, schema: str = "") -> Optional[str]:
        """
        Get cached Cypher query.
        
        Args:
            question: Natural language question
            schema: Database schema
            
        Returns:
            Cached Cypher query if found, None otherwise
        """
        if not self.enable_cache:
            return None
            
        cache_key = self._get_cache_key(question, schema)
        result = self.cache.get(cache_key)
        
        if result is not None:
            self.stats.hits += 1
        else:
            self.stats.misses += 1
            
        return result
    
    def put(self, question: str, cypher_query: str, schema: str = ""):
        """
        Cache a Cypher query.
        
        Args:
            question: Natural language question
            cypher_query: Generated Cypher query
            schema: Database schema
        """
        if not self.enable_cache:
            return
            
        cache_key = self._get_cache_key(question, schema)
        self.cache.put(cache_key, cypher_query)
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            **self.stats.to_dict(),
            "cache_size": self.cache.size(),
            "max_size": self.cache.max_size,
            "evictions": self.cache.evictions
        }
    
    def print_stats(self):
        """Print human-readable cache statistics"""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("CACHE STATISTICS")
        print("="*60)
        
        print(f"\nCache Size: {stats['cache_size']}/{stats['max_size']}")
        print(f"\nHit Rate: {stats['hit_rate']*100:.1f}% "
              f"({stats['hits']} hits / {stats['total_requests']} total)")
        print(f"Cache Hits: {stats['hits']}")
        print(f"Cache Misses: {stats['misses']}")
        print(f"Evictions: {stats['evictions']}")
        
        print("="*60 + "\n")
    
    def clear(self):
        """Clear cache and reset statistics"""
        self.cache.clear()
        self.stats = CacheStats()


# Global cache instance (can be imported and used directly)
_global_cache = None


def get_global_cache(max_size: int = 256, enable_cache: bool = True) -> QueryCache:
    """
    Get or create global cache instance.
    
    Args:
        max_size: Maximum cache size
        enable_cache: Whether caching is enabled
        
    Returns:
        Global QueryCache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = QueryCache(max_size=max_size, enable_cache=enable_cache)
    return _global_cache


def clear_global_cache():
    """Clear global cache"""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()
