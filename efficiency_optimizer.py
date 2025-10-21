"""
Efficiency Optimization System
Maximizes performance through caching, indexing, and query optimization
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import hashlib
import json

class EfficiencyOptimizer:
    """
    Optimizes system efficiency through intelligent caching and database optimization
    """
    
    def __init__(self, db, coordinator):
        self.db = db
        self.coordinator = coordinator
        self.cache = {}  # In-memory cache
        self.cache_ttl = {}  # Time-to-live for cache entries
        self.cache_hits = 0
        self.cache_misses = 0
        self.optimizations_applied = []
        
    def _cache_key(self, prefix: str, identifier: str) -> str:
        """Generate cache key"""
        return f"{prefix}:{identifier}"
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is still valid"""
        if key not in self.cache_ttl:
            return False
        return datetime.utcnow() < self.cache_ttl[key]
    
    async def get_cached(self, prefix: str, identifier: str, ttl_seconds: int = 60) -> Optional[Any]:
        """Get value from cache"""
        key = self._cache_key(prefix, identifier)
        
        if key in self.cache and self._is_cache_valid(key):
            self.cache_hits += 1
            return self.cache[key]
        
        self.cache_misses += 1
        return None
    
    async def set_cached(self, prefix: str, identifier: str, value: Any, ttl_seconds: int = 60):
        """Set value in cache"""
        key = self._cache_key(prefix, identifier)
        self.cache[key] = value
        self.cache_ttl[key] = datetime.utcnow() + timedelta(seconds=ttl_seconds)
    
    async def invalidate_cache(self, prefix: str, identifier: Optional[str] = None):
        """Invalidate cache entries"""
        if identifier:
            key = self._cache_key(prefix, identifier)
            self.cache.pop(key, None)
            self.cache_ttl.pop(key, None)
        else:
            # Invalidate all entries with this prefix
            keys_to_remove = [k for k in self.cache.keys() if k.startswith(f"{prefix}:")]
            for key in keys_to_remove:
                self.cache.pop(key, None)
                self.cache_ttl.pop(key, None)
    
    async def cleanup_expired_cache(self):
        """Remove expired cache entries"""
        now = datetime.utcnow()
        expired_keys = [k for k, ttl in self.cache_ttl.items() if ttl < now]
        
        for key in expired_keys:
            self.cache.pop(key, None)
            self.cache_ttl.pop(key, None)
        
        if expired_keys:
            logging.debug(f"🗑️ Cleaned up {len(expired_keys)} expired cache entries")
    
    async def ensure_database_indexes(self) -> Dict:
        """Ensure all necessary database indexes exist"""
        indexes_created = []
        
        try:
            # Music Links indexes
            try:
                await self.db.music_links.create_index("user_id")
                indexes_created.append("music_links.user_id")
            except:
                pass  # Index might already exist
            
            try:
                await self.db.music_links.create_index("is_active")
                indexes_created.append("music_links.is_active")
            except:
                pass
            
            try:
                await self.db.music_links.create_index([("user_id", 1), ("is_active", 1)])
                indexes_created.append("music_links.user_id_is_active")
            except:
                pass
            
            # Short Code Map indexes
            try:
                await self.db.short_code_map.create_index("short_code", unique=True)
                indexes_created.append("short_code_map.short_code")
            except:
                pass
            
            try:
                await self.db.short_code_map.create_index("link_id")
                indexes_created.append("short_code_map.link_id")
            except:
                pass
            
            try:
                await self.db.short_code_map.create_index("last_accessed")
                indexes_created.append("short_code_map.last_accessed")
            except:
                pass
            
            # Click Events indexes
            try:
                await self.db.click_events.create_index("link_id")
                indexes_created.append("click_events.link_id")
            except:
                pass
            
            try:
                await self.db.click_events.create_index("timestamp")
                indexes_created.append("click_events.timestamp")
            except:
                pass
            
            try:
                await self.db.click_events.create_index([("link_id", 1), ("event_type", 1)])
                indexes_created.append("click_events.link_id_event_type")
            except:
                pass
            
            # Users indexes
            try:
                await self.db.users.create_index("email", unique=True)
                indexes_created.append("users.email")
            except:
                pass
            
            try:
                await self.db.users.create_index("username")
                indexes_created.append("users.username")
            except:
                pass
            
            self.optimizations_applied.append({
                "type": "database_indexes",
                "indexes_created": indexes_created,
                "timestamp": datetime.utcnow()
            })
            
            logging.info(f"📊 Ensured {len(indexes_created)} database indexes")
            
            return {
                "indexes_ensured": len(indexes_created),
                "indexes": indexes_created
            }
            
        except Exception as e:
            logging.error(f"❌ Error ensuring indexes: {str(e)}")
            return {"error": str(e)}
    
    async def optimize_link_lookups(self, link_id: str) -> Dict:
        """Optimize link lookups using caching"""
        # Check cache first
        cached_link = await self.get_cached("link", link_id, ttl_seconds=300)  # 5 min cache
        if cached_link:
            return cached_link
        
        # Fetch from database
        link = await self.db.music_links.find_one({"_id": link_id})
        
        if link:
            # Convert datetime to string for JSON serialization
            link_copy = link.copy()
            if "created_at" in link_copy and isinstance(link_copy["created_at"], datetime):
                link_copy["created_at"] = link_copy["created_at"].isoformat()
            
            # Cache the result
            await self.set_cached("link", link_id, link_copy, ttl_seconds=300)
        
        return link
    
    async def optimize_short_code_lookup(self, short_code: str) -> Optional[Dict]:
        """Optimize short code lookups using caching"""
        # Check cache first
        cached_mapping = await self.get_cached("short_code", short_code, ttl_seconds=600)  # 10 min cache
        if cached_mapping:
            return cached_mapping
        
        # Fetch from database
        mapping = await self.db.short_code_map.find_one({"short_code": short_code})
        
        if mapping:
            # Convert datetime to string for JSON serialization
            mapping_copy = mapping.copy()
            if "created_at" in mapping_copy and isinstance(mapping_copy["created_at"], datetime):
                mapping_copy["created_at"] = mapping_copy["created_at"].isoformat()
            if "last_accessed" in mapping_copy and isinstance(mapping_copy["last_accessed"], datetime):
                mapping_copy["last_accessed"] = mapping_copy["last_accessed"].isoformat()
            
            # Cache the result
            await self.set_cached("short_code", short_code, mapping_copy, ttl_seconds=600)
        
        return mapping
    
    async def batch_create_mappings(self, links: list) -> int:
        """Batch create short code mappings for efficiency"""
        from server import generate_short_code
        import uuid
        
        mappings_to_create = []
        
        for link in links:
            link_id = link["_id"]
            short_code = generate_short_code(link_id)
            
            # Check if mapping already exists
            existing = await self.db.short_code_map.find_one({"short_code": short_code})
            if not existing:
                mappings_to_create.append({
                    "_id": str(uuid.uuid4()),
                    "short_code": short_code,
                    "link_id": link_id,
                    "created_at": datetime.utcnow(),
                    "last_accessed": datetime.utcnow()
                })
        
        if mappings_to_create:
            result = await self.db.short_code_map.insert_many(mappings_to_create)
            return len(result.inserted_ids)
        
        return 0
    
    async def get_cache_statistics(self) -> Dict:
        """Get cache performance statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percentage": round(hit_rate, 2),
            "optimizations_applied": len(self.optimizations_applied)
        }
    
    async def run_optimization_cycle(self) -> Dict:
        """Run a complete optimization cycle"""
        logging.info("⚡ Starting optimization cycle")
        
        try:
            # Ensure indexes
            index_result = await self.ensure_database_indexes()
            
            # Cleanup expired cache
            await self.cleanup_expired_cache()
            
            # Get cache stats
            cache_stats = await self.get_cache_statistics()
            
            # Fire neural network signal
            await self.coordinator.fire_signal("optimization_complete", {
                "indexes": index_result,
                "cache_stats": cache_stats
            })
            
            logging.info(f"✅ Optimization cycle complete - Cache hit rate: {cache_stats['hit_rate_percentage']}%")
            
            return {
                "status": "success",
                "indexes": index_result,
                "cache_stats": cache_stats
            }
            
        except Exception as e:
            logging.error(f"❌ Optimization cycle error: {str(e)}")
            return {"error": str(e)}

# Global optimizer instance (will be initialized with db and coordinator)
optimizer = None
