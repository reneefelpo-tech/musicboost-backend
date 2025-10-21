"""
Real-Time Stream Tracking Service
Aggregates stream counts from multiple music platforms with live updates
"""
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import os
import asyncio
import requests
from motor.motor_asyncio import AsyncIOMotorDatabase
import logging

logger = logging.getLogger(__name__)

class StreamTrackingService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        
        # API credentials
        self.spotify_client_id = os.getenv("SPOTIFY_CLIENT_ID", "")
        self.spotify_client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "")
        self.apple_music_token = os.getenv("APPLE_MUSIC_TOKEN", "")
        self.youtube_api_key = os.getenv("YOUTUBE_API_KEY", "")
        self.soundcloud_client_id = os.getenv("SOUNDCLOUD_CLIENT_ID", "")
        
        # Spotify access token cache
        self.spotify_token = None
        self.spotify_token_expires = None
    
    # ==================== SPOTIFY TRACKING ====================
    
    async def get_spotify_token(self) -> str:
        """Get Spotify access token (client credentials flow)"""
        if self.spotify_token and self.spotify_token_expires > datetime.utcnow():
            return self.spotify_token
        
        auth_url = "https://accounts.spotify.com/api/token"
        auth_response = requests.post(auth_url, {
            "grant_type": "client_credentials",
            "client_id": self.spotify_client_id,
            "client_secret": self.spotify_client_secret
        })
        auth_response.raise_for_status()
        
        token_data = auth_response.json()
        self.spotify_token = token_data["access_token"]
        self.spotify_token_expires = datetime.utcnow() + timedelta(seconds=token_data.get("expires_in", 3600))
        
        return self.spotify_token
    
    async def get_spotify_track_stats(self, track_id: str) -> Dict:
        """
        Get Spotify track statistics
        Note: Spotify API doesn't provide play counts directly. 
        This is a placeholder for third-party API integration (e.g., Spotontrack)
        """
        try:
            token = await self.get_spotify_token()
            
            # Get track details from Spotify API
            headers = {"Authorization": f"Bearer {token}"}
            track_url = f"https://api.spotify.com/v1/tracks/{track_id}"
            response = requests.get(track_url, headers=headers)
            response.raise_for_status()
            
            track_data = response.json()
            
            # Spotify doesn't provide play counts via public API
            # We'll use track popularity as a proxy and simulate counts
            popularity = track_data.get("popularity", 0)
            
            # Estimated streams based on popularity (rough approximation)
            estimated_streams = popularity * 10000  # 0-100 popularity -> 0-1M streams
            
            return {
                "platform": "spotify",
                "track_id": track_id,
                "track_name": track_data.get("name"),
                "artist": track_data.get("artists", [{}])[0].get("name"),
                "popularity": popularity,
                "estimated_streams": estimated_streams,
                "followers": track_data.get("artists", [{}])[0].get("followers", {}).get("total", 0),
                "last_updated": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Spotify tracking error: {e}")
            return {
                "platform": "spotify",
                "error": str(e),
                "estimated_streams": 0
            }
    
    # ==================== YOUTUBE MUSIC TRACKING ====================
    
    async def get_youtube_video_stats(self, video_id: str) -> Dict:
        """Get YouTube video statistics"""
        try:
            url = "https://www.googleapis.com/youtube/v3/videos"
            params = {
                "part": "statistics,snippet",
                "id": video_id,
                "key": self.youtube_api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data.get("items"):
                return {
                    "platform": "youtube",
                    "error": "Video not found",
                    "view_count": 0
                }
            
            item = data["items"][0]
            stats = item.get("statistics", {})
            snippet = item.get("snippet", {})
            
            return {
                "platform": "youtube",
                "video_id": video_id,
                "title": snippet.get("title"),
                "channel": snippet.get("channelTitle"),
                "view_count": int(stats.get("viewCount", 0)),
                "like_count": int(stats.get("likeCount", 0)),
                "comment_count": int(stats.get("commentCount", 0)),
                "last_updated": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"YouTube tracking error: {e}")
            return {
                "platform": "youtube",
                "error": str(e),
                "view_count": 0
            }
    
    # ==================== SOUNDCLOUD TRACKING ====================
    
    async def get_soundcloud_track_stats(self, track_url: str) -> Dict:
        """Get SoundCloud track statistics"""
        try:
            # Resolve track URL to get track ID
            resolve_url = f"https://api.soundcloud.com/resolve"
            params = {
                "url": track_url,
                "client_id": self.soundcloud_client_id
            }
            
            response = requests.get(resolve_url, params=params)
            response.raise_for_status()
            track_data = response.json()
            
            return {
                "platform": "soundcloud",
                "track_id": track_data.get("id"),
                "title": track_data.get("title"),
                "artist": track_data.get("user", {}).get("username"),
                "play_count": track_data.get("playback_count", 0),
                "like_count": track_data.get("likes_count", 0),
                "comment_count": track_data.get("comment_count", 0),
                "download_count": track_data.get("download_count", 0),
                "last_updated": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"SoundCloud tracking error: {e}")
            return {
                "platform": "soundcloud",
                "error": str(e),
                "play_count": 0
            }
    
    # ==================== APPLE MUSIC (LIMITED) ====================
    
    async def get_apple_music_stats(self, track_id: str) -> Dict:
        """
        Get Apple Music statistics (very limited public API)
        Note: Apple Music doesn't provide public stream counts
        """
        return {
            "platform": "apple_music",
            "note": "Apple Music does not provide public stream count API",
            "estimated_streams": 0,
            "last_updated": datetime.utcnow()
        }
    
    # ==================== AGGREGATED TRACKING ====================
    
    async def track_link_streams(self, user_id: str, link_id: str, platform: str, platform_id: str) -> Dict:
        """Track and store stream statistics for a music link"""
        stats = {}
        
        if platform == "spotify":
            stats = await self.get_spotify_track_stats(platform_id)
        elif platform == "youtube":
            stats = await self.get_youtube_video_stats(platform_id)
        elif platform == "soundcloud":
            stats = await self.get_soundcloud_track_stats(platform_id)
        elif platform == "apple_music":
            stats = await self.get_apple_music_stats(platform_id)
        
        # Store in database
        stream_record = {
            "_id": f"{link_id}_{platform}_{datetime.utcnow().strftime('%Y%m%d%H%M')}",
            "user_id": user_id,
            "link_id": link_id,
            "platform": platform,
            "platform_id": platform_id,
            "stats": stats,
            "timestamp": datetime.utcnow()
        }
        
        await self.db.stream_stats.insert_one(stream_record)
        
        return stats
    
    async def get_aggregated_stats(self, user_id: str, link_id: str = None) -> Dict:
        """Get aggregated statistics across all platforms"""
        query = {"user_id": user_id}
        if link_id:
            query["link_id"] = link_id
        
        # Get latest stats for each platform
        pipeline = [
            {"$match": query},
            {"$sort": {"timestamp": -1}},
            {"$group": {
                "_id": {"link_id": "$link_id", "platform": "$platform"},
                "latest_stats": {"$first": "$stats"},
                "timestamp": {"$first": "$timestamp"}
            }}
        ]
        
        results = await self.db.stream_stats.aggregate(pipeline).to_list(1000)
        
        # Aggregate totals
        total_streams = 0
        total_views = 0
        total_likes = 0
        platform_breakdown = {}
        
        for result in results:
            platform = result["_id"]["platform"]
            stats = result["latest_stats"]
            
            # Add to totals
            if "estimated_streams" in stats:
                total_streams += stats.get("estimated_streams", 0)
            if "view_count" in stats:
                total_views += stats.get("view_count", 0)
            if "play_count" in stats:
                total_streams += stats.get("play_count", 0)
            if "like_count" in stats:
                total_likes += stats.get("like_count", 0)
            
            # Platform breakdown
            if platform not in platform_breakdown:
                platform_breakdown[platform] = {
                    "streams": 0,
                    "engagement": 0
                }
            
            platform_breakdown[platform]["streams"] = stats.get("estimated_streams") or stats.get("play_count") or stats.get("view_count", 0)
            platform_breakdown[platform]["engagement"] = stats.get("like_count", 0) + stats.get("comment_count", 0)
        
        return {
            "total_streams": total_streams + total_views,
            "total_likes": total_likes,
            "platform_breakdown": platform_breakdown,
            "last_updated": datetime.utcnow()
        }
    
    # ==================== REAL-TIME REFRESH ====================
    
    async def start_realtime_tracking(self):
        """Background task for real-time stream tracking (every 60 seconds)"""
        while True:
            try:
                # Get all active music links with platform info
                links = await self.db.music_links.find({"is_active": True}).to_list(1000)
                
                for link in links:
                    try:
                        # Extract platform ID from URL (simplified)
                        platform = link.get("platform", "").lower()
                        url = link.get("url", "")
                        
                        # Extract ID based on platform
                        platform_id = None
                        if "spotify.com" in url and "/track/" in url:
                            platform_id = url.split("/track/")[1].split("?")[0]
                            platform = "spotify"
                        elif "youtube.com" in url or "youtu.be" in url:
                            if "v=" in url:
                                platform_id = url.split("v=")[1].split("&")[0]
                            elif "youtu.be/" in url:
                                platform_id = url.split("youtu.be/")[1].split("?")[0]
                            platform = "youtube"
                        elif "soundcloud.com" in url:
                            platform_id = url
                            platform = "soundcloud"
                        
                        if platform_id and platform:
                            stats = await self.track_link_streams(
                                link["user_id"],
                                link["_id"],
                                platform,
                                platform_id
                            )
                            
                            # Update link with latest stats
                            await self.db.music_links.update_one(
                                {"_id": link["_id"]},
                                {"$set": {
                                    "latest_stream_count": stats.get("estimated_streams") or stats.get("play_count") or stats.get("view_count", 0),
                                    "last_tracked": datetime.utcnow()
                                }}
                            )
                    
                    except Exception as e:
                        logger.error(f"Error tracking link {link.get('_id')}: {e}")
                        continue
                
                logger.info(f"✅ Tracked {len(links)} music links")
                
                # Wait 60 seconds before next update
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Real-time tracking error: {e}")
                await asyncio.sleep(60)


# Global instance
stream_tracking_service = None

def get_stream_tracking_service(db: AsyncIOMotorDatabase) -> StreamTrackingService:
    global stream_tracking_service
    if stream_tracking_service is None:
        stream_tracking_service = StreamTrackingService(db)
    return stream_tracking_service
