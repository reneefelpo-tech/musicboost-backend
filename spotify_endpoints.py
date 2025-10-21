"""
Spotify API Endpoints - OAuth, Stats, Auto-Promotion
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/spotify", tags=["spotify"])

# These endpoints will be integrated into server.py

# ==================== SPOTIFY OAUTH ENDPOINTS ====================

@router.get("/connect")
async def spotify_connect(current_user: dict):
    """Generate Spotify OAuth URL"""
    from backend.spotify_service import get_spotify_service
    from backend.server import db
    
    spotify_service = get_spotify_service(db)
    state = str(uuid.uuid4())
    
    # Store state for verification
    await db.oauth_states.insert_one({
        "_id": state,
        "user_id": current_user["_id"],
        "platform": "spotify",
        "created_at": datetime.utcnow()
    })
    
    auth_url = spotify_service.get_authorization_url(state)
    
    return {
        "auth_url": auth_url,
        "state": state,
        "message": "Redirect user to auth_url to connect Spotify"
    }

@router.get("/callback")
async def spotify_callback(code: str, state: str):
    """Handle Spotify OAuth callback"""
    from backend.spotify_service import get_spotify_service
    from backend.server import db
    
    # Verify state
    oauth_state = await db.oauth_states.find_one({"_id": state, "platform": "spotify"})
    if not oauth_state:
        raise HTTPException(status_code=400, detail="Invalid state")
    
    user_id = oauth_state["user_id"]
    spotify_service = get_spotify_service(db)
    
    try:
        # Exchange code for tokens
        token_data = await spotify_service.exchange_code_for_token(code)
        
        # Store connection
        connection = await spotify_service.store_spotify_connection(user_id, token_data)
        
        # Cleanup state
        await db.oauth_states.delete_one({"_id": state})
        
        # Return success (in production, redirect to app)
        return {
            "success": True,
            "message": "Spotify connected successfully!",
            "spotify_id": connection.get("spotify_id"),
            "connected_at": connection.get("connected_at")
        }
        
    except Exception as e:
        logger.error(f"Spotify callback error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to connect Spotify: {str(e)}")

@router.get("/status")
async def spotify_connection_status(current_user: dict):
    """Check if user has Spotify connected"""
    from backend.server import db
    
    connection = await db.spotify_connections.find_one({"user_id": current_user["_id"]})
    
    if not connection:
        return {
            "connected": False,
            "message": "Spotify not connected"
        }
    
    profile = await db.spotify_artist_profiles.find_one({"user_id": current_user["_id"]})
    
    return {
        "connected": True,
        "display_name": profile.get("display_name") if profile else None,
        "followers": profile.get("followers") if profile else 0,
        "connected_at": connection.get("connected_at"),
        "last_synced": connection.get("last_synced")
    }

@router.post("/disconnect")
async def spotify_disconnect(current_user: dict):
    """Disconnect Spotify account"""
    from backend.server import db
    
    # Delete connection
    await db.spotify_connections.delete_one({"user_id": current_user["_id"]})
    
    # Delete profile
    await db.spotify_artist_profiles.delete_one({"user_id": current_user["_id"]})
    
    # Deactivate campaigns
    await db.spotify_promotion_campaigns.update_many(
        {"user_id": current_user["_id"], "status": "active"},
        {"$set": {"status": "inactive"}}
    )
    
    return {
        "success": True,
        "message": "Spotify disconnected successfully"
    }

# ==================== STATS & ANALYTICS ENDPOINTS ====================

@router.get("/stats/realtime")
async def get_realtime_spotify_stats(current_user: dict):
    """Get real-time Spotify statistics"""
    from backend.spotify_service import get_spotify_service
    from backend.server import db
    
    spotify_service = get_spotify_service(db)
    
    try:
        stats = await spotify_service.get_artist_realtime_stats(current_user["_id"])
        
        if "error" in stats:
            raise HTTPException(status_code=400, detail=stats["error"])
        
        return stats
        
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/track/{track_id}")
async def get_track_stats(track_id: str, current_user: dict):
    """Get detailed stats for a specific track"""
    from backend.spotify_service import get_spotify_service
    from backend.server import db
    
    spotify_service = get_spotify_service(db)
    
    # Get user's token
    token = await spotify_service.get_valid_token(current_user["_id"])
    
    try:
        stats = await spotify_service.get_track_detailed_stats(track_id, token)
        
        if "error" in stats:
            raise HTTPException(status_code=400, detail=stats["error"])
        
        return stats
        
    except Exception as e:
        logger.error(f"Error fetching track stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sync")
async def sync_spotify_profile(current_user: dict):
    """Manually sync Spotify profile"""
    from backend.spotify_service import get_spotify_service
    from backend.server import db
    
    spotify_service = get_spotify_service(db)
    token = await spotify_service.get_valid_token(current_user["_id"])
    
    if not token:
        raise HTTPException(status_code=400, detail="Spotify not connected")
    
    try:
        profile = await spotify_service.sync_artist_profile(current_user["_id"], token)
        
        return {
            "success": True,
            "message": "Profile synced successfully",
            "profile": profile
        }
        
    except Exception as e:
        logger.error(f"Sync error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== AUTO-PROMOTION ENDPOINTS ====================

@router.post("/promotion/create")
async def create_promotion_campaign(
    track_ids: List[str],
    current_user: dict
):
    """Create auto-promotion campaign"""
    from backend.spotify_service import get_spotify_service
    from backend.server import db
    
    # Verify Spotify is connected
    connection = await db.spotify_connections.find_one({"user_id": current_user["_id"]})
    if not connection:
        raise HTTPException(status_code=400, detail="Connect Spotify first")
    
    spotify_service = get_spotify_service(db)
    
    try:
        campaign = await spotify_service.create_promotion_campaign(current_user["_id"], track_ids)
        
        return {
            "success": True,
            "message": "Promotion campaign created!",
            "campaign": campaign
        }
        
    except Exception as e:
        logger.error(f"Campaign creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/promotion/campaigns")
async def get_promotion_campaigns(current_user: dict):
    """Get all promotion campaigns"""
    from backend.server import db
    
    campaigns = await db.spotify_promotion_campaigns.find({
        "user_id": current_user["_id"]
    }).to_list(100)
    
    return {"campaigns": campaigns}

@router.post("/promotion/execute/{campaign_id}")
async def execute_promotion(campaign_id: str, current_user: dict):
    """Manually execute promotion campaign"""
    from backend.spotify_service import get_spotify_service
    from backend.server import db
    
    # Verify campaign belongs to user
    campaign = await db.spotify_promotion_campaigns.find_one({
        "_id": campaign_id,
        "user_id": current_user["_id"]
    })
    
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    spotify_service = get_spotify_service(db)
    
    try:
        result = await spotify_service.execute_auto_promotion(campaign_id)
        
        return {
            "success": True,
            "message": "Promotion executed!",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Promotion execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/promotion/pause/{campaign_id}")
async def pause_campaign(campaign_id: str, current_user: dict):
    """Pause a promotion campaign"""
    from backend.server import db
    
    result = await db.spotify_promotion_campaigns.update_one(
        {"_id": campaign_id, "user_id": current_user["_id"]},
        {"$set": {"status": "paused"}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    return {
        "success": True,
        "message": "Campaign paused"
    }

@router.post("/promotion/resume/{campaign_id}")
async def resume_campaign(campaign_id: str, current_user: dict):
    """Resume a paused campaign"""
    from backend.server import db
    
    result = await db.spotify_promotion_campaigns.update_one(
        {"_id": campaign_id, "user_id": current_user["_id"]},
        {"$set": {"status": "active"}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    return {
        "success": True,
        "message": "Campaign resumed"
    }

# ==================== EXPOSURE MAXIMIZATION ENDPOINTS ====================

@router.get("/exposure/maximize")
async def maximize_exposure(current_user: dict):
    """Get exposure maximization strategies"""
    from backend.spotify_service import get_spotify_service
    from backend.server import db
    
    # Verify Spotify is connected
    connection = await db.spotify_connections.find_one({"user_id": current_user["_id"]})
    if not connection:
        raise HTTPException(status_code=400, detail="Connect Spotify first")
    
    spotify_service = get_spotify_service(db)
    
    try:
        exposure_plan = await spotify_service.maximize_exposure(current_user["_id"])
        
        return {
            "success": True,
            "message": "Exposure maximization plan generated",
            "plan": exposure_plan
        }
        
    except Exception as e:
        logger.error(f"Exposure maximization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/exposure/recommendations")
async def get_growth_recommendations(current_user: dict):
    """Get personalized growth recommendations"""
    from backend.server import db
    
    # Get artist profile
    profile = await db.spotify_artist_profiles.find_one({"user_id": current_user["_id"]})
    if not profile:
        raise HTTPException(status_code=400, detail="No Spotify profile found")
    
    followers = profile.get("followers", 0)
    top_tracks = profile.get("top_tracks", [])
    
    recommendations = []
    
    # Follower-based recommendations
    if followers < 100:
        recommendations.append({
            "priority": "high",
            "category": "foundation",
            "title": "Build Your Foundation",
            "actions": [
                "Share music on all social platforms daily",
                "Engage with similar artists",
                "Post consistently (3-5 times/week)",
                "Use relevant hashtags"
            ],
            "estimated_impact": "+50-100 followers/month"
        })
    elif followers < 1000:
        recommendations.append({
            "priority": "high",
            "category": "growth",
            "title": "Accelerate Growth",
            "actions": [
                "Collaborate with artists in your niche",
                "Submit to Spotify playlists",
                "Run targeted Instagram/TikTok ads",
                "Create viral content (challenges, trends)"
            ],
            "estimated_impact": "+200-500 followers/month"
        })
    else:
        recommendations.append({
            "priority": "high",
            "category": "scale",
            "title": "Scale Your Reach",
            "actions": [
                "Launch merchandise",
                "Start a YouTube channel",
                "Host live streaming sessions",
                "Partner with brands"
            ],
            "estimated_impact": "+1000+ followers/month"
        })
    
    # Content recommendations
    if len(top_tracks) > 0:
        recommendations.append({
            "priority": "medium",
            "category": "content",
            "title": "Optimize Your Content",
            "actions": [
                f"Promote your top track: {top_tracks[0]['name']}",
                "Create behind-the-scenes content",
                "Share songwriting process",
                "Post audio snippets on TikTok/Reels"
            ],
            "estimated_impact": "+30% engagement"
        })
    
    # Timing recommendations
    recommendations.append({
        "priority": "medium",
        "category": "timing",
        "title": "Optimal Posting Times",
        "actions": [
            "Post on weekdays: 6-9 PM (local time)",
            "Weekend posts: 10 AM - 2 PM",
            "New music: Friday releases (Spotify algorithm boost)",
            "Stories/Reels: 12-1 PM, 7-9 PM"
        ],
        "estimated_impact": "+200% reach"
    })
    
    return {
        "success": True,
        "current_followers": followers,
        "recommendations": recommendations,
        "next_milestone": next((m for m in [100, 500, 1000, 5000, 10000] if m > followers), "100k+")
    }
