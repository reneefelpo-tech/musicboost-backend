"""
New API Endpoints for Social Media, Stream Tracking, and Follower Growth
To be integrated into main server.py
"""

# Add these endpoints to your server.py after the existing endpoints

# ==================== SOCIAL MEDIA ENDPOINTS ====================

@api_router.get("/social/connect/{platform}")
async def social_connect(platform: str, current_user: dict = Depends(get_current_user)):
    """Get OAuth URL for connecting social media platform"""
    state = str(uuid.uuid4())
    
    # Store state temporarily
    await db.oauth_states.insert_one({
        "_id": state,
        "user_id": current_user["_id"],
        "platform": platform,
        "created_at": datetime.utcnow()
    })
    
    if platform == "facebook":
        auth_url = await social_service.get_facebook_auth_url(state)
    elif platform == "instagram":
        auth_url = await social_service.get_instagram_auth_url(state)
    elif platform == "twitter":
        auth_url = await social_service.get_twitter_auth_url(state)
    elif platform == "tiktok":
        auth_url = await social_service.get_tiktok_auth_url(state)
    elif platform == "snapchat":
        auth_url = await social_service.get_snapchat_auth_url(state)
    elif platform == "google":
        # Use Emergent Google OAuth integration
        auth_url = f"{os.getenv('BASE_URL')}/api/social/google/auth"
    else:
        raise HTTPException(status_code=400, detail="Unsupported platform")
    
    return {"auth_url": auth_url, "state": state}

@api_router.get("/social/callback/{platform}")
async def social_callback(platform: str, code: str, state: str):
    """Handle OAuth callback from social media platforms"""
    # Verify state
    oauth_state = await db.oauth_states.find_one({"_id": state})
    if not oauth_state:
        raise HTTPException(status_code=400, detail="Invalid state")
    
    user_id = oauth_state["user_id"]
    
    # Exchange code for token
    if platform == "facebook":
        token_data = await social_service.exchange_facebook_code(code)
        await social_service.store_social_token(user_id, "facebook", token_data)
    
    # Cleanup state
    await db.oauth_states.delete_one({"_id": state})
    
    return RedirectResponse(url=f"{os.getenv('BASE_URL')}/app/profile?connected={platform}")

@api_router.get("/social/connections")
async def get_social_connections(current_user: dict = Depends(get_current_user)):
    """Get list of connected social media platforms"""
    tokens = await db.social_tokens.find({"user_id": current_user["_id"]}).to_list(10)
    
    connections = []
    for token in tokens:
        connections.append({
            "platform": token["platform"],
            "connected_at": token["created_at"],
            "expires_at": token.get("expires_at")
        })
    
    return {"connections": connections}

@api_router.post("/social/schedule-post")
async def schedule_social_post(
    platforms: List[str],
    content: str,
    scheduled_time: Optional[datetime] = None,
    link_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Schedule a post to multiple social media platforms"""
    # Get link URL if provided
    link_url = None
    if link_id:
        link = await db.music_links.find_one({"_id": link_id, "user_id": current_user["_id"]})
        if link:
            short_code = generate_short_code(link_id)
            base_url = os.getenv("BASE_URL", "https://musicboost.app")
            link_url = f"{base_url}/api/l/{short_code}"
    
    # If no scheduled time, post immediately or AI determines optimal time
    if not scheduled_time:
        scheduled_time = datetime.utcnow() + timedelta(minutes=5)  # Schedule 5 min from now
    
    post = await social_service.schedule_social_post(
        current_user["_id"],
        platforms,
        content,
        scheduled_time,
        link_url
    )
    
    return post

@api_router.get("/social/scheduled-posts")
async def get_scheduled_posts(current_user: dict = Depends(get_current_user)):
    """Get all scheduled social media posts"""
    posts = await db.scheduled_social_posts.find({
        "user_id": current_user["_id"]
    }).sort("scheduled_time", 1).to_list(100)
    
    return {"posts": posts}

# ==================== STREAM TRACKING ENDPOINTS ====================

@api_router.get("/streams/stats")
async def get_stream_stats(link_id: Optional[str] = None, current_user: dict = Depends(get_current_user)):
    """Get aggregated stream statistics"""
    stats = await stream_service.get_aggregated_stats(current_user["_id"], link_id)
    return stats

@api_router.post("/streams/track/{link_id}")
async def track_streams(link_id: str, current_user: dict = Depends(get_current_user)):
    """Manually trigger stream tracking for a link"""
    link = await db.music_links.find_one({"_id": link_id, "user_id": current_user["_id"]})
    if not link:
        raise HTTPException(status_code=404, detail="Link not found")
    
    # Extract platform and ID from URL
    url = link["url"]
    platform = link.get("platform", "").lower()
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
    
    if not platform_id:
        raise HTTPException(status_code=400, detail="Could not extract platform ID from URL")
    
    stats = await stream_service.track_link_streams(
        current_user["_id"],
        link_id,
        platform,
        platform_id
    )
    
    return stats

@api_router.get("/streams/realtime")
async def get_realtime_streams(current_user: dict = Depends(get_current_user)):
    """Get real-time stream counts for all user's links"""
    links = await db.music_links.find({
        "user_id": current_user["_id"],
        "is_active": True
    }).to_list(100)
    
    realtime_data = []
    for link in links:
        realtime_data.append({
            "link_id": link["_id"],
            "title": link.get("title"),
            "platform": link.get("platform"),
            "latest_stream_count": link.get("latest_stream_count", 0),
            "last_tracked": link.get("last_tracked")
        })
    
    return {"links": realtime_data}

# ==================== FOLLOWER GROWTH ENDPOINTS ====================

@api_router.get("/growth/stats")
async def get_growth_stats(current_user: dict = Depends(get_current_user)):
    """Get follower growth statistics"""
    social_proof = await growth_service.generate_social_proof(current_user["_id"])
    milestones = await growth_service.check_follower_milestones(current_user["_id"])
    
    return {
        **social_proof,
        "milestones": milestones
    }

@api_router.get("/growth/referral")
async def get_referral_info(current_user: dict = Depends(get_current_user)):
    """Get referral campaign information"""
    campaign = await db.referral_campaigns.find_one({"user_id": current_user["_id"]})
    
    if not campaign:
        campaign = await growth_service.create_referral_campaign(current_user["_id"])
    
    return campaign

@api_router.post("/growth/referral/apply")
async def apply_referral(referral_code: str, current_user: dict = Depends(get_current_user)):
    """Apply a referral code"""
    result = await growth_service.process_referral(referral_code, current_user["_id"])
    return result

@api_router.post("/growth/auto-follow")
async def create_auto_follow(
    target_audience: List[str],
    daily_limit: int = 200,
    current_user: dict = Depends(get_current_user)
):
    """Create an auto-follow growth campaign"""
    campaign = await growth_service.create_follow_for_follow_campaign(
        current_user["_id"],
        target_audience
    )
    return campaign

@api_router.get("/growth/campaigns")
async def get_growth_campaigns(current_user: dict = Depends(get_current_user)):
    """Get all active growth campaigns"""
    campaigns = await db.growth_campaigns.find({
        "user_id": current_user["_id"],
        "status": "active"
    }).to_list(100)
    
    return {"campaigns": campaigns}

@api_router.post("/growth/sync-followers")
async def sync_social_followers(current_user: dict = Depends(get_current_user)):
    """Sync followers from connected social media platforms"""
    result = await growth_service.sync_social_followers(current_user["_id"])
    return result

# ==================== EARNINGS OPTIMIZATION ENDPOINTS ====================

@api_router.get("/earnings/optimization")
async def get_earnings_optimization(current_user: dict = Depends(get_current_user)):
    """Get earnings optimization statistics"""
    # Calculate revenue from different sources
    total_clicks = await db.click_events.count_documents({"user_id": current_user["_id"]})
    
    # Get stream counts
    stream_stats = await stream_service.get_aggregated_stats(current_user["_id"])
    total_streams = stream_stats.get("total_streams", 0)
    
    # Get follower count
    follower_count = await db.followers.count_documents({"artist_id": current_user["_id"]})
    
    # Revenue calculations (simulated - based on industry averages)
    # Spotify: $0.003 - $0.005 per stream
    # Premium targeting: 13x more ($0.004 average)
    stream_revenue = total_streams * 0.004
    
    # Click revenue: $0.05 per click (ad revenue + conversions)
    click_revenue = total_clicks * 0.05
    
    # Follower value: $0.10 per follower per month
    follower_revenue = follower_count * 0.10
    
    # Geo-targeting premium markets (US, UK, Norway) = 13x multiplier
    geo_optimized_revenue = stream_revenue * 1.5  # Simulated 50% in premium markets
    
    total_optimized_revenue = stream_revenue + click_revenue + follower_revenue + geo_optimized_revenue
    
    return {
        "total_revenue": total_optimized_revenue,
        "revenue_breakdown": {
            "streams": stream_revenue,
            "clicks": click_revenue,
            "followers": follower_revenue,
            "geo_optimization": geo_optimized_revenue
        },
        "optimization_tips": [
            "Target premium Spotify users for 13x higher revenue",
            "Focus on US, UK, Norway markets",
            "Increase follower engagement for recurring revenue",
            "Use peak-time posting for maximum clicks"
        ],
        "total_streams": total_streams,
        "total_clicks": total_clicks,
        "total_followers": follower_count
    }
