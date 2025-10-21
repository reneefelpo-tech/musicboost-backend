"""
Meta (Facebook & Instagram) API Endpoints
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from datetime import datetime
import uuid
import logging

# Router instance will be imported in server.py
# These endpoints will be added to the main API router

async def setup_meta_endpoints(api_router, db, meta_service, get_current_user):
    """Setup Meta OAuth and API endpoints"""
    
    # ==================== META OAUTH ENDPOINTS ====================
    
    @api_router.get("/meta/connect")
    async def meta_connect(platform: str = "facebook", current_user: dict = Depends(get_current_user)):
        """Generate Meta OAuth URL for Facebook or Instagram"""
        state = str(uuid.uuid4())
        
        # Store state for verification
        await db.oauth_states.insert_one({
            "_id": state,
            "user_id": current_user["_id"],
            "platform": f"meta_{platform}",
            "created_at": datetime.utcnow()
        })
        
        auth_url = meta_service.get_authorization_url(state, platform)
        
        return {
            "auth_url": auth_url,
            "state": state,
            "platform": platform,
            "message": f"Redirect user to auth_url to connect {platform.title()}"
        }
    
    @api_router.get("/meta/callback")
    async def meta_callback(code: str, state: str):
        """Handle Meta OAuth callback"""
        from fastapi.responses import HTMLResponse
        
        # Verify state
        oauth_state = await db.oauth_states.find_one({"_id": state})
        if not oauth_state:
            return HTMLResponse(content="""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Connection Error</title>
                <style>
                    body { font-family: Arial; text-align: center; padding: 50px; background: #000; color: #fff; }
                    .error { background: #f44336; padding: 20px; border-radius: 10px; max-width: 400px; margin: 0 auto; }
                    .button { background: #1877F2; color: white; padding: 15px 30px; border-radius: 25px; 
                             text-decoration: none; display: inline-block; margin-top: 20px; }
                </style>
            </head>
            <body>
                <div class="error">
                    <h2>⚠️ Connection Error</h2>
                    <p>Invalid or expired authorization request.</p>
                    <p>Please try connecting again from the app.</p>
                </div>
                <a href="musicboost://meta" class="button">Back to App</a>
            </body>
            </html>
            """, status_code=400)
        
        user_id = oauth_state["user_id"]
        platform = oauth_state["platform"].replace("meta_", "")
        
        try:
            # Exchange code for tokens
            token_data = await meta_service.exchange_code_for_token(code)
            
            # Store connection
            connection = await meta_service.store_meta_connection(user_id, token_data, platform)
            
            # Cleanup state
            await db.oauth_states.delete_one({"_id": state})
            
            logging.info(f"✅ {platform.title()} connected successfully for user {user_id}")
            
            # Return success page with auto-redirect to app
            return HTMLResponse(content=f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{platform.title()} Connected!</title>
                <style>
                    body {{ font-family: Arial; text-align: center; padding: 50px; background: #000; color: #fff; }}
                    .success {{ background: #1877F2; padding: 30px; border-radius: 15px; max-width: 400px; margin: 0 auto; }}
                    .icon {{ font-size: 60px; margin-bottom: 20px; }}
                    .message {{ font-size: 18px; margin: 20px 0; }}
                    .redirect {{ color: #ccc; font-size: 14px; margin-top: 20px; }}
                </style>
                <script>
                    // Auto-redirect to app after 2 seconds
                    setTimeout(function() {{
                        window.location.href = 'musicboost://meta/connected?platform={platform}';
                    }}, 2000);
                </script>
            </head>
            <body>
                <div class="success">
                    <div class="icon">🎉</div>
                    <h2>{platform.title()} Connected!</h2>
                    <div class="message">Your {platform.title()} account has been linked successfully.</div>
                    <div class="redirect">Redirecting you back to Music Boost...</div>
                </div>
            </body>
            </html>
            """)
            
        except Exception as e:
            logging.error(f"Meta callback error: {e}")
            
            # Return error page
            return HTMLResponse(content=f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Connection Failed</title>
                <style>
                    body {{ font-family: Arial; text-align: center; padding: 50px; background: #000; color: #fff; }}
                    .error {{ background: #f44336; padding: 30px; border-radius: 15px; max-width: 400px; margin: 0 auto; }}
                    .icon {{ font-size: 60px; margin-bottom: 20px; }}
                    .message {{ font-size: 18px; margin: 20px 0; }}
                    .details {{ color: #ccc; font-size: 14px; margin-top: 20px; }}
                    .button {{ background: #1877F2; color: white; padding: 15px 30px; border-radius: 25px; 
                             text-decoration: none; display: inline-block; margin-top: 20px; }}
                </style>
            </head>
            <body>
                <div class="error">
                    <div class="icon">❌</div>
                    <h2>Connection Failed</h2>
                    <div class="message">Could not connect your {platform.title()} account.</div>
                    <div class="details">Error: {str(e)}</div>
                </div>
                <a href="musicboost://meta" class="button">Try Again</a>
            </body>
            </html>
            """)
    
    @api_router.get("/meta/status")
    async def meta_connection_status(platform: str = "facebook", current_user: dict = Depends(get_current_user)):
        """Check if user has Meta platform connected"""
        connection = await db.meta_connections.find_one({
            "user_id": current_user["_id"],
            "platform": platform
        })
        
        if not connection:
            return {
                "connected": False,
                "platform": platform,
                "message": f"{platform.title()} not connected"
            }
        
        profile = await db.meta_profiles.find_one({
            "user_id": current_user["_id"],
            "platform": platform
        })
        
        return {
            "connected": True,
            "platform": platform,
            "profile": profile or {},
            "connected_at": connection.get("connected_at"),
            "last_synced": connection.get("last_synced")
        }
    
    @api_router.post("/meta/disconnect")
    async def meta_disconnect(platform: str, current_user: dict = Depends(get_current_user)):
        """Disconnect Meta platform account"""
        # Delete connection
        await db.meta_connections.delete_one({
            "user_id": current_user["_id"],
            "platform": platform
        })
        
        # Delete profile
        await db.meta_profiles.delete_one({
            "user_id": current_user["_id"],
            "platform": platform
        })
        
        # Deactivate scheduled posts
        await db.scheduled_posts.update_many(
            {
                "user_id": current_user["_id"],
                "platform": platform,
                "status": "scheduled"
            },
            {"$set": {"status": "cancelled"}}
        )
        
        return {
            "success": True,
            "platform": platform,
            "message": f"{platform.title()} disconnected successfully"
        }
    
    @api_router.get("/meta/stats/realtime")
    async def get_realtime_meta_stats(platform: str = "facebook", current_user: dict = Depends(get_current_user)):
        """Get real-time statistics from Meta platforms"""
        try:
            stats = await meta_service.get_realtime_stats(current_user["_id"], platform)
            
            if "error" in stats:
                raise HTTPException(status_code=400, detail=stats["error"])
            
            return stats
            
        except Exception as e:
            logging.error(f"Error fetching {platform} stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @api_router.post("/meta/sync")
    async def sync_meta_profile(platform: str, current_user: dict = Depends(get_current_user)):
        """Manually sync Meta platform profile"""
        token = await meta_service.get_valid_token(current_user["_id"], platform)
        
        if not token:
            raise HTTPException(status_code=400, detail=f"{platform.title()} not connected")
        
        try:
            await meta_service.sync_user_profile(current_user["_id"], token, platform)
            
            return {
                "success": True,
                "platform": platform,
                "message": f"{platform.title()} profile synced successfully"
            }
            
        except Exception as e:
            logging.error(f"Sync error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @api_router.post("/meta/post")
    async def create_meta_post(
        platform: str,
        page_id: str,
        content: str,
        link: Optional[str] = None,
        current_user: dict = Depends(get_current_user)
    ):
        """Post content to Meta platform"""
        try:
            if platform == "facebook":
                result = await meta_service.post_to_facebook(
                    current_user["_id"],
                    page_id,
                    content,
                    link
                )
                
                return {
                    "success": True,
                    "platform": platform,
                    "post_id": result.get("id"),
                    "message": "Posted successfully!"
                }
            else:
                raise HTTPException(status_code=400, detail=f"Posting not yet implemented for {platform}")
                
        except Exception as e:
            logging.error(f"Post creation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @api_router.post("/meta/schedule")
    async def schedule_meta_post(
        platform: str,
        page_id: str,
        content: str,
        scheduled_time: datetime,
        link: Optional[str] = None,
        current_user: dict = Depends(get_current_user)
    ):
        """Schedule a post for later"""
        try:
            post = await meta_service.schedule_post(
                current_user["_id"],
                platform,
                page_id,
                content,
                scheduled_time,
                link
            )
            
            return {
                "success": True,
                "platform": platform,
                "scheduled_time": scheduled_time,
                "message": "Post scheduled successfully!"
            }
            
        except Exception as e:
            logging.error(f"Post scheduling error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @api_router.get("/meta/posts/scheduled")
    async def get_scheduled_posts(platform: Optional[str] = None, current_user: dict = Depends(get_current_user)):
        """Get all scheduled posts"""
        query = {"user_id": current_user["_id"], "status": "scheduled"}
        
        if platform:
            query["platform"] = platform
        
        posts = await db.scheduled_posts.find(query).sort("scheduled_time", 1).to_list(100)
        
        return {"scheduled_posts": posts}
    
    logging.info("🌐 Meta endpoints registered successfully")
