"""
Social Media Integration Service
Handles OAuth, token management, and auto-posting for multiple platforms
"""
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import os
import jwt
import uuid
import asyncio
from motor.motor_asyncio import AsyncIOMotorDatabase
import requests
from requests_oauthlib import OAuth1Session, OAuth2Session
import logging

logger = logging.getLogger(__name__)

class SocialMediaService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        
        # OAuth credentials (to be configured via environment variables)
        self.facebook_app_id = os.getenv("FACEBOOK_APP_ID", "")
        self.facebook_app_secret = os.getenv("FACEBOOK_APP_SECRET", "")
        self.instagram_app_id = os.getenv("INSTAGRAM_APP_ID", "")  # Uses Facebook Graph API
        self.instagram_app_secret = os.getenv("INSTAGRAM_APP_SECRET", "")
        self.twitter_api_key = os.getenv("TWITTER_API_KEY", "")
        self.twitter_api_secret = os.getenv("TWITTER_API_SECRET", "")
        self.tiktok_client_key = os.getenv("TIKTOK_CLIENT_KEY", "")
        self.tiktok_client_secret = os.getenv("TIKTOK_CLIENT_SECRET", "")
        self.snapchat_client_id = os.getenv("SNAPCHAT_CLIENT_ID", "")
        self.snapchat_client_secret = os.getenv("SNAPCHAT_CLIENT_SECRET", "")
        
        # Redirect URIs
        self.redirect_uri = os.getenv("BASE_URL", "http://localhost:8001") + "/api/social/callback"
    
    # ==================== OAUTH FLOWS ====================
    
    async def get_facebook_auth_url(self, state: str) -> str:
        """Generate Facebook OAuth URL"""
        scope = "email,public_profile,pages_manage_posts,pages_read_engagement,instagram_basic,instagram_content_publish"
        auth_url = (
            f"https://www.facebook.com/v18.0/dialog/oauth?"
            f"client_id={self.facebook_app_id}&"
            f"redirect_uri={self.redirect_uri}/facebook&"
            f"state={state}&"
            f"scope={scope}"
        )
        return auth_url
    
    async def exchange_facebook_code(self, code: str) -> Dict:
        """Exchange Facebook authorization code for access token"""
        token_url = "https://graph.facebook.com/v18.0/oauth/access_token"
        params = {
            "client_id": self.facebook_app_id,
            "client_secret": self.facebook_app_secret,
            "redirect_uri": f"{self.redirect_uri}/facebook",
            "code": code
        }
        response = requests.get(token_url, params=params)
        response.raise_for_status()
        return response.json()
    
    async def get_instagram_auth_url(self, state: str) -> str:
        """Generate Instagram OAuth URL (via Facebook)"""
        scope = "instagram_basic,instagram_content_publish,pages_show_list,pages_read_engagement"
        auth_url = (
            f"https://api.instagram.com/oauth/authorize?"
            f"client_id={self.instagram_app_id}&"
            f"redirect_uri={self.redirect_uri}/instagram&"
            f"scope={scope}&"
            f"response_type=code&"
            f"state={state}"
        )
        return auth_url
    
    async def get_twitter_auth_url(self, state: str) -> str:
        """Generate Twitter OAuth 2.0 URL"""
        scope = "tweet.read,tweet.write,users.read,follows.read,follows.write"
        auth_url = (
            f"https://twitter.com/i/oauth2/authorize?"
            f"response_type=code&"
            f"client_id={self.twitter_api_key}&"
            f"redirect_uri={self.redirect_uri}/twitter&"
            f"scope={scope}&"
            f"state={state}&"
            f"code_challenge=challenge&"
            f"code_challenge_method=plain"
        )
        return auth_url
    
    async def get_tiktok_auth_url(self, state: str) -> str:
        """Generate TikTok OAuth URL"""
        scope = "user.info.basic,video.list,video.upload"
        auth_url = (
            f"https://www.tiktok.com/auth/authorize/?"
            f"client_key={self.tiktok_client_key}&"
            f"response_type=code&"
            f"scope={scope}&"
            f"redirect_uri={self.redirect_uri}/tiktok&"
            f"state={state}"
        )
        return auth_url
    
    async def get_snapchat_auth_url(self, state: str) -> str:
        """Generate Snapchat OAuth URL"""
        scope = "https://auth.snapchat.com/oauth2/api/user.display_name"
        auth_url = (
            f"https://accounts.snapchat.com/accounts/oauth2/auth?"
            f"response_type=code&"
            f"client_id={self.snapchat_client_id}&"
            f"redirect_uri={self.redirect_uri}/snapchat&"
            f"scope={scope}&"
            f"state={state}"
        )
        return auth_url
    
    # ==================== TOKEN STORAGE ====================
    
    async def store_social_token(self, user_id: str, platform: str, token_data: Dict):
        """Store social media access token"""
        token_doc = {
            "_id": str(uuid.uuid4()),
            "user_id": user_id,
            "platform": platform,
            "access_token": token_data.get("access_token"),
            "refresh_token": token_data.get("refresh_token"),
            "expires_at": datetime.utcnow() + timedelta(seconds=token_data.get("expires_in", 3600)),
            "scope": token_data.get("scope", ""),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Upsert: Update if exists, insert if not
        await self.db.social_tokens.update_one(
            {"user_id": user_id, "platform": platform},
            {"$set": token_doc},
            upsert=True
        )
        
        return token_doc
    
    async def get_social_token(self, user_id: str, platform: str) -> Optional[Dict]:
        """Get social media access token"""
        token = await self.db.social_tokens.find_one({
            "user_id": user_id,
            "platform": platform
        })
        
        if not token:
            return None
        
        # Check if token is expired
        if token.get("expires_at") and token["expires_at"] < datetime.utcnow():
            # Try to refresh token
            refreshed = await self.refresh_token(user_id, platform)
            if refreshed:
                return refreshed
            return None
        
        return token
    
    async def refresh_token(self, user_id: str, platform: str) -> Optional[Dict]:
        """Refresh expired access token"""
        token = await self.db.social_tokens.find_one({
            "user_id": user_id,
            "platform": platform
        })
        
        if not token or not token.get("refresh_token"):
            return None
        
        try:
            if platform == "facebook":
                # Facebook long-lived token exchange
                token_url = "https://graph.facebook.com/v18.0/oauth/access_token"
                params = {
                    "grant_type": "fb_exchange_token",
                    "client_id": self.facebook_app_id,
                    "client_secret": self.facebook_app_secret,
                    "fb_exchange_token": token["access_token"]
                }
                response = requests.get(token_url, params=params)
                response.raise_for_status()
                new_token_data = response.json()
                
                return await self.store_social_token(user_id, platform, new_token_data)
            
            # Add refresh logic for other platforms as needed
            
        except Exception as e:
            logger.error(f"Token refresh error for {platform}: {e}")
            return None
    
    # ==================== AUTO-POSTING ====================
    
    async def post_to_facebook(self, user_id: str, content: str, link: str = None) -> Dict:
        """Post content to Facebook"""
        token = await self.get_social_token(user_id, "facebook")
        if not token:
            raise ValueError("Facebook not connected")
        
        # Get user's pages
        pages_url = f"https://graph.facebook.com/v18.0/me/accounts"
        params = {"access_token": token["access_token"]}
        pages_response = requests.get(pages_url, params=params)
        pages_response.raise_for_status()
        pages = pages_response.json().get("data", [])
        
        if not pages:
            raise ValueError("No Facebook pages found")
        
        # Post to first page
        page = pages[0]
        page_id = page["id"]
        page_access_token = page["access_token"]
        
        post_url = f"https://graph.facebook.com/v18.0/{page_id}/feed"
        post_data = {
            "message": content,
            "access_token": page_access_token
        }
        
        if link:
            post_data["link"] = link
        
        response = requests.post(post_url, data=post_data)
        response.raise_for_status()
        
        return response.json()
    
    async def post_to_instagram(self, user_id: str, image_url: str, caption: str) -> Dict:
        """Post content to Instagram (requires business account)"""
        token = await self.get_social_token(user_id, "facebook")  # Instagram uses Facebook token
        if not token:
            raise ValueError("Instagram not connected")
        
        # Get Instagram business account ID
        accounts_url = f"https://graph.facebook.com/v18.0/me/accounts"
        params = {"access_token": token["access_token"]}
        accounts_response = requests.get(accounts_url, params=params)
        accounts_response.raise_for_status()
        
        # This is a simplified version - actual implementation requires:
        # 1. Get Instagram Business Account ID
        # 2. Create media container
        # 3. Publish media container
        
        return {"status": "Instagram posting requires business account setup"}
    
    async def schedule_social_post(
        self,
        user_id: str,
        platforms: List[str],
        content: str,
        scheduled_time: datetime,
        link: str = None
    ) -> Dict:
        """Schedule a post to multiple platforms"""
        scheduled_post = {
            "_id": str(uuid.uuid4()),
            "user_id": user_id,
            "platforms": platforms,
            "content": content,
            "link": link,
            "scheduled_time": scheduled_time,
            "status": "scheduled",
            "created_at": datetime.utcnow()
        }
        
        await self.db.scheduled_social_posts.insert_one(scheduled_post)
        
        return scheduled_post
    
    async def execute_scheduled_posts(self):
        """Background task to execute scheduled posts"""
        while True:
            try:
                # Find posts that are due
                current_time = datetime.utcnow()
                due_posts = await self.db.scheduled_social_posts.find({
                    "scheduled_time": {"$lte": current_time},
                    "status": "scheduled"
                }).to_list(100)
                
                for post in due_posts:
                    try:
                        # Post to each platform
                        results = {}
                        for platform in post["platforms"]:
                            if platform == "facebook":
                                result = await self.post_to_facebook(
                                    post["user_id"],
                                    post["content"],
                                    post.get("link")
                                )
                                results[platform] = result
                        
                        # Mark as posted
                        await self.db.scheduled_social_posts.update_one(
                            {"_id": post["_id"]},
                            {
                                "$set": {
                                    "status": "posted",
                                    "posted_at": datetime.utcnow(),
                                    "results": results
                                }
                            }
                        )
                        
                    except Exception as e:
                        logger.error(f"Error executing post {post['_id']}: {e}")
                        await self.db.scheduled_social_posts.update_one(
                            {"_id": post["_id"]},
                            {
                                "$set": {
                                    "status": "failed",
                                    "error": str(e)
                                }
                            }
                        )
                
                # Wait 60 seconds before next check
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Scheduled posts execution error: {e}")
                await asyncio.sleep(60)


# Global instance
social_media_service = None

def get_social_media_service(db: AsyncIOMotorDatabase) -> SocialMediaService:
    global social_media_service
    if social_media_service is None:
        social_media_service = SocialMediaService(db)
    return social_media_service
