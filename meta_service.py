"""
Meta (Facebook & Instagram) OAuth and Social Media Service
Handles OAuth authentication, data fetching, and auto-posting
"""

import os
import logging
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)

class MetaService:
    def __init__(self, db):
        self.db = db
        self.client_id = os.getenv('META_APP_ID', '')
        self.client_secret = os.getenv('META_APP_SECRET', '')
        self.redirect_uri = os.getenv('META_REDIRECT_URI', os.getenv('BASE_URL', 'http://localhost:8001') + '/api/meta/callback')
        
        # Facebook Graph API
        self.graph_api_base = 'https://graph.facebook.com/v19.0'
        
        if not self.client_id or not self.client_secret:
            logging.warning("⚠️ Meta credentials not configured")
    
    def get_authorization_url(self, state: str, platform: str = 'facebook') -> str:
        """Generate Meta OAuth URL for Facebook or Instagram"""
        
        if platform == 'instagram':
            # Instagram Basic Display API scopes
            scopes = [
                'instagram_basic',
                'instagram_manage_insights',
                'pages_show_list',
                'pages_read_engagement',
                'instagram_content_publish'
            ]
        else:
            # Facebook scopes (email removed - requires app review)
            scopes = [
                'pages_show_list',
                'pages_read_engagement',
                'pages_manage_posts',
                'pages_read_user_content',
                'public_profile'
            ]
        
        scope_string = ','.join(scopes)
        
        auth_url = (
            f'https://www.facebook.com/v19.0/dialog/oauth?'
            f'client_id={self.client_id}&'
            f'redirect_uri={self.redirect_uri}&'
            f'scope={scope_string}&'
            f'state={state}&'
            f'response_type=code'
        )
        
        return auth_url
    
    async def exchange_code_for_token(self, code: str) -> Dict:
        """Exchange authorization code for access token"""
        
        token_url = f'{self.graph_api_base}/oauth/access_token'
        
        params = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'redirect_uri': self.redirect_uri,
            'code': code
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(token_url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Token exchange failed: {error_text}")
                
                data = await response.json()
                
                # Get long-lived token
                long_lived_token = await self.get_long_lived_token(data['access_token'])
                
                return {
                    'access_token': long_lived_token['access_token'],
                    'token_type': data.get('token_type', 'bearer'),
                    'expires_in': long_lived_token.get('expires_in', 5184000)  # 60 days
                }
    
    async def get_long_lived_token(self, short_token: str) -> Dict:
        """Exchange short-lived token for long-lived token (60 days)"""
        
        url = f'{self.graph_api_base}/oauth/access_token'
        
        params = {
            'grant_type': 'fb_exchange_token',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'fb_exchange_token': short_token
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logging.warning("Could not get long-lived token, using short-lived")
                    return {'access_token': short_token, 'expires_in': 3600}
                
                return await response.json()
    
    async def store_meta_connection(self, user_id: str, token_data: Dict, platform: str) -> Dict:
        """Store Meta OAuth connection"""
        
        connection_doc = {
            'user_id': user_id,
            'platform': platform,
            'access_token': token_data['access_token'],
            'token_type': token_data['token_type'],
            'expires_at': datetime.utcnow() + timedelta(seconds=token_data.get('expires_in', 5184000)),
            'connected_at': datetime.utcnow(),
            'last_synced': datetime.utcnow()
        }
        
        # Upsert connection
        await self.db.meta_connections.update_one(
            {'user_id': user_id, 'platform': platform},
            {'$set': connection_doc},
            upsert=True
        )
        
        # Fetch and store profile data
        await self.sync_user_profile(user_id, token_data['access_token'], platform)
        
        return connection_doc
    
    async def get_valid_token(self, user_id: str, platform: str) -> Optional[str]:
        """Get valid access token for user"""
        
        connection = await self.db.meta_connections.find_one({
            'user_id': user_id,
            'platform': platform
        })
        
        if not connection:
            return None
        
        # Check if token expired
        if connection['expires_at'] < datetime.utcnow():
            logging.warning(f"Meta token expired for user {user_id}")
            return None
        
        return connection['access_token']
    
    async def sync_user_profile(self, user_id: str, token: str, platform: str):
        """Sync user profile data from Meta"""
        
        try:
            if platform == 'facebook':
                profile = await self.get_facebook_profile(token)
                pages = await self.get_facebook_pages(token)
                
                profile_doc = {
                    'user_id': user_id,
                    'platform': 'facebook',
                    'profile_id': profile.get('id'),
                    'name': profile.get('name'),
                    'email': profile.get('email'),
                    'pages': pages,
                    'synced_at': datetime.utcnow()
                }
                
            elif platform == 'instagram':
                profile = await self.get_instagram_profile(token)
                
                profile_doc = {
                    'user_id': user_id,
                    'platform': 'instagram',
                    'profile_id': profile.get('id'),
                    'username': profile.get('username'),
                    'followers_count': profile.get('followers_count', 0),
                    'follows_count': profile.get('follows_count', 0),
                    'media_count': profile.get('media_count', 0),
                    'synced_at': datetime.utcnow()
                }
            
            await self.db.meta_profiles.update_one(
                {'user_id': user_id, 'platform': platform},
                {'$set': profile_doc},
                upsert=True
            )
            
            logging.info(f"✅ Synced {platform} profile for user {user_id}")
            
        except Exception as e:
            logging.error(f"Error syncing {platform} profile: {e}")
    
    async def get_facebook_profile(self, token: str) -> Dict:
        """Get Facebook user profile"""
        
        url = f'{self.graph_api_base}/me'
        params = {
            'access_token': token,
            'fields': 'id,name,email'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception("Failed to fetch Facebook profile")
                return await response.json()
    
    async def get_facebook_pages(self, token: str) -> List[Dict]:
        """Get user's Facebook pages"""
        
        url = f'{self.graph_api_base}/me/accounts'
        params = {
            'access_token': token,
            'fields': 'id,name,access_token,followers_count'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
                return data.get('data', [])
    
    async def get_instagram_profile(self, token: str) -> Dict:
        """Get Instagram Business Account profile"""
        
        # First get Facebook pages
        pages = await self.get_facebook_pages(token)
        
        if not pages:
            return {}
        
        # Get Instagram account linked to first page
        page_id = pages[0]['id']
        page_token = pages[0]['access_token']
        
        url = f'{self.graph_api_base}/{page_id}'
        params = {
            'access_token': page_token,
            'fields': 'instagram_business_account'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    return {}
                
                data = await response.json()
                ig_account_id = data.get('instagram_business_account', {}).get('id')
                
                if not ig_account_id:
                    return {}
                
                # Get Instagram profile details
                ig_url = f'{self.graph_api_base}/{ig_account_id}'
                ig_params = {
                    'access_token': page_token,
                    'fields': 'id,username,followers_count,follows_count,media_count'
                }
                
                async with session.get(ig_url, params=ig_params) as ig_response:
                    if ig_response.status != 200:
                        return {}
                    return await ig_response.json()
    
    async def post_to_facebook(self, user_id: str, page_id: str, content: str, link: Optional[str] = None) -> Dict:
        """Post content to Facebook page"""
        
        connection = await self.db.meta_connections.find_one({
            'user_id': user_id,
            'platform': 'facebook'
        })
        
        if not connection:
            raise Exception("Facebook not connected")
        
        # Get page access token
        profile = await self.db.meta_profiles.find_one({
            'user_id': user_id,
            'platform': 'facebook'
        })
        
        page_token = None
        for page in profile.get('pages', []):
            if page['id'] == page_id:
                page_token = page['access_token']
                break
        
        if not page_token:
            raise Exception("Page not found")
        
        # Create post
        url = f'{self.graph_api_base}/{page_id}/feed'
        
        payload = {
            'access_token': page_token,
            'message': content
        }
        
        if link:
            payload['link'] = link
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=payload) as response:
                if response.status not in [200, 201]:
                    error_text = await response.text()
                    raise Exception(f"Failed to post: {error_text}")
                
                result = await response.json()
                
                # Log post
                await self.db.meta_posts.insert_one({
                    'user_id': user_id,
                    'platform': 'facebook',
                    'page_id': page_id,
                    'post_id': result.get('id'),
                    'content': content,
                    'link': link,
                    'posted_at': datetime.utcnow()
                })
                
                return result
    
    async def schedule_post(self, user_id: str, platform: str, page_id: str, 
                          content: str, scheduled_time: datetime, link: Optional[str] = None) -> Dict:
        """Schedule a post for later"""
        
        post_doc = {
            'user_id': user_id,
            'platform': platform,
            'page_id': page_id,
            'content': content,
            'link': link,
            'scheduled_time': scheduled_time,
            'status': 'scheduled',
            'created_at': datetime.utcnow()
        }
        
        await self.db.scheduled_posts.insert_one(post_doc)
        
        logging.info(f"📅 Scheduled {platform} post for {scheduled_time}")
        
        return post_doc
    
    async def get_realtime_stats(self, user_id: str, platform: str) -> Dict:
        """Get real-time statistics from Meta platforms"""
        
        token = await self.get_valid_token(user_id, platform)
        
        if not token:
            return {'error': f'{platform} not connected'}
        
        try:
            if platform == 'facebook':
                profile = await self.db.meta_profiles.find_one({
                    'user_id': user_id,
                    'platform': 'facebook'
                })
                
                pages_data = []
                for page in profile.get('pages', []):
                    page_stats = await self.get_facebook_page_stats(page['access_token'], page['id'])
                    pages_data.append({
                        'name': page['name'],
                        'followers': page.get('followers_count', 0),
                        **page_stats
                    })
                
                return {
                    'platform': 'facebook',
                    'pages': pages_data,
                    'total_followers': sum(p.get('followers', 0) for p in pages_data)
                }
                
            elif platform == 'instagram':
                profile = await self.db.meta_profiles.find_one({
                    'user_id': user_id,
                    'platform': 'instagram'
                })
                
                return {
                    'platform': 'instagram',
                    'username': profile.get('username'),
                    'followers': profile.get('followers_count', 0),
                    'following': profile.get('follows_count', 0),
                    'posts': profile.get('media_count', 0)
                }
        
        except Exception as e:
            logging.error(f"Error fetching {platform} stats: {e}")
            return {'error': str(e)}
    
    async def get_facebook_page_stats(self, page_token: str, page_id: str) -> Dict:
        """Get Facebook page statistics"""
        
        url = f'{self.graph_api_base}/{page_id}/insights'
        params = {
            'access_token': page_token,
            'metric': 'page_engaged_users,page_impressions,page_post_engagements',
            'period': 'day'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        return {}
                    
                    data = await response.json()
                    
                    stats = {}
                    for item in data.get('data', []):
                        metric_name = item['name']
                        values = item.get('values', [])
                        if values:
                            stats[metric_name] = values[-1].get('value', 0)
                    
                    return {
                        'engaged_users': stats.get('page_engaged_users', 0),
                        'impressions': stats.get('page_impressions', 0),
                        'post_engagements': stats.get('page_post_engagements', 0)
                    }
        except Exception as e:
            logging.error(f"Error fetching page stats: {e}")
            return {}
    
    async def background_sync_all_users(self):
        """Background task to sync all connected Meta accounts"""
        
        try:
            connections = await self.db.meta_connections.find({}).to_list(1000)
            
            for conn in connections:
                try:
                    token = conn['access_token']
                    await self.sync_user_profile(conn['user_id'], token, conn['platform'])
                except Exception as e:
                    logging.error(f"Error syncing Meta user {conn['user_id']}: {e}")
            
            logging.info(f"✅ Synced {len(connections)} Meta accounts")
            
        except Exception as e:
            logging.error(f"Background Meta sync error: {e}")


# Singleton instance
_meta_service_instance = None

def get_meta_service(db):
    global _meta_service_instance
    if _meta_service_instance is None:
        _meta_service_instance = MetaService(db)
    return _meta_service_instance
