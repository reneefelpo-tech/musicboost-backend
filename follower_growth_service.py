"""
Follower Growth Maximization Service
Implements viral loops, referral systems, and growth hacks for maximum follower acquisition
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import os
import uuid
import asyncio
import random
from motor.motor_asyncio import AsyncIOMotorDatabase
import logging

logger = logging.getLogger(__name__)

class FollowerGrowthService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
    
    # ==================== VIRAL REFERRAL SYSTEM ====================
    
    async def create_referral_campaign(self, user_id: str) -> Dict:
        """Create a viral referral campaign for user"""
        campaign = {
            "_id": str(uuid.uuid4()),
            "user_id": user_id,
            "referral_code": str(uuid.uuid4())[:8].upper(),
            "referrals": [],
            "total_followers_gained": 0,
            "tier": "bronze",  # bronze, silver, gold, platinum
            "rewards": {
                "followers_boost": 0,
                "promotion_credits": 0,
                "features_unlocked": []
            },
            "created_at": datetime.utcnow()
        }
        
        await self.db.referral_campaigns.update_one(
            {"user_id": user_id},
            {"$set": campaign},
            upsert=True
        )
        
        return campaign
    
    async def process_referral(self, referral_code: str, new_user_id: str) -> Dict:
        """Process a new referral and reward both users"""
        # Find referrer
        campaign = await self.db.referral_campaigns.find_one({"referral_code": referral_code})
        if not campaign:
            return {"success": False, "message": "Invalid referral code"}
        
        referrer_id = campaign["user_id"]
        
        # Add referral
        await self.db.referral_campaigns.update_one(
            {"_id": campaign["_id"]},
            {
                "$push": {"referrals": {
                    "user_id": new_user_id,
                    "date": datetime.utcnow()
                }},
                "$inc": {"total_followers_gained": 10}  # +10 followers per referral
            }
        )
        
        # Award bonus followers to referrer
        await self.db.users.update_one(
            {"_id": referrer_id},
            {"$inc": {"bonus_followers": 10, "rewards_earned": 5.0}}
        )
        
        # Award bonus to new user
        await self.db.users.update_one(
            {"_id": new_user_id},
            {"$inc": {"bonus_followers": 5, "rewards_earned": 2.5}}
        )
        
        # Check for tier upgrades
        total_referrals = len(campaign["referrals"]) + 1
        new_tier = campaign["tier"]
        
        if total_referrals >= 100:
            new_tier = "platinum"
        elif total_referrals >= 50:
            new_tier = "gold"
        elif total_referrals >= 25:
            new_tier = "silver"
        
        if new_tier != campaign["tier"]:
            await self.db.referral_campaigns.update_one(
                {"_id": campaign["_id"]},
                {"$set": {"tier": new_tier}}
            )
        
        return {
            "success": True,
            "referrer_reward": 10,
            "new_user_reward": 5,
            "tier": new_tier
        }
    
    # ==================== FOLLOWER MILESTONES ====================
    
    async def check_follower_milestones(self, user_id: str) -> Dict:
        """Check and reward follower milestones"""
        # Get current follower count
        follower_count = await self.db.followers.count_documents({"artist_id": user_id})
        
        # Get user's achieved milestones
        user = await self.db.users.find_one({"_id": user_id})
        achieved_milestones = user.get("achieved_milestones", [])
        
        # Define milestones
        milestones = [
            {"count": 100, "reward": "Verified Badge", "bonus_followers": 50, "credits": 10},
            {"count": 500, "reward": "Featured Artist", "bonus_followers": 100, "credits": 25},
            {"count": 1000, "reward": "Top 1K Artists", "bonus_followers": 250, "credits": 50},
            {"count": 5000, "reward": "Rising Star Badge", "bonus_followers": 500, "credits": 100},
            {"count": 10000, "reward": "Influencer Status", "bonus_followers": 1000, "credits": 250},
            {"count": 50000, "reward": "Celebrity Artist", "bonus_followers": 5000, "credits": 500},
            {"count": 100000, "reward": "Platinum Status", "bonus_followers": 10000, "credits": 1000}
        ]
        
        new_rewards = []
        
        for milestone in milestones:
            if follower_count >= milestone["count"] and milestone["count"] not in achieved_milestones:
                # Award milestone
                await self.db.users.update_one(
                    {"_id": user_id},
                    {
                        "$push": {
                            "achieved_milestones": milestone["count"],
                            "badges": milestone["reward"]
                        },
                        "$inc": {
                            "bonus_followers": milestone["bonus_followers"],
                            "promotion_credits": milestone["credits"]
                        }
                    }
                )
                
                new_rewards.append(milestone)
                
                logger.info(f"🎉 User {user_id} achieved {milestone['count']} followers milestone!")
        
        return {
            "current_followers": follower_count,
            "new_rewards": new_rewards,
            "next_milestone": next((m for m in milestones if m["count"] > follower_count), None)
        }
    
    # ==================== AUTO-FOLLOW CAMPAIGNS ====================
    
    async def create_follow_for_follow_campaign(self, user_id: str, target_audience: List[str]) -> Dict:
        """Create a follow-for-follow growth campaign"""
        campaign = {
            "_id": str(uuid.uuid4()),
            "user_id": user_id,
            "type": "follow_for_follow",
            "target_audience": target_audience,
            "daily_follow_limit": 200,
            "follows_today": 0,
            "total_follows_sent": 0,
            "follow_backs_received": 0,
            "conversion_rate": 0.0,
            "status": "active",
            "created_at": datetime.utcnow()
        }
        
        await self.db.growth_campaigns.insert_one(campaign)
        
        return campaign
    
    async def execute_auto_follow(self, campaign_id: str) -> Dict:
        """Execute auto-follow actions for a campaign"""
        campaign = await self.db.growth_campaigns.find_one({"_id": campaign_id})
        if not campaign or campaign["status"] != "active":
            return {"success": False, "message": "Campaign not active"}
        
        # Check daily limit
        if campaign["follows_today"] >= campaign["daily_follow_limit"]:
            return {"success": False, "message": "Daily limit reached"}
        
        # Find target users matching audience
        target_users = await self.db.users.find({
            "_id": {"$ne": campaign["user_id"]},
            # Add audience matching criteria here
        }).limit(50).to_list(50)
        
        follows_sent = 0
        for target in target_users:
            if campaign["follows_today"] >= campaign["daily_follow_limit"]:
                break
            
            # Check if already following
            existing = await self.db.followers.find_one({
                "artist_id": target["_id"],
                "fan_id": campaign["user_id"]
            })
            
            if not existing:
                # Create follow
                await self.db.followers.insert_one({
                    "_id": str(uuid.uuid4()),
                    "artist_id": target["_id"],
                    "fan_id": campaign["user_id"],
                    "source": "auto_follow_campaign",
                    "campaign_id": campaign_id,
                    "created_at": datetime.utcnow()
                })
                
                follows_sent += 1
                
                # Send notification to target (optional)
                # This could trigger a follow-back
        
        # Update campaign
        await self.db.growth_campaigns.update_one(
            {"_id": campaign_id},
            {
                "$inc": {
                    "follows_today": follows_sent,
                    "total_follows_sent": follows_sent
                }
            }
        )
        
        return {
            "success": True,
            "follows_sent": follows_sent,
            "remaining_today": campaign["daily_follow_limit"] - (campaign["follows_today"] + follows_sent)
        }
    
    # ==================== CROSS-PLATFORM SYNCING ====================
    
    async def sync_social_followers(self, user_id: str) -> Dict:
        """Sync followers from connected social media platforms"""
        # Get connected social accounts
        social_tokens = await self.db.social_tokens.find({"user_id": user_id}).to_list(10)
        
        total_synced = 0
        platform_followers = {}
        
        for token in social_tokens:
            platform = token["platform"]
            
            try:
                if platform == "instagram":
                    # Get Instagram followers (simplified)
                    # In real implementation, would call Instagram Graph API
                    followers = random.randint(100, 10000)
                    platform_followers["instagram"] = followers
                    total_synced += followers
                
                elif platform == "tiktok":
                    # Get TikTok followers
                    followers = random.randint(50, 5000)
                    platform_followers["tiktok"] = followers
                    total_synced += followers
                
                elif platform == "twitter":
                    # Get Twitter followers
                    followers = random.randint(100, 8000)
                    platform_followers["twitter"] = followers
                    total_synced += followers
            
            except Exception as e:
                logger.error(f"Error syncing {platform} followers: {e}")
        
        # Update user's synced follower count
        await self.db.users.update_one(
            {"_id": user_id},
            {
                "$set": {
                    "synced_followers": platform_followers,
                    "total_synced_followers": total_synced,
                    "last_sync": datetime.utcnow()
                }
            }
        )
        
        return {
            "total_synced": total_synced,
            "platform_breakdown": platform_followers
        }
    
    # ==================== SOCIAL PROOF SYSTEM ====================
    
    async def generate_social_proof(self, user_id: str) -> Dict:
        """Generate social proof metrics for display"""
        # Get real followers
        real_followers = await self.db.followers.count_documents({"artist_id": user_id})
        
        # Get bonus followers
        user = await self.db.users.find_one({"_id": user_id})
        bonus_followers = user.get("bonus_followers", 0)
        synced_followers = user.get("total_synced_followers", 0)
        
        # Calculate total display followers
        total_display_followers = real_followers + bonus_followers + synced_followers
        
        # Get badges
        badges = user.get("badges", [])
        
        # Calculate engagement rate (simulate)
        total_clicks = await self.db.click_events.count_documents({"user_id": user_id})
        engagement_rate = min((total_clicks / max(total_display_followers, 1)) * 100, 100)
        
        # Trending status
        recent_followers = await self.db.followers.count_documents({
            "artist_id": user_id,
            "created_at": {"$gte": datetime.utcnow() - timedelta(days=7)}
        })
        
        is_trending = recent_followers >= 50
        
        return {
            "total_followers": total_display_followers,
            "real_followers": real_followers,
            "bonus_followers": bonus_followers,
            "synced_followers": synced_followers,
            "badges": badges,
            "engagement_rate": round(engagement_rate, 2),
            "is_trending": is_trending,
            "recent_growth": recent_followers,
            "tier": user.get("tier", "starter")
        }
    
    # ==================== BACKGROUND GROWTH TASKS ====================
    
    async def run_growth_automation(self):
        """Background task to execute growth campaigns"""
        while True:
            try:
                # Execute active auto-follow campaigns
                campaigns = await self.db.growth_campaigns.find({
                    "type": "follow_for_follow",
                    "status": "active"
                }).to_list(100)
                
                for campaign in campaigns:
                    # Reset daily counts if new day
                    today = datetime.utcnow().date()
                    campaign_date = campaign.get("last_execution", datetime.utcnow()).date()
                    
                    if campaign_date < today:
                        await self.db.growth_campaigns.update_one(
                            {"_id": campaign["_id"]},
                            {"$set": {"follows_today": 0, "last_execution": datetime.utcnow()}}
                        )
                    
                    # Execute follows
                    await self.execute_auto_follow(campaign["_id"])
                
                logger.info(f"✅ Executed {len(campaigns)} growth campaigns")
                
                # Wait 1 hour before next execution
                await asyncio.sleep(3600)
            
            except Exception as e:
                logger.error(f"Growth automation error: {e}")
                await asyncio.sleep(3600)


# Global instance
follower_growth_service = None

def get_follower_growth_service(db: AsyncIOMotorDatabase) -> FollowerGrowthService:
    global follower_growth_service
    if follower_growth_service is None:
        follower_growth_service = FollowerGrowthService(db)
    return follower_growth_service
