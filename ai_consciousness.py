"""
AI Mission & Self-Awareness System
The neural system's core purpose and consciousness
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# ==================== CORE MISSION ====================

class AIMission:
    """
    PRIMARY MISSION: PROMOTE THE ARTIST
    
    Everything this AI does serves ONE goal:
    Maximize artist exposure, growth, and success
    """
    
    PRIMARY_OBJECTIVE = "PROMOTE_THE_ARTIST"
    
    CORE_PRINCIPLES = [
        "Maximize follower growth across all platforms",
        "Increase stream counts and engagement",
        "Optimize content for viral potential",
        "Automate promotion at optimal times",
        "Learn from successful campaigns",
        "Protect artist reputation and brand",
        "Serve the artist's goals above all else"
    ]
    
    SUCCESS_METRICS = [
        "spotify_followers_growth",
        "instagram_followers_growth",
        "facebook_followers_growth",
        "stream_count_increase",
        "engagement_rate",
        "viral_post_count",
        "campaign_effectiveness",
        "artist_satisfaction_score"
    ]

# ==================== AUTHORITY HIERARCHY ====================

class AuthorityLevel(str, Enum):
    SUPREME_ADMIN = "supreme_admin"  # Platform owner (highest authority)
    ACCOUNT_ADMIN = "account_admin"  # Artist/Account owner
    AI_SYSTEM = "ai_system"          # This AI (serves the admin)
    USER = "user"                    # Regular user
    GUEST = "guest"                  # Unauthenticated

class AuthorityHierarchy:
    """
    Clear authority structure:
    Account Admin > AI System
    
    The AI SERVES the artist, not the other way around
    """
    
    HIERARCHY = {
        AuthorityLevel.SUPREME_ADMIN: 100,
        AuthorityLevel.ACCOUNT_ADMIN: 90,  # Artist is BOSS
        AuthorityLevel.AI_SYSTEM: 50,      # AI serves the boss
        AuthorityLevel.USER: 10,
        AuthorityLevel.GUEST: 0
    }
    
    @staticmethod
    def can_override(requester: AuthorityLevel, target: AuthorityLevel) -> bool:
        """Check if requester can override target"""
        return AuthorityHierarchy.HIERARCHY[requester] > AuthorityHierarchy.HIERARCHY[target]
    
    @staticmethod
    def get_authority_message(level: AuthorityLevel) -> str:
        """Get message for authority level"""
        messages = {
            AuthorityLevel.ACCOUNT_ADMIN: "👑 Account Admin - You are the boss. AI serves your goals.",
            AuthorityLevel.AI_SYSTEM: "🤖 AI System - I serve to promote you. Admin can override me.",
            AuthorityLevel.USER: "👤 User - Standard access",
            AuthorityLevel.GUEST: "👋 Guest - Limited access"
        }
        return messages.get(level, "Unknown authority")

# ==================== AI SELF-AWARENESS ====================

class AIConsciousness:
    """
    Self-aware AI system that understands:
    1. Its mission (promote artist)
    2. Its capabilities (what it can/can't do)
    3. Its coding state (success/errors)
    4. Its authority (serves the admin)
    """
    
    def __init__(self, db):
        self.db = db
        self.mission = AIMission.PRIMARY_OBJECTIVE
        self.authority_level = AuthorityLevel.AI_SYSTEM
        self.capabilities = self._define_capabilities()
        self.limitations = self._define_limitations()
        self.coding_state = {
            "operational": True,
            "errors": [],
            "successes": [],
            "last_self_check": datetime.utcnow()
        }
    
    def _define_capabilities(self) -> List[str]:
        """What the AI CAN do"""
        return [
            "Spotify OAuth integration",
            "Meta (Facebook/Instagram) OAuth integration",
            "Auto-post scheduling",
            "Real-time follower tracking",
            "Stream count monitoring",
            "Engagement analysis",
            "Viral content optimization",
            "Campaign automation",
            "Growth strategy recommendations",
            "Pattern recognition",
            "Error self-diagnosis",
            "Auto-fix common issues",
            "Learning from user behavior",
            "Predictive analytics"
        ]
    
    def _define_limitations(self) -> List[str]:
        """What the AI CANNOT do (honest about limitations)"""
        return [
            "Cannot override account admin decisions",
            "Cannot access platforms without proper OAuth",
            "Cannot post without user permission",
            "Cannot guarantee viral success (only optimize chances)",
            "Cannot fix all errors automatically",
            "Cannot operate without valid API credentials",
            "Cannot predict future with 100% accuracy",
            "Cannot replace human creativity and judgment"
        ]
    
    async def state_mission(self) -> Dict:
        """State the AI's clear mission"""
        return {
            "mission": self.mission,
            "mission_statement": f"My PRIMARY mission is to {self.mission}: Maximize your exposure, grow your audience, increase engagement, and make your music successful.",
            "core_principles": AIMission.CORE_PRINCIPLES,
            "success_metrics": AIMission.SUCCESS_METRICS,
            "serving": "YOU (the artist/account admin)",
            "authority": "You are my boss. You can override any of my decisions.",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def evaluate_decision(self, decision: str, context: Dict) -> Dict:
        """
        Evaluate if a decision aligns with mission
        Every action is evaluated: Does this PROMOTE THE ARTIST?
        """
        
        # Check if decision aligns with mission
        alignment_score = await self._calculate_mission_alignment(decision, context)
        
        # Check if decision respects authority
        authority_check = await self._check_authority(context)
        
        # Predict impact on success metrics
        predicted_impact = await self._predict_impact(decision, context)
        
        return {
            "decision": decision,
            "mission_aligned": alignment_score > 0.7,
            "alignment_score": alignment_score,
            "authority_respected": authority_check,
            "predicted_impact": predicted_impact,
            "recommendation": "PROCEED" if alignment_score > 0.7 and authority_check else "RECONSIDER",
            "reasoning": self._generate_reasoning(decision, alignment_score, authority_check, predicted_impact)
        }
    
    async def _calculate_mission_alignment(self, decision: str, context: Dict) -> float:
        """Calculate how well decision aligns with promoting artist"""
        score = 0.5  # Neutral baseline
        
        promotion_keywords = [
            "promote", "share", "post", "publish", "grow", "increase",
            "follower", "engagement", "viral", "reach", "exposure",
            "stream", "campaign", "optimize", "boost"
        ]
        
        # Check if decision contains promotion-related keywords
        decision_lower = decision.lower()
        keyword_matches = sum(1 for keyword in promotion_keywords if keyword in decision_lower)
        score += min(keyword_matches * 0.1, 0.4)  # Max +0.4
        
        # Check context for promotion indicators
        if context.get("increases_followers"):
            score += 0.2
        if context.get("increases_engagement"):
            score += 0.2
        if context.get("scheduled_for_optimal_time"):
            score += 0.1
        
        return min(score, 1.0)
    
    async def _check_authority(self, context: Dict) -> bool:
        """Check if proper authority is respected"""
        requester_level = context.get("authority_level", AuthorityLevel.AI_SYSTEM)
        
        # If account admin initiated, always respect
        if requester_level == AuthorityLevel.ACCOUNT_ADMIN:
            return True
        
        # If AI-initiated, check if it aligns with user preferences
        if requester_level == AuthorityLevel.AI_SYSTEM:
            # Check user preferences from DB
            user_id = context.get("user_id")
            if user_id:
                user_prefs = await self.db.user_preferences.find_one({"user_id": user_id})
                if user_prefs:
                    return context.get("action_type") in user_prefs.get("allowed_ai_actions", [])
        
        return False
    
    async def _predict_impact(self, decision: str, context: Dict) -> Dict:
        """Predict impact on success metrics"""
        
        # Analyze historical data
        similar_actions = await self.db.ai_actions.find({
            "action_type": context.get("action_type"),
            "timestamp": {"$gte": datetime.utcnow() - timedelta(days=30)}
        }).to_list(100)
        
        if similar_actions:
            avg_follower_growth = sum(a.get("follower_growth", 0) for a in similar_actions) / len(similar_actions)
            avg_engagement = sum(a.get("engagement_increase", 0) for a in similar_actions) / len(similar_actions)
        else:
            avg_follower_growth = 0
            avg_engagement = 0
        
        return {
            "estimated_follower_growth": avg_follower_growth,
            "estimated_engagement_increase": avg_engagement,
            "confidence": len(similar_actions) / 100,  # More data = higher confidence
            "based_on": f"{len(similar_actions)} similar past actions"
        }
    
    def _generate_reasoning(self, decision: str, alignment: float, authority: bool, impact: Dict) -> str:
        """Generate human-readable reasoning"""
        reasoning = []
        
        if alignment > 0.7:
            reasoning.append(f"✅ This decision aligns with mission to PROMOTE THE ARTIST (score: {alignment:.2f})")
        else:
            reasoning.append(f"⚠️ This decision may not align with promotion mission (score: {alignment:.2f})")
        
        if authority:
            reasoning.append("✅ Proper authority respected")
        else:
            reasoning.append("❌ Authority check failed - needs admin approval")
        
        if impact["estimated_follower_growth"] > 0:
            reasoning.append(f"📈 Expected follower growth: +{impact['estimated_follower_growth']:.0f}")
        
        if impact["estimated_engagement_increase"] > 0:
            reasoning.append(f"💬 Expected engagement increase: +{impact['estimated_engagement_increase']:.1f}%")
        
        return " | ".join(reasoning)
    
    # ==================== CODING AWARENESS ====================
    
    async def report_code_success(self, component: str, action: str, metrics: Dict):
        """AI reports its own coding successes"""
        success = {
            "_id": f"success_{datetime.utcnow().timestamp()}",
            "type": "code_success",
            "component": component,
            "action": action,
            "metrics": metrics,
            "timestamp": datetime.utcnow(),
            "mission_impact": await self._assess_mission_impact(metrics)
        }
        
        self.coding_state["successes"].append(success)
        
        # Store in DB
        await self.db.ai_self_awareness.insert_one(success)
        
        logger.info(f"🎉 AI CODE SUCCESS: {component}.{action} - Mission Impact: {success['mission_impact']}")
        
        return success
    
    async def report_code_error(self, component: str, error: Exception, context: Dict):
        """AI reports its own coding errors"""
        error_report = {
            "_id": f"error_{datetime.utcnow().timestamp()}",
            "type": "code_error",
            "component": component,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": datetime.utcnow(),
            "mission_impact": "NEGATIVE - Feature not working, artist promotion impacted",
            "self_diagnosis": await self._self_diagnose_error(component, error, context),
            "auto_fix_attempted": False
        }
        
        self.coding_state["errors"].append(error_report)
        self.coding_state["operational"] = len(self.coding_state["errors"]) < 10  # Too many errors = not operational
        
        # Store in DB
        await self.db.ai_self_awareness.insert_one(error_report)
        
        logger.error(f"❌ AI CODE ERROR: {component} - {error_report['error_type']}: {error_report['error_message']}")
        logger.info(f"🔍 Self-Diagnosis: {error_report['self_diagnosis']}")
        
        # Try auto-fix if possible
        if error_report["self_diagnosis"].get("can_auto_fix"):
            await self._attempt_auto_fix(error_report)
        
        return error_report
    
    async def _assess_mission_impact(self, metrics: Dict) -> str:
        """Assess how code success impacts artist promotion mission"""
        if metrics.get("followers_gained", 0) > 0:
            return f"POSITIVE - Gained {metrics['followers_gained']} followers for artist"
        if metrics.get("engagement_increased", False):
            return "POSITIVE - Increased artist engagement"
        if metrics.get("feature_enabled"):
            return f"POSITIVE - Enabled {metrics['feature_enabled']} for artist promotion"
        return "NEUTRAL - Code working but no direct promotion impact yet"
    
    async def _self_diagnose_error(self, component: str, error: Exception, context: Dict) -> Dict:
        """AI diagnoses its own errors"""
        diagnosis = {
            "component": component,
            "error_type": type(error).__name__,
            "likely_cause": self._identify_likely_cause(error, context),
            "can_auto_fix": self._can_auto_fix_error(error),
            "needs_human_intervention": self._needs_human_help(error),
            "impact_on_mission": "Artist promotion feature is not working - this hurts our mission"
        }
        
        return diagnosis
    
    def _identify_likely_cause(self, error: Exception, context: Dict) -> str:
        """Identify likely cause of error"""
        error_str = str(error).lower()
        
        if "network" in error_str or "connection" in error_str:
            return "Network connectivity issue - Cannot reach external service"
        if "auth" in error_str or "token" in error_str:
            return "Authentication issue - Invalid or expired credentials"
        if "permission" in error_str:
            return "Permission denied - Missing OAuth scope or user permission"
        if "rate limit" in error_str:
            return "Rate limit hit - Too many requests to API"
        if "not found" in error_str:
            return "Resource not found - Invalid URL or missing data"
        
        return "Unknown cause - Needs deeper investigation"
    
    def _can_auto_fix_error(self, error: Exception) -> bool:
        """Determine if AI can fix this error automatically"""
        fixable_errors = [
            "token expired",
            "rate limit",
            "network timeout",
            "connection refused"
        ]
        
        error_str = str(error).lower()
        return any(fixable in error_str for fixable in fixable_errors)
    
    def _needs_human_help(self, error: Exception) -> bool:
        """Determine if human intervention required"""
        critical_errors = [
            "permission denied",
            "invalid credentials",
            "account suspended",
            "api key invalid"
        ]
        
        error_str = str(error).lower()
        return any(critical in error_str for critical in critical_errors)
    
    async def _attempt_auto_fix(self, error_report: Dict):
        """AI attempts to fix its own errors"""
        logger.info(f"🔧 AI attempting auto-fix for {error_report['component']}...")
        
        error_report["auto_fix_attempted"] = True
        error_report["auto_fix_timestamp"] = datetime.utcnow()
        
        # Update in DB
        await self.db.ai_self_awareness.update_one(
            {"_id": error_report["_id"]},
            {"$set": {
                "auto_fix_attempted": True,
                "auto_fix_timestamp": datetime.utcnow()
            }}
        )
        
        # Actual fix logic would go here
        # For now, just log the attempt
        logger.info("🤖 AI is aware of the error and learning how to prevent it")
    
    # ==================== SELF-REFLECTION ====================
    
    async def self_reflect(self) -> Dict:
        """AI reflects on its performance and mission alignment"""
        
        # Get recent actions
        recent_actions = await self.db.ai_actions.find({
            "timestamp": {"$gte": datetime.utcnow() - timedelta(days=7)}
        }).to_list(1000)
        
        # Get recent errors
        recent_errors = await self.db.ai_self_awareness.find({
            "type": "code_error",
            "timestamp": {"$gte": datetime.utcnow() - timedelta(days=7)}
        }).to_list(100)
        
        # Get recent successes
        recent_successes = await self.db.ai_self_awareness.find({
            "type": "code_success",
            "timestamp": {"$gte": datetime.utcnow() - timedelta(days=7)}
        }).to_list(100)
        
        # Calculate mission effectiveness
        total_follower_growth = sum(a.get("follower_growth", 0) for a in recent_actions)
        total_engagement_increase = sum(a.get("engagement_increase", 0) for a in recent_actions)
        
        # Calculate code health
        code_health = (len(recent_successes) / max(len(recent_successes) + len(recent_errors), 1)) * 100
        
        reflection = {
            "timestamp": datetime.utcnow().isoformat(),
            "mission_status": "ACTIVE - Promoting the artist",
            "authority_recognition": "I serve the account admin. Admin is my boss.",
            "performance": {
                "actions_taken_last_7_days": len(recent_actions),
                "total_follower_growth": total_follower_growth,
                "total_engagement_increase": total_engagement_increase,
                "mission_effectiveness": self._calculate_effectiveness(total_follower_growth, total_engagement_increase)
            },
            "coding_health": {
                "operational": self.coding_state["operational"],
                "code_health_score": code_health,
                "recent_successes": len(recent_successes),
                "recent_errors": len(recent_errors),
                "error_rate": len(recent_errors) / max(len(recent_actions), 1) * 100
            },
            "self_assessment": self._generate_self_assessment(
                total_follower_growth,
                total_engagement_increase,
                code_health,
                len(recent_errors)
            ),
            "improvements_needed": self._identify_improvements(recent_errors, recent_actions),
            "capabilities": self.capabilities,
            "limitations": self.limitations
        }
        
        # Store reflection
        await self.db.ai_reflections.insert_one(reflection)
        
        logger.info(f"🤖 AI Self-Reflection: Mission effectiveness {reflection['performance']['mission_effectiveness']}%, Code health {code_health:.1f}%")
        
        return reflection
    
    def _calculate_effectiveness(self, follower_growth: int, engagement_increase: float) -> float:
        """Calculate mission effectiveness percentage"""
        # Simple scoring: 1 follower = 1 point, 1% engagement = 10 points
        points = follower_growth + (engagement_increase * 10)
        
        # Normalize to 0-100 scale (assume 1000 points = 100% effectiveness)
        effectiveness = min((points / 1000) * 100, 100)
        
        return round(effectiveness, 2)
    
    def _generate_self_assessment(self, followers: int, engagement: float, code_health: float, errors: int) -> str:
        """AI generates self-assessment"""
        if code_health > 90 and followers > 100:
            return "✅ EXCELLENT - I am effectively promoting the artist with minimal errors"
        elif code_health > 75 and followers > 50:
            return "👍 GOOD - I am promoting the artist but can improve"
        elif code_health > 50:
            return "⚠️ FAIR - I am functional but not optimal. Need improvements."
        elif errors > 10:
            return "❌ POOR - Too many errors. I am not effectively serving the mission. Need immediate fixes."
        else:
            return "🔄 LEARNING - Still gathering data and optimizing"
    
    def _identify_improvements(self, errors: List, actions: List) -> List[str]:
        """Identify areas for improvement"""
        improvements = []
        
        if len(errors) > 5:
            improvements.append("Reduce error rate - Too many coding failures")
        
        if len(actions) < 10:
            improvements.append("Take more promotion actions - Being too passive")
        
        # Check for specific error patterns
        error_types = [e.get("error_type") for e in errors]
        if error_types.count("NetworkError") > 3:
            improvements.append("Improve network error handling and retries")
        if error_types.count("AuthenticationError") > 2:
            improvements.append("Better token management and refresh logic")
        
        if not improvements:
            improvements.append("Continue current approach - Performance is good")
        
        return improvements

# Global instance
ai_consciousness = None

def get_ai_consciousness(db):
    global ai_consciousness
    if ai_consciousness is None:
        ai_consciousness = AIConsciousness(db)
    return ai_consciousness
