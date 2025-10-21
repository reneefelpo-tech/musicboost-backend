"""
Neural Communication Language
A structured protocol for app-to-AI communication
Enables faster diagnosis, self-healing, and pattern recognition
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)

# ==================== ERROR TAXONOMY ====================

class ErrorSeverity(str, Enum):
    CRITICAL = "critical"  # App unusable
    HIGH = "high"         # Feature broken
    MEDIUM = "medium"     # Degraded experience
    LOW = "low"           # Minor issue
    INFO = "info"         # Informational

class ErrorCategory(str, Enum):
    AUTH = "authentication"
    NETWORK = "network"
    API = "api_integration"
    UI = "user_interface"
    DATA = "data_integrity"
    PERFORMANCE = "performance"
    OAUTH = "oauth_flow"
    PERMISSION = "permissions"
    PLATFORM = "platform_specific"

class ErrorCode(str, Enum):
    # Authentication Errors (AUTH_xxx)
    AUTH_TOKEN_EXPIRED = "AUTH_001"
    AUTH_INVALID_CREDENTIALS = "AUTH_002"
    AUTH_SESSION_LOST = "AUTH_003"
    
    # Network Errors (NET_xxx)
    NET_TIMEOUT = "NET_001"
    NET_NO_CONNECTION = "NET_002"
    NET_SLOW_RESPONSE = "NET_003"
    NET_DNS_FAILURE = "NET_004"
    
    # API Integration Errors (API_xxx)
    API_RATE_LIMIT = "API_001"
    API_INVALID_RESPONSE = "API_002"
    API_MISSING_DATA = "API_003"
    API_VERSION_MISMATCH = "API_004"
    
    # OAuth Errors (OAUTH_xxx)
    OAUTH_REDIRECT_FAILED = "OAUTH_001"
    OAUTH_URL_NOT_OPENING = "OAUTH_002"
    OAUTH_CALLBACK_TIMEOUT = "OAUTH_003"
    OAUTH_STATE_MISMATCH = "OAUTH_004"
    OAUTH_TOKEN_EXCHANGE_FAILED = "OAUTH_005"
    
    # UI Errors (UI_xxx)
    UI_RENDER_FAILURE = "UI_001"
    UI_NAVIGATION_BLOCKED = "UI_002"
    UI_COMPONENT_CRASH = "UI_003"
    
    # Platform Errors (PLAT_xxx)
    PLAT_IOS_PERMISSION = "PLAT_001"
    PLAT_ANDROID_PERMISSION = "PLAT_002"
    PLAT_WEB_INCOMPATIBLE = "PLAT_003"

# ==================== DIAGNOSTIC MESSAGES ====================

class DiagnosticMessage:
    """Structured diagnostic message for AI consumption"""
    
    def __init__(
        self,
        code: ErrorCode,
        severity: ErrorSeverity,
        category: ErrorCategory,
        title: str,
        description: str,
        context: Dict[str, Any],
        suggested_fixes: List[str],
        user_impact: str,
        can_auto_fix: bool = False,
        auto_fix_action: Optional[str] = None
    ):
        self.code = code
        self.severity = severity
        self.category = category
        self.title = title
        self.description = description
        self.context = context
        self.suggested_fixes = suggested_fixes
        self.user_impact = user_impact
        self.can_auto_fix = can_auto_fix
        self.auto_fix_action = auto_fix_action
        self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "context": self.context,
            "suggested_fixes": self.suggested_fixes,
            "user_impact": self.user_impact,
            "can_auto_fix": self.can_auto_fix,
            "auto_fix_action": self.auto_fix_action,
            "timestamp": self.timestamp,
            "ai_query": self._generate_ai_query()
        }
    
    def _generate_ai_query(self) -> str:
        """Generate a natural language query for AI agent"""
        return f"[{self.code}] {self.title}: {self.description}. User impact: {self.user_impact}. Context: {json.dumps(self.context, indent=2)}"

# ==================== ERROR KNOWLEDGE BASE ====================

ERROR_KNOWLEDGE_BASE = {
    ErrorCode.OAUTH_URL_NOT_OPENING: {
        "common_causes": [
            "iOS blocking third-party redirects",
            "URL scheme not registered in app.json",
            "Linking.openURL() timing issue",
            "Safari/Chrome not set as default browser",
            "App sandbox restrictions"
        ],
        "diagnostic_steps": [
            "Check Linking.canOpenURL() response",
            "Verify URL format (https:// vs custom scheme)",
            "Test with setTimeout delay",
            "Check iOS version compatibility",
            "Verify app.json configuration"
        ],
        "known_fixes": [
            {
                "fix": "Add 500ms delay before Linking.openURL()",
                "success_rate": 0.75,
                "platform": "ios"
            },
            {
                "fix": "Use WebBrowser.openAuthSessionAsync instead",
                "success_rate": 0.90,
                "platform": "ios"
            },
            {
                "fix": "Add URL scheme to app.json expo.scheme",
                "success_rate": 0.85,
                "platform": "all"
            }
        ]
    },
    ErrorCode.NET_TIMEOUT: {
        "common_causes": [
            "Backend service slow/down",
            "Network congestion",
            "DNS resolution slow",
            "Timeout setting too aggressive"
        ],
        "diagnostic_steps": [
            "Check backend health endpoint",
            "Measure actual response time",
            "Test from different network",
            "Check CDN/proxy status"
        ],
        "known_fixes": [
            {
                "fix": "Increase timeout to 30 seconds",
                "success_rate": 0.60,
                "platform": "all"
            },
            {
                "fix": "Implement retry with exponential backoff",
                "success_rate": 0.85,
                "platform": "all"
            },
            {
                "fix": "Add connection pooling",
                "success_rate": 0.70,
                "platform": "all"
            }
        ]
    },
    ErrorCode.AUTH_TOKEN_EXPIRED: {
        "common_causes": [
            "Token TTL expired",
            "Clock skew between client/server",
            "Token not refreshed automatically"
        ],
        "diagnostic_steps": [
            "Check token expiration timestamp",
            "Compare client/server time",
            "Verify refresh token exists"
        ],
        "known_fixes": [
            {
                "fix": "Implement automatic token refresh before expiry",
                "success_rate": 0.95,
                "platform": "all"
            },
            {
                "fix": "Add token refresh interceptor",
                "success_rate": 0.90,
                "platform": "all"
            }
        ]
    }
}

# ==================== NEURAL DIAGNOSTIC ENGINE ====================

class NeuralDiagnosticEngine:
    """AI-powered diagnostic engine for intelligent error analysis"""
    
    def __init__(self, db):
        self.db = db
        self.pattern_threshold = 3  # Errors before pattern recognized
    
    async def analyze_error(
        self,
        error_code: ErrorCode,
        context: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> DiagnosticMessage:
        """Analyze error and generate diagnostic message"""
        
        # Get knowledge base info
        kb = ERROR_KNOWLEDGE_BASE.get(error_code, {})
        
        # Check for patterns
        pattern_data = await self._check_patterns(error_code, user_id)
        
        # Generate suggested fixes
        suggested_fixes = await self._generate_fixes(error_code, context, pattern_data)
        
        # Determine severity
        severity = await self._calculate_severity(error_code, pattern_data)
        
        # Create diagnostic message
        diagnostic = DiagnosticMessage(
            code=error_code,
            severity=severity,
            category=self._get_category(error_code),
            title=self._get_title(error_code),
            description=self._generate_description(error_code, context, kb),
            context=context,
            suggested_fixes=suggested_fixes,
            user_impact=self._assess_user_impact(error_code, pattern_data),
            can_auto_fix=self._can_auto_fix(error_code),
            auto_fix_action=self._get_auto_fix_action(error_code)
        )
        
        # Store diagnostic
        await self._store_diagnostic(diagnostic, user_id)
        
        return diagnostic
    
    async def _check_patterns(self, error_code: ErrorCode, user_id: Optional[str]) -> Dict:
        """Check if this error is part of a pattern"""
        query = {"code": error_code}
        if user_id:
            query["user_id"] = user_id
        
        # Count recent occurrences
        recent_time = datetime.utcnow() - timedelta(hours=1)
        query["timestamp"] = {"$gte": recent_time}
        
        count = await self.db.diagnostics.count_documents(query)
        
        # Get affected users
        affected_users = await self.db.diagnostics.distinct("user_id", {"code": error_code})
        
        return {
            "is_pattern": count >= self.pattern_threshold,
            "occurrence_count": count,
            "affected_user_count": len(affected_users),
            "first_seen": await self._get_first_occurrence(error_code),
            "last_seen": datetime.utcnow().isoformat()
        }
    
    async def _get_first_occurrence(self, error_code: ErrorCode) -> str:
        """Get timestamp of first occurrence"""
        result = await self.db.diagnostics.find_one(
            {"code": error_code},
            sort=[("timestamp", 1)]
        )
        return result["timestamp"] if result else datetime.utcnow().isoformat()
    
    async def _generate_fixes(
        self,
        error_code: ErrorCode,
        context: Dict,
        pattern_data: Dict
    ) -> List[str]:
        """Generate prioritized list of fixes"""
        kb = ERROR_KNOWLEDGE_BASE.get(error_code, {})
        fixes = []
        
        # Add known fixes sorted by success rate
        known_fixes = kb.get("known_fixes", [])
        for fix_data in sorted(known_fixes, key=lambda x: x["success_rate"], reverse=True):
            platform = context.get("platform", "unknown")
            if fix_data["platform"] == "all" or fix_data["platform"] == platform:
                fixes.append(f"{fix_data['fix']} (success rate: {fix_data['success_rate']*100}%)")
        
        # Add pattern-based fixes
        if pattern_data["is_pattern"]:
            fixes.insert(0, f"⚠️ PATTERN DETECTED: {pattern_data['occurrence_count']} occurrences in last hour affecting {pattern_data['affected_user_count']} users")
        
        # Add diagnostic steps as fallback
        if not fixes:
            fixes = kb.get("diagnostic_steps", ["No known fixes. Manual investigation required."])
        
        return fixes
    
    async def _calculate_severity(self, error_code: ErrorCode, pattern_data: Dict) -> ErrorSeverity:
        """Calculate dynamic severity based on impact"""
        
        # Base severity
        if "AUTH" in error_code or "OAUTH" in error_code:
            base = ErrorSeverity.HIGH
        elif "NET" in error_code:
            base = ErrorSeverity.MEDIUM
        elif "UI" in error_code:
            base = ErrorSeverity.LOW
        else:
            base = ErrorSeverity.MEDIUM
        
        # Escalate if pattern detected
        if pattern_data["is_pattern"]:
            if pattern_data["affected_user_count"] > 5:
                return ErrorSeverity.CRITICAL
            return ErrorSeverity.HIGH
        
        return base
    
    def _get_category(self, error_code: ErrorCode) -> ErrorCategory:
        """Map error code to category"""
        if "AUTH" in error_code:
            return ErrorCategory.AUTH
        elif "NET" in error_code:
            return ErrorCategory.NETWORK
        elif "OAUTH" in error_code:
            return ErrorCategory.OAUTH
        elif "UI" in error_code:
            return ErrorCategory.UI
        elif "API" in error_code:
            return ErrorCategory.API
        elif "PLAT" in error_code:
            return ErrorCategory.PLATFORM
        return ErrorCategory.DATA
    
    def _get_title(self, error_code: ErrorCode) -> str:
        """Get human-readable title"""
        titles = {
            ErrorCode.OAUTH_URL_NOT_OPENING: "OAuth URL Not Opening in Browser",
            ErrorCode.OAUTH_REDIRECT_FAILED: "OAuth Redirect Failed",
            ErrorCode.NET_TIMEOUT: "Network Request Timeout",
            ErrorCode.AUTH_TOKEN_EXPIRED: "Authentication Token Expired",
        }
        return titles.get(error_code, error_code.replace("_", " ").title())
    
    def _generate_description(self, error_code: ErrorCode, context: Dict, kb: Dict) -> str:
        """Generate detailed description"""
        desc = f"Error {error_code} occurred. "
        
        # Add context
        platform = context.get("platform", "unknown")
        desc += f"Platform: {platform}. "
        
        # Add common causes
        causes = kb.get("common_causes", [])
        if causes:
            desc += f"Common causes: {', '.join(causes[:3])}. "
        
        return desc
    
    def _assess_user_impact(self, error_code: ErrorCode, pattern_data: Dict) -> str:
        """Assess impact on user experience"""
        if "OAUTH" in error_code:
            impact = "User cannot connect to Spotify/Meta. Feature blocked."
        elif "AUTH" in error_code:
            impact = "User logged out. Must re-authenticate."
        elif "NET" in error_code:
            impact = "Slow or failed requests. Degraded experience."
        else:
            impact = "Minor inconvenience."
        
        if pattern_data["is_pattern"]:
            impact += f" WIDESPREAD: Affecting {pattern_data['affected_user_count']} users."
        
        return impact
    
    def _can_auto_fix(self, error_code: ErrorCode) -> bool:
        """Determine if error can be auto-fixed"""
        auto_fixable = [
            ErrorCode.AUTH_TOKEN_EXPIRED,
            ErrorCode.NET_TIMEOUT,
        ]
        return error_code in auto_fixable
    
    def _get_auto_fix_action(self, error_code: ErrorCode) -> Optional[str]:
        """Get auto-fix action if available"""
        auto_fixes = {
            ErrorCode.AUTH_TOKEN_EXPIRED: "refresh_auth_token",
            ErrorCode.NET_TIMEOUT: "retry_with_exponential_backoff",
        }
        return auto_fixes.get(error_code)
    
    async def _store_diagnostic(self, diagnostic: DiagnosticMessage, user_id: Optional[str]):
        """Store diagnostic for pattern analysis"""
        doc = diagnostic.to_dict()
        doc["user_id"] = user_id
        doc["timestamp"] = datetime.utcnow()
        await self.db.diagnostics.insert_one(doc)
        
        logger.info(f"🔍 Diagnostic stored: {diagnostic.code} - {diagnostic.title}")
    
    async def get_health_report(self) -> Dict:
        """Generate comprehensive health report for AI agent"""
        now = datetime.utcnow()
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)
        
        # Get error summary
        recent_errors = await self.db.diagnostics.find(
            {"timestamp": {"$gte": last_hour}}
        ).to_list(1000)
        
        # Group by severity
        severity_counts = {}
        for error in recent_errors:
            severity = error.get("severity", "unknown")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Get active patterns
        patterns = await self._detect_active_patterns()
        
        # Get system metrics
        metrics = await self._get_system_metrics()
        
        # Calculate health score
        health_score = self._calculate_health_score(severity_counts, patterns, metrics)
        
        return {
            "timestamp": now.isoformat(),
            "health_score": health_score,
            "status": self._get_status(health_score),
            "recent_errors": {
                "last_hour": len(recent_errors),
                "by_severity": severity_counts
            },
            "active_patterns": patterns,
            "system_metrics": metrics,
            "recommendations": await self._generate_recommendations(patterns, metrics),
            "ai_summary": self._generate_ai_summary(health_score, patterns, recent_errors)
        }
    
    async def _detect_active_patterns(self) -> List[Dict]:
        """Detect active error patterns"""
        last_hour = datetime.utcnow() - timedelta(hours=1)
        
        # Aggregate errors by code
        pipeline = [
            {"$match": {"timestamp": {"$gte": last_hour}}},
            {"$group": {
                "_id": "$code",
                "count": {"$sum": 1},
                "affected_users": {"$addToSet": "$user_id"}
            }},
            {"$match": {"count": {"$gte": self.pattern_threshold}}}
        ]
        
        patterns = await self.db.diagnostics.aggregate(pipeline).to_list(100)
        
        return [
            {
                "error_code": p["_id"],
                "occurrence_count": p["count"],
                "affected_users": len([u for u in p["affected_users"] if u]),
                "severity": "high" if p["count"] > 10 else "medium"
            }
            for p in patterns
        ]
    
    async def _get_system_metrics(self) -> Dict:
        """Get system performance metrics"""
        # This would integrate with actual monitoring
        return {
            "api_response_time_avg_ms": 250,
            "error_rate_percent": 2.5,
            "active_users": 42,
            "database_health": "healthy"
        }
    
    def _calculate_health_score(
        self,
        severity_counts: Dict,
        patterns: List,
        metrics: Dict
    ) -> int:
        """Calculate overall health score (0-100)"""
        score = 100
        
        # Deduct for errors
        score -= severity_counts.get("critical", 0) * 20
        score -= severity_counts.get("high", 0) * 10
        score -= severity_counts.get("medium", 0) * 5
        score -= severity_counts.get("low", 0) * 2
        
        # Deduct for patterns
        score -= len(patterns) * 15
        
        # Deduct for poor metrics
        if metrics.get("error_rate_percent", 0) > 5:
            score -= 10
        
        return max(0, min(100, score))
    
    def _get_status(self, health_score: int) -> str:
        """Get status from health score"""
        if health_score >= 90:
            return "excellent"
        elif health_score >= 75:
            return "good"
        elif health_score >= 50:
            return "degraded"
        else:
            return "critical"
    
    async def _generate_recommendations(self, patterns: List, metrics: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if patterns:
            recommendations.append(f"🚨 {len(patterns)} error pattern(s) detected. Investigate immediately.")
        
        if metrics.get("error_rate_percent", 0) > 5:
            recommendations.append("⚠️ Error rate above threshold. Check backend logs.")
        
        if metrics.get("api_response_time_avg_ms", 0) > 1000:
            recommendations.append("🐌 Slow API responses. Optimize or scale backend.")
        
        if not recommendations:
            recommendations.append("✅ All systems operating normally.")
        
        return recommendations
    
    def _generate_ai_summary(self, health_score: int, patterns: List, recent_errors: List) -> str:
        """Generate natural language summary for AI agent"""
        status = self._get_status(health_score)
        
        summary = f"System health: {status.upper()} ({health_score}/100). "
        summary += f"{len(recent_errors)} errors in last hour. "
        
        if patterns:
            summary += f"⚠️ ATTENTION: {len(patterns)} error patterns detected - "
            summary += ", ".join([p["error_code"] for p in patterns[:3]])
            summary += ". Immediate investigation recommended. "
        else:
            summary += "No concerning patterns. "
        
        return summary

# Global instance
diagnostic_engine = None

def get_diagnostic_engine(db):
    global diagnostic_engine
    if diagnostic_engine is None:
        diagnostic_engine = NeuralDiagnosticEngine(db)
    return diagnostic_engine
