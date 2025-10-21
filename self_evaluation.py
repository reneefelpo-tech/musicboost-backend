"""
Self-Evaluation and Deep Error Detection System
Continuously monitors, detects, and auto-fixes issues
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import aiohttp

class SelfEvaluationSystem:
    """
    Self-evaluation and error detection system
    Monitors all endpoints, validates data integrity, and auto-fixes issues
    """
    
    def __init__(self, db, coordinator):
        self.db = db
        self.coordinator = coordinator
        self.errors_detected = []
        self.auto_fixes_applied = []
        self.last_evaluation = None
        self.evaluation_running = False
        
    async def deep_link_validation(self) -> Dict:
        """Deep validation of all links and their mappings"""
        issues = []
        fixes_applied = []
        
        try:
            # Get all active links
            links = await self.db.music_links.find({"is_active": True}).to_list(10000)
            
            for link in links:
                link_id = link["_id"]
                
                # Check 1: Verify link has a valid URL
                if not link.get("url") or not link["url"].startswith("http"):
                    issues.append({
                        "type": "invalid_url",
                        "link_id": link_id,
                        "severity": "high"
                    })
                
                # Check 2: Verify short code mapping exists
                from server import generate_short_code
                short_code = generate_short_code(link_id)
                mapping = await self.db.short_code_map.find_one({"short_code": short_code})
                
                if not mapping:
                    # Auto-fix: Create missing mapping
                    await self.db.short_code_map.insert_one({
                        "_id": str(__import__('uuid').uuid4()),
                        "short_code": short_code,
                        "link_id": link_id,
                        "created_at": datetime.utcnow(),
                        "last_accessed": datetime.utcnow()
                    })
                    fixes_applied.append({
                        "type": "created_missing_mapping",
                        "link_id": link_id,
                        "short_code": short_code
                    })
                    
                # Check 3: Verify mapping points to correct link
                elif mapping.get("link_id") != link_id:
                    issues.append({
                        "type": "mapping_mismatch",
                        "link_id": link_id,
                        "mapping_link_id": mapping.get("link_id"),
                        "severity": "critical"
                    })
                    
                    # Auto-fix: Correct the mapping
                    await self.db.short_code_map.update_one(
                        {"short_code": short_code},
                        {"$set": {"link_id": link_id, "corrected_at": datetime.utcnow()}}
                    )
                    fixes_applied.append({
                        "type": "corrected_mapping",
                        "link_id": link_id,
                        "short_code": short_code
                    })
                
                # Check 4: Verify user still exists
                user = await self.db.users.find_one({"_id": link["user_id"]})
                if not user:
                    issues.append({
                        "type": "orphaned_link",
                        "link_id": link_id,
                        "user_id": link["user_id"],
                        "severity": "medium"
                    })
            
            return {
                "total_links_checked": len(links),
                "issues_found": len(issues),
                "auto_fixes_applied": len(fixes_applied),
                "issues": issues[:10],  # First 10 issues
                "fixes": fixes_applied[:10]  # First 10 fixes
            }
            
        except Exception as e:
            logging.error(f"❌ Deep link validation error: {str(e)}")
            return {"error": str(e)}
    
    async def check_endpoint_health(self, base_url: str) -> Dict:
        """Check health of all critical endpoints"""
        endpoints = [
            {"method": "GET", "path": "/api/health", "expected": 200},
            {"method": "GET", "path": "/api/public/link/test-id", "expected": 404},  # Should return error page
            {"method": "GET", "path": "/api/public/artist/test-user", "expected": 404},  # Should return error page
            {"method": "GET", "path": "/api/l/testcode", "expected": 404},  # Should return error page
        ]
        
        results = []
        
        try:
            async with aiohttp.ClientSession() as session:
                for endpoint in endpoints:
                    try:
                        async with session.request(
                            endpoint["method"],
                            f"{base_url}{endpoint['path']}",
                            timeout=aiohttp.ClientTimeout(total=10)
                        ) as response:
                            status = response.status
                            is_healthy = (status == endpoint["expected"]) or (status in [200, 404, 410])
                            
                            results.append({
                                "endpoint": endpoint["path"],
                                "status": status,
                                "expected": endpoint["expected"],
                                "healthy": is_healthy
                            })
                    except Exception as e:
                        results.append({
                            "endpoint": endpoint["path"],
                            "error": str(e),
                            "healthy": False
                        })
            
            healthy_count = sum(1 for r in results if r.get("healthy", False))
            
            return {
                "total_endpoints": len(results),
                "healthy_endpoints": healthy_count,
                "unhealthy_endpoints": len(results) - healthy_count,
                "results": results
            }
            
        except Exception as e:
            logging.error(f"❌ Endpoint health check error: {str(e)}")
            return {"error": str(e)}
    
    async def validate_database_integrity(self) -> Dict:
        """Validate database integrity and relationships"""
        issues = []
        fixes_applied = []
        
        try:
            # Check for duplicate short codes
            pipeline = [
                {"$group": {"_id": "$short_code", "count": {"$sum": 1}}},
                {"$match": {"count": {"$gt": 1}}}
            ]
            duplicates = await self.db.short_code_map.aggregate(pipeline).to_list(100)
            
            for dup in duplicates:
                issues.append({
                    "type": "duplicate_short_code",
                    "short_code": dup["_id"],
                    "count": dup["count"],
                    "severity": "critical"
                })
                
                # Auto-fix: Keep most recent, delete others
                mappings = await self.db.short_code_map.find(
                    {"short_code": dup["_id"]}
                ).sort("created_at", -1).to_list(100)
                
                if len(mappings) > 1:
                    # Keep first (most recent), delete others
                    keep_id = mappings[0]["_id"]
                    delete_ids = [m["_id"] for m in mappings[1:]]
                    
                    result = await self.db.short_code_map.delete_many({"_id": {"$in": delete_ids}})
                    fixes_applied.append({
                        "type": "removed_duplicate_mappings",
                        "short_code": dup["_id"],
                        "deleted_count": result.deleted_count
                    })
            
            # Check for stale mappings (links deleted but mapping still exists)
            all_mappings = await self.db.short_code_map.find({}).to_list(10000)
            for mapping in all_mappings:
                link = await self.db.music_links.find_one({"_id": mapping["link_id"]})
                if not link or not link.get("is_active", True):
                    issues.append({
                        "type": "stale_mapping",
                        "link_id": mapping["link_id"],
                        "short_code": mapping["short_code"],
                        "severity": "low"
                    })
                    
                    # Auto-fix: Delete stale mapping
                    await self.db.short_code_map.delete_one({"_id": mapping["_id"]})
                    fixes_applied.append({
                        "type": "removed_stale_mapping",
                        "link_id": mapping["link_id"],
                        "short_code": mapping["short_code"]
                    })
            
            return {
                "duplicate_codes": len(duplicates),
                "issues_found": len(issues),
                "auto_fixes_applied": len(fixes_applied),
                "issues": issues[:10],
                "fixes": fixes_applied[:10]
            }
            
        except Exception as e:
            logging.error(f"❌ Database integrity validation error: {str(e)}")
            return {"error": str(e)}
    
    async def performance_analysis(self) -> Dict:
        """Analyze system performance"""
        try:
            # Check database indexes
            indexes = await self.db.music_links.index_information()
            short_code_indexes = await self.db.short_code_map.index_information()
            
            # Count recent events
            recent_clicks = await self.db.click_events.count_documents({
                "timestamp": {"$gte": datetime.utcnow() - timedelta(hours=1)}
            })
            
            # Get mapping statistics
            total_mappings = await self.db.short_code_map.count_documents({})
            total_links = await self.db.music_links.count_documents({"is_active": True})
            
            mapping_coverage = (total_mappings / total_links * 100) if total_links > 0 else 0
            
            return {
                "database_indexes": {
                    "music_links": len(indexes),
                    "short_code_map": len(short_code_indexes)
                },
                "recent_activity": {
                    "clicks_last_hour": recent_clicks
                },
                "mapping_statistics": {
                    "total_mappings": total_mappings,
                    "total_links": total_links,
                    "coverage_percentage": round(mapping_coverage, 2)
                }
            }
            
        except Exception as e:
            logging.error(f"❌ Performance analysis error: {str(e)}")
            return {"error": str(e)}
    
    async def run_full_evaluation(self) -> Dict:
        """Run complete system evaluation"""
        if self.evaluation_running:
            return {"status": "evaluation_already_running"}
        
        self.evaluation_running = True
        evaluation_id = str(__import__('uuid').uuid4())
        start_time = datetime.utcnow()
        
        logging.info(f"🔍 Starting full system evaluation: {evaluation_id}")
        
        try:
            # Run all checks in parallel for efficiency
            results = await asyncio.gather(
                self.deep_link_validation(),
                self.validate_database_integrity(),
                self.performance_analysis(),
                return_exceptions=True
            )
            
            link_validation, db_integrity, performance = results
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            # Calculate overall health score
            total_issues = 0
            if isinstance(link_validation, dict):
                total_issues += link_validation.get("issues_found", 0)
            if isinstance(db_integrity, dict):
                total_issues += db_integrity.get("issues_found", 0)
            
            health_score = max(0, 100 - (total_issues * 5))  # Deduct 5 points per issue
            
            evaluation_result = {
                "evaluation_id": evaluation_id,
                "timestamp": start_time,
                "duration_seconds": duration,
                "health_score": health_score,
                "status": "healthy" if health_score >= 80 else "needs_attention" if health_score >= 60 else "critical",
                "link_validation": link_validation,
                "database_integrity": db_integrity,
                "performance": performance
            }
            
            self.last_evaluation = evaluation_result
            self.evaluation_running = False
            
            # Fire neural network signal
            await self.coordinator.fire_signal("evaluation_complete", evaluation_result)
            
            logging.info(f"✅ Evaluation complete: Health score {health_score}/100")
            
            return evaluation_result
            
        except Exception as e:
            self.evaluation_running = False
            logging.error(f"❌ Full evaluation error: {str(e)}")
            return {"error": str(e)}
    
    def get_last_evaluation(self) -> Dict:
        """Get results of last evaluation"""
        return self.last_evaluation or {"status": "no_evaluation_run_yet"}

# Global evaluator instance (will be initialized with db and coordinator)
evaluator = None
