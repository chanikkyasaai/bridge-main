"""
Bank Adapter for Behavioral Authentication
Integrates with banking systems and transactions
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np

from ..core.enhanced_behavioral_processor import EnhancedBehavioralProcessor
from ..data.models import AuthenticationDecision, RiskLevel

logger = logging.getLogger(__name__)

class BankAdapter:
    """
    Adapter for banking system integration
    Handles transaction risk assessment and behavioral analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.behavioral_processor = EnhancedBehavioralProcessor()
        
        # Risk thresholds for banking transactions
        self.risk_thresholds = {
            "low_value": 1000,      # Below $1000
            "medium_value": 10000,  # $1000 - $10000  
            "high_value": 50000,    # $10000 - $50000
            "critical_value": 50000 # Above $50000
        }
        
        self.logger.info("Bank Adapter initialized")
    
    async def assess_transaction_risk(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess risk level for a banking transaction
        
        Args:
            transaction_data: Transaction details including amount, type, etc.
            
        Returns:
            Risk assessment result
        """
        try:
            amount = transaction_data.get('amount', 0)
            transaction_type = transaction_data.get('transaction_type', 'unknown')
            user_id = transaction_data.get('user_id')
            
            # Calculate base risk score
            base_risk = self._calculate_base_risk(amount, transaction_type)
            
            # Add behavioral risk factors
            behavioral_risk = await self._assess_behavioral_risk(transaction_data)
            
            # Combine risks
            total_risk = min(1.0, base_risk + behavioral_risk)
            
            # Determine risk level
            if total_risk < 0.3:
                risk_level = RiskLevel.LOW
                action = "allow"
            elif total_risk < 0.6:
                risk_level = RiskLevel.MEDIUM
                action = "challenge"
            elif total_risk < 0.8:
                risk_level = RiskLevel.HIGH
                action = "step_up_auth"
            else:
                risk_level = RiskLevel.CRITICAL
                action = "block"
            
            result = {
                "user_id": user_id,
                "transaction_id": transaction_data.get('transaction_id'),
                "risk_score": total_risk,
                "risk_level": risk_level.value,
                "recommended_action": action,
                "risk_factors": {
                    "amount_risk": base_risk,
                    "behavioral_risk": behavioral_risk
                },
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Transaction risk assessed: {risk_level.value} (score: {total_risk:.3f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Error assessing transaction risk: {e}")
            return {
                "error": str(e),
                "risk_score": 1.0,  # Fail secure
                "recommended_action": "block"
            }
    
    def _calculate_base_risk(self, amount: float, transaction_type: str) -> float:
        """Calculate base risk score based on amount and type"""
        
        # Amount-based risk
        if amount < self.risk_thresholds["low_value"]:
            amount_risk = 0.1
        elif amount < self.risk_thresholds["medium_value"]:
            amount_risk = 0.3
        elif amount < self.risk_thresholds["high_value"]:
            amount_risk = 0.6
        else:
            amount_risk = 0.9
        
        # Transaction type risk
        type_risk_map = {
            "deposit": 0.1,
            "withdrawal": 0.3,
            "transfer": 0.4,
            "wire_transfer": 0.7,
            "international": 0.8,
            "unknown": 0.5
        }
        
        type_risk = type_risk_map.get(transaction_type, 0.5)
        
        return min(1.0, amount_risk + type_risk * 0.3)
    
    async def _assess_behavioral_risk(self, transaction_data: Dict[str, Any]) -> float:
        """Assess behavioral risk factors"""
        
        behavioral_risk = 0.0
        
        # Time-based risk (unusual hours)
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 23:
            behavioral_risk += 0.2
        
        # Frequency risk (multiple transactions)
        user_id = transaction_data.get('user_id')
        if user_id:
            # Simulate checking recent transaction frequency
            # In real implementation, query transaction history
            behavioral_risk += 0.1
        
        # Location risk (if available)
        location = transaction_data.get('location')
        if location:
            # Simulate location-based risk assessment
            behavioral_risk += 0.05
        
        return min(0.5, behavioral_risk)
    
    async def get_behavioral_context(self, user_id: str, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get behavioral context for transaction analysis
        
        Args:
            user_id: User identifier
            transaction_data: Transaction details
            
        Returns:
            Behavioral context information
        """
        try:
            context = {
                "user_id": user_id,
                "transaction_patterns": await self._analyze_transaction_patterns(user_id),
                "behavioral_profile": await self._get_user_behavioral_profile(user_id),
                "risk_indicators": await self._identify_risk_indicators(transaction_data),
                "timestamp": datetime.now().isoformat()
            }
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error getting behavioral context: {e}")
            return {"error": str(e)}
    
    async def _analyze_transaction_patterns(self, user_id: str) -> Dict[str, Any]:
        """Analyze user's historical transaction patterns"""
        
        # Simulate pattern analysis
        # In real implementation, analyze historical data
        return {
            "average_transaction_amount": 2500.0,
            "frequent_transaction_types": ["transfer", "deposit"],
            "typical_transaction_hours": [9, 10, 11, 14, 15, 16],
            "monthly_transaction_count": 45,
            "risk_score": 0.2
        }
    
    async def _get_user_behavioral_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user's behavioral profile"""
        
        # Simulate behavioral profile
        return {
            "user_type": "regular_customer",
            "account_age_days": 365,
            "trust_score": 0.8,
            "historical_risk_level": "low",
            "behavioral_consistency": 0.9
        }
    
    async def _identify_risk_indicators(self, transaction_data: Dict[str, Any]) -> List[str]:
        """Identify risk indicators in transaction"""
        
        indicators = []
        
        amount = transaction_data.get('amount', 0)
        if amount > self.risk_thresholds["high_value"]:
            indicators.append("high_value_transaction")
        
        transaction_type = transaction_data.get('transaction_type')
        if transaction_type in ["wire_transfer", "international"]:
            indicators.append("high_risk_transaction_type")
        
        # Check for unusual timing
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 23:
            indicators.append("unusual_transaction_time")
        
        return indicators

    async def process_transaction_decision(self, transaction_data: Dict[str, Any], auth_decision: AuthenticationDecision) -> Dict[str, Any]:
        """
        Process final transaction decision based on authentication result
        
        Args:
            transaction_data: Transaction details
            auth_decision: Authentication decision from behavioral system
            
        Returns:
            Final transaction processing result
        """
        try:
            # Combine transaction risk with authentication decision
            transaction_risk = await self.assess_transaction_risk(transaction_data)
            
            final_decision = {
                "transaction_id": transaction_data.get('transaction_id'),
                "user_id": transaction_data.get('user_id'),
                "transaction_allowed": False,
                "authentication_result": auth_decision.action.value,
                "transaction_risk": transaction_risk,
                "final_action": "block",
                "reason": [],
                "timestamp": datetime.now().isoformat()
            }
            
            # Decision logic
            if auth_decision.action.value == "allow" and transaction_risk["risk_level"] in ["low", "medium"]:
                final_decision["transaction_allowed"] = True
                final_decision["final_action"] = "allow"
            elif auth_decision.action.value == "challenge" or transaction_risk["risk_level"] == "high":
                final_decision["final_action"] = "challenge"
                final_decision["reason"].append("additional_verification_required")
            else:
                final_decision["reason"].append("high_risk_detected")
            
            self.logger.info(f"Transaction decision: {final_decision['final_action']}")
            return final_decision
            
        except Exception as e:
            self.logger.error(f"Error processing transaction decision: {e}")
            return {
                "error": str(e),
                "transaction_allowed": False,
                "final_action": "block"
            }
