"""
E-Commerce Adapter for Behavioral Authentication
Integrates with e-commerce platforms and fraud detection
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np

from ..core.enhanced_behavioral_processor import EnhancedBehavioralProcessor
from ..data.models import AuthenticationDecision, RiskLevel

logger = logging.getLogger(__name__)

class ECommerceAdapter:
    """
    Adapter for e-commerce platform integration
    Handles order fraud detection and shopping behavior analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.behavioral_processor = EnhancedBehavioralProcessor()
        
        # Risk thresholds for e-commerce orders
        self.risk_thresholds = {
            "low_value": 100,       # Below $100
            "medium_value": 500,    # $100 - $500
            "high_value": 2000,     # $500 - $2000
            "critical_value": 2000  # Above $2000
        }
        
        # Suspicious product categories
        self.high_risk_categories = [
            "electronics", "luxury_goods", "gift_cards", 
            "gaming", "software", "cryptocurrency"
        ]
        
        self.logger.info("E-Commerce Adapter initialized")
    
    async def detect_fraud(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect potential fraud in e-commerce orders
        
        Args:
            order_data: Order details including items, amount, user info
            
        Returns:
            Fraud detection result
        """
        try:
            user_id = order_data.get('user_id')
            order_total = self._calculate_order_total(order_data.get('items', []))
            
            # Calculate fraud risk score
            fraud_score = await self._calculate_fraud_score(order_data)
            
            # Analyze order patterns
            pattern_risk = await self._analyze_order_patterns(order_data)
            
            # Check behavioral anomalies
            behavioral_risk = await self._check_behavioral_anomalies(order_data)
            
            # Combine all risk factors
            total_risk = min(1.0, fraud_score + pattern_risk + behavioral_risk)
            
            # Determine fraud level and action
            if total_risk < 0.3:
                fraud_level = "low"
                action = "allow"
            elif total_risk < 0.6:
                fraud_level = "medium"
                action = "review"
            elif total_risk < 0.8:
                fraud_level = "high"
                action = "challenge"
            else:
                fraud_level = "critical"
                action = "block"
            
            result = {
                "user_id": user_id,
                "order_id": order_data.get('order_id'),
                "fraud_score": total_risk,
                "fraud_level": fraud_level,
                "recommended_action": action,
                "risk_factors": {
                    "order_fraud_risk": fraud_score,
                    "pattern_risk": pattern_risk,
                    "behavioral_risk": behavioral_risk
                },
                "order_total": order_total,
                "suspicious_indicators": await self._identify_suspicious_indicators(order_data),
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Fraud detection: {fraud_level} (score: {total_risk:.3f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Error detecting fraud: {e}")
            return {
                "error": str(e),
                "fraud_score": 1.0,  # Fail secure
                "recommended_action": "block"
            }
    
    def _calculate_order_total(self, items: List[Dict[str, Any]]) -> float:
        """Calculate total order amount"""
        total = 0.0
        for item in items:
            price = item.get('price', 0)
            quantity = item.get('quantity', 1)
            total += price * quantity
        return total
    
    async def _calculate_fraud_score(self, order_data: Dict[str, Any]) -> float:
        """Calculate base fraud score"""
        
        fraud_score = 0.0
        items = order_data.get('items', [])
        order_total = self._calculate_order_total(items)
        
        # Order value risk
        if order_total < self.risk_thresholds["low_value"]:
            fraud_score += 0.1
        elif order_total < self.risk_thresholds["medium_value"]:
            fraud_score += 0.2
        elif order_total < self.risk_thresholds["high_value"]:
            fraud_score += 0.4
        else:
            fraud_score += 0.7
        
        # High-risk product categories
        for item in items:
            category = item.get('category', '').lower()
            if any(risk_cat in category for risk_cat in self.high_risk_categories):
                fraud_score += 0.2
                break
        
        # Multiple high-value items
        high_value_items = [item for item in items if item.get('price', 0) > 500]
        if len(high_value_items) > 2:
            fraud_score += 0.3
        
        # Shipping address risk
        shipping_address = order_data.get('shipping_address', '')
        if 'po box' in shipping_address.lower() or 'freight forwarder' in shipping_address.lower():
            fraud_score += 0.2
        
        return min(0.8, fraud_score)
    
    async def _analyze_order_patterns(self, order_data: Dict[str, Any]) -> float:
        """Analyze order patterns for fraud indicators"""
        
        pattern_risk = 0.0
        user_id = order_data.get('user_id')
        
        # Simulate pattern analysis
        # In real implementation, analyze user's order history
        
        # Multiple orders in short time
        # This would check recent order frequency
        pattern_risk += 0.1
        
        # Unusual order time
        current_hour = datetime.now().hour
        if current_hour < 2 or current_hour > 23:
            pattern_risk += 0.15
        
        # Payment method risk
        payment_method = order_data.get('payment_method', '')
        if payment_method in ['prepaid_card', 'cryptocurrency', 'wire_transfer']:
            pattern_risk += 0.2
        
        return min(0.3, pattern_risk)
    
    async def _check_behavioral_anomalies(self, order_data: Dict[str, Any]) -> float:
        """Check for behavioral anomalies in shopping behavior"""
        
        behavioral_risk = 0.0
        
        # Rapid checkout (too fast for human)
        # This would check session duration and behavior
        behavioral_risk += 0.1
        
        # Inconsistent shipping/billing addresses
        shipping = order_data.get('shipping_address', '')
        billing = order_data.get('billing_address', '')
        if shipping and billing and shipping != billing:
            behavioral_risk += 0.1
        
        # Device/location anomalies would be checked here
        # This requires integration with device fingerprinting
        
        return min(0.2, behavioral_risk)
    
    async def _identify_suspicious_indicators(self, order_data: Dict[str, Any]) -> List[str]:
        """Identify specific suspicious indicators"""
        
        indicators = []
        items = order_data.get('items', [])
        order_total = self._calculate_order_total(items)
        
        if order_total > self.risk_thresholds["high_value"]:
            indicators.append("high_value_order")
        
        # Check for high-risk categories
        for item in items:
            category = item.get('category', '').lower()
            if any(risk_cat in category for risk_cat in self.high_risk_categories):
                indicators.append("high_risk_product_category")
                break
        
        # Multiple expensive items
        expensive_items = [item for item in items if item.get('price', 0) > 500]
        if len(expensive_items) > 1:
            indicators.append("multiple_expensive_items")
        
        # Unusual shipping
        shipping_address = order_data.get('shipping_address', '').lower()
        if 'po box' in shipping_address:
            indicators.append("po_box_shipping")
        
        # Rush delivery
        shipping_method = order_data.get('shipping_method', '').lower()
        if 'overnight' in shipping_method or 'express' in shipping_method:
            indicators.append("rush_delivery")
        
        return indicators
    
    async def analyze_shopping_behavior(self, user_id: str, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze shopping behavior patterns
        
        Args:
            user_id: User identifier
            order_data: Current order data
            
        Returns:
            Shopping behavior analysis
        """
        try:
            analysis = {
                "user_id": user_id,
                "behavior_profile": await self._build_behavior_profile(user_id),
                "current_order_analysis": await self._analyze_current_order(order_data),
                "deviation_score": await self._calculate_behavior_deviation(user_id, order_data),
                "shopping_patterns": await self._identify_shopping_patterns(user_id),
                "timestamp": datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing shopping behavior: {e}")
            return {"error": str(e)}
    
    async def _build_behavior_profile(self, user_id: str) -> Dict[str, Any]:
        """Build user's shopping behavior profile"""
        
        # Simulate behavior profile building
        # In real implementation, analyze historical data
        return {
            "customer_type": "regular",
            "average_order_value": 150.0,
            "typical_categories": ["clothing", "home", "books"],
            "shopping_frequency": "weekly",
            "preferred_payment_method": "credit_card",
            "trust_score": 0.8,
            "account_age_days": 180
        }
    
    async def _analyze_current_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current order characteristics"""
        
        items = order_data.get('items', [])
        order_total = self._calculate_order_total(items)
        
        categories = list(set([item.get('category', 'unknown') for item in items]))
        
        return {
            "order_total": order_total,
            "item_count": len(items),
            "categories": categories,
            "payment_method": order_data.get('payment_method'),
            "shipping_method": order_data.get('shipping_method'),
            "order_complexity": len(categories) / len(items) if items else 0
        }
    
    async def _calculate_behavior_deviation(self, user_id: str, order_data: Dict[str, Any]) -> float:
        """Calculate how much current behavior deviates from normal"""
        
        # Simulate deviation calculation
        # Compare current order with historical patterns
        
        deviation_score = 0.0
        
        # Order value deviation
        current_total = self._calculate_order_total(order_data.get('items', []))
        avg_order_value = 150.0  # Would come from historical data
        
        if current_total > avg_order_value * 3:
            deviation_score += 0.4
        elif current_total < avg_order_value * 0.3:
            deviation_score += 0.2
        
        # Category deviation
        current_categories = set([item.get('category', '') for item in order_data.get('items', [])])
        typical_categories = set(["clothing", "home", "books"])  # From profile
        
        if not current_categories.intersection(typical_categories):
            deviation_score += 0.3
        
        return min(1.0, deviation_score)
    
    async def _identify_shopping_patterns(self, user_id: str) -> Dict[str, Any]:
        """Identify user's shopping patterns"""
        
        # Simulate pattern identification
        return {
            "shopping_days": ["monday", "friday", "saturday"],
            "peak_hours": [12, 13, 19, 20],
            "seasonal_preferences": ["summer_clothing", "winter_accessories"],
            "brand_loyalty": 0.6,
            "price_sensitivity": "medium",
            "impulse_buying_tendency": 0.3
        }
    
    async def process_order_decision(self, order_data: Dict[str, Any], auth_decision: AuthenticationDecision) -> Dict[str, Any]:
        """
        Process final order decision based on authentication and fraud analysis
        
        Args:
            order_data: Order details
            auth_decision: Authentication decision from behavioral system
            
        Returns:
            Final order processing result
        """
        try:
            # Get fraud analysis
            fraud_result = await self.detect_fraud(order_data)
            
            final_decision = {
                "order_id": order_data.get('order_id'),
                "user_id": order_data.get('user_id'),
                "order_approved": False,
                "authentication_result": auth_decision.action.value,
                "fraud_analysis": fraud_result,
                "final_action": "block",
                "reason": [],
                "timestamp": datetime.now().isoformat()
            }
            
            # Decision logic combining auth and fraud results
            auth_ok = auth_decision.action.value == "allow"
            fraud_low = fraud_result.get("fraud_level") in ["low", "medium"]
            
            if auth_ok and fraud_low:
                final_decision["order_approved"] = True
                final_decision["final_action"] = "approve"
            elif auth_decision.action.value == "challenge" or fraud_result.get("fraud_level") == "high":
                final_decision["final_action"] = "review"
                final_decision["reason"].append("additional_verification_required")
            else:
                final_decision["reason"].append("high_fraud_risk_detected")
            
            self.logger.info(f"Order decision: {final_decision['final_action']}")
            return final_decision
            
        except Exception as e:
            self.logger.error(f"Error processing order decision: {e}")
            return {
                "error": str(e),
                "order_approved": False,
                "final_action": "block"
            }
