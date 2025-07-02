"""
Token Management System
Handles refresh token storage, validation, and revocation
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Set
import asyncio
from app.core.config import settings

class TokenManager:
    """Manages refresh tokens and token blacklisting"""
    
    def __init__(self):
        # In-memory storage for demo - use Redis/Database in production
        self.active_refresh_tokens: Dict[str, Dict] = {}  # jti -> token_info
        self.blacklisted_tokens: Set[str] = set()  # Blacklisted token JTIs
        self.user_tokens: Dict[str, Set[str]] = {}  # user_id -> set of JTIs
        
    def store_refresh_token(self, jti: str, user_id: str, device_id: str, expires_at: datetime):
        """Store refresh token information"""
        self.active_refresh_tokens[jti] = {
            "user_id": user_id,
            "device_id": device_id,
            "expires_at": expires_at,
            "created_at": datetime.utcnow()
        }
        
        # Track user tokens for bulk operations
        if user_id not in self.user_tokens:
            self.user_tokens[user_id] = set()
        self.user_tokens[user_id].add(jti)
    
    def is_token_valid(self, jti: str) -> bool:
        """Check if refresh token is valid (not blacklisted and exists)"""
        if jti in self.blacklisted_tokens:
            return False
        
        if jti not in self.active_refresh_tokens:
            return False
        
        token_info = self.active_refresh_tokens[jti]
        if datetime.utcnow() > token_info["expires_at"]:
            # Token expired, remove it
            self.revoke_token(jti)
            return False
        
        return True
    
    def revoke_token(self, jti: str):
        """Revoke a specific refresh token"""
        self.blacklisted_tokens.add(jti)
        
        if jti in self.active_refresh_tokens:
            token_info = self.active_refresh_tokens[jti]
            user_id = token_info["user_id"]
            
            # Remove from active tokens
            del self.active_refresh_tokens[jti]
            
            # Remove from user tokens
            if user_id in self.user_tokens:
                self.user_tokens[user_id].discard(jti)
    
    def revoke_user_tokens(self, user_id: str):
        """Revoke all refresh tokens for a user"""
        if user_id in self.user_tokens:
            user_token_jtis = self.user_tokens[user_id].copy()
            for jti in user_token_jtis:
                self.revoke_token(jti)
    
    def revoke_device_tokens(self, user_id: str, device_id: str):
        """Revoke all refresh tokens for a specific device"""
        if user_id in self.user_tokens:
            user_token_jtis = self.user_tokens[user_id].copy()
            for jti in user_token_jtis:
                if jti in self.active_refresh_tokens:
                    token_info = self.active_refresh_tokens[jti]
                    if token_info["device_id"] == device_id:
                        self.revoke_token(jti)
    
    def get_user_active_tokens(self, user_id: str) -> list:
        """Get all active refresh tokens for a user"""
        if user_id not in self.user_tokens:
            return []
        
        active_tokens = []
        for jti in self.user_tokens[user_id]:
            if self.is_token_valid(jti):
                token_info = self.active_refresh_tokens[jti]
                active_tokens.append({
                    "jti": jti,
                    "device_id": token_info["device_id"],
                    "created_at": token_info["created_at"],
                    "expires_at": token_info["expires_at"]
                })
        
        return active_tokens
    
    async def cleanup_expired_tokens(self):
        """Clean up expired tokens (background task)"""
        current_time = datetime.utcnow()
        expired_jtis = []
        
        for jti, token_info in self.active_refresh_tokens.items():
            if current_time > token_info["expires_at"]:
                expired_jtis.append(jti)
        
        for jti in expired_jtis:
            self.revoke_token(jti)
        
        if expired_jtis:
            print(f"Cleaned up {len(expired_jtis)} expired refresh tokens")
    
    def get_stats(self) -> Dict:
        """Get token manager statistics"""
        return {
            "active_refresh_tokens": len(self.active_refresh_tokens),
            "blacklisted_tokens": len(self.blacklisted_tokens),
            "users_with_tokens": len(self.user_tokens),
            "total_user_token_mappings": sum(len(tokens) for tokens in self.user_tokens.values())
        }

# Global token manager instance
token_manager = TokenManager()

async def token_cleanup_task():
    """Background task to cleanup expired tokens"""
    while True:
        await asyncio.sleep(3600)  # Run every hour
        await token_manager.cleanup_expired_tokens()
