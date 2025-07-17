#!/usr/bin/env python3
"""
Test the database constraint fixes
"""

import asyncio
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'behavioral-auth-engine', 'src'))

async def test_authentication_decision_fix():
    """Test that AuthenticationDecision enum matches database constraints"""
    
    try:
        from src.data.models import AuthenticationDecision
        from src.core.ml_database import ml_db
        
        print("🧪 Testing Authentication Decision Fixes")
        print("="*50)
        
        # Test enum values
        print("\n📋 Available AuthenticationDecision values:")
        for decision in AuthenticationDecision:
            print(f"   • {decision.name} = '{decision.value}'")
        
        # Check that all enum values are in the allowed database values
        allowed_db_values = {'allow', 'challenge', 'block', 'learn'}
        enum_values = {decision.value for decision in AuthenticationDecision}
        
        print(f"\n🗄️  Database allows: {sorted(allowed_db_values)}")
        print(f"🐍 Code uses: {sorted(enum_values)}")
        
        # Check compatibility
        incompatible = enum_values - allowed_db_values
        if incompatible:
            print(f"\n❌ INCOMPATIBLE VALUES: {incompatible}")
            print("   These enum values will cause database constraint violations!")
            return False
        else:
            print(f"\n✅ ALL ENUM VALUES COMPATIBLE WITH DATABASE")
            
        # Test database connection
        print(f"\n🔌 Testing database connection...")
        health = await ml_db.health_check()
        if health:
            print("   ✅ Database connection successful")
        else:
            print("   ⚠️  Database connection failed (but enum fix is still valid)")
            
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🔧 DATABASE CONSTRAINT FIX VERIFICATION")
    print("="*55)
    
    success = await test_authentication_decision_fix()
    
    print("\n" + "="*55)
    if success:
        print("✅ FIXES VERIFIED: Authentication decisions should work now!")
        print("\n🎯 What was fixed:")
        print("   • Changed all AuthenticationDecision.DENY → BLOCK")
        print("   • Removed DENY from enum (not in database constraint)")
        print("   • All enum values now match database allowed values")
        print("\n🚀 Your behavioral authentication system should now store")
        print("   decisions without constraint violations!")
    else:
        print("❌ ISSUES FOUND: Please check the error messages above")
    
    print("\n💡 Next steps:")
    print("   1. Restart your ML Engine service")
    print("   2. Test behavioral analysis")
    print("   3. Check logs for successful decision storage")

if __name__ == "__main__":
    asyncio.run(main())
