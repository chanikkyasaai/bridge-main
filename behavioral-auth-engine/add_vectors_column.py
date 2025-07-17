#!/usr/bin/env python3
"""
Add vectors_collected column to user_profiles table
"""

import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.ml_database import ml_db

async def add_vectors_column():
    """Add vectors_collected column to user_profiles table"""
    try:
        # Try to add the column using raw SQL
        result = ml_db.supabase.rpc('exec_sql', {
            'sql': 'ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS vectors_collected INTEGER DEFAULT 0;'
        }).execute()
        
        print("✅ Successfully added vectors_collected column to user_profiles table")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"❌ Error adding column: {e}")
        
        # Alternative approach - try to update table structure
        try:
            # Check if column exists by trying to query it
            test_result = ml_db.supabase.table('user_profiles').select('vectors_collected').limit(1).execute()
            print("✅ vectors_collected column already exists")
        except Exception as e2:
            print(f"❌ Column doesn't exist and couldn't be added: {e2}")

if __name__ == "__main__":
    asyncio.run(add_vectors_column())
