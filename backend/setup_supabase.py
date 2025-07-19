"""
Setup script for Supabase integration
This script helps set up the required database tables and storage bucket
"""

import asyncio
import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

async def setup_supabase():
    """Setup Supabase database and storage for the behavioral logging system"""
    
    print("🚀 Setting up Supabase for Canara AI Behavioral Logging")
    print("=" * 60)
    
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("❌ Error: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env file")
        return
    
    try:
        # Initialize Supabase client
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        print("✅ Connected to Supabase successfully")
        
        # SQL commands to create tables
        create_users_table = """
        CREATE TABLE IF NOT EXISTS users (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            phone TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            mpin_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """
        
        create_sessions_table = """
        CREATE TABLE IF NOT EXISTS sessions (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            user_id UUID REFERENCES users(id),
            started_at TIMESTAMP DEFAULT NOW(),
            ended_at TIMESTAMP,
            device_info TEXT,
            is_escalated BOOLEAN DEFAULT FALSE,
            final_decision TEXT,
            anomaly_score FLOAT,
            session_token TEXT,
            log_file_url TEXT
        );
        """
        
        create_security_events_table = """
        CREATE TABLE IF NOT EXISTS security_events (
            id SERIAL PRIMARY KEY,
            session_id UUID REFERENCES sessions(id),
            level INTEGER,
            decision TEXT,
            reason TEXT,
            timestamp TIMESTAMP DEFAULT NOW(),
            model_used TEXT,
            match_score FLOAT
        );
        """
        
        print("\n📊 Creating database tables...")
        
        # Create tables
        try:
            supabase.rpc('exec_sql', {'sql': create_users_table}).execute()
            print("✅ Users table created/verified")
        except Exception as e:
            print(f"⚠️ Users table: {e}")
        
        try:
            supabase.rpc('exec_sql', {'sql': create_sessions_table}).execute()
            print("✅ Sessions table created/verified")
        except Exception as e:
            print(f"⚠️ Sessions table: {e}")
        
        try:
            supabase.rpc('exec_sql', {'sql': create_security_events_table}).execute()
            print("✅ Security events table created/verified")
        except Exception as e:
            print(f"⚠️ Security events table: {e}")
        
        print("\n💾 Setting up storage bucket...")
        
        # Create storage bucket for behavioral logs
        try:
            bucket_name = "behavior-logs"
            
            # Try to create bucket (will fail if already exists)
            try:
                result = supabase.storage.create_bucket(bucket_name, {
                    "public": True,
                    "file_size_limit": 52428800,  # 50MB
                    "allowed_mime_types": ["application/json"]
                })
                print(f"✅ Storage bucket '{bucket_name}' created successfully")
            except Exception as e:
                if "already exists" in str(e).lower():
                    print(f"✅ Storage bucket '{bucket_name}' already exists")
                else:
                    print(f"⚠️ Storage bucket creation: {e}")
            
            # Verify bucket exists
            buckets = supabase.storage.list_buckets()
            bucket_exists = any(bucket.name == bucket_name for bucket in buckets)
            
            if bucket_exists:
                print(f"✅ Storage bucket '{bucket_name}' is ready")
                
                # Test upload to verify permissions
                test_data = '{"test": "setup verification"}'
                test_path = "setup_test.json"
                
                try:
                    supabase.storage.from_(bucket_name).upload(
                        test_path, 
                        test_data.encode('utf-8'),
                        file_options={"content-type": "application/json"}
                    )
                    print("✅ Storage upload test successful")
                    
                    # Clean up test file
                    supabase.storage.from_(bucket_name).remove([test_path])
                    print("✅ Storage permissions verified")
                    
                except Exception as e:
                    print(f"⚠️ Storage upload test failed: {e}")
            else:
                print(f"❌ Storage bucket '{bucket_name}' not found")
        
        except Exception as e:
            print(f"❌ Storage setup error: {e}")
        
        print("\n🔍 Verifying setup...")
        
        # Test database connection by checking tables
        try:
            # Check if tables exist
            users_check = supabase.table('users').select('count', count='exact').execute()
            sessions_check = supabase.table('sessions').select('count', count='exact').execute()
            events_check = supabase.table('security_events').select('count', count='exact').execute()
            
            print("✅ Database verification:")
            print(f"  - Users table: accessible")
            print(f"  - Sessions table: accessible") 
            print(f"  - Security events table: accessible")
            
        except Exception as e:
            print(f"⚠️ Database verification failed: {e}")
        
        print("\n🎉 Supabase setup completed!")
        print("\n📋 Next steps:")
        print("1. Start the FastAPI backend: python main.py")
        print("2. Run the demo client: python supabase_demo_client.py")
        print("3. Check your Supabase dashboard for data")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")

def print_manual_setup_instructions():
    """Print manual setup instructions if automatic setup fails"""
    print("\n📖 MANUAL SETUP INSTRUCTIONS")
    print("=" * 40)
    print("\n1. 🗄️ Create these SQL tables in your Supabase SQL editor:")
    print("""
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    phone TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    mpin_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Sessions table  
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    started_at TIMESTAMP DEFAULT NOW(),
    ended_at TIMESTAMP,
    device_info TEXT,
    is_escalated BOOLEAN DEFAULT FALSE,
    final_decision TEXT,
    anomaly_score FLOAT,
    session_token TEXT,
    log_file_url TEXT
);

-- Security events table
CREATE TABLE security_events (
    id SERIAL PRIMARY KEY,
    session_id UUID REFERENCES sessions(id),
    level INTEGER,
    decision TEXT,
    reason TEXT,
    timestamp TIMESTAMP DEFAULT NOW(),
    model_used TEXT,
    match_score FLOAT
);
""")
    
    print("\n2. 💾 Create storage bucket:")
    print("   - Go to Storage in Supabase dashboard")
    print("   - Create bucket named: 'behavior-logs'")
    print("   - Set as public bucket")
    print("   - Set file size limit: 50MB")
    print("   - Allowed MIME types: application/json")
    
    print("\n3. 🔑 Environment setup:")
    print("   - Ensure .env file has:")
    print("     SUPABASE_URL=your_supabase_url")
    print("     SUPABASE_SERVICE_KEY=your_service_key")

if __name__ == "__main__":
    print("🔧 Supabase Setup for Canara AI Behavioral Logging")
    print("This will create the required database tables and storage bucket")
    print("\nPress Enter to continue...")
    input()
    
    try:
        asyncio.run(setup_supabase())
    except Exception as e:
        print(f"\n❌ Automatic setup failed: {e}")
        print_manual_setup_instructions()
