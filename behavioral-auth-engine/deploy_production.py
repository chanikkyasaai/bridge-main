#!/usr/bin/env python3
"""
Production Deployment Script for Enhanced Behavioral Authentication System
Deploys Layers G, H, and J for national-level fraud detection
"""

import sys
import os
import subprocess
import json
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'numpy',
        'networkx',
        'fastapi',
        'uvicorn',
        'supabase',
        'faiss-cpu',  # or faiss-gpu for production
        'pytest'
    ]
    
    optional_packages = [
        'torch',
        'torch-geometric',
        'pytorch'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - MISSING")
    
    for package in optional_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} (optional)")
        except ImportError:
            print(f"⚠️  {package} (optional) - for GNN functionality")
    
    if missing_packages:
        print(f"\n❌ Missing required packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All required dependencies available")
    return True

def run_layer_tests():
    """Run comprehensive tests for all layers"""
    print("\n🧪 Running layer tests...")
    
    try:
        # Run the verification script
        result = subprocess.run([
            sys.executable, "verify_layers.py"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("✅ All layer tests passed")
            print(result.stdout)
            return True
        else:
            print("❌ Layer tests failed")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def create_production_config():
    """Create production configuration files"""
    print("\n⚙️  Creating production configuration...")
    
    config = {
        "system": {
            "environment": "production",
            "log_level": "INFO",
            "max_concurrent_users": 10000,
            "session_timeout": 1800
        },
        "layers": {
            "session_graph_generator": {
                "enabled": True,
                "rapid_threshold_ms": 500,
                "delayed_threshold_ms": 3000,
                "max_nodes_per_session": 1000
            },
            "gnn_anomaly_detector": {
                "enabled": True,
                "model_path": "/models/behavioral_gnn.pth",
                "node_features": 10,
                "edge_features": 4,
                "hidden_dim": 64,
                "num_layers": 3,
                "batch_size": 32
            },
            "policy_orchestration": {
                "enabled": True,
                "default_policy_level": "level_2_enhanced",
                "high_value_threshold": 50000,
                "max_failures_per_hour": 5
            }
        },
        "thresholds": {
            "level_1_basic": {
                "allow_threshold": 0.8,
                "challenge_threshold": 0.6,
                "block_threshold": 0.3
            },
            "level_2_enhanced": {
                "allow_threshold": 0.75,
                "challenge_threshold": 0.55,
                "block_threshold": 0.35
            },
            "level_3_advanced": {
                "allow_threshold": 0.7,
                "challenge_threshold": 0.5,
                "block_threshold": 0.4
            },
            "level_4_maximum": {
                "allow_threshold": 0.65,
                "challenge_threshold": 0.45,
                "block_threshold": 0.45
            }
        },
        "database": {
            "supabase_url": "YOUR_SUPABASE_URL",
            "supabase_key": "YOUR_SUPABASE_KEY",
            "connection_pool_size": 20,
            "query_timeout": 30
        },
        "monitoring": {
            "enable_metrics": True,
            "metrics_port": 9090,
            "log_file": "/var/log/behavioral_auth.log",
            "alert_webhook": "YOUR_WEBHOOK_URL"
        }
    }
    
    try:
        config_path = Path("production_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Production config created: {config_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating config: {e}")
        return False

def create_docker_deployment():
    """Create Docker deployment files"""
    print("\n🐳 Creating Docker deployment files...")
    
    dockerfile_content = """# Behavioral Authentication Engine - Production
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch for GNN (optional)
RUN pip install torch torch-geometric --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 authuser && chown -R authuser:authuser /app
USER authuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
  CMD curl -f http://localhost:8001/health || exit 1

# Expose port
EXPOSE 8001

# Run application
CMD ["python", "main.py"]
"""
    
    docker_compose_content = """version: '3.8'
services:
  behavioral-auth:
    build: .
    ports:
      - "8001:8001"
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    volumes:
      - ./logs:/app/logs
      - ./models:/app/models
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
    
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: behavioral_auth
      POSTGRES_USER: auth_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

volumes:
  postgres_data:
"""
    
    try:
        with open("Dockerfile", 'w') as f:
            f.write(dockerfile_content)
        
        with open("docker-compose.yml", 'w') as f:
            f.write(docker_compose_content)
        
        print("✅ Docker files created: Dockerfile, docker-compose.yml")
        return True
        
    except Exception as e:
        print(f"❌ Error creating Docker files: {e}")
        return False

def create_monitoring_scripts():
    """Create monitoring and deployment scripts"""
    print("\n📊 Creating monitoring scripts...")
    
    monitoring_script = """#!/bin/bash
# Behavioral Authentication System Monitoring

echo "Behavioral Authentication System Status"
echo "========================================"

# Check API health
echo "API Health:"
curl -s http://localhost:8001/health | jq .

# Check system status
echo -e "\\nSystem Status:"
curl -s http://localhost:8001/api/v1/system/status | jq .

# Check recent decisions
echo -e "\\nRecent Decisions:"
curl -s http://localhost:8001/api/v1/system/stats | jq .

# Check layer performance
echo -e "\\nLayer Performance:"
curl -s http://localhost:8001/api/v1/layers/statistics | jq .

# Check logs for errors
echo -e "\\nRecent Errors:"
tail -n 20 /var/log/behavioral_auth.log | grep ERROR
"""
    
    deployment_script = """#!/bin/bash
# Production Deployment Script

set -e

echo "Deploying Behavioral Authentication System"
echo "============================================="

# Pull latest code
echo "Pulling latest code..."
git pull origin main

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run tests
echo "Running tests..."
python verify_layers.py

# Update database schema if needed
echo "Updating database..."
# Add database migration commands here

# Restart services
echo "Restarting services..."
sudo systemctl restart behavioral-auth
sudo systemctl restart nginx

# Verify deployment
echo "Verifying deployment..."
sleep 10
curl -f http://localhost:8001/health || exit 1

echo "Deployment successful!"
"""
    
    try:
        with open("scripts/monitor.sh", 'w') as f:
            f.write(monitoring_script)
        os.chmod("scripts/monitor.sh", 0o755)
        
        with open("scripts/deploy.sh", 'w') as f:
            f.write(deployment_script)
        os.chmod("scripts/deploy.sh", 0o755)
        
        print("✅ Monitoring scripts created in scripts/")
        return True
        
    except Exception as e:
        print(f"❌ Error creating monitoring scripts: {e}")
        return False

def main():
    """Main deployment function"""
    print("🚀 BEHAVIORAL AUTHENTICATION SYSTEM DEPLOYMENT")
    print("=" * 60)
    print("Deploying Layers G, H, J for National-Level Security")
    print("=" * 60)
    
    # Create scripts directory
    os.makedirs("scripts", exist_ok=True)
    
    # Run deployment steps
    steps = [
        ("Dependencies Check", check_dependencies),
        ("Layer Tests", run_layer_tests),
        ("Production Config", create_production_config),
        ("Docker Deployment", create_docker_deployment),
        ("Monitoring Scripts", create_monitoring_scripts)
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        print(f"\n📋 Step: {step_name}")
        print("-" * 40)
        
        try:
            if step_func():
                print(f"✅ {step_name} completed successfully")
            else:
                print(f"❌ {step_name} failed")
                failed_steps.append(step_name)
        except Exception as e:
            print(f"❌ {step_name} failed with error: {e}")
            failed_steps.append(step_name)
    
    # Final status
    print("\n" + "=" * 60)
    if not failed_steps:
        print("🎉 DEPLOYMENT PREPARATION COMPLETE!")
        print("✅ All layers implemented and tested")
        print("✅ Production configuration ready")
        print("✅ Docker deployment files created")
        print("✅ Monitoring scripts ready")
        print("\n📋 NEXT STEPS:")
        print("1. Update production_config.json with your database credentials")
        print("2. Train GNN models on historical data")
        print("3. Run: docker-compose up -d")
        print("4. Monitor with: ./scripts/monitor.sh")
        print("\n🚀 READY FOR NATIONAL DEPLOYMENT!")
    else:
        print(f"❌ DEPLOYMENT PREPARATION FAILED")
        print(f"Failed steps: {failed_steps}")
        print("Please fix the issues and run again.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
