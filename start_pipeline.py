#!/usr/bin/env python3
"""
Startup script for the integrated anomaly detection pipeline
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import pandas
        import numpy
        import sklearn
        print("✅ Core dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install -r requirements.txt")
        return False

def check_alert_system():
    """Check if alert system is properly set up"""
    alert_system_path = Path("alert_system/alert_system")
    if not alert_system_path.exists():
        print("❌ Alert system not found")
        return False
    
    # Check if Prisma is set up
    prisma_schema = alert_system_path / "prisma" / "schema.prisma"
    if not prisma_schema.exists():
        print("❌ Prisma schema not found")
        return False
    
    print("✅ Alert system found")
    return True

def setup_alert_system():
    """Set up the alert system database"""
    print("🔧 Setting up alert system...")
    
    # Change to alert system directory
    os.chdir("alert_system/alert_system")
    
    try:
        # Generate Prisma client
        subprocess.run(["npx", "prisma", "generate"], check=True)
        print("✅ Prisma client generated")
        
        # Push database schema (if DATABASE_URL is set)
        if os.getenv("DATABASE_URL"):
            subprocess.run(["npx", "prisma", "db", "push"], check=True)
            print("✅ Database schema pushed")
        else:
            print("⚠️  DATABASE_URL not set, skipping database setup")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to setup alert system: {e}")
        return False
    except FileNotFoundError:
        print("❌ Node.js/npx not found. Install Node.js to use the alert system")
        return False
    finally:
        # Change back to root directory
        os.chdir("../..")
    
    return True

def main():
    """Main startup function"""
    print("🚀 Anomaly Detection Pipeline Startup")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check alert system
    alert_system_available = check_alert_system()
    
    if alert_system_available:
        setup_alert_system()
    
    # Start the main application
    print("\n🎯 Starting integrated pipeline...")
    print("📊 Anomaly Detection: http://localhost:8000")
    print("🚨 Alert System: http://localhost:8001")
    print("📖 API Docs: http://localhost:8000/docs")
    print("🎛️  Alert Dashboard: http://localhost:8001")
    print("\nPress Ctrl+C to stop")
    
    # Import and run main
    from main import main as run_main
    run_main()

if __name__ == "__main__":
    main()
