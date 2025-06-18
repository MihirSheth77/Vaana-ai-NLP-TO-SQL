#!/usr/bin/env python3
"""
Health check script for Vanna AI application
"""
import sys
import requests
import time

def check_backend(max_retries=3):
    """Check if backend API is healthy"""
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/", timeout=5)
            if response.status_code == 200:
                print("✅ Backend API is healthy")
                return True
        except Exception as e:
            print(f"❌ Backend API check failed (attempt {i+1}/{max_retries}): {e}")
            if i < max_retries - 1:
                time.sleep(2)
    return False

def check_frontend(max_retries=3):
    """Check if frontend is healthy"""
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8501/", timeout=5)
            if response.status_code == 200:
                print("✅ Frontend UI is healthy")
                return True
        except Exception as e:
            print(f"❌ Frontend UI check failed (attempt {i+1}/{max_retries}): {e}")
            if i < max_retries - 1:
                time.sleep(2)
    return False

if __name__ == "__main__":
    backend_ok = check_backend()
    frontend_ok = check_frontend()
    
    if backend_ok and frontend_ok:
        print("✅ All services are healthy!")
        sys.exit(0)
    else:
        print("❌ Some services are not healthy")
        sys.exit(1) 