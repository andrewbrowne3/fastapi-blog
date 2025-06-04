#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from jose import JWTError, jwt
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# JWT Configuration (same as jwt_handler.py)
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"

def test_jwt():
    print("=== Simple JWT Test ===")
    
    # Check environment variables
    print(f"JWT_SECRET_KEY from env: {SECRET_KEY[:20]}...")
    print(f"Full secret length: {len(SECRET_KEY)}")
    
    # Create a test token (same logic as JWTHandler)
    test_data = {"sub": "5", "username": "andrewbrowne"}
    expire = datetime.utcnow() + timedelta(minutes=30)
    to_encode = test_data.copy()
    to_encode.update({"exp": expire, "type": "access"})
    
    token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    print(f"Created token: {token[:50]}...")
    
    # Try to verify the token
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        print(f"✅ Self-created token verified: {payload}")
    except JWTError as e:
        print(f"❌ Self-created token failed: {e}")
    
    # Test with the actual token from login
    actual_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOjUsInVzZXJuYW1lIjoiYW5kcmV3YnJvd25lIiwiZXhwIjoxNzQ5MDYxMTk0LCJ0eXBlIjoiYWNjZXNzIn0.gdAZj56lg5XrsU22gL6vu5PJdf0XHm4SPkYGwZxYHEA"
    print(f"\n=== Testing actual login token ===")
    
    # Decode without verification first
    try:
        decoded = jwt.decode(actual_token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_signature": False})
        print(f"Decoded without verification: {decoded}")
    except JWTError as e:
        print(f"❌ Decode failed: {e}")
    
    # Try to verify with current secret
    try:
        payload = jwt.decode(actual_token, SECRET_KEY, algorithms=[ALGORITHM])
        print(f"✅ Actual token verified: {payload}")
    except JWTError as e:
        print(f"❌ Actual token verification failed: {e}")
        
    # Try with the default secret
    default_secret = "your-secret-key-change-this-in-production"
    print(f"\n=== Testing with default secret ===")
    try:
        payload = jwt.decode(actual_token, default_secret, algorithms=[ALGORITHM])
        print(f"✅ Verified with default secret: {payload}")
    except JWTError as e:
        print(f"❌ Default secret failed: {e}")

if __name__ == "__main__":
    test_jwt() 