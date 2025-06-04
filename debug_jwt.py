#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from auth.jwt_handler import JWTHandler

# Load environment variables
load_dotenv()

# Test JWT functionality
def test_jwt():
    print("=== JWT Debug Test ===")
    
    # Check environment variables
    jwt_secret = os.getenv("JWT_SECRET_KEY", "default-not-found")
    print(f"JWT_SECRET_KEY from env: {jwt_secret[:20]}...")
    
    # Create a test token
    test_data = {"sub": 5, "username": "andrewbrowne"}
    token = JWTHandler.create_access_token(test_data)
    print(f"Created token: {token[:50]}...")
    
    # Try to verify the token
    payload = JWTHandler.verify_token(token)
    print(f"Verified payload: {payload}")
    
    # Decode without verification
    decoded = JWTHandler.decode_token(token)
    print(f"Decoded without verification: {decoded}")
    
    # Test with the actual token from login
    actual_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOjUsInVzZXJuYW1lIjoiYW5kcmV3YnJvd25lIiwiZXhwIjoxNzQ5MDYxMTk0LCJ0eXBlIjoiYWNjZXNzIn0.gdAZj56lg5XrsU22gL6vu5PJdf0XHm4SPkYGwZxYHEA"
    print(f"\n=== Testing actual token ===")
    actual_payload = JWTHandler.verify_token(actual_token)
    print(f"Actual token verified: {actual_payload}")
    
    actual_decoded = JWTHandler.decode_token(actual_token)
    print(f"Actual token decoded: {actual_decoded}")

if __name__ == "__main__":
    test_jwt() 