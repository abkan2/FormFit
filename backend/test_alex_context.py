#!/usr/bin/env python3
"""
Simple test script to verify ALEX user context functionality
"""

import asyncio
import httpx
import json

BACKEND_URL = "http://localhost:8000/api/v1"

async def test_user_context():
    """Test user context retrieval and chat with personalization"""
    
    async with httpx.AsyncClient() as client:
        print("Testing ALEX User Context Integration\n")
        
        # Test 1: Get user context
        print("1. Testing user context retrieval...")
        try:
            response = await client.get(f"{BACKEND_URL}/user/tADFea1gzlRBZbjMyPSCiLw7DXj2")
            if response.status_code == 200:
                user_data = response.json()
                print("✅ User context retrieved successfully!")
                print(f"   User: {user_data['user_data']['personalInfo']['name']}")
                print(f"   Goals: {', '.join(user_data['user_data']['goals'])}")
                print(f"   Context length: {len(user_data['formatted_context'])} characters")
            else:
                print(f"❌ Failed to get user context: {response.status_code}")
                return
        except Exception as e:
            print(f"❌ Error getting user context: {e}")
            return
        
        # Test 2: Test chat with user context
        print("\n2. Testing personalized chat...")
        try:
            chat_payload = {
                "messages": [
                    {"role": "user", "content": "Hi ALEX! I just finished my onboarding. Can you welcome me?"}
                ],
                "user_id": "test_user_123"
            }
            
            response = await client.post(f"{BACKEND_URL}/chat", json=chat_payload)
            if response.status_code == 200:
                chat_response = response.json()
                print("✅ Personalized chat response received!")
                print(f"   Response: {chat_response['text'][:200]}...")
                
                # Check if response contains user name
                if "Alex Test User" in chat_response['text']:
                    print("✅ Response is personalized with user name!")
                else:
                    print("⚠️  Response doesn't seem to include user name")
                    
            else:
                print(f"❌ Chat failed: {response.status_code}")
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"❌ Error in chat: {e}")
        
        # Test 3: Test chat without user context
        print("\n3. Testing chat without user context...")
        try:
            chat_payload = {
                "messages": [
                    {"role": "user", "content": "Hi ALEX! Can you tell me about FormFit?"}
                ]
                # No user_id provided
            }
            
            response = await client.post(f"{BACKEND_URL}/chat", json=chat_payload)
            if response.status_code == 200:
                chat_response = response.json()
                print("✅ General chat response received!")
                print(f"   Response: {chat_response['text'][:200]}...")
            else:
                print(f"❌ General chat failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Error in general chat: {e}")

if __name__ == "__main__":
    asyncio.run(test_user_context())
