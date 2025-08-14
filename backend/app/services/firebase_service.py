"""Firebase service for user data management"""

import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
from typing import Optional, Dict, Any
from fastapi import HTTPException

class FirebaseService:
    def __init__(self):
        self.db = None
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase Admin SDK"""
        try:
            # Check if Firebase is already initialized
            if firebase_admin._apps:
                app = firebase_admin.get_app()
            else:
                # For development/testing - use emulator or skip Firebase
                firebase_config = os.getenv("FIREBASE_CONFIG")
                if firebase_config:
                    # If config is provided as JSON string
                    cred_dict = json.loads(firebase_config)
                    cred = credentials.Certificate(cred_dict)
                    app = firebase_admin.initialize_app(cred)
                else:
                    # Try service account file
                    service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "serviceAccountKey.json")
                    if os.path.exists(service_account_path):
                        cred = credentials.Certificate(service_account_path)
                        app = firebase_admin.initialize_app(cred)
                    else:
                        # For development - create a mock setup
                        print("Warning: No Firebase credentials found. Running in development mode.")
                        self.db = None
                        return
            
            self.db = firestore.client()
            print("Firebase initialized successfully")
            
        except Exception as e:
            print(f"Warning: Firebase initialization failed: {e}")
            # Continue without Firebase for development
            self.db = None
    
    async def get_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Fetch user data from Firestore"""
        if not self.db:
            # Return mock data for development
            return self._get_mock_user_data(user_id)
            
        try:
            user_ref = self.db.collection('users').document(user_id)
            user_doc = user_ref.get()
            
            if user_doc.exists:
                return user_doc.to_dict()
            else:
                return None
                
        except Exception as e:
            print(f"Error fetching user data: {e}")
            return None
    
    def _get_mock_user_data(self, user_id: str) -> Dict[str, Any]:
        """Return mock user data for development"""
        return {
            'personalInfo': {
                'name': 'Alex Test User',
                'age': '25',
                'gender': 'Male',
                'heightFeet': '5',
                'heightInches': '10',
                'weight': '150',
                'birthday': '1999-01-15'
            },
            'activityLevel': 'Moderately Active',
            'goals': ['Lose Weight', 'Gain Muscle', 'Improve Endurance'],
            'createdAt': '2025-08-13T14:30:00.000Z'
        }
    
    def format_user_context(self, user_data: Dict[str, Any]) -> str:
        """Format user data into context string for ALEX"""
        if not user_data:
            return "No user data available."
        
        context_parts = []
        
        # Basic info
        if 'personalInfo' in user_data:
            personal = user_data['personalInfo']
            name = personal.get('name', 'User')
            age = personal.get('age', 'Unknown')
            gender = personal.get('gender', 'Unknown')
            weight = personal.get('weight', 'Unknown')
            height_feet = personal.get('heightFeet', '')
            height_inches = personal.get('heightInches', '')
            
            context_parts.append(f"User Profile:")
            context_parts.append(f"- Name: {name}")
            context_parts.append(f"- Age: {age} years old")
            context_parts.append(f"- Gender: {gender}")
            if height_feet and height_inches:
                context_parts.append(f"- Height: {height_feet}'{height_inches}\"")
            if weight != 'Unknown':
                context_parts.append(f"- Weight: {weight} lbs")
        
        # Activity level
        if 'activityLevel' in user_data:
            activity = user_data['activityLevel']
            context_parts.append(f"- Activity Level: {activity}")
        
        # Goals
        if 'goals' in user_data:
            goals = user_data['goals']
            if isinstance(goals, list) and goals:
                context_parts.append(f"- Fitness Goals: {', '.join(goals)}")
        
        # Account info
        if 'createdAt' in user_data:
            context_parts.append(f"- Account created: {user_data['createdAt']}")
        
        return '\n'.join(context_parts)

# Global instance
firebase_service = FirebaseService()
