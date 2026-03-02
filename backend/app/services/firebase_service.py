"""Firebase service for user data management"""

import os
import json
import firebase_admin
from firebase_admin import credentials, firestore, auth
from typing import Optional, Dict, Any
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# ✅ FIXED: Security should be at module level, not inside class
security = HTTPBearer()

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
                        print("⚠️ Warning: No Firebase credentials found. Running in development mode.")
                        self.db = None
                        return
            
            self.db = firestore.client()
            print("Firebase initialized successfully")
            
        except Exception as e:
            print(f"⚠️ Warning: Firebase initialization failed: {e}")
            # Continue without Firebase for development
            self.db = None
    
    async def get_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Fetch user data from Firestore"""
        if not self.db:
            print("⚠️ Firebase not initialized - cannot fetch user data")
            return None
            
        try:
            user_ref = self.db.collection('users').document(user_id)
            user_doc = user_ref.get()
            
            if user_doc.exists:
                return user_doc.to_dict()
            else:
                print(f"⚠️ User {user_id} not found in Firestore")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching user data: {e}")
            return None
    
    async def save_user_data(self, user_id: str, data: Dict[str, Any]) -> bool:
        """Save/update user data in Firestore"""
        if not self.db:
            print("⚠️ Firebase not initialized - cannot save user data")
            return False
            
        try:
            user_ref = self.db.collection('users').document(user_id)
            user_ref.set(data, merge=True)  # merge=True updates existing fields
            print(f"✅ User data saved for {user_id}")
            return True
                
        except Exception as e:
            print(f"❌ Error saving user data: {e}")
            return False
    
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


# ✅ FIXED: Auth functions at MODULE level (not inside class)
async def verify_firebase_token(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Dict[str, Any]:
    """
    Verify Firebase ID token and return decoded token
    
    Usage in route:
        from app.services.firebase_service import verify_firebase_token
        
        @router.get("/protected")
        async def protected_route(token_data = Depends(verify_firebase_token)):
            user_id = token_data['uid']
    """
    try:
        token = credentials.credentials
        
        # Verify the Firebase ID token
        decoded_token = auth.verify_id_token(token)
        
        return decoded_token  # Contains: uid, email, email_verified, etc.
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Dict[str, Any]:
    """
    Get current authenticated user data
    
    Returns decoded Firebase token with user info:
    {
        'uid': 'firebase_user_id',
        'email': 'user@example.com',
        'email_verified': True,
        'name': 'User Name',
        ...
    }
    
    Usage in route:
        from app.services.firebase_service import get_current_user
        
        @router.get("/profile")
        async def get_profile(current_user = Depends(get_current_user)):
            user_id = current_user['uid']
            email = current_user['email']
    """
    decoded_token = await verify_firebase_token(credentials)
    return decoded_token


# ✅ Global instance for easy import
firebase_service = FirebaseService()