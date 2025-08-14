# FormFit 🏋️‍♂️

**Your AI-Powered Personal Fitness Trainer**

FormFit is an intelligent fitness application that combines computer vision, machine learning, and conversational AI to provide personalized workout coaching. Using real-time pose detection and form analysis, FormFit helps users perform exercises correctly while tracking their progress through an interactive AI coach named A.L.E.X.

## 🚀 Vision

FormFit aims to democratize access to professional fitness coaching by leveraging cutting-edge AI technology. Our vision is to create an inclusive, intelligent, and interactive fitness companion that:

- **Makes fitness accessible** to everyone, regardless of experience level
- **Provides real-time form correction** to prevent injuries and maximize effectiveness
- **Offers personalized coaching** through conversational AI interactions
- **Tracks progress intelligently** with computer vision-based rep counting
- **Creates an engaging experience** that motivates users to achieve their fitness goals

## 🎯 Current Features

### 🤖 A.L.E.X - AI Fitness Coach
- **Conversational Interface**: Natural language interactions for workout guidance
- **Exercise Selection**: Intelligent recommendations for squats, push-ups, and planks
- **Personalized Sessions**: Tailored workouts based on user preferences and history
- **Voice Integration**: Talk to A.L.E.X during workouts for hands-free coaching

### 📱 Real-Time Pose Detection
- **Computer Vision**: MediaPipe-powered pose estimation and analysis
- **Exercise Recognition**: Automatic detection of squats, push-ups, and planks
- **Form Analysis**: Real-time evaluation of exercise technique with feedback
- **Rep Counting**: Intelligent repetition counting with beginner-friendly thresholds

### 👤 User Management
- **Firebase Authentication**: Secure user registration and login
- **Personal Profiles**: Comprehensive onboarding with health and fitness data
- **Progress Tracking**: Workout history and performance analytics
- **Goal Setting**: Customizable fitness objectives and milestones

### 💪 Workout Experience
- **Exercise-Specific Modes**: Focused detection for selected exercises
- **Real-Time Feedback**: Instant form corrections and encouragement
- **Progress Visualization**: Live workout stats and rep counters
- **Pause/Resume**: Flexible workout controls with motivational messaging

## 🛠️ Technology Stack

### Frontend
- **React Native** with Expo Router for cross-platform mobile development
- **Firebase SDK** for authentication and data storage
- **Expo Camera** for real-time video capture and processing
- **WebSocket** connections for real-time AI communication

### Backend
- **FastAPI** for high-performance API development
- **MediaPipe** for pose detection and landmark extraction
- **Scikit-learn** for machine learning model inference
- **WebSocket** for real-time pose analysis streaming

### AI/ML Pipeline
- **Exercise Detection Models**: Classification of different exercise types
- **Form Analysis Models**: Quality assessment for each exercise
- **OpenAI Integration**: Conversational AI for workout coaching
- **Feature Engineering**: Advanced pose-based feature extraction

## 📁 Project Structure

```
AI-Fitness-trainer/
├── frontend/                 # React Native mobile app
│   ├── app/                 # Expo Router app directory
│   │   ├── (auth)/         # Authentication screens
│   │   ├── (onboarding)/   # User onboarding flow
│   │   ├── (tabs)/         # Main app navigation
│   │   └── (workout)/      # Workout screens
│   ├── assets/             # Images, fonts, and styles
│   ├── components/         # Reusable UI components
│   ├── constants/          # App constants and colors
│   └── src/               # Core app logic
│       ├── context/       # React context providers
│       └── lib/          # Firebase and utility functions
├── backend/                # FastAPI backend server
│   ├── app/               # Application code
│   │   ├── models/        # ML models and encoders
│   │   ├── routes/        # API endpoints
│   │   └── services/      # Core services
│   ├── ml/                # Machine learning pipeline
│   │   ├── data/          # Training data
│   │   ├── notebooks/     # Jupyter notebooks
│   │   └── training/      # Model training scripts
│   └── tests/             # Unit tests
└── scripts/               # Utility scripts
```

## 🚀 Getting Started

### Prerequisites

- **Node.js** (v18 or higher)
- **Python** (3.8 or higher)
- **Expo CLI** (`npm install -g @expo/cli`)
- **iOS Simulator** or **Android Emulator** (for testing)
- **Firebase Project** with Authentication and Firestore enabled

### Backend Setup

1. **Navigate to backend directory**
   ```bash
   cd backend
   ```

2. **Create Python virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install additional ML dependencies**
   ```bash
   pip install mediapipe opencv-python scikit-learn joblib numpy
   ```

5. **Set up environment variables**
   ```bash
   # Create .env file in backend directory
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

6. **Start the backend server**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

   The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Configure Firebase**
   - Create a Firebase project at [Firebase Console](https://console.firebase.google.com)
   - Enable Authentication (Email/Password) and Firestore
   - Download the configuration file and update `src/lib/firebase.js`

4. **Update API endpoint**
   - In `app/(workout)/camera.jsx`, update the WebSocket URL:
   ```javascript
   const wsUrl = `ws://YOUR_LOCAL_IP:8000/api/v1/pose_detection?exercise=${selectedExercise}`;
   ```

5. **Start the development server**
   ```bash
   npx expo start
   ```

6. **Run on device/simulator**
   - Press `i` for iOS simulator
   - Press `a` for Android emulator
   - Scan QR code with Expo Go app for physical device testing

### Network Configuration

For real-time pose detection to work on physical devices:

1. **Find your local IP address**
   ```bash
   # macOS/Linux
   ifconfig | grep "inet " | grep -v 127.0.0.1
   
   # Windows
   ipconfig | findstr "IPv4"
   ```

2. **Update WebSocket URLs** in the frontend code to use your local IP
3. **Ensure devices are on the same network** as your development machine

## 🔧 Development

### Running Tests
```bash
# Backend tests
cd backend && python -m pytest tests/

# Frontend linting
cd frontend && npm run lint
```

### ML Model Training
```bash
cd backend/ml
python train.py  # Train exercise detection models
```

### Database Migrations
Firebase Firestore is schema-less, but ensure proper security rules are configured in the Firebase Console.

## 📊 API Documentation

When the backend is running, visit `http://localhost:8000/docs` for interactive API documentation powered by FastAPI's automatic OpenAPI generation.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🐛 Known Issues

- Camera permissions required for pose detection
- WebSocket connections may need firewall configuration
- ML models require sufficient device performance
- Network latency affects real-time feedback quality

## 🔮 Roadmap

- [ ] Additional exercise support (lunges, burpees, etc.)
- [ ] Advanced analytics and progress tracking
- [ ] Social features and challenges
- [ ] Offline mode capabilities
- [ ] Wearable device integration
- [ ] Custom workout creation tools

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe** for robust pose detection capabilities
- **OpenAI** for conversational AI technology
- **Firebase** for backend infrastructure
- **Expo** for cross-platform mobile development
- **FastAPI** for high-performance API framework

---

**Built with ❤️ for the fitness community**

For questions, issues, or contributions, please visit our [GitHub repository](https://github.com/abkan2/FormFit).