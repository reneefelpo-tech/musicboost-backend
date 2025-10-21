# Music Boost Backend API

FastAPI backend for Music Boost - Free music promotion platform with AI-driven optimization.

## Features

- 🎵 Spotify Integration & OAuth
- 📱 Meta (Facebook/Instagram/Threads) Integration & OAuth
- 🤖 AI-Powered Music Promotion
- 📊 Real-time Analytics & Stream Tracking
- 🎯 Smart Targeting & Audience Matching
- 🔐 Secure Authentication (JWT)
- 💾 MongoDB Database
- ⚡ Background Task Processing

## Tech Stack

- **Framework**: FastAPI
- **Database**: MongoDB (Motor async driver)
- **Authentication**: JWT, OAuth 2.0
- **Background Tasks**: AsyncIO, APScheduler
- **API Integrations**: Spotify, Meta Graph API

## Quick Start

### Prerequisites

- Python 3.9+
- MongoDB
- Spotify Developer Account
- Meta Developer Account

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your credentials

# Run the server
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

### Environment Variables

Required environment variables (see `.env.example`):

```bash
MONGO_URL=mongodb://localhost:27017
DB_NAME=musicboost_production
JWT_SECRET=your-secret-key
BASE_URL=https://yourdomain.com
SPOTIFY_CLIENT_ID=your-spotify-client-id
SPOTIFY_CLIENT_SECRET=your-spotify-client-secret
META_APP_ID=your-meta-app-id
META_APP_SECRET=your-meta-app-secret
META_REDIRECT_URI=https://yourdomain.com/api/meta/callback
```

## Deployment

### Railway

This project is configured for Railway deployment:

1. Push to GitHub
2. Connect repository to Railway
3. Add MongoDB database
4. Configure environment variables
5. Deploy!

See `railway.json` and `Procfile` for configuration.

## API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`

## Key Endpoints

- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `GET /api/links` - Get music links
- `POST /api/links` - Create music link with auto-boost
- `GET /api/spotify/connect` - Spotify OAuth
- `GET /api/meta/connect` - Meta OAuth

## Project Structure

```
backend/
├── server.py                 # Main FastAPI application
├── requirements.txt          # Python dependencies
├── railway.json             # Railway configuration
├── Procfile                 # Process definition
├── runtime.txt              # Python version
├── meta_service.py          # Meta API integration
├── meta_endpoints.py        # Meta API routes
├── spotify_service.py       # Spotify API integration
├── spotify_endpoints.py     # Spotify API routes
├── neural_coordinator.py    # AI coordination system
├── ai_consciousness.py      # AI mission & awareness
└── spiritual_consciousness.py # AI values & ethics
```

## License

Proprietary - All Rights Reserved

## Support

For issues and questions, contact the development team.
