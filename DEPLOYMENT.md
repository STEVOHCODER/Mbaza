# Youth Sexuality Hub - Deployment Guide

## Recent Updates

### ✅ Fixed Vercel Deployment Errors
- Removed unused TypeScript imports/variables that were blocking the build
- Build now succeeds with zero errors

### ✅ Added Authentication System
- **Email/Password Login** - Users must register and login before accessing the app
- **User Management** - Backend stores user data with password hashing using bcrypt
- **Token-Based Auth** - JWT tokens valid for 7 days
- **Protected Routes** - Frontend gates access behind auth checks
- **Logout Button** - Displays current user email with logout option

### ✅ Mobile-Friendly Design
- Already using Tailwind CSS with responsive breakpoints
- All UI elements work on mobile, tablet, and desktop
- Touch-friendly button sizes and spacing

## Deployment to Vercel (Frontend)

### Prerequisites
- Vercel account (https://vercel.com)
- GitHub repository with this code pushed

### Steps

1. **Connect to Vercel**
   - Go to https://vercel.com
   - Click "Add New..." → "Project"
   - Select your GitHub repository

2. **Configure Environment Variables**
   - In Vercel dashboard, go to Settings → Environment Variables
   - Add: `VITE_API_URL=https://your-api-domain.com` (your backend URL)
   - Leave `VITE_GOOGLE_CLIENT_ID` empty for now (optional)

3. **Deploy**
   - Vercel will auto-deploy on every push to main branch
   - Build command: `npm run build`
   - Output directory: `dist`

### Notes
- The app requires a backend API running to function
- Users will see login screen if API is unreachable
- Build time: ~30 seconds

## Deployment of Backend (API)

### Option 1: Deploy to Render (Recommended for beginners)

1. **Prepare Backend**
   - Ensure `.env` file has: `GEMINI_API_KEY=your_key`, `JWT_SECRET=your_secret`
   - Requirements already in `requirements.txt`

2. **Connect to Render**
   - Go to https://render.com
   - Click "New+" → "Web Service"
   - Connect your GitHub repository

3. **Configure**
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app`
   - Environment variables from `.env`

### Option 2: Deploy to Railway
Similar process, but uses `python main.py` as start command.

## Environment Variables

### Frontend (.env.local)
```
VITE_API_URL=http://localhost:8000  # Local development
VITE_API_URL=https://your-api.com   # Production
```

### Backend (.env)
```
GEMINI_API_KEY=your_api_key
JWT_SECRET=your_secret_key_for_tokens
DATABASE_URL=sqlite:///./autodev_enterprise.db  # or PostgreSQL URL
REDIS_URL=redis://localhost:6379/0  # Optional
```

## Database Setup

The app uses SQLite by default (no setup needed for local dev).

For production, use PostgreSQL:
1. Create a PostgreSQL database
2. Set `DATABASE_URL=postgresql://user:password@host/dbname` in backend `.env`
3. Tables are auto-created on first run

## Testing Locally

### Start Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
# Server runs on http://localhost:8000
```

### Start Frontend
```bash
cd frontend
npm install
npm run dev
# App opens on http://localhost:5173
```

### Test Login
1. Open http://localhost:5173
2. Click "Sign Up" to create account
3. Enter email, password, name
4. Click "Sign In" with credentials

## Mobile App Deployment (Android)

To host on Google Play Store, use Capacitor or React Native:

### Option 1: Capacitor (Easiest - wraps web app)
```bash
npm install @capacitor/core @capacitor/cli
npm install @capacitor/android @capacitor/ios

npx cap init "Youth Hub" "com.youthhub.app"
npm run build
npx cap add android
# Opens Android Studio - configure and build APK
```

Then upload APK to Google Play Console.

### Option 2: React Native
Rewrite app using React Native for better native performance (more work).

## Security Notes

⚠️ **Production Checklist:**
- [ ] Change `JWT_SECRET` to a strong random value
- [ ] Enable HTTPS on both frontend and backend
- [ ] Set `CORS` to specific domain (not `*`)
- [ ] Use environment variables for all secrets
- [ ] Enable database encryption if using sensitive data
- [ ] Add rate limiting on login endpoint

## API Endpoints

### Authentication
- `POST /api/auth/register` - Create new account
- `POST /api/auth/login` - Login and get token

### Period Tracking
- `GET /api/period/{session_id}` - Get period entries
- `POST /api/period/{session_id}` - Add period entry

### Calendar
- `GET /api/calendar/{session_id}` - Get events
- `POST /api/calendar/{session_id}` - Add event

### AI Chat (WebSocket)
- `ws://localhost:8000/chat` - Connect for real-time chat

## Troubleshooting

### "Cannot find VITE_API_URL"
- Create `.env.local` in frontend folder
- Add: `VITE_API_URL=http://localhost:8000`

### "Login always fails"
- Check backend is running: `curl http://localhost:8000/api/health`
- Check database: `ls autodev_enterprise.db`
- Check CORS in backend

### Build fails on Vercel
- Run `npm run build` locally first
- Check TypeScript errors: `npx tsc --noEmit`

## Support

For issues or questions, check:
1. Frontend console (F12) for errors
2. Backend logs in terminal
3. Network tab for failed API requests
