# 🔧 Authentication Debugging Guide

## ✅ Issues Fixed

### **1. Endpoint Path Issue** ✓
- **Problem:** Frontend was calling `/auth/register` but backend expects `/api/auth/register`
- **Status:** FIXED - Now using correct paths

### **2. Pydantic Model Issue** ✓  
- **Problem:** `Optional[str] = None` wasn't properly defined
- **Status:** FIXED - Now using proper `Optional[str]` type

### **3. Better Error Messages** ✓
- **Added:** Console logging to track auth flow
- **Added:** Better error messages that display to user

---

## 🧪 Step-by-Step Testing

### **Step 1: Start Backend**

```bash
cd backend
pip install -r requirements.txt
python main.py
```

**Expected Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Test if it's running:**
```bash
curl http://localhost:8000/api/health
```

Should return:
```json
{"status":"ok","version":"1.0.0","uptime":"99.9%"}
```

---

### **Step 2: Start Frontend**

```bash
cd frontend
npm install  # Only first time
npm run dev
```

**Expected Output:**
```
Local:   http://localhost:5173/
```

---

### **Step 3: Test Registration**

1. **Open Browser Console** (F12 → Console tab)
2. **Go to** http://localhost:5173
3. **Click "Sign Up"**
4. **Enter:**
   - Name: `Test User`
   - Email: `test@example.com`
   - Password: `password123`
   - Confirm Password: `password123`
5. **Click "Sign Up"**

**What to look for in Console:**
```
Register attempt with: {email: 'test@example.com', name: 'Test User', password: '...'}
Calling register endpoint: http://localhost:8000/api/auth/register
Response status: 200
Register response data: {access_token: '...', token_type: 'bearer', user: {...}}
Registration successful!
```

**Expected Result:**
- ✅ No error messages appear
- ✅ Page redirects to main app
- ✅ You see "YouthHub" header with your email in top right

---

### **Step 4: Test Login After Logout**

1. **Click "Logout"** button (top right)
2. **You should see login screen again**
3. **Enter same credentials:**
   - Email: `test@example.com`
   - Password: `password123`
4. **Click "Sign In"**

**What to look for in Console:**
```
Login attempt with: {email: 'test@example.com', password: '...'}
Calling login endpoint: http://localhost:8000/api/auth/login
Response status: 200
Login response data: {access_token: '...', token_type: 'bearer', user: {...}}
Login successful!
```

---

### **Step 5: Test Invalid Credentials**

1. **Try login with wrong password:**
   - Email: `test@example.com`
   - Password: `wrongpassword`
2. **Click "Sign In"**

**Expected:**
- ❌ Error message appears: "Incorrect email or password"
- Console shows:
  ```
  Response status: 401
  Register error response: {detail: 'Incorrect email or password'}
  ```

---

### **Step 6: Test Session Persistence**

1. **After successful login, refresh page** (F5)
2. **Should NOT go back to login screen**
3. **Should stay logged in**
4. **Check localStorage:**
   - Open DevTools → Application → Local Storage
   - Should see `auth_token` and `user_data`

---

## 🐛 Common Issues & Solutions

### **Issue: "Cannot connect to server" Error**

**Solution:**
1. Check if backend is running: `curl http://localhost:8000/api/health`
2. If not running, start it: `cd backend && python main.py`
3. Verify no other app is using port 8000

### **Issue: "Email already registered"**

**Solution:**
1. Use a different email: `test2@example.com`, `test3@example.com`, etc.
2. Or delete the database and restart:
   ```bash
   rm backend/autodev_enterprise.db
   python main.py
   ```

### **Issue: Credentials disappear but nothing happens**

**Solution:**
1. Open browser console (F12)
2. Look for red errors
3. Check if API URL is correct:
   - Go to console, type: `console.log(import.meta.env.VITE_API_URL)`
   - Should show: `http://localhost:8000`

### **Issue: "Password must be at least 6 characters"**

**Solution:**
- Use password with 6+ characters
- Example: `password123` not `pass`

---

## 🔍 Advanced Debugging

### **Check Database**

```bash
# On Windows or any OS, use SQLite viewer
# Or check if tables exist:
python -c "from backend.main import *; import sqlite3; conn = sqlite3.connect('backend/autodev_enterprise.db'); print(conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall())"
```

### **Test API Directly with curl**

**Register:**
```bash
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"curl@test.com\",\"password\":\"password123\",\"name\":\"Curl User\"}"
```

**Login:**
```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"curl@test.com\",\"password\":\"password123\"}"
```

### **Enable Backend Logging**

Edit `backend/main.py`, add at top:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Then restart backend to see detailed logs.

---

## ✅ Verification Checklist

- [ ] Backend runs without errors
- [ ] Frontend runs without errors
- [ ] Can navigate to http://localhost:5173
- [ ] Can see login screen
- [ ] Can create new account with "Sign Up"
- [ ] After signup, redirected to main app
- [ ] Can see email in top right corner
- [ ] Can click "Logout"
- [ ] After logout, back at login screen
- [ ] Can login with credentials
- [ ] Period tracking works while logged in
- [ ] AI chat works while logged in
- [ ] Page refresh keeps you logged in
- [ ] Password validation works (rejects short passwords)

---

## 📱 Testing on Mobile

### **Using Your Computer's IP**

1. Find your computer's IP: `ipconfig` (Windows) or `ifconfig` (Mac/Linux)
2. Find IPv4 address (e.g., `192.168.x.x`)
3. On your phone, go to: `http://192.168.x.x:5173`
4. Should see login screen on mobile

---

## 🚀 After Everything Works Locally

1. **Commit changes:**
   ```bash
   git add -A
   git commit -m "fix: correct auth endpoints and improve error handling"
   git push origin main
   ```

2. **Deploy frontend to Vercel**
3. **Deploy backend to Render**
4. **Update frontend's VITE_API_URL** to production backend URL
5. **Test on production URLs**

---

## 📞 If Still Having Issues

1. **Screenshot console errors** (F12 → Console)
2. **Check both frontend AND backend terminal** for errors
3. **Verify backend is running** on port 8000
4. **Verify frontend is running** on port 5173
5. **Try clearing browser cache** (Ctrl+Shift+Delete)
6. **Try incognito/private window** (rules out cache issues)

---

**You should now be able to register, login, and use the app!** 🎉
