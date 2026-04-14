# 🔧 Port Management Guide

## Current Running Services

```
Frontend (React):    http://localhost:5175
Backend (FastAPI):   http://localhost:8000
```

## If Ports Get Stuck

### Frontend Port 5175 Stuck
```powershell
# Find process
netstat -ano | Select-String ":5175.*LISTENING"

# Kill it (replace PID)
taskkill /PID <PID> /F

# Restart
cd frontend
npm run dev
```

### Backend Port 8000 Stuck
```powershell
# Find process
netstat -ano | Select-String ":8000.*LISTENING"

# Kill it (replace PID)
taskkill /PID <PID> /F

# Restart
cd backend
python main.py
```

## Common Issue: "Port already in use"

This happens when a process doesn't fully close. Solution:

```powershell
# 1. Find which process is using the port
netstat -ano | Select-String ":8000.*LISTENING"

# 2. Kill the process
taskkill /PID 7560 /F

# 3. Wait 2 seconds
Start-Sleep -Seconds 2

# 4. Restart the service
cd backend
python main.py
```

## Quick Check - Are Services Running?

```powershell
Write-Host "Frontend:" ((netstat -ano | Select-String ":5175.*LISTENING") ? "✅ Running" : "❌ Not running")
Write-Host "Backend:" ((netstat -ano | Select-String ":8000.*LISTENING") ? "✅ Running" : "❌ Not running")
```

## Manual Port Cleanup Script

Save as `cleanup-ports.ps1`:

```powershell
# Kill port 8000
$pid8000 = (netstat -ano | Select-String ":8000.*LISTENING" | ForEach-Object { $_ -split '\s+' })[-1]
if ($pid8000 -match '^\d+$') { taskkill /PID $pid8000 /F }

# Kill port 5175
$pid5175 = (netstat -ano | Select-String ":5175.*LISTENING" | ForEach-Object { $_ -split '\s+' })[-1]
if ($pid5175 -match '^\d+$') { taskkill /PID $pid5175 /F }

Write-Host "Ports cleaned. Wait 2 seconds before restarting services."
Start-Sleep -Seconds 2
```

Run with: `.\cleanup-ports.ps1`

## Verify Services After Restart

```bash
# Test backend
curl http://localhost:8000/api/health

# Test frontend
# Open http://localhost:5175 in browser
```

## If Still Issues: Full Reset

```powershell
# 1. Kill all Python processes
Get-Process python | Stop-Process -Force

# 2. Kill Node processes  
Get-Process node | Stop-Process -Force

# 3. Wait
Start-Sleep -Seconds 3

# 4. Restart services in new terminals
cd backend
python main.py

# In another terminal:
cd frontend
npm run dev
```

---

**Remember:** Always give services 3-5 seconds to fully start after killing them.
