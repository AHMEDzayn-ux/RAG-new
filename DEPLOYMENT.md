# Production Deployment Guide

## 🚀 Quick Start

1. **Deploy Backend to Render** (Step-by-step below)
2. **Configure Vercel with Backend URL**
3. **Test Production System**

---

## 📦 Backend Deployment (Render.com - FREE)

### Step 1: Commit Deployment Files

```bash
cd "F:\My projects\RAG"
git add backend/Procfile render.yaml backend/main.py
git commit -m "feat: Add production deployment configuration"
git push origin main
```

### Step 2: Create Render Account

1. Go to **https://render.com**
2. Click **"Get Started for Free"**
3. Sign up with GitHub (easiest option)

### Step 3: Deploy Backend

1. **In Render Dashboard**, click **"New +"** → **"Web Service"**

2. **Connect Repository:**
   - Click **"Connect a repository"**
   - Authorize Render to access your GitHub
   - Select: **`AHMEDzayn-ux/RAG-new`**

3. **Configure Service:**

   ```
   Name: rag-backend
   Region: Oregon (or closest to you)
   Branch: main
   Root Directory: backend
   Runtime: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
   Instance Type: Free
   ```

4. **Environment Variables** (Click "Add Environment Variable"):

   ```
   GROQ_API_KEY = your_actual_groq_api_key_here
   ENVIRONMENT = production
   LOG_LEVEL = INFO
   ```

   ⚠️ **IMPORTANT:** Copy your actual Groq API key from your local `.env` file!

5. Click **"Create Web Service"**

6. **Wait for deployment** (5-10 minutes first time)
   - Watch the logs in Render dashboard
   - Status will change to "Live" when ready

7. **Copy your Backend URL**
   - Example: `https://rag-backend.onrender.com`
   - You'll need this for Vercel configuration

---

## 🌐 Frontend Configuration (Vercel)

### Step 1: Set Environment Variable

1. Go to **https://vercel.com/dashboard**

2. Select your project: **`rag-new-henna`**

3. Click **"Settings"** → **"Environment Variables"**

4. **Add New Variable:**

   ```
   Key: VITE_API_URL
   Value: https://rag-backend.onrender.com
          ☝️ Use YOUR actual Render URL from above

   Environment: Production, Preview, Development (check all)
   ```

5. Click **"Save"**

### Step 2: Redeploy Frontend

1. Go to **"Deployments"** tab

2. Find latest deployment → Click **"..."** → **"Redeploy"**

3. Wait for rebuild (2-3 minutes)

4. Frontend will now connect to your production backend!

---

## ✅ Testing Production Deployment

### Test Backend

```bash
# Test backend health (replace with YOUR URL)
curl https://rag-backend.onrender.com/health

# Expected response:
{"status":"healthy","message":"RAG Chatbot API is running"}
```

### Test Frontend

1. Visit: **https://rag-new-henna.vercel.app**

2. **Check Browser Console** (F12):
   - Look for: `🔗 API Base URL: https://rag-backend.onrender.com`
   - Should NOT see CORS errors
   - Clients should load successfully

3. **Test Full Flow:**
   - Create a client
   - Upload a document
   - Ask a question
   - Verify response works

---

## 🎯 Architecture Overview

```
┌─────────────────────────────────────────┐
│   Vercel (Frontend)                     │
│   https://rag-new-henna.vercel.app     │
│   - React + Vite                        │
│   - Static files                        │
│   - Animated UI                         │
└─────────────┬───────────────────────────┘
              │ HTTPS requests
              ↓
┌─────────────────────────────────────────┐
│   Render (Backend)                      │
│   https://rag-backend.onrender.com     │
│   - FastAPI + Python                    │
│   - RAG Pipeline                        │
│   - Rate Limiting                       │
│   - Security Middleware                 │
│   - Advanced Retrieval                  │
└─────────────┬───────────────────────────┘
              │ API calls
              ↓
┌─────────────────────────────────────────┐
│   Groq API (LLM)                        │
│   https://api.groq.com                  │
│   - Llama 3.3 70B                       │
│   - Fast inference                      │
└─────────────────────────────────────────┘
```

---

## ⚙️ Configuration Files

### backend/Procfile

```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

### render.yaml

```yaml
services:
  - type: web
    name: rag-backend
    runtime: python
    region: oregon
    plan: free
    rootDir: backend
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Vercel Environment Variable

```
VITE_API_URL=https://rag-backend.onrender.com
```

---

## 🔒 Security Notes

### Production Settings

- ✅ CORS restricted to Vercel domains only
- ✅ Rate limiting active (30 req/min for queries)
- ✅ Security headers enabled
- ✅ Input validation active
- ✅ API key stored as environment variable
- ✅ HTTPS enforced

### Free Tier Limitations

**Render Free Tier:**

- Backend sleeps after 15 minutes of inactivity
- First request after sleep takes ~30-60 seconds (cold start)
- 750 hours/month of runtime (enough for most use)
- **Solution:** Keep-alive ping service (optional)

**Keep Backend Awake (Optional):**

```bash
# Add this to a cron job or UptimeRobot
curl https://rag-backend.onrender.com/health
```

Run every 10 minutes to prevent sleep.

---

## 🐛 Troubleshooting

### Issue: "Service Unavailable"

**Cause:** Backend is sleeping (Render free tier)

**Solution:**

- Wait 30-60 seconds and refresh
- First request wakes up the service
- Subsequent requests are fast

### Issue: CORS Error

**Symptoms:**

```
Access to XMLHttpRequest blocked by CORS policy
```

**Solutions:**

1. Check Render logs: Is backend running?
2. Verify ENVIRONMENT=production set in Render
3. Check Vercel has correct VITE_API_URL
4. Redeploy both frontend and backend

**Debug:**

```bash
# Check CORS headers
curl -I https://rag-backend.onrender.com/api/clients/ \
  -H "Origin: https://rag-new-henna.vercel.app"

# Should include:
# access-control-allow-origin: https://rag-new-henna.vercel.app
```

### Issue: "Unable to load clients"

**Checklist:**

- [ ] Backend deployed and live
- [ ] Environment variable VITE_API_URL set in Vercel
- [ ] Frontend redeployed after setting env var
- [ ] Check browser console for actual error

### Issue: Rate Limiting

**Symptoms:**

```json
{
  "detail": "Rate limit exceeded. Please try again later.",
  "retry_after": "45"
}
```

**Cause:** Exceeded 30 requests/minute

**Solution:** Wait 45 seconds or adjust rate limits in `backend/security.py`

---

## 📊 Monitoring

### Render Dashboard

- **Logs:** Real-time application logs
- **Metrics:** CPU, Memory, Response times
- **Events:** Deployments, crashes, restarts

### Vercel Analytics

- **Performance:** Page load times
- **Errors:** Runtime errors
- **Traffic:** User visits, geographic distribution

---

## 🔄 Updating Production

### Backend Updates

```bash
# Make changes to backend
git add backend/
git commit -m "Update backend feature"
git push origin main

# Render auto-deploys from main branch
# Check deployment progress in Render dashboard
```

### Frontend Updates

```bash
# Make changes to frontend
git add frontend/
git commit -m "Update frontend feature"
git push origin main

# Vercel auto-deploys from main branch
# Check deployment progress in Vercel dashboard
```

---

## 💰 Cost Breakdown

### Current Setup (FREE!)

| Service  | Plan  | Cost  | Limits                         |
| -------- | ----- | ----- | ------------------------------ |
| Render   | Free  | $0/mo | 750 hrs/mo, sleeps after 15min |
| Vercel   | Hobby | $0/mo | 100 GB bandwidth/mo            |
| Groq API | Free  | $0/mo | Rate limited                   |
| GitHub   | Free  | $0/mo | Unlimited public repos         |

**Total: $0/month** 🎉

### Upgrade Options (Optional)

**Render Starter ($7/mo):**

- No sleep (always on)
- Better performance
- More RAM

**Vercel Pro ($20/mo):**

- More bandwidth
- Analytics
- Better support

---

## 📚 Next Steps

1. ✅ Deploy backend to Render
2. ✅ Configure Vercel environment variable
3. ✅ Test production deployment
4. 📊 Monitor logs and metrics
5. 🎨 Customize frontend branding
6. 📄 Add more documents to knowledge base
7. 🚀 Share with users!

---

## 🆘 Support

**Render Support:**

- Docs: https://render.com/docs
- Community: https://community.render.com

**Vercel Support:**

- Docs: https://vercel.com/docs
- Discord: https://vercel.com/discord

**Project Issues:**

- GitHub: https://github.com/AHMEDzayn-ux/RAG-new/issues

---

**🎉 Your production RAG system is ready to launch!**

Follow the steps above and you'll have a fully functional, secure, production-grade AI chatbot system deployed in 15-20 minutes.
