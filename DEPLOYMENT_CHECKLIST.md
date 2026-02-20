# ✅ WhatsApp Bot - Deployment Checklist

## Current Status

✅ **Backend Code**: Complete  
✅ **Credentials Configured**: Done  
✅ **Server Running**: http://localhost:8000

### Your Credentials (Now Configured)

```
Phone Number ID: 933812406490507
Access Token: EAAgKSZB... (configured in .env)
Verify Token: ragbot_verify_token_2026_secure
```

---

## 🚀 Next Steps (Required for WhatsApp to Work)

### Step 1: Deploy with HTTPS ⚠️ REQUIRED

WhatsApp webhooks **require HTTPS**. Choose one option:

#### Option A: Quick Testing with ngrok (5 minutes) ✨ RECOMMENDED FOR TESTING

1. **Download ngrok**: https://ngrok.com/download
2. **Extract and run**:
   ```bash
   # In a new terminal
   ngrok http 8000
   ```
3. **Copy the HTTPS URL** (e.g., `https://abc123.ngrok-free.app`)
4. **Use this URL** in Step 2 below

#### Option B: Deploy to Render (Production)

1. Go to https://render.com
2. Create new **Web Service**
3. Connect your GitHub repo
4. Configure:
   - **Build Command**: `cd backend && pip install -r requirements.txt`
   - **Start Command**: `cd backend && uvicorn main:app --host 0.0.0.0 --port 8000`
   - **Environment Variables**: Copy all from `backend/.env`
5. Deploy and copy the HTTPS URL

---

### Step 2: Configure Webhook in Meta Developer Console ⚠️ REQUIRED

1. Go to **Meta for Developers**: https://developers.facebook.com/apps
2. Select your app
3. Click **WhatsApp** → **Configuration**
4. Find **Webhook** section → Click **Edit**
5. Enter:

   ```
   Callback URL: https://YOUR-DOMAIN/webhooks/whatsapp
   Verify Token: ragbot_verify_token_2026_secure
   ```

   **Examples**:
   - ngrok: `https://abc123.ngrok-free.app/webhooks/whatsapp`
   - Render: `https://your-app.onrender.com/webhooks/whatsapp`

6. Click **Verify and Save**
   - ✅ Should show "Success" or similar
   - ❌ If fails, check URL and verify token match

7. **Subscribe to webhook fields**:
   - Check **messages** ✓
   - Click **Subscribe**

---

### Step 3: Test Your Bot 🎉

1. Open **WhatsApp** on your phone
2. Send a message to the test number: **+1 555 025 5745** (or your Meta test number)
3. You should receive:
   - ✅ Welcome message from bot
   - ✅ Answer to your question from RAG system

**Test Messages**:

```
help
How do I track my order?
What are your business hours?
clear
```

---

## 📋 Quick Verification Checklist

Before testing, verify:

- [ ] Backend running (locally or on Render)
- [ ] HTTPS URL obtained (ngrok or Render)
- [ ] Webhook configured in Meta dashboard
- [ ] Webhook shows "Success" verification
- [ ] "messages" field subscribed
- [ ] Your phone number added to test numbers in Meta
- [ ] Client `customer_care_test` has documents uploaded

---

## 🔍 Troubleshooting

### Issue: Webhook verification fails

**Check**:

1. URL is HTTPS (not HTTP)
2. URL ends with `/webhooks/whatsapp`
3. Verify token: `ragbot_verify_token_2026_secure`
4. Backend is running and accessible

**Test webhook locally**:

```bash
curl "http://localhost:8000/webhooks/whatsapp?hub.mode=subscribe&hub.challenge=TEST123&hub.verify_token=ragbot_verify_token_2026_secure"
```

Should return: `TEST123`

### Issue: Messages not delivered to bot

**Check**:

1. Webhook subscribed to "messages" field
2. Your phone number added in Meta console
3. Message sent to correct test number
4. Check backend logs for incoming webhooks

### Issue: Bot doesn't respond

**Check**:

1. Backend logs for errors
2. Client `customer_care_test` exists: http://localhost:8000/api/clients
3. Client has documents uploaded
4. Access token is valid (24h for temporary token)

---

## 📊 Monitor Your Bot

### Check Status

```bash
curl http://localhost:8000/webhooks/whatsapp/status
```

### View Logs

Backend logs will show:

- `Webhook verification request` - Meta verifying your webhook
- `Received webhook` - Incoming messages
- `Processed message from` - Bot responses

### Update Client Data

Make sure `customer_care_test` has documents:

1. Go to frontend: http://localhost:5173
2. Select "customer_care_test" client
3. Upload documents (PDF or JSON)
4. Test queries

---

## 🎯 What You Need to Do NOW

1. **Choose deployment method**:
   - ✨ **For quick test**: Download and run ngrok
   - 🚀 **For production**: Deploy to Render

2. **Get HTTPS URL** (from ngrok or Render)

3. **Configure webhook** in Meta console:
   - URL: `https://your-domain/webhooks/whatsapp`
   - Token: `ragbot_verify_token_2026_secure`
   - Subscribe to "messages"

4. **Test** by sending WhatsApp message!

---

## 📞 Testing with ngrok (Easiest Method)

### Complete ngrok Setup:

1. Download: https://ngrok.com/download
2. Extract the ZIP file
3. Open terminal and run:
   ```bash
   ngrok http 8000
   ```
4. You'll see:
   ```
   Forwarding  https://abc123.ngrok-free.app -> http://localhost:8000
   ```
5. Copy that HTTPS URL
6. Go to Meta console and set webhook:
   ```
   https://abc123.ngrok-free.app/webhooks/whatsapp
   ```
7. Done! Test by sending WhatsApp message

---

## ✅ Success Indicators

You'll know it's working when:

1. ✅ Webhook verification shows "Success" in Meta
2. ✅ Backend logs show "Webhook verified successfully"
3. ✅ You receive welcome message in WhatsApp
4. ✅ Bot answers your questions
5. ✅ Sources are cited in responses

---

## 🔗 Quick Links

- **Meta Dashboard**: https://developers.facebook.com/apps
- **ngrok Download**: https://ngrok.com/download
- **Render Deploy**: https://render.com
- **Local Status**: http://localhost:8000/webhooks/whatsapp/status
- **API Docs**: http://localhost:8000/docs

---

## 💡 Pro Tips

1. **Use ngrok first** - Fastest way to test (no deployment needed)
2. **Keep ngrok running** - As long as ngrok runs, webhook works
3. **Check logs** - Backend shows all webhook activity
4. **Test commands** - Try `help`, `clear` to verify features
5. **Access token expires** - Temporary token lasts 24h, generate permanent one for production

---

**You're almost there! Just need HTTPS deployment + webhook configuration.**
