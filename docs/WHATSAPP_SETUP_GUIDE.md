# WhatsApp Chatbot Integration - Setup Guide

## Overview

This guide will help you integrate your RAG system with WhatsApp Business API to create a customer support chatbot.

## Prerequisites

1. **Meta Developer Account**: Sign up at [developers.facebook.com](https://developers.facebook.com/)
2. **WhatsApp Business Account**: Required for production use
3. **HTTPS Deployment**: Webhooks require HTTPS (use Render, Railway, or ngrok for testing)

## Step 1: Setup Meta Developer App

1. Go to [Meta for Developers](https://developers.facebook.com/)
2. Click **My Apps** → **Create App**
3. Select **Business** as the app type
4. Fill in app details:
   - **App Name**: Your RAG Chatbot
   - **App Contact Email**: Your email
5. Click **Create App**

## Step 2: Add WhatsApp Product

1. In your app dashboard, click **Add Product**
2. Find **WhatsApp** and click **Set Up**
3. You'll see the WhatsApp setup page

## Step 3: Get Test Phone Number

Meta provides a test number for development:

1. In the WhatsApp setup page, find **From** section
2. You'll see a test phone number (Phone Number ID)
3. Copy the **Phone Number ID** - you'll need this for `.env`
4. Add your personal phone number to receive test messages:
   - Click **Add phone number**
   - Enter your WhatsApp number
   - Verify with the code sent to WhatsApp

## Step 4: Get Access Token

1. In WhatsApp setup page, find **Temporary access token**
2. Copy the token (valid for 24 hours for testing)
3. For production, generate a **System User Token**:
   - Go to **Business Settings** → **System Users**
   - Create system user
   - Generate token with `whatsapp_business_messaging` permission

## Step 5: Configure Environment Variables

Create or update `backend/.env` file:

```env
# Existing settings
GROQ_API_KEY=your_groq_api_key
DATABASE_URL=sqlite:///./rag_system.db
ADMIN_PASSWORD=your_admin_password

# WhatsApp Configuration
WHATSAPP_VERIFY_TOKEN=your_custom_verify_token_12345
WHATSAPP_ACCESS_TOKEN=your_meta_access_token
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
WHATSAPP_BUSINESS_ACCOUNT_ID=your_business_account_id
WHATSAPP_CLIENT_ID=customer_care_test
WHATSAPP_BOT_NAME=Customer Support
```

**Important Notes:**

- `WHATSAPP_VERIFY_TOKEN`: Create your own random string (e.g., "mybot_verify_2024")
- `WHATSAPP_ACCESS_TOKEN`: Copy from Meta dashboard
- `WHATSAPP_PHONE_NUMBER_ID`: Copy from WhatsApp setup page
- `WHATSAPP_CLIENT_ID`: Must be an existing client in your RAG system with indexed documents

## Step 6: Deploy Your Backend

### Option A: Quick Testing with ngrok

```bash
# Install ngrok
# Download from https://ngrok.com/download

# Start your backend
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000

# In another terminal, start ngrok
ngrok http 8000

# Copy the HTTPS URL (e.g., https://abc123.ngrok.io)
```

### Option B: Production Deployment (Render)

1. Create account at [render.com](https://render.com)
2. Click **New** → **Web Service**
3. Connect your GitHub repository
4. Configure:
   - **Build Command**: `cd backend && pip install -r requirements.txt`
   - **Start Command**: `cd backend && uvicorn main:app --host 0.0.0.0 --port 8000`
   - **Environment Variables**: Add all variables from `.env`
5. Deploy and copy the HTTPS URL

## Step 7: Configure Webhook

1. In Meta Developers → WhatsApp → **Configuration**
2. Find **Webhook** section
3. Click **Edit**
4. Enter:
   - **Callback URL**: `https://your-domain.com/webhooks/whatsapp`
   - **Verify Token**: Same as `WHATSAPP_VERIFY_TOKEN` in `.env`
5. Click **Verify and Save**
6. Subscribe to **messages** webhook field

## Step 8: Test Your Bot

1. Open WhatsApp on your phone
2. Send a message to the test number provided by Meta
3. You should receive:
   - Welcome message from the bot
   - Answer to your question from the RAG system

**Test Commands:**

- `help` - Show help message
- `clear` - Clear conversation history
- Any question - Get answer from RAG knowledge base

## Step 9: Monitor and Debug

### Check Bot Status

```bash
curl https://your-domain.com/webhooks/whatsapp/status
```

Response includes:

- Active sessions count
- Client ID being used
- Bot configuration

### View Logs

Backend logs will show:

- Incoming messages
- RAG queries
- Responses sent
- Any errors

### Common Issues

**Issue**: Webhook verification fails

- **Solution**: Check `WHATSAPP_VERIFY_TOKEN` matches in both `.env` and Meta dashboard

**Issue**: Messages not received

- **Solution**: Ensure webhook subscribed to "messages" field in Meta dashboard

**Issue**: Bot doesn't respond

- **Solution**:
  - Check backend logs for errors
  - Verify `WHATSAPP_CLIENT_ID` client exists and has documents indexed
  - Check `WHATSAPP_ACCESS_TOKEN` is valid

**Issue**: "Client not found" error

- **Solution**: Create client and upload documents:
  ```bash
  # Via API or frontend
  POST /api/clients
  POST /api/clients/{client_id}/documents
  ```

## Production Checklist

Before going live:

- [ ] Replace temporary access token with permanent system user token
- [ ] Get verified WhatsApp Business Account
- [ ] Use production phone number (not test number)
- [ ] Enable proper error handling and monitoring
- [ ] Setup rate limiting if needed
- [ ] Configure proper CORS for your domain
- [ ] Enable HTTPS on production server
- [ ] Test all commands (help, clear, queries)
- [ ] Test long messages (auto-split feature)
- [ ] Test conversation history persistence

## Features Implemented

✅ **Session Management**: 30-minute conversation timeout
✅ **Message Formatting**: WhatsApp markdown support (bold, italic)
✅ **Long Message Splitting**: Auto-splits messages over 4096 characters
✅ **Commands**: help, clear
✅ **Welcome Messages**: Greets new users
✅ **Source Citations**: Shows document sources
✅ **Error Handling**: User-friendly error messages
✅ **Conversation History**: Maintains context across messages
✅ **Read Receipts**: Marks messages as read

## API Endpoints

- `GET /webhooks/whatsapp` - Webhook verification (Meta calls this)
- `POST /webhooks/whatsapp` - Message webhook (receives messages)
- `GET /webhooks/whatsapp/status` - Bot status and statistics

## Security Notes

- Never commit `.env` file to Git
- Rotate access tokens regularly
- Use environment variables for all secrets
- Validate webhook signatures (Meta provides signature header)
- Rate limit webhook endpoint if needed

## Next Steps

1. Customize welcome message in `whatsapp_formatter.py`
2. Add more commands as needed
3. Implement webhook signature validation
4. Add analytics/logging
5. Setup monitoring alerts
6. Create admin dashboard

## Support

For issues:

- Check backend logs: `tail -f logs/app.log`
- Test webhook: Send test message from Meta dashboard
- Verify client has documents: `GET /api/clients/{client_id}`

## Resources

- [Meta WhatsApp Business API Docs](https://developers.facebook.com/docs/whatsapp/cloud-api)
- [WhatsApp Message Templates](https://developers.facebook.com/docs/whatsapp/api/messages/message-templates)
- [Webhook Setup Guide](https://developers.facebook.com/docs/whatsapp/cloud-api/webhooks)
