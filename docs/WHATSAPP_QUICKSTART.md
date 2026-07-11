# WhatsApp Chatbot - Quick Reference

## ✅ Implementation Complete!

Your RAG system now has WhatsApp Business API integration for customer support.

## 📁 Files Created

```
backend/
├── integrations/
│   ├── __init__.py              # Package initialization
│   ├── session_manager.py       # Conversation history (30min timeout)
│   ├── whatsapp_formatter.py    # Message formatting for WhatsApp
│   └── whatsapp_bot.py          # Webhook handler (main logic)
├── config.py                    # ✏️ Updated with WhatsApp settings
├── main.py                      # ✏️ Added WhatsApp router
└── requirements.txt             # ✏️ Added requests library
```

## 🚀 Current Status

**Server Running**: ✅ http://localhost:8000  
**WhatsApp Endpoints**: ✅ Active  
**Default Client**: `customer_care_test`  
**Active Sessions**: 0

## 📡 API Endpoints

| Endpoint                    | Method | Purpose                     |
| --------------------------- | ------ | --------------------------- |
| `/webhooks/whatsapp`        | GET    | Webhook verification (Meta) |
| `/webhooks/whatsapp`        | POST   | Receive messages            |
| `/webhooks/whatsapp/status` | GET    | Bot status & stats          |

## ⚙️ Configuration (.env)

Add these variables to `backend/.env`:

```env
# WhatsApp Configuration
WHATSAPP_VERIFY_TOKEN=your_custom_verify_token_12345
WHATSAPP_ACCESS_TOKEN=your_meta_access_token_here
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id_here
WHATSAPP_BUSINESS_ACCOUNT_ID=your_business_account_id_here
WHATSAPP_CLIENT_ID=customer_care_test
WHATSAPP_BOT_NAME=Customer Support
```

## 🎯 Features

✅ **Session Management**: 30-minute conversation timeout  
✅ **Message Formatting**: WhatsApp markdown (bold, italic)  
✅ **Long Message Splitting**: Auto-splits > 4096 characters  
✅ **Commands**: `help`, `clear`  
✅ **Welcome Messages**: Greets new users  
✅ **Source Citations**: Shows document sources  
✅ **Error Handling**: User-friendly error messages  
✅ **Conversation History**: Maintains context  
✅ **Read Receipts**: Marks messages as read

## 💬 User Commands

- **help** - Show usage instructions
- **clear** - Clear conversation history
- **Any question** - Get answer from RAG system

## 📝 Next Steps

### 1. Get Meta Developer Account

Visit [developers.facebook.com](https://developers.facebook.com/)

### 2. Create WhatsApp App

- Create new app → Add WhatsApp product
- Get test phone number
- Copy credentials (Access Token, Phone Number ID)

### 3. Deploy with HTTPS

Choose one:

- **Testing**: Use ngrok → `ngrok http 8000`
- **Production**: Deploy to Render/Railway

### 4. Configure Webhook

In Meta Dashboard:

- Callback URL: `https://your-domain.com/webhooks/whatsapp`
- Verify Token: Same as `WHATSAPP_VERIFY_TOKEN`
- Subscribe to: `messages`

### 5. Test

Send WhatsApp message to test number → Get RAG response!

## 🔧 Testing Locally

```bash
# Check status
curl http://localhost:8000/webhooks/whatsapp/status

# Simulate webhook verification
curl "http://localhost:8000/webhooks/whatsapp?hub.mode=subscribe&hub.challenge=test123&hub.verify_token=your_verify_token"
```

## 📚 Documentation

Full setup guide: [WHATSAPP_SETUP_GUIDE.md](WHATSAPP_SETUP_GUIDE.md)

## 🎨 Customization

### Change Welcome Message

Edit `backend/integrations/whatsapp_formatter.py`:

```python
@staticmethod
def format_welcome_message(client_name: str = None) -> str:
    return "Your custom welcome message!"
```

### Change Session Timeout

Edit `backend/integrations/session_manager.py`:

```python
session_manager = SessionManager(session_timeout_minutes=60)  # 60 minutes
```

### Change Bot Name

Edit `backend/.env`:

```env
WHATSAPP_BOT_NAME=My Custom Bot Name
```

## 🐛 Troubleshooting

**Problem**: Webhook verification fails  
**Solution**: Check `WHATSAPP_VERIFY_TOKEN` matches in `.env` and Meta dashboard

**Problem**: Messages not received  
**Solution**: Ensure webhook subscribed to "messages" in Meta dashboard

**Problem**: Bot doesn't respond  
**Solution**:

- Check logs for errors
- Verify `WHATSAPP_CLIENT_ID` client exists
- Confirm documents are indexed for that client

## 📊 Monitor Usage

```bash
# Check active conversations
curl http://localhost:8000/webhooks/whatsapp/status

# View logs
tail -f backend/logs/app.log
```

## 🔐 Security Notes

- Never commit `.env` file
- Use environment variables for secrets
- Enable webhook signature validation (recommended for production)
- Rate limit webhook endpoint

## 💡 Tips

1. **Use test number first**: Meta provides free test number for development
2. **Test with ngrok**: Quick HTTPS for local testing
3. **Monitor logs**: Watch for errors during testing
4. **Clear sessions**: Users can type "clear" to reset conversation
5. **Customize responses**: Edit formatter for your brand voice

## 📞 Support

For detailed setup instructions, see [WHATSAPP_SETUP_GUIDE.md](WHATSAPP_SETUP_GUIDE.md)

---

**Status**: ✅ Ready to configure with Meta credentials and deploy!
