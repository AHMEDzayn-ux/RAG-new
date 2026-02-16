# Voice Chat Feature Documentation

## Overview

The RAG chatbot now includes voice interaction capabilities, allowing users to speak their questions and hear responses spoken back.

## Features

### 1. Speech-to-Text (Voice Input)
- **Microphone Button**: Click the 🎙️ button to start recording your question
- **Smart Transcription**: Automatically converts speech to text using Web Speech API
- **Visual Feedback**: Microphone turns red (🎤) and pulses while listening
- **Hands-free Input**: Perfect for accessibility or multitasking

### 2. Text-to-Speech (Voice Output)
- **Auto-Play Responses**: Enable voice mode (🔊) to hear all responses automatically
- **Natural Voice**: Uses browser's native text-to-speech engine
- **Stop Control**: Click ⏹️ to stop speaking at any time
- **Toggle On/Off**: Click 🔇/🔊 to enable/disable voice responses

## How to Use

### Voice Input (Speaking Questions)

1. **Start Recording**: Click the 🎙️ microphone button
2. **Speak Clearly**: Ask your question (e.g., "What are the leadership roles?")
3. **Auto-Stop**: Recording stops automatically when you finish speaking
4. **Review & Send**: Your transcribed text appears in the input box - review and click Send

### Voice Output (Hearing Responses)

1. **Enable Voice**: Click the 🔇 button to enable voice mode (turns to 🔊)
2. **Ask Question**: Type or speak your question
3. **Listen**: The response will be spoken automatically
4. **Stop Anytime**: Click ⏹️ to interrupt the speech
5. **Disable**: Click 🔊 to turn off auto-speaking

## Browser Compatibility

### Fully Supported
- ✅ **Chrome** (Desktop & Mobile) - Best experience
- ✅ **Edge** (Desktop)
- ✅ **Safari** (Desktop & Mobile)
- ✅ **Opera** (Desktop)

### Limited/No Support
- ❌ **Firefox** - No Web Speech API support yet
- ❌ **Internet Explorer** - Not supported

**Note**: If speech features are unavailable, the voice buttons will not appear.

## Technical Details

### Speech Recognition
- **API**: Web Speech API (`SpeechRecognition`)
- **Language**: English (US) by default
- **Mode**: Single-shot (stops after one sentence)
- **Accuracy**: Depends on microphone quality and background noise

### Text-to-Speech
- **API**: Web Speech Synthesis API
- **Voice**: System default voice
- **Settings**:
  - Rate: 1.0 (normal speed)
  - Pitch: 1.0 (normal pitch)
  - Volume: 1.0 (100%)

## Use Cases

### 1. Accessibility
- Users with visual impairments can hear responses
- Users with motor difficulties can speak instead of typing

### 2. Multitasking
- Get answers while working on other tasks
- Hands-free operation in labs, workshops, or kitchens

### 3. Mobile Users
- Faster input on mobile devices
- Better experience while driving (as passenger)

### 4. Customer Support
- More natural, conversational interaction
- Reduces typing fatigue for support agents

## Tips for Best Results

### Voice Input
1. **Speak clearly** and at a moderate pace
2. **Reduce background noise** for better accuracy
3. **Use a quality microphone** if possible
4. **Review transcription** before sending (fix any errors)

### Voice Output
1. **Use headphones** in shared spaces
2. **Adjust system volume** for comfort
3. **Stop speaking** if you need to pause and think

## Troubleshooting

### "Speech recognition is not supported in your browser"
- **Solution**: Switch to Chrome, Edge, or Safari
- **Alternative**: Use keyboard input normally

### Microphone not working
- **Check permissions**: Allow microphone access in browser settings
- **Check hardware**: Ensure microphone is connected and working
- **Try refresh**: Reload the page and grant permission again

### Voice sounds robotic or unclear
- **System voice**: Change your operating system's default TTS voice
- **Browser limitation**: Some browsers have better voices than others

### Voice interrupts too early
- **Manual control**: Click the microphone button to stop/start manually
- **Type instead**: Use keyboard input for complex multi-part questions

## Privacy & Security

- ✅ **No audio recording**: Speech is processed by your browser only
- ✅ **No data sent**: Voice processing happens locally on your device
- ✅ **Transcription only**: Only the text transcription is sent to the server
- ✅ **User control**: Voice features are opt-in (disabled by default)

## Future Enhancements

Potential improvements for future versions:

1. **Voice Selection**: Choose different voices (male/female, accents)
2. **Speed Control**: Adjust speaking rate
3. **Multi-language**: Support for languages beyond English
4. **Wake Word**: "Hey Assistant" to activate voice input
5. **Continuous Mode**: Keep listening for follow-up questions
6. **Emotion Detection**: Detect user sentiment from voice tone

## Examples

### Example 1: CV Questions
**User (speaks)**: "Has the person done any leadership roles?"  
**System (speaks)**: "Yes, the candidate has held several leadership positions including Team Lead at XYZ Corp and Project Manager at ABC Inc."

### Example 2: Document Search
**User (speaks)**: "What are the system requirements?"  
**System (speaks)**: "The system requires Python 3.8 or higher, 8GB RAM minimum, and 50GB free disk space."

### Example 3: Clarification
**User (speaks)**: "Tell me more about their education"  
**System (speaks)**: "The candidate holds a Master's degree in Computer Science from Stanford University, completed in 2019."

---

**Enjoy hands-free, natural conversations with your RAG chatbot! 🎙️🔊**
