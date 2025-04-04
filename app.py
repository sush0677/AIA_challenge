import streamlit as st
from transformers import pipeline
from groq import Groq
from datetime import datetime

# ------------------------
# Load Emotion Classifier
# ------------------------
emotion_classifier = pipeline(
    "text-classification",
    model="michellejieli/emotion_text_classifier",
    return_all_scores=False
)

# ------------------------
# Initialize Groq Client
# ------------------------
GROQ_API_KEY = "your_groq_api_key_here"  # <-- Replace with your API key
client = Groq(api_key=GROQ_API_KEY)

# ------------------------
# Setup session state
# ------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "emotion_tags" not in st.session_state:
    st.session_state.emotion_tags = []

# ------------------------
# UI Header
# ------------------------
st.title("🧠 Emotion-Aware Chatbot with Memory + Groq 🤖")
st.markdown("A smart chatbot that adapts responses based on detected emotions, remembers your messages, and lets you export chats.")

# ------------------------
# Reset Chat Button
# ------------------------
if st.button("🔄 Reset Chat"):
    st.session_state.history = []
    st.session_state.emotion_tags = []
    st.experimental_rerun()

# ------------------------
# Display Chat History with Emotion Tags
# ------------------------
emotion_colors = {
    "joy": "#fdfd96",
    "sadness": "#aec6cf",
    "anger": "#ff6961",
    "fear": "#cfcfc4",
    "surprise": "#ffb347",
    "disgust": "#77dd77",
    "neutral": "#dddddd"
}

for idx, msg in enumerate(st.session_state.history):
    role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
    color = "#ffffff"

    if msg["role"] == "user" and idx // 2 < len(st.session_state.emotion_tags):
        emotion = st.session_state.emotion_tags[idx // 2]
        color = emotion_colors.get(emotion, "#ffffff")
        role += f" ({emotion})"

    with st.chat_message(role):
        st.markdown(
            f"<div style='background-color:{color}; padding:10px; border-radius:10px'>{msg['content']}</div>",
            unsafe_allow_html=True
        )

# ------------------------
# User Input Field
# ------------------------
user_input = st.chat_input("Say something...")

if user_input:
    # Add to chat history
    st.session_state.history.append({"role": "user", "content": user_input})

    # Detect Emotion
    emotion_result = emotion_classifier(user_input)
    detected_emotion = emotion_result[0]['label']
    st.session_state.emotion_tags.append(detected_emotion)

    # Construct Groq prompt
    prompt = f"""
You are a kind and emotionally aware AI assistant.
A user just said: "{user_input}"
The emotion detected in their message is: {detected_emotion}.

Respond in a way that reflects their emotional state:
- sadness → comfort them.
- joy → be cheerful and positive.
- anger → stay calm and supportive.
- fear → be reassuring.
- surprise → be curious and amazed.
- disgust → acknowledge and change topic.
- neutral → be professional and friendly.

Now reply to the user:
"""

    # Bot response with streaming
    with st.chat_message("🤖 Bot"):
        response_placeholder = st.empty()
        streamed_response = ""

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a friendly, emotionally intelligent chatbot."},
                {"role": "user", "content": prompt}
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )

        for chunk in completion:
            content_piece = chunk.choices[0].delta.content or ""
            streamed_response += content_piece
            response_placeholder.markdown(streamed_response)

        # Save response to history
        st.session_state.history.append({"role": "assistant", "content": streamed_response})

# ------------------------
# Export Chat Option
# ------------------------
if st.session_state.history:
    full_chat = ""
    for i, msg in enumerate(st.session_state.history):
        role = "You" if msg["role"] == "user" else "Bot"
        emotion = st.session_state.emotion_tags[i // 2] if msg["role"] == "user" else ""
        line = f"{role} ({emotion}): {msg['content']}" if emotion else f"{role}: {msg['content']}"
        full_chat += line + "\n\n"

    filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    st.download_button("💾 Download Chat", full_chat, file_name=filename)
