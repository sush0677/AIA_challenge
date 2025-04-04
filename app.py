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
GROQ_API_KEY = "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"  # Replace this with your actual key
client = Groq(api_key=GROQ_API_KEY)

# ------------------------
# Setup session state
# ------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "emotion_tags" not in st.session_state:
    st.session_state.emotion_tags = []

# ------------------------
# Streamlit UI Header for AIA
# ------------------------
st.set_page_config(page_title="AIA - Emotion-Aware Chatbot", page_icon="🤖")

st.title("🤖 AIA - Emotion-Aware Chatbot")
st.markdown("""
### 🧠 Meet AIA
**AIA** is your Emotion-Aware AI Assistant powered by LLaMA 3 and Hugging Face.

It detects how you're feeling — sadness, joy, anger, and more — and replies with empathy and intelligence.

Type a message below and see AIA respond to your emotion in real time.
""")

# ------------------------
# Reset Chat Button
# ------------------------
if st.button("🔄 Reset Chat"):
    st.session_state.history = []
    st.session_state.emotion_tags = []
    st.experimental_rerun()

# ------------------------
# Emotion Color Mapping
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

# ------------------------
# Display Chat History with Emotion + Response
# ------------------------
for i in range(0, len(st.session_state.history), 2):
    if i + 1 >= len(st.session_state.history):
        break  # Skip if bot response isn't yet available

    user_msg = st.session_state.history[i]["content"]
    bot_msg = st.session_state.history[i + 1]["content"]
    emotion = st.session_state.emotion_tags[i // 2]
    bg_color = emotion_colors.get(emotion, "#ffffff")

    # Display User Message
    with st.chat_message("🧑 You"):
        st.markdown(
            f"<div style='background-color:{bg_color}; padding:10px; border-radius:10px'>"
            f"<b>Emotion Detected:</b> <code>{emotion}</code><br><br>"
            f"{user_msg}</div>",
            unsafe_allow_html=True
        )

    # Display Bot Message
    with st.chat_message("🤖 AIA"):
        st.markdown(
            f"<div style='background-color:#f0f0f0; padding:10px; border-radius:10px'>"
            f"{bot_msg}</div>",
            unsafe_allow_html=True
        )

# ------------------------
# User Input Field
# ------------------------
user_input = st.chat_input("Say something...")

if user_input:
    # Append user message to history
    st.session_state.history.append({"role": "user", "content": user_input})

    # Emotion detection
    emotion_result = emotion_classifier(user_input)
    detected_emotion = emotion_result[0]['label']
    st.session_state.emotion_tags.append(detected_emotion)

    # Prompt to Groq
    prompt = f"""
You are AIA, an emotionally intelligent AI assistant that adapts to the user's emotional tone.

The user just said: "{user_input}"
The emotion detected in their message is: {detected_emotion}

Please respond as AIA in a way that reflects their emotional state:
- If sadness → comfort them gently.
- If joy → be cheerful and excited.
- If anger → stay calm and helpful.
- If fear → be warm and reassuring.
- If surprise → show curiosity and excitement.
- If disgust → acknowledge and shift focus.
- If neutral → be friendly and professional.

Now, respond to the user:
"""


    # Stream Groq response
    with st.chat_message("🤖 AIA"):
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

        st.session_state.history.append({"role": "assistant", "content": streamed_response})

# ------------------------
# Download Conversation Button
# ------------------------
if st.session_state.history:
    full_chat = ""
    for i in range(0, len(st.session_state.history), 2):
        if i + 1 >= len(st.session_state.history): break
        user = st.session_state.history[i]["content"]
        bot = st.session_state.history[i + 1]["content"]
        emotion = st.session_state.emotion_tags[i // 2]
        full_chat += f"You ({emotion}): {user}\nBot: {bot}\n\n"

    filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    st.download_button("💾 Download Chat", full_chat, file_name=filename)
