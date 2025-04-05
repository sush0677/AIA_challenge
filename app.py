import streamlit as st
from transformers import pipeline
from groq import Groq
import os
from dotenv import load_dotenv
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
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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
        break  # Skip incomplete pairs

    user_msg = st.session_state.history[i]["content"]
    bot_msg = st.session_state.history[i + 1]["content"]
    emotion = st.session_state.emotion_tags[i // 2]
    bg_color = emotion_colors.get(emotion, "#ffffff")

    with st.chat_message("🧑 You"):
        st.markdown(
            f"<div style='background-color:{bg_color}; padding:10px; border-radius:10px'>"
            f"<b>Emotion Detected:</b> <code>{emotion}</code><br><br>"
            f"{user_msg}</div>",
            unsafe_allow_html=True
        )

    with st.chat_message("🤖 AIA"):
        st.markdown(
            f"<div style='background-color:#f0f0f0; padding:10px; border-radius:10px'>"
            f"{bot_msg}</div>",
            unsafe_allow_html=True
        )

# ------------------------
# Sidebar: Model Selection
# ------------------------
st.sidebar.title("🛠️ AIA Settings")
model_choice = st.sidebar.selectbox("🧬 Choose Groq Model", [
    "LLaMA 3.1 (8B) - Groq",
    "Qwen-2.5 (32B) - Groq",
    "DeepSeek-R1 Distill Qwen (32B) - Groq",
    "Gemma2-9B-IT - Groq"
])

# Map UI name to actual Groq model ID
model_map = {
    "LLaMA 3.1 (8B) - Groq": "llama-3.1-8b-instant",
    "Qwen-2.5 (32B) - Groq": "qwen-2.5-32b",
    "DeepSeek-R1 Distill Qwen (32B) - Groq": "deepseek-r1-distill-qwen-32b",
    "Gemma2-9B-IT - Groq": "gemma2-9b-it"
}
selected_model_id = model_map[model_choice]

# ------------------------
# User Input Field
# ------------------------
user_input = st.chat_input("Say something...")

if user_input:
    # Append user input to chat history
    st.session_state.history.append({"role": "user", "content": user_input})

    # Detect Emotion
    emotion_result = emotion_classifier(user_input)
    detected_emotion = emotion_result[0]['label']
    st.session_state.emotion_tags.append(detected_emotion)

    # Create Prompt for AIA
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
            model=selected_model_id,
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a friendly, emotionally intelligent chatbot."},
                {"role": "user", "content": prompt}
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
        )

        for chunk in completion:
            content_piece = chunk.choices[0].delta.content or ""
            streamed_response += content_piece
            response_placeholder.markdown(streamed_response)

    # Append bot response to history
    st.session_state.history.append({"role": "assistant", "content": streamed_response})

    # Display latest interaction
    st.markdown("### 💬 Latest Interaction")
    st.markdown(f"""
    <div style="padding:15px; border-radius:10px; background-color:#e8f0fe;">
        <b>🧑 You:</b><br>
        {user_input}<br><br>
        <b>🧠 Emotion Detected:</b> <code>{detected_emotion}</code><br><br>
        <b>🤖 AIA:</b><br>
        {streamed_response}
    </div>
    """, unsafe_allow_html=True)

# ------------------------
# Download Chat History Button
# ------------------------
if st.session_state.history:
    full_chat = ""
    for i in range(0, len(st.session_state.history), 2):
        if i + 1 >= len(st.session_state.history):
            break
        user = st.session_state.history[i]["content"]
        bot = st.session_state.history[i + 1]["content"]
        emotion = st.session_state.emotion_tags[i // 2]
        full_chat += f"You ({emotion}): {user}\nAIA: {bot}\n\n"

    filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    st.download_button("💾 Download Chat with AIA", full_chat, file_name=filename)

