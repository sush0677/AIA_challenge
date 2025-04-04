import streamlit as st
from transformers import pipeline
from groq import Groq

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
GROQ_API_KEY = "your_groq_api_key_here"  # <-- Replace this with your API key
client = Groq(api_key=GROQ_API_KEY)

# ------------------------
# Initialize Chat History
# ------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------------
# Streamlit UI
# ------------------------
st.title("🧠 Emotion-Aware Chatbot with Memory 🤖")
st.markdown("Talk to the bot and it will respond based on your emotion. Your conversation will be remembered during this session.")

# Show chat history
for msg in st.session_state.history:
    role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
    with st.chat_message(role):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Say something...")

if user_input:
    # Add user input to chat history
    st.session_state.history.append({"role": "user", "content": user_input})

    # Detect Emotion
    emotion_result = emotion_classifier(user_input)
    detected_emotion = emotion_result[0]['label']

    # Construct dynamic prompt
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

    # Display bot response (streaming)
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

        # Save bot reply to history
        st.session_state.history.append({"role": "assistant", "content": streamed_response})
