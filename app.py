import streamlit as st
from transformers import pipeline
from groq import Groq
import os

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
GROQ_API_KEY = "your_groq_api_key_here"  # <-- Replace this with your actual API key
client = Groq(api_key=GROQ_API_KEY)

# ------------------------
# Streamlit UI
# ------------------------
st.title("🧠 Emotion-Aware Chatbot with Groq 🤖")
st.markdown("This chatbot adapts its tone based on the emotion detected in your message.")

user_input = st.text_input("You:", placeholder="Type something...")

if user_input:
    # Detect Emotion
    emotion_result = emotion_classifier(user_input)
    detected_emotion = emotion_result[0]['label']

    # Show detected emotion
    st.markdown(f"**Detected Emotion:** `{detected_emotion}`")

    # Construct dynamic prompt for Groq
    prompt = f"""
You are a kind and emotionally aware AI assistant.
A user just sent this message: "{user_input}"
The emotion detected in their message is: {detected_emotion}.

Please respond in a way that reflects their emotional state:
- If sadness → comfort them.
- If joy → encourage them.
- If anger → stay calm and offer help.
- If fear → reassure them.
- If surprise → show curiosity.
- If disgust → acknowledge and move on.
- If neutral → respond professionally.

Now, reply to the user:
"""

    # Display streamed response
    st.markdown("**Bot Response:**")
    response_placeholder = st.empty()

    streamed_response = ""
    with st.spinner("Thinking..."):
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

