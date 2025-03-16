import streamlit as st
import speech_recognition as sr
import numpy as np
import librosa
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
import pandas as pd
import time
import random
import matplotlib.pyplot as plt
import threading
import queue
import pyaudio

# Download NLTK data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Initialize recognizer
recognizer = sr.Recognizer()
recognizer.energy_threshold = 150  # Adjustable sensitivity


# Cache AI tools
@st.cache_resource
def load_sentiment_analyzer():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="distilgpt2", max_length=50)


# Global state
if "state" not in st.session_state:
    st.session_state.state = {"scores": [], "progress": [], "user_level": 1, "language": "en"}
state = st.session_state.state

# Queue for async speech
speech_queue = queue.Queue()


# Free Google Speech Recognition
def free_speech_to_text(audio, language="en"):
    try:
        return recognizer.recognize_google(audio, language=language)
    except sr.UnknownValueError:
        return "Could not understand audio."
    except sr.RequestError as e:
        return f"API request error: {e}"


def process_speech(audio, language, result_container):
    result = free_speech_to_text(audio, language)
    result_container.put(result)


# Emotion detection
def detect_emotion(audio_data, sr=16000):
    y = librosa.util.buf_to_float(audio_data)
    energy = np.mean(librosa.feature.rms(y=y))
    return "Positive" if energy > 0.01 else "Negative" if energy < 0.005 else "Neutral"


# Adaptive difficulty
def adjust_difficulty():
    if state["scores"]:
        avg_score = np.mean([s["score"] for s in state["scores"][-5:]])
        state["user_level"] = min(3, max(1, int(avg_score * 3)))


# Test microphone access
def test_microphone():
    try:
        with sr.Microphone(sample_rate=16000) as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            st.write("Microphone detected and calibrated successfully.")
            return True
    except sr.RequestError as e:
        st.error(f"Microphone error: {e}. Check if it’s connected and allowed.")
        return False
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return False


# Exercise 1: Rapid Fire Analogies
def rapid_fire_analogies(sentiment_analyzer, generator):
    analogies = {1: ["Success is like..."], 2: ["Fear is like..."], 3: ["Hope is like..."]}
    st.header("Rapid Fire Analogies")
    level = state["user_level"]
    analogy = random.choice(analogies[level])
    st.write(f"Analogy (Level {level}): {analogy}")

    if st.button("Start Speaking", key="rf"):
        if not test_microphone():
            return

        with sr.Microphone(sample_rate=16000) as source:
            st.write("Speak now (5 sec)...")
            try:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                start_time = time.time()
                audio = recognizer.listen(source, timeout=5 + level, phrase_time_limit=5 + level)
                end_time = time.time()

                result_container = queue.Queue()
                thread = threading.Thread(target=process_speech, args=(audio, state["language"], result_container))
                thread.start()
                thread.join()

                response = result_container.get()
                audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)

                if "error" not in response.lower() and "could not" not in response.lower():
                    timing_score = max(0, (5 + level) - (end_time - start_time)) / (5 + level)
                    continuity = 1.0 if len(response.split()) > 3 + level else 0.5
                    relevance = sentiment_analyzer(response)[0]['score']
                    creativity = len(set(response.split()) - stop_words) / (10 + level)
                    emotion = detect_emotion(audio_data)

                    total_score = (timing_score + continuity + relevance + creativity) / 4
                    st.write(f"Response: {response}")
                    st.write(
                        f"Score: {total_score:.2f} (Timing: {timing_score:.2f}, Continuity: {continuity:.2f}, Relevance: {relevance:.2f}, Creativity: {creativity:.2f})")
                    st.write(f"Emotion Detected: {emotion}")
                    state["scores"].append({"exercise": "Rapid Fire", "score": total_score, "emotion": emotion})

                    feedback = generator(f"Improve this analogy response: {response}")[0]['generated_text']
                    st.write(f"AI Coach: {feedback}")
                else:
                    st.write(response)
            except sr.WaitTimeoutError:
                st.write("No speech detected. Please speak louder or check microphone permissions.")
            except sr.RequestError as e:
                st.write(f"Microphone access error: {e}. Ensure permissions are granted.")
            except Exception as e:
                st.write(f"Error: {e}")
    adjust_difficulty()


# Exercise 2: Triple Step
def triple_step(sentiment_analyzer, generator):
    topics = {1: ["Teamwork"], 2: ["Innovation"], 3: ["Leadership"]}
    st.header("Triple Step")
    level = state["user_level"]
    topic = random.choice(topics[level])
    distractors = [generator(f"Distractor related to {topic}: ")[0]['generated_text'].split(": ")[-1] for _ in
                   range(level + 1)]
    st.write(f"Topic (Level {level}): {topic} | Distractors: {', '.join(distractors)}")

    if st.button("Start Speaking", key="ts"):
        if not test_microphone():
            return

        with sr.Microphone(sample_rate=16000) as source:
            st.write("Speak now (10 sec)...")
            try:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source, timeout=10 + level, phrase_time_limit=10 + level)

                result_container = queue.Queue()
                thread = threading.Thread(target=process_speech, args=(audio, state["language"], result_container))
                thread.start()
                thread.join()

                response = result_container.get()
                audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)

                if "error" not in response.lower() and "could not" not in response.lower():
                    coherence = 1.0 if len(response.split()) > 10 + level else 0.5
                    adherence = sentiment_analyzer(f"{topic} {response}")[0]['score']
                    distraction_handling = 0.8 if not any(d.lower() in response.lower() for d in distractors) else 0.4
                    emotion = detect_emotion(audio_data)

                    total_score = (coherence + adherence + distraction_handling) / 3
                    st.write(f"Response: {response}")
                    st.write(
                        f"Score: {total_score:.2f} (Coherence: {coherence:.2f}, Adherence: {adherence:.2f}, Distraction Handling: {distraction_handling:.2f})")
                    st.write(f"Emotion Detected: {emotion}")
                    state["scores"].append({"exercise": "Triple Step", "score": total_score, "emotion": emotion})

                    feedback = generator(f"Improve coherence on {topic}: {response}")[0]['generated_text']
                    st.write(f"AI Coach: {feedback}")
                else:
                    st.write(response)
            except sr.WaitTimeoutError:
                st.write("No speech detected. Please speak louder or check microphone permissions.")
            except sr.RequestError as e:
                st.write(f"Microphone access error: {e}. Ensure permissions are granted.")
            except Exception as e:
                st.write(f"Error: {e}")
    adjust_difficulty()


# Exercise 3: Conductor
def conductor(sentiment_analyzer, generator):
    moods = {1: [("Calm", 0.3)], 2: [("High Energy", 1.0)], 3: [("Excited", 0.7)]}
    st.header("Conductor")
    level = state["user_level"]
    mood, energy_level = random.choice(moods[level])
    st.write(f"Mood (Level {level}): {mood} (Target Energy: {energy_level})")

    if st.button("Start Speaking", key="cd"):
        if not test_microphone():
            return

        with sr.Microphone(sample_rate=16000) as source:
            st.write("Speak now (10 sec)...")
            try:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source, timeout=10 + level, phrase_time_limit=10 + level)
                audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)

                result_container = queue.Queue()
                thread = threading.Thread(target=process_speech, args=(audio, state["language"], result_container))
                thread.start()
                thread.join()

                response = result_container.get()

                if "error" not in response.lower() and "could not" not in response.lower():
                    energy = np.mean(np.abs(audio_data)) / 10000
                    variety = len(set(response.split())) / (10 + level)
                    mood_match = 1.0 - abs(energy - energy_level)
                    emotion = detect_emotion(audio_data)

                    total_score = (energy + variety + mood_match) / 3
                    st.write(f"Response: {response}")
                    st.write(
                        f"Score: {total_score:.2f} (Energy: {energy:.2f}, Variety: {variety:.2f}, Mood Match: {mood_match:.2f})")
                    st.write(f"Emotion Detected: {emotion}")
                    suggestion = "Speak louder" if energy < energy_level else "Softer tone"
                    st.write(f"Suggestion: {suggestion} for {mood}")
                    state["scores"].append({"exercise": "Conductor", "score": total_score, "emotion": emotion})

                    feedback = generator(f"Improve vocal variety for {mood}: {response}")[0]['generated_text']
                    st.write(f"AI Coach: {feedback}")
                else:
                    st.write(response)
            except sr.WaitTimeoutError:
                st.write("No speech detected. Please speak louder or check microphone permissions.")
            except sr.RequestError as e:
                st.write(f"Microphone access error: {e}. Ensure permissions are granted.")
            except Exception as e:
                st.write(f"Error: {e}")
    adjust_difficulty()


# Analytics Dashboard
def analytics_dashboard():
    st.header("Advanced Analytics Dashboard")
    if state["scores"]:
        df = pd.DataFrame(state["scores"])
        st.dataframe(df)

        fig, ax = plt.subplots()
        for exercise in df["exercise"].unique():
            scores = df[df["exercise"] == exercise]["score"]
            ax.plot(scores, label=exercise)
        ax.set_xlabel("Attempts")
        ax.set_ylabel("Score")
        ax.legend()
        st.pyplot(fig)

        st.write(f"Current Level: {state['user_level']}")
        emotions = df["emotion"].value_counts()
        st.bar_chart(emotions)


# Streamlit Frontend
def main():
    st.set_page_config(page_title="AI Public Speaking Trainer", layout="wide")
    st.title("AI-Powered Public Speaking Training Platform")

    # Load models
    sentiment_analyzer = load_sentiment_analyzer()
    generator = load_generator()

    # Microphone test button
    if st.button("Test Microphone"):
        test_microphone()

    # Sidebar
    state["language"] = st.sidebar.selectbox("Language", ["en", "es", "fr"], index=0)

    # Tabs
    tabs = st.tabs(["Rapid Fire Analogies", "Triple Step", "Conductor", "Analytics"])
    with tabs[0]:
        rapid_fire_analogies(sentiment_analyzer, generator)
    with tabs[1]:
        triple_step(sentiment_analyzer, generator)
    with tabs[2]:
        conductor(sentiment_analyzer, generator)
    with tabs[3]:
        analytics_dashboard()


if __name__ == "__main__":
    main()
