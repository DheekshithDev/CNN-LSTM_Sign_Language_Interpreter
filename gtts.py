#from gtts import gTTS
import pyttsx3

# Initialize the TTS engine
engine = pyttsx3.init()

# Set properties before adding anything to speak
engine.setProperty('rate', 150)    # Speed percent (can go over 100)
engine.setProperty('volume', 1.0)  # Volume 0-1

# Adding text to speak
engine.say("a")
engine.say("B")

# Blocks while processing all the currently queued commands
engine.runAndWait()