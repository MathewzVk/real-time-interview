from flask import Flask, request, jsonify, render_template
import replicate
import tempfile
import os
import pyaudio
import wave



app = Flask(__name__)

model = replicate.Client(api_token="api_token")
audio = pyaudio.PyAudio()

@app.route("/")
def index():
    return render_template("index.html")


app.route("/process-audio", method=["POST"])
def process_audio_data():
    print("processing audio....")
    try:
        audio_data = record_audio()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_data)
            temp_audio.flush()

        output = model.run(
            "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c",
            input={
                "task": "transcribe",
                "audio": temp_audio,
                "language": "english",
                "timestamp": "chunk",
                "batch_size": 64,
                "diarise_audio": False
            }
        )

        print(output)

        results = output["text"]

        return jsonify({"trascript": results})
    except Exception as e:
        print(f"Error running replicate model: {e}")
        return None
    
def record_audio():
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 5

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")

    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()

    audio_data = b''.join(frames)
    return audio_data
    
@app.route("/get-suggestion", methods=["POST"])
def get_suggestion():
    print("Getting Suggestions...")
    data = request.get_json()
    transcript = data.get("transcript", "")
    prompt_text = data.get("prompt", "")

    prompt = f"""
    {transcript}
    ------
    {prompt_text}
    """

    suggestion = ""
    for event in model.stream(
        "mistralai/mixtral-8x7b-instruct-v0.1",
        input={
            "top_k": 50,
            "top_p": 0.9,
            "prompt": prompt,
            "temperature": 0.6,
            "max_new_tokens": 512,
            "prompt_template": "<s>[INST] {prompt} [/INST] ",
            "presence_penalty": 0,
            "frequency_penalty": 0
        },
    ):
        suggestion += str(event)
    return jsonify({"suggestion" : suggestion})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)