import os
import flask
from flask import Flask, render_template, request
from predict_demo import predict_file  # your existing backend model

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print(">>> Flask App Loaded")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return render_template("index.html", prediction="No file uploaded")

    audio = request.files["audio"]

    # Save file
    save_path = os.path.join(UPLOAD_FOLDER, audio.filename)
    audio.save(save_path)

    # Get prediction from backend model
    raw = predict_file(save_path)  # returns ('human', numpy.float64(...))

    label = raw[0]                      # human/fake
    confidence = float(raw[1]) * 100    # convert np.float64 → Python float

    formatted = f"{label.title()} ({confidence:.2f}% confidence)"

    # This will allow audio playback
    audio_url = f"/uploads/{audio.filename}"

    return render_template("index.html",
                           prediction=formatted,
                           audio_path=audio_url)


# Serve uploaded files
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return flask.send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    print(">>> Starting Flask server...")
    app.run(debug=True)
