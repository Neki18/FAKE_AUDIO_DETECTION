import os
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from predict_demo import predict_file

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return render_template("index.html", error="No file uploaded")

    audio = request.files["audio"]
    if audio.filename == '':
        return render_template("index.html", error="No file selected")

    filename = secure_filename(audio.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio.save(save_path)

    # Model Prediction
    raw = predict_file(save_path)
    label = str(raw[0]).upper()
    confidence = float(raw[1]) * 100

    status_color = "#4ade80" if "HUMAN" in label else "#f87171"

    # Pass the filename back to the template
    return render_template(
        "index.html",
        prediction=label,
        confidence=f"{confidence:.2f}%",
        audio_path=f"/uploads/{filename}",
        status_color=status_color,
        filename=filename  # This keeps the UI updated after analysis
    )

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)