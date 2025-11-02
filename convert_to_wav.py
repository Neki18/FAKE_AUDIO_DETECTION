import os
import subprocess
import sys

# -------------------------
# FOLDERS TO PROCESS
# -------------------------
folders = [
    ("data/human/", "data/human_wav/"),
    ("data/ai/", "data/ai_wav/")
]

# -------------------------
# FFmpeg PATH
# -------------------------
ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"  # update if different

if not os.path.isfile(ffmpeg_path):
    print(f"ERROR: ffmpeg.exe not found at {ffmpeg_path}")
    sys.exit(1)

# -------------------------
# PROCESS EACH FOLDER
# -------------------------
for input_path, output_path in folders:
    os.makedirs(output_path, exist_ok=True)
    print(f"\nProcessing folder: {input_path}")

    mp3_files = [f for f in os.listdir(input_path) if f.lower().endswith(".mp3")]

    if not mp3_files:
        print(f"No MP3 files found in {input_path}")
        continue

    for file in mp3_files:
        mp3_file = os.path.join(input_path, file)
        wav_file = os.path.join(output_path, os.path.splitext(file)[0] + ".wav")
        try:
            command = [
                ffmpeg_path,
                "-y",          # overwrite if exists
                "-i", mp3_file,
                "-ar", "16000",  # set sample rate
                wav_file
            ]
            subprocess.run(command, check=True)
            print(f"Converted: {file} → {wav_file}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to convert {file}: {e}")

print("\nAll conversions done!")
