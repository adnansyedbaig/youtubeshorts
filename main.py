from flask import Flask, request, render_template
import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips
from pytube import YouTube
import speech_recognition as sr
from pydub import AudioSegment

app = Flask(__name__)

# Function to download YouTube video
def download_youtube_video(url):
    yt = YouTube(url)
    stream = yt.streams.filter(file_extension='webm').first()
    if not os.path.exists('videos'):
        os.makedirs('videos')
    output_path = stream.download(output_path='videos')
    return output_path

# Function to extract audio from video
def extractAudio(video_path):
    audio_path = video_path.replace(".mp4", ".wav")
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)
    return audio_path

# Function to transcribe audio
def transcribeAudio(audio_path):
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_wav(audio_path)
    audio.export("temp.wav", format="wav")
    with sr.AudioFile("temp.wav") as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    os.remove("temp.wav")
    return [(text, 0, len(audio))]

# Function to get highlights from transcription
def GetHighlight(transcription):
    # Dummy implementation for highlight extraction
    return 0, 10

# Function to crop video
def crop_video(input_video_path, output_video_path, start, stop):
    video = VideoFileClip(input_video_path).subclip(start, stop)
    video.write_videofile(output_video_path)

# Function to crop video to vertical format
def crop_to_vertical(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    vertical_height = original_height
    vertical_width = int(vertical_height * 9 / 16)  # 16:9 aspect ratio

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (vertical_width, vertical_height))

    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            center_x = x + w // 2
            start_x = max(0, center_x - vertical_width // 2)
            end_x = min(original_width, center_x + vertical_width // 2)
            cropped_frame = frame[:, start_x:end_x]
        else:
            cropped_frame = frame[:, :vertical_width]

        out.write(cropped_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Function to combine videos
def combine_videos(video1_path, video2_path, output_path):
    clip1 = VideoFileClip(video1_path)
    clip2 = VideoFileClip(video2_path)
    final_clip = concatenate_videoclips([clip1, clip2])
    final_clip.write_videofile(output_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    url = request.form['url']
    Vid = download_youtube_video(url)
    if Vid:
        Vid = Vid.replace(".webm", ".mp4")
        Audio = extractAudio(Vid)
        if Audio:
            transcriptions = transcribeAudio(Audio)
            if len(transcriptions) > 0:
                TransText = ""
                for text, start, end in transcriptions:
                    TransText += (f"{start} - {end}: {text}")
                start, stop = GetHighlight(TransText)
                if start != 0 and stop != 0:
                    Output = "Out.mp4"
                    crop_video(Vid, Output, start, stop)
                    croped = "croped.mp4"
                    crop_to_vertical("Out.mp4", croped)
                    combine_videos("Out.mp4", croped, "Final.mp4")
                    return "Video processed successfully!"
                else:
                    return "Error in getting highlight"
            else:
                return "No transcriptions found"
        else:
            return "No audio file found"
    else:
        return "Unable to download the video"

if __name__ == '__main__':
    app.run(debug=True)