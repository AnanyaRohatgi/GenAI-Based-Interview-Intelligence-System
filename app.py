import os
import time
import tempfile
import numpy as np
import cv2
import mediapipe as mp
import librosa
from moviepy.editor import VideoFileClip, AudioFileClip
from openai import OpenAI
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from fpdf import FPDF
import re

app = Flask(__name__)
CORS(app)

DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

def extract_audio_from_video(video_path, audio_path):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec='libmp3lame')
    video.close()

def transcribe_audio(audio_path, api_key):
    client = OpenAI(api_key=api_key)
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            response_format="text"
        )
    return transcript

def extract_qa_pairs(transcript):
    # This regex matches Q1:, Q2:, ... and A1:, A2:, ... style pairs
    qa_pattern = re.compile(r'(Q\d+[:\.\)]\s*)(.*?)(?=(?:A\d+[:\.\)]|Q\d+[:\.\)]|$))', re.DOTALL | re.IGNORECASE)
    a_pattern = re.compile(r'(A\d+[:\.\)]\s*)(.*?)(?=(?:Q\d+[:\.\)]|A\d+[:\.\)]|$))', re.DOTALL | re.IGNORECASE)
    questions = qa_pattern.findall(transcript)
    answers = a_pattern.findall(transcript)
    qa_pairs = []
    for i in range(max(len(questions), len(answers))):
        q = questions[i][1].strip() if i < len(questions) else ""
        a = answers[i][1].strip() if i < len(answers) else ""
        if q or a:
            qa_pairs.append({"question": q, "answer": a})
    # Fallback: If no Q/A tags, split by lines in pairs
    if not qa_pairs:
        lines = [l.strip() for l in transcript.split('\n') if l.strip()]
        for i in range(0, len(lines), 2):
            q = lines[i]
            a = lines[i+1] if i+1 < len(lines) else ""
            qa_pairs.append({"question": q, "answer": a})
    return qa_pairs

def verify_answer_with_gpt(question, answer, api_key):
    client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are an expert grader. For each question and answer pair, reply with only 'Yes' if the answer is correct, "
        "and 'No' if it is incorrect. Do not provide any explanation."
    )
    user_prompt = f"Question: {question}\nAnswer: {answer}\nIs the answer correct? Reply only Yes or No."
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def answer_coverage_gpt(question, answer, api_key):
    client = OpenAI(api_key=api_key)
    prompt = (
        "For the following question and answer, reply with only one of these: 'Fully covered', 'Partially covered', or 'Missed'.\n"
        f"Question: {question}\nAnswer: {answer}\nCoverage:"
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def generate_pdf_report(report_data, qa_pairs, output_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Assessment Report", ln=True, align='C')
    pdf.ln(10)
    for idx, qa in enumerate(qa_pairs):
        pdf.multi_cell(0, 10, f"Q{idx+1}: {qa['question']}")
        pdf.multi_cell(0, 10, f"A: {qa['answer']}")
        pdf.multi_cell(0, 10, f"Correct? {qa['is_correct']}")
        pdf.multi_cell(0, 10, f"Coverage: {qa['coverage']}")
        pdf.ln(5)
    pdf.ln(10)
    for key, value in report_data.items():
        pdf.multi_cell(0, 10, f"{key}: {value}")
    pdf.output(output_path)

def rate_of_speech(transcript, audio_path):
    word_count = len(transcript.split())
    duration = AudioFileClip(audio_path).duration / 60  # in minutes
    return word_count / duration if duration > 0 else 0

def analyze_eye_movement(video_path):
    cap = cv2.VideoCapture(video_path)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    frame_count = 0
    eye_contact_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            eye_contact_frames += 1
    cap.release()
    percent_eye_contact = (eye_contact_frames / frame_count) * 100 if frame_count > 0 else 0
    return f"{percent_eye_contact:.2f}% ({classify_visual_confidence(percent_eye_contact)})"

def classify_visual_confidence(percent_eye_contact):
    if percent_eye_contact > 70:
        return "High"
    elif percent_eye_contact > 40:
        return "Medium"
    else:
        return "Low"

def analyze_voice_modulation(audio_path):
    y, sr = librosa.load(audio_path)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    energy = np.mean(librosa.feature.rms(y=y))
    return {"average_pitch": float(pitch), "average_energy": float(energy)}

def sentiment_analysis_gpt(transcript, api_key):
    client = OpenAI(api_key=api_key)
    prompt = f"Analyze the following answer and reply with the sentiment (e.g., 'confident', 'nervous', 'neutral'):\n{transcript}"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    temp_video = None
    temp_audio = None

    try:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        video_file.save(temp_video.name)
        temp_video.close()

        if not os.path.getsize(temp_video.name):
            raise ValueError("Empty video file")

        extract_audio_from_video(temp_video.name, temp_audio.name)
        temp_audio.close()

        transcript = transcribe_audio(temp_audio.name, OPENAI_API_KEY)
        qa_pairs = extract_qa_pairs(transcript)

        # Analyze each Q/A pair
        for qa in qa_pairs:
            qa['is_correct'] = verify_answer_with_gpt(qa['question'], qa['answer'], OPENAI_API_KEY)
            qa['coverage'] = answer_coverage_gpt(qa['question'], qa['answer'], OPENAI_API_KEY)

        speech_rate = rate_of_speech(transcript, temp_audio.name)
        voice_metrics = analyze_voice_modulation(temp_audio.name)
        eye_movement = analyze_eye_movement(temp_video.name)
        sentiment = sentiment_analysis_gpt(transcript, OPENAI_API_KEY)

        report_data = {
            "Speech Rate (words/min)": f"{speech_rate:.2f}",
            "Voice Modulation": voice_metrics,
            "Eye Movement": eye_movement,
            "Speaker Sentiment": sentiment
        }
        pdf_path = os.path.join(DATASET_DIR, f"analysis_report_{int(time.time())}.pdf")
        generate_pdf_report(report_data, qa_pairs, pdf_path)

        return jsonify({
            'qa_pairs': qa_pairs,
            'speech_rate': speech_rate,
            'voice_metrics': voice_metrics,
            'eye_movement': eye_movement,
            'sentiment': sentiment,
            'pdf_report': f'/download_report/{os.path.basename(pdf_path)}'
        })

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error details: {error_details}")
        return jsonify({'error': str(e)}), 500

    finally:
        if temp_video and os.path.exists(temp_video.name):
            try:
                os.unlink(temp_video.name)
            except:
                pass
        if temp_audio and os.path.exists(temp_audio.name):
            try:
                os.unlink(temp_audio.name)
            except:
                pass

@app.route('/download_report/<filename>')
def download_report(filename):
    path = os.path.join(DATASET_DIR, filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    else:
        return "Report not found", 404

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

