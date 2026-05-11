from flask import Flask, render_template, request
import pdfplumber
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Skills database
SKILLS = [
    'python',
    'java',
    'sql',
    'flask',
    'machine learning',
    'html',
    'css',
    'javascript',
    'react',
    'mongodb',
    'mysql'
]

# Extract text from PDF
def extract_text(pdf_path):
    text = ''

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()

            if extracted:
                text += extracted

    return text.lower()

# Calculate similarity
def calculate_similarity(resume_text, job_description):

    documents = [resume_text, job_description]

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(vectors)[0][1]

    return round(similarity * 100, 2)

# Extract skills
def extract_skills(text):

    found_skills = []

    for skill in SKILLS:
        if skill in text:
            found_skills.append(skill)

    return found_skills

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():

    file = request.files['resume']
    job_description = request.form['job_description'].lower()

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    file.save(filepath)

    resume_text = extract_text(filepath)

    match_percentage = calculate_similarity(
        resume_text,
        job_description
    )

    resume_skills = extract_skills(resume_text)

    job_skills = extract_skills(job_description)

    missing_skills = []

    for skill in job_skills:
        if skill not in resume_skills:
            missing_skills.append(skill)

    return render_template(
        'result.html',
        match_percentage=match_percentage,
        resume_skills=resume_skills,
        missing_skills=missing_skills
    )

if __name__ == '__main__':
    app.run(debug=True)