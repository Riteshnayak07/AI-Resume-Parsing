import os
import re
import json
import uuid
import math
import nltk
import fitz  # PyMuPDF
import docx
import anthropic
from collections import Counter
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Auto-download required NLTK resources
for resource in ['punkt','punkt_tab','stopwords','averaged_perceptron_tagger',
                  'averaged_perceptron_tagger_eng','maxent_ne_chunker','words']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXT = {'pdf', 'docx', 'txt'}

client = anthropic.Anthropic()

# ─── Skill taxonomy ──────────────────────────────────────────────────────────
SKILL_TAXONOMY = {
    "Programming Languages": ["python","java","c++","c","javascript","typescript","kotlin","swift","r","go","rust","scala","php","ruby","matlab","perl"],
    "Web Technologies": ["html","css","react","angular","vue","node.js","django","flask","fastapi","spring","express","bootstrap","tailwind"],
    "Data & ML": ["machine learning","deep learning","nlp","computer vision","tensorflow","pytorch","keras","scikit-learn","pandas","numpy","matplotlib","seaborn","opencv","nltk","spacy","bert","gpt","llm","transformer"],
    "Databases": ["sql","mysql","postgresql","mongodb","redis","oracle","sqlite","cassandra","dynamodb","firebase","elasticsearch"],
    "Cloud & DevOps": ["aws","azure","gcp","docker","kubernetes","jenkins","git","github","gitlab","linux","bash","ci/cd","terraform","ansible"],
    "Data Analysis": ["power bi","tableau","excel","data visualization","statistics","hypothesis testing","regression","eda","data cleaning","feature engineering","a/b testing"],
    "Soft Skills": ["leadership","communication","teamwork","problem solving","critical thinking","project management","agile","scrum","time management"],
}

FLAT_SKILLS = {s.lower(): cat for cat, skills in SKILL_TAXONOMY.items() for s in skills}

JOB_ROLES = {
    "Data Analyst": ["sql","python","excel","power bi","tableau","statistics","data visualization","pandas","numpy","data cleaning","eda"],
    "ML Engineer": ["python","machine learning","tensorflow","pytorch","scikit-learn","deep learning","nlp","docker","kubernetes","aws"],
    "Full Stack Developer": ["javascript","react","node.js","html","css","sql","git","api","docker","mongodb"],
    "Data Scientist": ["python","statistics","machine learning","sql","pandas","numpy","matplotlib","scikit-learn","r","deep learning"],
    "Backend Developer": ["python","java","sql","api","docker","git","linux","postgresql","redis","spring"],
}

SECTION_HEADERS = {
    "education": ["education","academic","qualification","degree","university","college","school"],
    "experience": ["experience","employment","work history","professional","internship","career"],
    "skills": ["skills","technical skills","competencies","technologies","tools","expertise"],
    "projects": ["projects","portfolio","works","applications","developed"],
    "certifications": ["certification","certificate","credential","license","course","training"],
    "summary": ["summary","objective","profile","about","overview","statement"],
    "achievements": ["achievement","award","honor","recognition","publication","research"],
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def extract_text(filepath):
    ext = filepath.rsplit('.', 1)[1].lower()
    if ext == 'pdf':
        doc = fitz.open(filepath)
        return "\n".join(page.get_text() for page in doc)
    elif ext == 'docx':
        d = docx.Document(filepath)
        return "\n".join(p.text for p in d.paragraphs)
    else:
        with open(filepath, 'r', errors='ignore') as f:
            return f.read()

# ─── NLP Extraction ──────────────────────────────────────────────────────────

def extract_email(text):
    emails = re.findall(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', text)
    return emails[0] if emails else ""

def extract_phone(text):
    phones = re.findall(r'(?:\+91[\s\-]?)?[6-9]\d{9}|(?:\+\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}', text)
    return phones[0].strip() if phones else ""

def extract_name(text):
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for line in lines[:8]:
        if len(line.split()) in [2, 3] and not any(c in line for c in ['@', ':', '/', '|', '+']):
            if re.match(r'^[A-Za-z\s\.]+$', line) and len(line) > 3:
                return line
    return ""

def extract_linkedin(text):
    m = re.search(r'linkedin\.com/in/[\w\-]+', text, re.IGNORECASE)
    return m.group(0) if m else ""

def extract_github(text):
    m = re.search(r'github\.com/[\w\-]+', text, re.IGNORECASE)
    return m.group(0) if m else ""

def extract_skills(text):
    text_lower = text.lower()
    found = {}
    for skill, category in FLAT_SKILLS.items():
        # Match whole word / phrase
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.setdefault(category, [])
            if skill not in found[category]:
                found[category].append(skill)
    return found

def segment_sections(text):
    lines = text.split('\n')
    sections = {}
    current = 'other'
    buffer = []
    for line in lines:
        stripped = line.strip().lower()
        matched = None
        for sec, kws in SECTION_HEADERS.items():
            if any(kw in stripped for kw in kws) and len(stripped) < 50:
                matched = sec
                break
        if matched:
            if buffer:
                sections.setdefault(current, []).extend(buffer)
                buffer = []
            current = matched
        else:
            if line.strip():
                buffer.append(line.strip())
    if buffer:
        sections.setdefault(current, []).extend(buffer)
    return {k: '\n'.join(v) for k, v in sections.items()}

def extract_education(sections):
    edu_text = sections.get('education', '')
    degrees = re.findall(r'(?:B\.?Tech|M\.?Tech|B\.?E|M\.?E|MCA|BCA|B\.?Sc|M\.?Sc|B\.?Com|MBA|PhD|M\.?S|B\.?S)[\s\w\(\)]*', edu_text, re.IGNORECASE)
    years = re.findall(r'(?:20\d{2})', edu_text)
    institutions = re.findall(r'(?:University|College|Institute|School|Academy)\s+[A-Za-z\s]+', edu_text, re.IGNORECASE)
    cgpa = re.findall(r'(?:CGPA|GPA|Percentage|%)[:\s]*(\d+\.?\d*)', edu_text, re.IGNORECASE)
    return {
        "degrees": list(set(degrees))[:3],
        "years": list(set(years))[:4],
        "institutions": [i.strip() for i in institutions][:3],
        "cgpa": cgpa[0] if cgpa else ""
    }

def extract_experience(sections):
    exp_text = sections.get('experience', '')
    years_exp = re.findall(r'(\d+\.?\d*)\+?\s*(?:years?|yrs?)', exp_text, re.IGNORECASE)
    companies = re.findall(r'(?:at|@|with)\s+([A-Z][A-Za-z\s&\.]+(?:Inc|Ltd|Pvt|Corp|Technologies|Solutions|Systems)?)', exp_text)
    roles = re.findall(r'(?:as|position|role|title)[:\s]+([A-Za-z\s]+(?:Engineer|Analyst|Developer|Manager|Lead|Intern|Scientist))', exp_text, re.IGNORECASE)
    dates = re.findall(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\s*[-–]\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec\s+\d{4}|Present|Current)', exp_text, re.IGNORECASE)
    return {
        "total_years": years_exp[0] if years_exp else "0",
        "companies": [c.strip() for c in companies][:3],
        "roles": [r.strip() for r in roles][:3],
        "date_ranges": dates[:3]
    }

def extract_projects(sections):
    proj_text = sections.get('projects', '')
    lines = [l.strip() for l in proj_text.split('\n') if l.strip()]
    projects = []
    current_proj = None
    for line in lines:
        if re.match(r'^[A-Z][A-Za-z\s]+$', line) and len(line) < 60:
            if current_proj:
                projects.append(current_proj)
            current_proj = {"name": line, "desc": ""}
        elif current_proj:
            current_proj["desc"] += " " + line
    if current_proj:
        projects.append(current_proj)
    # Find tech stacks
    for p in projects:
        techs = []
        for skill in FLAT_SKILLS:
            if re.search(r'\b' + re.escape(skill) + r'\b', p.get("desc","").lower()):
                techs.append(skill)
        p["tech"] = techs[:6]
    return projects[:5]

def tfidf_keywords(text, top_n=15):
    sents = sent_tokenize(text)
    if len(sents) < 2:
        sents = text.split('.')
        sents = [s for s in sents if len(s) > 10]
    if not sents:
        return []
    try:
        vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=200)
        tfidf = vec.fit_transform(sents)
        scores = tfidf.toarray().sum(axis=0)
        terms = vec.get_feature_names_out()
        ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
        return [t for t, _ in ranked[:top_n]]
    except:
        return []

def ats_score(text, skills_found):
    score = 0
    reasons = []
    # Contact info
    if extract_email(text): score += 10; reasons.append(("✓ Email found", "good"))
    else: reasons.append(("✗ No email detected", "bad"))
    if extract_phone(text): score += 5; reasons.append(("✓ Phone found", "good"))
    else: reasons.append(("✗ No phone detected", "bad"))
    # Skills
    total_skills = sum(len(v) for v in skills_found.values())
    skill_score = min(25, total_skills * 2)
    score += skill_score
    reasons.append((f"✓ {total_skills} skills detected (+{skill_score}pts)", "good" if total_skills >= 8 else "warn"))
    # Sections
    text_lower = text.lower()
    for sec in ['experience','education','projects','skills']:
        if any(kw in text_lower for kw in SECTION_HEADERS.get(sec,[])):
            score += 5
            reasons.append((f"✓ '{sec.title()}' section present", "good"))
        else:
            reasons.append((f"✗ '{sec.title()}' section missing/unclear", "bad"))
    # Length
    word_count = len(text.split())
    if 300 <= word_count <= 1200:
        score += 10; reasons.append((f"✓ Good length ({word_count} words)", "good"))
    elif word_count < 300:
        reasons.append((f"⚠ Too short ({word_count} words)", "warn"))
    else:
        reasons.append((f"⚠ Too long ({word_count} words)", "warn"))
    # Numbers/quantification
    numbers = re.findall(r'\d+%|\d+\+|increased|reduced|improved|achieved', text, re.IGNORECASE)
    if len(numbers) >= 3:
        score += 10; reasons.append((f"✓ {len(numbers)} quantified achievements", "good"))
    else:
        reasons.append((f"⚠ Add quantified achievements (numbers, %)", "warn"))
    return min(score, 100), reasons

def job_fit(skills_found):
    flat_found = set(s for skills in skills_found.values() for s in skills)
    results = {}
    for role, required in JOB_ROLES.items():
        matched = [s for s in required if s in flat_found]
        pct = round(len(matched) / len(required) * 100)
        results[role] = {"match": pct, "matched": matched, "missing": [s for s in required if s not in flat_found]}
    return dict(sorted(results.items(), key=lambda x: x[1]['match'], reverse=True))

def word_freq(text, top=20):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    words = [w for w in tokens if w.isalpha() and w not in stop_words and len(w) > 3]
    return Counter(words).most_common(top)

def pos_distribution(text):
    tokens = word_tokenize(text[:3000])
    tagged = pos_tag(tokens)
    counts = Counter(tag for _, tag in tagged)
    simplified = {
        "Nouns": counts.get('NN',0) + counts.get('NNS',0) + counts.get('NNP',0) + counts.get('NNPS',0),
        "Verbs": counts.get('VB',0) + counts.get('VBD',0) + counts.get('VBG',0) + counts.get('VBN',0),
        "Adjectives": counts.get('JJ',0) + counts.get('JJR',0) + counts.get('JJS',0),
        "Adverbs": counts.get('RB',0) + counts.get('RBR',0) + counts.get('RBS',0),
        "Others": sum(counts.values()) - sum([counts.get('NN',0),counts.get('NNS',0),counts.get('NNP',0),counts.get('NNPS',0),counts.get('VB',0),counts.get('VBD',0),counts.get('VBG',0),counts.get('VBN',0),counts.get('JJ',0),counts.get('JJR',0),counts.get('JJS',0),counts.get('RB',0),counts.get('RBR',0),counts.get('RBS',0)])
    }
    return simplified

# ─── Claude AI Analysis ──────────────────────────────────────────────────────

def ai_analyze(text):
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system="""You are an expert resume analyst and NLP specialist. Analyze the resume and return ONLY a valid JSON object with these exact keys:
{
  "summary": "2-3 sentence professional summary of the candidate",
  "strengths": ["strength 1", "strength 2", "strength 3", "strength 4"],
  "weaknesses": ["gap 1", "gap 2", "gap 3"],
  "suggestions": ["suggestion 1", "suggestion 2", "suggestion 3", "suggestion 4", "suggestion 5"],
  "seniority_level": "Fresher/Junior/Mid-Level/Senior/Lead",
  "top_role": "Best matching job role",
  "nlp_insight": "Brief NLP-specific observation about writing style, action verbs usage, keyword density",
  "overall_rating": 7.5
}
Return ONLY the JSON, no markdown, no explanation.""",
            messages=[{"role": "user", "content": f"Analyze this resume:\n\n{text[:4000]}"}]
        )
        raw = msg.content[0].text.strip()
        raw = raw.replace('```json','').replace('```','').strip()
        return json.loads(raw)
    except Exception as e:
        return {
            "summary": "AI analysis unavailable. Manual NLP parsing completed successfully.",
            "strengths": ["Resume uploaded and parsed successfully", "NLP pipeline executed"],
            "weaknesses": ["AI model not reachable"],
            "suggestions": ["Ensure API key is valid for full AI analysis"],
            "seniority_level": "Unknown",
            "top_role": "Unknown",
            "nlp_insight": "Local NLP analysis completed using NLTK and sklearn.",
            "overall_rating": 0
        }

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/parse', methods=['POST'])
def parse_resume():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file. Use PDF, DOCX, or TXT"}), 400

    filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        text = extract_text(filepath)
        if not text.strip():
            return jsonify({"error": "Could not extract text from file"}), 400

        sections = segment_sections(text)
        skills = extract_skills(text)
        ats, ats_reasons = ats_score(text, skills)
        job_matches = job_fit(skills)
        keywords = tfidf_keywords(text)
        wf = word_freq(text)
        pos_dist = pos_distribution(text)
        education = extract_education(sections)
        experience = extract_experience(sections)
        projects = extract_projects(sections)
        ai_result = ai_analyze(text)

        # Sentence stats
        sents = sent_tokenize(text)
        avg_sent_len = round(sum(len(s.split()) for s in sents) / max(len(sents), 1), 1)

        result = {
            "contact": {
                "name": extract_name(text),
                "email": extract_email(text),
                "phone": extract_phone(text),
                "linkedin": extract_linkedin(text),
                "github": extract_github(text),
            },
            "skills": skills,
            "education": education,
            "experience": experience,
            "projects": projects,
            "sections_found": list(sections.keys()),
            "ats_score": ats,
            "ats_reasons": ats_reasons,
            "job_fit": job_matches,
            "tfidf_keywords": keywords,
            "word_frequency": wf,
            "pos_distribution": pos_dist,
            "nlp_stats": {
                "total_words": len(text.split()),
                "total_sentences": len(sents),
                "avg_sentence_length": avg_sent_len,
                "unique_words": len(set(text.lower().split())),
                "char_count": len(text),
                "readability": "Easy" if avg_sent_len < 15 else "Medium" if avg_sent_len < 25 else "Complex"
            },
            "ai_analysis": ai_result,
            "raw_text_preview": text[:500]
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/compare', methods=['POST'])
def compare_resumes():
    files = request.files.getlist('files')
    if len(files) < 2:
        return jsonify({"error": "Upload at least 2 resumes"}), 400
    texts = []
    names = []
    for f in files[:4]:
        if not allowed_file(f.filename): continue
        fname = secure_filename(f"{uuid.uuid4()}_{f.filename}")
        fp = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        f.save(fp)
        try:
            texts.append(extract_text(fp))
            names.append(f.filename)
        finally:
            if os.path.exists(fp): os.remove(fp)

    if len(texts) < 2:
        return jsonify({"error": "Could not extract text from files"}), 400

    # TF-IDF similarity matrix
    vec = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vec.fit_transform(texts)
    sim = cosine_similarity(tfidf_matrix).tolist()

    comparisons = []
    for i, (text, name) in enumerate(zip(texts, names)):
        skills = extract_skills(text)
        ats, _ = ats_score(text, skills)
        total_skills = sum(len(v) for v in skills.values())
        comparisons.append({
            "name": name,
            "ats_score": ats,
            "total_skills": total_skills,
            "word_count": len(text.split()),
            "top_skills": [s for skills_list in skills.values() for s in skills_list][:8],
        })

    return jsonify({"comparisons": comparisons, "similarity_matrix": sim, "names": names})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    import socket
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        local_ip = '0.0.0.0'
    print("\n" + "="*55)
    print("  ResumeIQ NLP Parser is RUNNING!")
    print("="*55)
    print(f"  Local PC  :  http://localhost:5000")
    print(f"  On Network:  http://{local_ip}:5000")
    print(f"  Use Network URL on your phone/tablet")
    print("="*55 + "\n")
    app.run(debug=True, port=5000, host='0.0.0.0')
