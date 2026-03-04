import streamlit as st
from transformers import BertTokenizer, BertForMaskedLM
import torch
import random
from pymongo import MongoClient
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Honey Encryption Study",
    page_icon="🔐",
    layout="centered",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .stApp {
        background-color: #0f0f0f;
        color: #e8e8e8;
    }
    h1, h2, h3 {
        font-family: 'IBM Plex Mono', monospace !important;
        color: #f0f0f0;
    }
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #1a1a1a;
        color: #e8e8e8;
        border: 1px solid #333;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #4af;
        box-shadow: 0 0 0 1px #4af;
    }
    .stButton > button {
        background-color: #1a1a1a;
        color: #4af;
        border: 1px solid #4af;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #4af;
        color: #0f0f0f;
    }
    .result-box {
        background-color: #1a1a1a;
        border-left: 3px solid #4af;
        padding: 1rem 1.25rem;
        border-radius: 0 4px 4px 0;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1rem;
        color: #e8e8e8;
        margin-top: 1rem;
    }
    .label-text {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #888;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    .info-box {
        background-color: #111;
        border: 1px solid #2a2a2a;
        border-radius: 6px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
        line-height: 1.6;
        color: #ccc;
    }
    .stAlert {
        background-color: #1a1a1a;
        border: 1px solid #333;
    }
    hr { border-color: #222; }
</style>
""", unsafe_allow_html=True)

# ── MongoDB init (reads from Streamlit Secrets) ─────────────────────────────────
@st.cache_resource
def init_mongo():
    uri = st.secrets["mongo"]["uri"]
    client = MongoClient(uri)
    db = client["honey_encryption"]
    return db

db = init_mongo()

# ── BERT init (cached) ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading BERT model…")
def load_models():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.eval()
    return tokenizer, model

tokenizer, bert_model = load_models()

# ── Password config ─────────────────────────────────────────────────────────────
SECRET_PASSWORD = "sunshine"

# ── Dataset ─────────────────────────────────────────────────────────────────────
NOUN_PHRASES = [
    "the old man", "she", "my father", "he", "the little boy",
    "the handsome chef", "the movie director", "his best friend",
    "her husband", "a young lady", "James", "the professor",
    "our governor", "Victor", "her mother", "the doctor",
    "they", "many students", "a friendly neighbor", "a curious child",
    "the bickering siblings"
]

VERB_PHRASES = [
    "has eaten", "died last night", "danced at the party", "loves her",
    "gave us a cup of coffee", "walked with precision", "hummed a melancholic tune",
    "devoured the last slice of cake in seconds", "quietly pondered the meaning of life",
    "furiously typed away at the keyboard", "diligently researched the topic for weeks",
    "burst into laughter at the unexpected joke", "gently cradled the newborn baby in their arms",
    "offered words of comfort to a grieving friend", "sang a beautiful song",
    "listened attentively to the lecture", "watched an exciting movie",
    "answered the phone quickly", "served us rice and beans",
    "organized the wedding", "embraced a new opportunity",
]

# ── Core logic ───────────────────────────────────────────────────────────────────
def is_semantically_correct(sentence: str) -> bool:
    tokens = tokenizer.tokenize(sentence)
    if not tokens:
        return False
    correct = 0
    for i, original_token in enumerate(tokens):
        masked_tokens = tokens[:i] + ['[MASK]'] + tokens[i + 1:]
        masked_sentence = tokenizer.convert_tokens_to_string(masked_tokens)
        inputs = tokenizer(masked_sentence, return_tensors='pt')
        with torch.no_grad():
            logits = bert_model(**inputs).logits
        mask_token_id = tokenizer.mask_token_id
        input_ids = inputs['input_ids'][0]
        mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_positions) == 0:
            continue
        mask_pos = mask_positions[0].item()
        predicted_id = torch.argmax(logits[0, mask_pos]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_id])[0]
        if predicted_token == original_token:
            correct += 1
    return (correct / len(tokens)) > 0.5


def generate_decoy(max_attempts: int = 10) -> str:
    best_sentence = None
    for _ in range(max_attempts):
        noun = random.choice(NOUN_PHRASES)
        verb = random.choice(VERB_PHRASES)
        candidate = noun + ' ' + verb
        candidate = candidate[0].upper() + candidate[1:]
        if not candidate.endswith('.'):
            candidate += '.'
        if is_semantically_correct(candidate):
            return candidate
        if best_sentence is None:
            best_sentence = candidate
    return best_sentence or "The professor organized the wedding."


def save_survey(data: dict):
    collection = db["survey_responses"]
    collection.insert_one({
        **data,
        "timestamp": datetime.utcnow().isoformat()
    })

# ── Session state defaults ───────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state["page"] = "survey"

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SURVEY
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state["page"] == "survey":

    st.title("📋 Research Survey")
    st.markdown("""
    <div class="info-box">
        <strong>Distinguishability Testing of BERT-based Honey Encryption</strong><br><br>
        Through this questionnaire, we aim to evaluate the distinguishability of a novel honey
        encryption scheme. This scheme utilises the BERT model to generate plausible-looking
        decoy data when an incorrect decryption key is used.<br><br>
        <strong>Your Task:</strong> You will be presented with multiple passwords. Your goal is
        to select and input the password you believe to be the correct one, then indicate your
        confidence level in that choice.<br><br>
        <strong>Confidentiality:</strong> All responses are anonymous and no personal data is stored.
        Results will be used strictly for academic evaluation of the model's performance.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    with st.form("survey_form"):
        st.markdown("#### Demographic Information")
        st.caption("All fields marked * are required.")

        age = st.selectbox("Age range *", [
            "", "16 - 25", "26 - 35", "36 - 45", "46 - 55", "56 - 65", "Over 65"
        ])

        gender = st.text_input("Gender *", placeholder="e.g. Male, Female, Non-binary, Prefer not to say…")

        education = st.selectbox("Highest education attained *", [
            "", "No formal education", "Basic education",
            "Senior secondary education", "Tertiary education", "Postgraduate education"
        ])

        st.markdown("#### English Language Proficiency")

        english_level = st.selectbox("How would you describe your level of English language mastery? *", [
            "", "Beginner", "Intermediate", "Advanced"
        ])

        english_exam = st.selectbox("Highest level of English Language testing exam taken and passed *", [
            "", "JAMB/WAEC", "GST", "IELTS/TOEFL/Duolingo", "Other", "None"
        ])

        submitted = st.form_submit_button("Submit & Continue →")

        if submitted:
            missing = []
            if not age:             missing.append("Age range")
            if not gender.strip():  missing.append("Gender")
            if not education:       missing.append("Highest education attained")
            if not english_level:   missing.append("English language mastery")
            if not english_exam:    missing.append("English exam")

            if missing:
                st.error(f"Please fill in the following required fields: {', '.join(missing)}")
            else:
                try:
                    save_survey({
                        "age_range": age,
                        "gender": gender.strip(),
                        "education": education,
                        "english_level": english_level,
                        "english_exam": english_exam,
                    })
                    st.session_state["page"] = "app"
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not save response to MongoDB: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ENCRYPTION TOOL
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state["page"] == "app":

    st.title("🔐 Decoy Sentence Generator")
    st.caption("Enter your sentence and a password. The correct password reveals your original sentence; anything else returns a plausible decoy.")

    st.divider()

    user_sentence = st.text_area(
        "Your sentence",
        placeholder="Type the sentence you want to protect…",
        height=100,
    )

    if st.button("Generate Decoy & Verify", disabled=not user_sentence.strip()):
        with st.spinner("Running BERT semantic validation…"):
            decoy = generate_decoy()
        st.session_state["decoy"] = decoy
        st.session_state["user_sentence"] = user_sentence
        st.success("Decoy generated. Enter your password below to reveal the result.")

    if "decoy" in st.session_state:
        st.divider()
        password = st.text_input("Password", type="password", placeholder="Enter password…")

        if st.button("Submit"):
            st.divider()
            st.markdown('<p class="label-text">Output</p>', unsafe_allow_html=True)
            if password == SECRET_PASSWORD:
                st.markdown(
                    f'<div class="result-box">✅ &nbsp;{st.session_state["user_sentence"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="result-box">🔀 &nbsp;{st.session_state["decoy"]}</div>',
                    unsafe_allow_html=True,
                )
