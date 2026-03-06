import streamlit as st
from transformers import BertTokenizer, BertForMaskedLM
import torch
import random
from pymongo import MongoClient
from datetime import datetime

st.set_page_config(
    page_title="Honey Encryption Study",
    page_icon="🔐",
    layout="centered",
)

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
        margin-bottom: 1rem;
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
    .step-indicator {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #555;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .step-active { color: #4af; }
    .thank-you-box {
        background-color: #111;
        border: 1px solid #1a3a1a;
        border-left: 3px solid #4a4;
        border-radius: 6px;
        padding: 2rem;
        text-align: center;
        color: #ccc;
        font-size: 1rem;
        line-height: 1.8;
    }
    .stAlert { background-color: #1a1a1a; border: 1px solid #333; }
    hr { border-color: #222; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_mongo():
    uri = st.secrets["mongo"]["uri"]
    client = MongoClient(uri)
    return client["honey_encryption"]

db = init_mongo()

@st.cache_resource(show_spinner="Loading BERT model…")
def load_models():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.eval()
    return tokenizer, model

tokenizer, bert_model = load_models()

SECRET_SENTENCE = "God does not play dice with the universe."
SECRET_PASSWORD = "Iam123@"

DECOY_PASSWORDS = [
    "password1", "123456789", "qwerty123", "iloveyou1",
    "admin@123", "letmein1!", "welcome12", "monkey123",
    "dragon123", "master123", "sunshine1", "shadow123",
    "michael1!", "jessica12", "football1", "baseball1",
    "abc123456", "trustno1!", "superman1", "batman123",
]

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


def generate_decoy_sentence(max_attempts: int = 10) -> str:
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


def build_password_options() -> list:
    decoys = random.sample(DECOY_PASSWORDS, 4)
    options = decoys + [SECRET_PASSWORD]
    random.shuffle(options)
    return options


def save_response(data: dict):
    db["survey_responses"].insert_one({
        **data,
        "timestamp": datetime.utcnow().isoformat()
    })


def render_steps(current: int):
    """Render a simple step indicator. Steps: 1=Survey, 2=Trial, 3=Confidence, 4=Done"""
    labels = ["Survey", "Password Trial", "Confidence", "Complete"]
    cols = st.columns(len(labels))
    for i, (col, label) in enumerate(zip(cols, labels)):
        step_num = i + 1
        if step_num == current:
            col.markdown(f'<div class="step-indicator step-active">● Step {step_num}<br>{label}</div>', unsafe_allow_html=True)
        elif step_num < current:
            col.markdown(f'<div class="step-indicator" style="color:#4a4;">✓ Step {step_num}<br>{label}</div>', unsafe_allow_html=True)
        else:
            col.markdown(f'<div class="step-indicator">○ Step {step_num}<br>{label}</div>', unsafe_allow_html=True)


if "page" not in st.session_state:
    st.session_state["page"] = "survey"
if "survey_data" not in st.session_state:
    st.session_state["survey_data"] = {}
if "trial_data" not in st.session_state:
    st.session_state["trial_data"] = {}
if "password_options" not in st.session_state:
    st.session_state["password_options"] = build_password_options()


if st.session_state["page"] == "survey":

    render_steps(1)
    st.divider()
    st.title("📋 Research Survey")
    st.markdown("""
    <div class="info-box">
        <strong>Distinguishability Testing of BERT-based Honey Encryption</strong><br><br>
        Through this questionnaire, we aim to evaluate the distinguishability of a novel honey
        encryption scheme. This scheme utilises the BERT model to generate plausible-looking
        decoy data when an incorrect decryption key is used.<br><br>
        <strong>Your Task:</strong> You will be presented with a list of passwords. Your goal is
        to select the password you believe to be the correct one. You will then see the decrypted
        output and be asked to rate your confidence.<br><br>
        <strong>Confidentiality:</strong> All responses are anonymous and no personal data is stored.
        Results will be used strictly for academic evaluation of the model's performance.
    </div>
    """, unsafe_allow_html=True)

    with st.form("survey_form"):
        st.markdown("#### Demographic Information")
        st.caption("All fields marked * are required.")

        age = st.selectbox("Age range *", [
            "", "16 - 25", "26 - 35", "36 - 45", "46 - 55", "56 - 65", "Over 65"
        ])
        gender = st.selectbox("Gender *", ["", "Male", "Female", "Prefer not to say"])
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
                st.session_state["survey_data"] = {
                    "age_range": age,
                    "gender": gender.strip(),
                    "education": education,
                    "english_level": english_level,
                    "english_exam": english_exam,
                }
                st.session_state["page"] = "trial"
                st.rerun()


elif st.session_state["page"] == "trial":

    render_steps(2)
    st.divider()
    st.title("🔐 Password Trial")
    st.caption("Select the password you believe is correct and submit to see the decrypted output.")

    st.divider()

    st.markdown("#### Encrypted message")
    st.markdown(
        '<div class="result-box">🔒 &nbsp;<em>[This message is encrypted. The correct password will reveal it.]</em></div>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("#### Select a password")
    password_options = st.session_state["password_options"]
    selected_password = st.radio(
        "Which password do you think is correct?",
        options=password_options,
        index=None,
        label_visibility="collapsed",
    )

    st.divider()

    if st.button("Decrypt →"):
        if not selected_password:
            st.error("Please select a password before continuing.")
        else:
            is_correct = selected_password == SECRET_PASSWORD

            if is_correct:
                output_sentence = SECRET_SENTENCE
            else:
                with st.spinner("Generating decoy sentence…"):
                    output_sentence = generate_decoy_sentence()

            st.session_state["trial_data"] = {
                "selected_password": selected_password,
                "correct_password_chosen": is_correct,
                "sentence_shown": output_sentence,
            }
            st.session_state["page"] = "confidence"
            st.rerun()



elif st.session_state["page"] == "confidence":

    render_steps(3)
    st.divider()
    st.title("📊 Decrypted Output")
    st.caption("This is the message produced by the password you selected.")

    st.divider()

    trial = st.session_state["trial_data"]
    is_correct = trial["correct_password_chosen"]
    output_sentence = trial["sentence_shown"]

    st.markdown('<p class="label-text">Decrypted Output</p>', unsafe_allow_html=True)
    icon = "✅" if is_correct else "🔀"
    st.markdown(
        f'<div class="result-box">{icon} &nbsp;{output_sentence}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(f"**Password used:** `{trial['selected_password']}`")

    st.divider()

    st.markdown("#### How confident were you in your password choice?")
    st.caption("Rate your confidence based on your choice, not on the output you see above.")

    confidence = st.radio(
        "Confidence level",
        options=["Not confident", "Somewhat confident", "Very confident"],
        index=None,
        label_visibility="collapsed",
    )

    st.divider()

    if st.button("Submit Response →"):
        if not confidence:
            st.error("Please rate your confidence level before submitting.")
        else:
            try:
                save_response({
                    **st.session_state["survey_data"],
                    **st.session_state["trial_data"],
                    "confidence": confidence,
                })
                st.session_state["page"] = "done"
                st.rerun()
            except Exception as e:
                st.error(f"Could not save to MongoDB: {e}")



elif st.session_state["page"] == "done":

    render_steps(4)
    st.divider()
    st.title("🎉 Thank You!")
    st.markdown("""
    <div class="thank-you-box">
        Your response has been recorded successfully.<br><br>
        Thank you for participating in this research study on<br>
        <strong>BERT-based Honey Encryption distinguishability</strong>.<br><br>
        Your contribution will help evaluate the effectiveness of the model
        in generating plausible decoy data. The results will be used
        strictly for academic purposes.
    </div>
    """, unsafe_allow_html=True)
