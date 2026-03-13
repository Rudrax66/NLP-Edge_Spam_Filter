"""
app.py — Streamlit Web App
Edge-Device Spam & Toxic Filter
Run: streamlit run app.py
"""

import streamlit as st
import sys
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Edge Spam & Toxic Filter",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Inline pipeline (no external files needed if running standalone) ──────────
# If you have the full project, it imports from pipeline.py automatically.
# Otherwise this file contains a self-contained mini version.

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pipeline import EdgeSpamToxicFilter
    FULL_PROJECT = True
except ImportError:
    FULL_PROJECT = False

# ── Fallback: self-contained mini pipeline ────────────────────────────────────
if not FULL_PROJECT:
    import re
    import string
    import unicodedata
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import LinearSVC
    from sklearn.naive_bayes import ComplementNB
    from sklearn.ensemble import VotingClassifier
    from sklearn.preprocessing import StandardScaler
    from scipy.sparse import hstack, csr_matrix
    from sklearn.model_selection import train_test_split
    import random

    LABEL_NAMES = {0: "HAM", 1: "SPAM", 2: "TOXIC"}
    STOPWORDS = {
        "a","an","the","and","or","but","in","on","at","to","for","of","with",
        "by","from","is","are","was","were","be","been","have","has","do","does",
        "did","will","would","could","should","this","that","i","me","my","we",
        "you","your","he","she","it","they","them","not","no","so","just","as"
    }
    SPAM_PATS = [
        r'\b(free|win|winner|won|prize|claim|reward)\b',
        r'\b(click here|click now|buy now|order now|act now)\b',
        r'\b(limited time|limited offer|expires?)\b',
        r'\b(100%|guaranteed|money back)\b',
        r'\b(earn \$|make money|extra income|work from home)\b',
        r'(http[s]?://\S+)', r'(\$\d+[\.,]?\d*)',
        r'([A-Z]{4,})', r'(!{2,})',
    ]
    TOXIC_PATS = [
        r'\b(hate|kill|die|murder|attack|destroy)\b',
        r'\b(idiot|stupid|moron|dumb|loser|worthless)\b',
        r'\b(threat|threaten|violence|violent)\b',
        r'\b(ugly|disgusting|horrible|terrible|awful)\b.*\b(you|your|him|her)\b',
    ]

    HAM = [
        "Hey, are you free to grab coffee tomorrow morning?",
        "Please find the attached report for your review.",
        "The meeting has been rescheduled to 3 PM on Friday.",
        "Can you pick up some milk on your way home?",
        "I really enjoyed the book you recommended last week.",
        "The project deadline has been extended by two days.",
        "Happy birthday! Hope you have a wonderful day.",
        "Just checking in to see how you're doing.",
        "The weather forecast says it will rain tomorrow.",
        "Thanks for your help with the presentation yesterday.",
        "I'll be working from home on Monday.",
        "The new software update fixed the login issue.",
        "Dinner is ready when you get home.",
        "Please review the attached contract before signing.",
        "The quarterly results exceeded our expectations this year.",
        "Let me know if you need any help with the assignment.",
        "The flight has been delayed by 30 minutes.",
        "Could you send me the file we discussed earlier?",
        "I'll be in the office by 9 AM tomorrow.",
        "Great job on the presentation today everyone!",
        "The library closes at 8 PM on weekdays.",
        "Your package has been shipped and will arrive Thursday.",
        "We've scheduled the team retrospective for next week.",
        "The new café around the corner has great pastries.",
        "Please remember to submit your timesheet by Friday.",
        "The conference call starts in 10 minutes.",
        "I left my umbrella in the conference room.",
        "The annual performance reviews begin next month.",
        "Could we reschedule our appointment to the afternoon?",
        "The kids school play is on Saturday at 6 PM.",
        "I am reading a really interesting book on machine learning.",
        "The doctor said everything looks good in my checkup.",
        "We need to order more printer paper for the office.",
        "The project documentation is now available on the wiki.",
        "Mom called she wants to know if you are coming for dinner.",
        "The gym is offering a free trial membership this week.",
        "I have uploaded the latest design files to the shared folder.",
        "Our internet was down this morning but it is back now.",
        "Please add your vacation days to the shared calendar.",
        "The report contains three main sections: intro, analysis, conclusion.",
        "We are happy to announce that our product launch was successful.",
        "Can you review the pull request I submitted this morning?",
        "The training session starts at 10 AM in the main hall.",
        "I have been learning Spanish for about three months now.",
        "The new employee handbook has been sent to everyone.",
        "Your subscription has been renewed successfully.",
        "We are planning a hiking trip next weekend.",
        "Please let us know your dietary preferences before the event.",
        "The budget has been approved for the next quarter.",
        "I will see you at the conference next Tuesday.",
    ]
    SPAM = [
        "CONGRATULATIONS! You've WON a FREE iPhone 15! CLICK HERE to claim NOW!!!",
        "You have been selected as the winner of our $1,000,000 lottery! Verify now.",
        "FREE OFFER: Buy 1 get 10 FREE! Limited time only. Act NOW before it expires!!!",
        "Urgent: Your account has been compromised. Click here to verify your password.",
        "Make $500 a day working from home! No experience needed. Limited spots left!",
        "EARN EXTRA INCOME from home! Guaranteed $5,000 per week. Click to learn more!",
        "Your PayPal account has been suspended. Confirm your details at http://paypa1.verify.com",
        "Hot singles in your area! Click here to meet them tonight. FREE registration!",
        "You've been chosen for a special prize! Call 1-800-555-0199 to claim your reward.",
        "Buy Cheap Meds Online! No prescription needed! 90% off retail prices! ORDER NOW!",
        "INVESTMENT OPPORTUNITY: 500% returns guaranteed! Send $100 to receive $500 back!",
        "FINAL NOTICE: Your computer has a virus! Download our FREE scanner immediately!",
        "Dear Friend, I am Prince Johnson and I need your help transferring $15,000,000 USD.",
        "Lose 30 lbs in 30 days! Our miracle pill GUARANTEED to work or money back!!!",
        "Your Netflix subscription expires TODAY. Update payment at http://netf1ix-bill.com",
        "URGENT: IRS final notice. You owe back taxes. Call 1-888-555-0123 IMMEDIATELY!",
        "Win a free vacation to Cancun! Just answer 3 questions! No purchase necessary!",
        "EXCLUSIVE OFFER for YOU only! Get rich quick with our proven investment system!",
        "Refinance your mortgage NOW! Lowest rates ever! Limited time offer. APPLY TODAY!",
        "FREE casino chips! Sign up now and get $500 bonus no deposit required!!!",
        "DEAR LUCKY WINNER! Your email was selected. Claim your $750,000 prize NOW!",
        "We have a business proposal for you. Reply to claim your share of $45 million.",
        "Your Amazon order has an issue. Verify your account: http://amaz0n-secure.xyz",
        "MIRACLE CURE discovered by doctors! Big pharma does not want you to know this!",
        "Work online, earn thousands per week! Join 10,000 happy members TODAY for FREE!",
        "ALERT! Your bank account will be CLOSED unless you verify at http://bank-secure.info",
        "FREE iPhone! Complete a survey and get yours shipped FREE! Limited stock!!!",
        "Herbal supplements 100% natural 100% GUARANTEED! Buy 2 get 1 FREE NOW!",
        "Unclaimed inheritance funds in your name: $3.2 million. Contact agent John for details.",
        "SPECIAL: Click NOW to unlock your premium account for FREE! Offer expires midnight!",
        "Credit card debt relief pay nothing! We will erase your debt GUARANTEED! Call now.",
        "WARNING: Your computer is infected with 5 viruses! Download our cleaner IMMEDIATELY!",
        "You have won a $250 Best Buy gift card! Click here: http://bestbuy-rewards.net/claim",
        "Make money fast with crypto! 1000% profit in 30 days! Only 10 spots available!",
        "SEXY SINGLES want to chat! Join FREE today over 1 million members online now!",
    ]
    TOXIC = [
        "You are the most stupid and worthless person I have ever met.",
        "I hate you so much. You should just disappear from everyone's life.",
        "You are an absolute idiot. How can someone be this dumb?",
        "Nobody likes you. Everyone thinks you are a loser and a failure.",
        "You should be ashamed of yourself. You are disgusting and horrible.",
        "I cannot believe how pathetic and useless you are at everything.",
        "You are nothing but a liar and a manipulative piece of trash.",
        "Everyone hates you and no one will ever care about you.",
        "Stop talking. Your opinions are stupid and nobody wants to hear them.",
        "You are a waste of space and the world would be better without you.",
        "I despise you with every fiber of my being. Get out of my sight.",
        "You are the worst person alive. Completely disgusting and vile.",
        "Your existence is a mistake. Nobody could ever love something like you.",
        "You are so ugly and repulsive it is honestly hard to look at you.",
        "I will make your life miserable. You deserve to suffer for what you did.",
        "You call yourself a professional? You are an embarrassment to the field.",
        "Shut up you brain-dead moron. Your ideas are garbage and always will be.",
        "You are a bigot and a coward who hides behind a keyboard spreading hate.",
        "Go away. No one wants you here. You contribute absolutely nothing.",
        "You are literally the dumbest person I have ever had to deal with.",
        "I have never met someone as vile and disgusting as you in my entire life.",
        "You are a toxic hateful manipulative person who hurts everyone around you.",
        "You are pathetic. A complete failure who cannot do anything right.",
        "Your stupidity is only matched by your arrogance. Truly insufferable.",
        "You are a racist hateful person and you make everyone around you miserable.",
    ]

    def _clean(text):
        text = unicodedata.normalize('NFKD', text).encode('ascii','ignore').decode('ascii')
        text = re.sub(r'<[^>]+>',' ',text)
        text = re.sub(r'\s+',' ',text)
        return text.strip()

    def _get_features(text):
        raw = text
        return [
            len(raw), len(raw.split()),
            sum(c.isdigit() for c in raw)/max(len(raw),1),
            sum(c.isupper() for c in raw)/max(len(raw),1),
            sum(c in string.punctuation for c in raw)/max(len(raw),1),
            raw.count('!'), raw.count('?'),
            len(re.findall(r'https?://\S+', raw)),
            len(re.findall(r'\$\d+', raw)),
            len(re.findall(r'\b[A-Z]{3,}\b', raw)),
            sum(1 for p in [re.compile(x,re.I) for x in SPAM_PATS] if p.search(raw)),
            sum(1 for p in [re.compile(x,re.I) for x in TOXIC_PATS] if p.search(raw)),
            int(bool(re.search(r'https?://', raw))),
            int(bool(re.search(r'\S+@\S+\.\S+', raw))),
            len(re.findall(r'(.)\1{2,}', raw)),
            len(set(raw.lower().split()))/max(len(raw.split()),1),
        ]

    def _gen_data():
        random.seed(42)
        texts, labels = [], []
        for _ in range(200):
            texts.append(random.choice(HAM)); labels.append(0)
        for _ in range(200):
            t = random.choice(SPAM)
            if random.random()<0.3: t=t.upper()
            texts.append(t); labels.append(1)
        for _ in range(200):
            texts.append(random.choice(TOXIC)); labels.append(2)
        c = list(zip(texts,labels)); random.shuffle(c)
        return zip(*c)

    class MiniFilter:
        def __init__(self):
            self.wtfidf = TfidfVectorizer(analyzer='word',ngram_range=(1,3),
                max_features=10000,sublinear_tf=True,min_df=1)
            self.ctfidf = TfidfVectorizer(analyzer='char_wb',ngram_range=(2,4),
                max_features=8000,sublinear_tf=True,min_df=1)
            self.scaler = StandardScaler(with_mean=False)
            self.model  = VotingClassifier([
                ('lr', LogisticRegression(C=5,max_iter=1000,class_weight='balanced',random_state=42)),
                ('svm', CalibratedClassifierCV(
                    LinearSVC(C=1,max_iter=3000,class_weight='balanced',random_state=42),cv=3)),
                ('nb', ComplementNB(alpha=0.1)),
            ], voting='soft', weights=[3,2,1])

        def train(self):
            texts, labels = _gen_data()
            texts, labels = list(texts), list(labels)
            hc = np.array([_get_features(t) for t in texts], dtype=np.float32)
            wf = self.wtfidf.fit_transform(texts)
            cf = self.ctfidf.fit_transform(texts)
            hcf= csr_matrix(self.scaler.fit_transform(hc))
            X  = hstack([wf, cf, hcf])
            self.model.fit(X, labels)

        def predict(self, text):
            cleaned = _clean(text)
            hc  = np.array([_get_features(text)], dtype=np.float32)
            wf  = self.wtfidf.transform([cleaned])
            cf  = self.ctfidf.transform([cleaned])
            hcf = csr_matrix(self.scaler.transform(hc))
            X   = hstack([wf, cf, hcf])
            label = int(self.model.predict(X)[0])
            proba = self.model.predict_proba(X)[0]
            confs = {LABEL_NAMES[i]: round(float(p),4) for i,p in enumerate(proba)}
            return LABEL_NAMES[label], confs

    # Wrapper to match EdgeSpamToxicFilter interface
    class EdgeSpamToxicFilter:
        def __init__(self, **kwargs): self._m = MiniFilter()
        def train(self, **kwargs): self._m.train()
        def analyze(self, text):
            label, confs = self._m.predict(text)
            conf = confs[label]
            action = "ALLOW" if label=="HAM" else ("BLOCK" if conf>0.75 else "FLAG")
            return {
                "prediction": label,
                "confidence": conf,
                "all_scores": confs,
                "action": action,
                "features": {
                    "spam_pattern_hits": sum(1 for p in SPAM_PATS if re.search(p,text,re.I)),
                    "toxic_pattern_hits": sum(1 for p in TOXIC_PATS if re.search(p,text,re.I)),
                    "url_count": len(re.findall(r'https?://\S+', text)),
                    "exclamation_count": text.count('!'),
                    "all_caps_words": len(re.findall(r'\b[A-Z]{3,}\b', text)),
                }
            }


# ─────────────────────────────────────────────────────────────────────────────
# Load / train model (cached across sessions)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    f = EdgeSpamToxicFilter(model_type="ensemble", verbose=False)
    f.train(n_samples=600)
    return f


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.stApp { background: #0d0f14; color: #e8e8e8; }

.result-box {
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    line-height: 1.8;
}
.ham-box  { background: #0d2d1a; border: 1px solid #1a6b3a; color: #4ade80; }
.spam-box { background: #2d0d0d; border: 1px solid #8b1a1a; color: #f87171; }
.toxic-box{ background: #2d2000; border: 1px solid #8b6000; color: #fbbf24; }

.badge {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 1rem;
    font-family: 'Syne', sans-serif;
    letter-spacing: 0.08em;
}
.badge-ham   { background: #14532d; color: #4ade80; }
.badge-spam  { background: #450a0a; color: #f87171; }
.badge-toxic { background: #451a00; color: #fbbf24; }

.score-bar-wrap { margin: 0.4rem 0; }
.score-label { display: inline-block; width: 70px; font-size: 0.8rem; }
.score-bar {
    display: inline-block;
    height: 8px;
    border-radius: 4px;
    vertical-align: middle;
    margin-left: 6px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# 🛡️ Edge Spam & Toxic Filter")
st.markdown("**NLP classifier · runs 100% on-device · no API key**")
st.divider()

# Train model
with st.spinner("⚙️ Training model... (first load only, ~15 sec)"):
    model = load_model()
st.success("✅ Model ready", icon="✅")

st.markdown("### Analyze a message")
text_input = st.text_area(
    label="Enter any text:",
    placeholder="Type or paste a message here...",
    height=120,
    label_visibility="collapsed"
)

col1, col2 = st.columns([1, 4])
with col1:
    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

if analyze_btn:
    if not text_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        result = model.analyze(text_input.strip())
        pred   = result["prediction"]
        conf   = result["confidence"]
        action = result["action"]
        scores = result["all_scores"]
        feats  = result["features"]

        # Badge + action
        badge_class = {"HAM":"badge-ham","SPAM":"badge-spam","TOXIC":"badge-toxic"}[pred]
        icon  = {"HAM":"✅","SPAM":"🚫","TOXIC":"⚠️"}[pred]
        color = {"HAM":"ham","SPAM":"spam","TOXIC":"toxic"}[pred]

        st.markdown(f"""
        <div class="result-box {color}-box">
            <div style="margin-bottom:0.8rem">
                <span class="badge {badge_class}">{icon} {pred}</span>
                &nbsp;&nbsp;
                <strong>Confidence:</strong> {conf*100:.1f}% &nbsp;&nbsp;
                <strong>Action:</strong> {action}
            </div>
            <div>
        """, unsafe_allow_html=True)

        # Score bars
        bar_colors = {"HAM":"#4ade80","SPAM":"#f87171","TOXIC":"#fbbf24"}
        for label, score in scores.items():
            width = int(score * 180)
            st.markdown(f"""
                <div class="score-bar-wrap">
                    <span class="score-label">{label}</span>
                    <span style="font-size:0.8rem">{score*100:.1f}%</span>
                    <span class="score-bar" style="width:{width}px;background:{bar_colors[label]}"></span>
                </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
            <br/>
            <strong>Signals detected:</strong><br/>
            &nbsp;• Spam pattern hits: {feats['spam_pattern_hits']}<br/>
            &nbsp;• Toxic pattern hits: {feats['toxic_pattern_hits']}<br/>
            &nbsp;• URLs found: {feats['url_count']}<br/>
            &nbsp;• Exclamations: {feats['exclamation_count']}<br/>
            &nbsp;• ALL-CAPS words: {feats['all_caps_words']}
        </div></div>
        """, unsafe_allow_html=True)


# ── Try example messages ──────────────────────────────────────────────────────
st.divider()
st.markdown("### 💡 Try example messages")

examples = {
    "✅ Ham": "The project deadline has been extended. Please review the updated timeline.",
    "🚫 Spam": "CONGRATULATIONS!!! You've WON a FREE iPhone 15! CLICK HERE to claim NOW!!!",
    "⚠️ Toxic": "You are the most stupid and worthless person I have ever met in my life.",
    "🚫 Phishing": "Your PayPal account is suspended. Verify now: http://paypa1-secure.xyz",
    "⚠️ Hate": "I hate you. Nobody likes you and everyone thinks you are a complete loser.",
}

cols = st.columns(len(examples))
for col, (label, msg) in zip(cols, examples.items()):
    with col:
        if st.button(label, use_container_width=True):
            st.session_state["example_text"] = msg

if "example_text" in st.session_state:
    ex = st.session_state["example_text"]
    result = model.analyze(ex)
    pred   = result["prediction"]
    icon   = {"HAM":"✅","SPAM":"🚫","TOXIC":"⚠️"}[pred]
    color  = {"HAM":"ham","SPAM":"spam","TOXIC":"toxic"}[pred]
    conf   = result["confidence"]
    st.markdown(f"""
    <div class="result-box {color}-box">
        <strong>Text:</strong> {ex[:100]}<br/>
        <strong>Result:</strong>
        <span class="badge badge-{color.lower()}">{icon} {pred}</span>
        &nbsp; {conf*100:.1f}% confidence &nbsp; → {result['action']}
    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;opacity:0.4;font-size:0.75rem;font-family:'JetBrains Mono',monospace">
Edge NLP · scikit-learn · TF-IDF · Ensemble Voting · No API Key
</div>
""", unsafe_allow_html=True)
