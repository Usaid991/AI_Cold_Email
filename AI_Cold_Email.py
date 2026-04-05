import streamlit as st  # UI
from langchain.embeddings import OpenAIEmbeddings  # text embeddings
from langchain.vectorstores import Chroma  # vector database
from groq import Groq  # Groq AI
from fpdf import FPDF  # PDF generation
from io import BytesIO  # memory buffer

# ------------------ KEYS ------------------
GROQ_API_KEY = "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"  # Groq
OPENAI_API_KEY = "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"  # OpenAI

# ------------------ INIT ------------------
st.set_page_config(page_title="AI Cold Email Generator", layout="wide")  # page layout
if "groq_client" not in st.session_state: st.session_state.groq_client = Groq(api_key=GROQ_API_KEY)  # init Groq
if "portfolio" not in st.session_state: st.session_state.portfolio = []  # init portfolio

# ------------------ SIDEBAR ------------------
st.sidebar.title("Settings")
theme = st.sidebar.radio("Theme", ["Light", "Dark"])  # theme
lang = st.sidebar.selectbox("Language", ["English", "French", "Spanish"])
tone = st.sidebar.selectbox("Tone", ["Professional", "Casual", "Friendly"])
num_emails = st.sidebar.slider("Emails", 1, 5, 3)
top_k = st.sidebar.slider("Top Projects", 1, 5, 3)

# portfolio form
with st.sidebar.form("portfolio_form"):
    t = st.text_input("Project Title")
    d = st.text_area("Description")
    if st.form_submit_button("Add") and t.strip() and d.strip(): st.session_state.portfolio.append({"title": t.strip(),"desc":d.strip()})

# ------------------ CSS ------------------
st.markdown(f"<style>{'textarea{background:#fff;color:#000}' if theme=='Light' else 'textarea{background:#1e1e1e;color:#fff}'}</style>", unsafe_allow_html=True)

# ------------------ MAIN ------------------
st.markdown("<h1 style='text-align:center;color:#1f77b4;'>AI Cold Email Generator</h1>", unsafe_allow_html=True)
job_desc = st.text_area("Enter Job Description", height=150)

# ------------------ FUNCTIONS ------------------
def get_projects(desc, k=3):  # get top projects
    if not st.session_state.portfolio: return []
    try:
        texts = [f"{p['title']}: {p['desc']}" for p in st.session_state.portfolio]
        emb = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
        db = Chroma.from_texts(texts, emb)
        results = db.similarity_search(desc, k=k)
        return [{"title": r.page_content.split(":")[0], "desc": ":".join(r.page_content.split(":")[1:])} for r in results]
    except Exception as e: st.error(f"Embedding error: {e}"); return []

def call_groq(prompt):  # Groq email
    try:
        r = st.session_state.groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user","content":prompt}]
        )
        return r.choices[0].message.content
    except Exception as e: return f"Error: {e}"

def gen_emails(desc, projects, lang, tone, n=3):
    port_txt = "\n".join([f"- {p['title']}: {p['desc']}" for p in projects]) if projects else "No portfolio."
    emails = []
    for _ in range(n):
        prompt=f"""Language:{lang}\nTone:{tone}\nJob Description:\n{desc}\nRelevant Portfolio:\n{port_txt}\nWrite a professional cold email."""
        emails.append(call_groq(prompt))
    return emails

def download_pdf(emails):
    pdf = FPDF(); pdf.add_page(); pdf.set_auto_page_break(True,15); pdf.set_font("Arial",12)
    for i,e in enumerate(emails): pdf.multi_cell(0,8,f"Email {i+1}:\n{e}\n\n")
    return BytesIO(pdf.output(dest='S').encode('latin-1'))

# ------------------ GENERATE ------------------
if st.button("Generate Emails") and job_desc.strip():
    with st.spinner("Generating..."):
        projects = get_projects(job_desc, top_k)
        emails = gen_emails(job_desc, projects, lang, tone, num_emails)

    st.markdown("**Top Relevant Projects:**")
    for p in projects: st.markdown(f"- {p['title']}")

    tabs = st.tabs([f"Email {i+1}" for i in range(len(emails))])
    for i, tab in enumerate(tabs):
        with tab: st.text_area("Email", emails[i], height=250, key=f"text_{i}")

    st.download_button("Download PDF", download_pdf(emails))
    st.download_button("Download All TXT", "\n\n".join(emails))
