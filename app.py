import streamlit as st
from chatbot import FAQChatbot

st.set_page_config(page_title="FAQ AI Chatbot", page_icon="💬", layout="centered")
st.title("💬 Intelligent FAQ Chatbot")
st.caption("Ask a question and get the closest answer from the FAQ knowledge base.")

@st.cache_resource
def load_bot():
    return FAQChatbot("data/faqs.csv")

try:
    bot = load_bot()
except Exception as e:
    st.error(f"Could not load the FAQ knowledge base: {e}")
    st.stop()
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("About")
    st.write("This chatbot uses TF-IDF and cosine similarity to match your question with a stored FAQ.")
    st.write("**Similarity threshold:**", f"{bot.threshold:.2f}")
    st.divider()
    if st.button("Clear chat"):
        st.session_state.history = []
        st.rerun()

for item in st.session_state.history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

prompt = st.chat_input("Ask something about the sample topic (student services)...")
if prompt:
    st.session_state.history.append({"role":"user","content":prompt})
    result = bot.answer(prompt)
    if result["matched"]:
        reply = result["answer"]
        st.session_state.history.append({"role":"assistant","content":reply})
    else:
        reply = "I couldn't find a close match in my FAQ knowledge base. Try rephrasing, or ask about admissions, fees, courses, scholarships, library, or office hours."
        st.session_state.history.append({"role":"assistant","content":reply})
    st.rerun()

with st.expander("View FAQ knowledge base"):
    st.dataframe(bot.data, hide_index=True)
