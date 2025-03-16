import streamlit as st
import openai
import PyPDF2
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#####################
# 1. 기본 설정 및 API 키
#####################
st.set_page_config(page_title="약수초등학교 업무용 챗봇", layout="wide")


# API 키 설정 (환경 변수 사용을 추천)
api_key= st.secrets["api_key"]

# OpenAI 클라이언트 생성
client = openai.Client(api_key=api_key)

#####################
# 2. 사이드바 (사용 설명)
#####################
with st.sidebar:
    st.title("사용 설명")
    st.markdown("""
    - **챗봇 소개:** 약수초등학교 업무 관련 정보를 제공하는 챗봇입니다.
    - **사용 방법:**
      1. 하단의 입력창에 질문을 입력하세요.
      2. 엔터를 누르거나 '전송' 버튼을 누르면, 답변을 받을 수 있습니다.
      © 2025 Jang Se Man <jangseman12@gmail.com>
    """)

#####################
# 3. 메인 화면 헤더
#####################
st.title("약수초등학교 업무용 챗봇")
st.caption('약수초등학교의 업무에 대해 자유롭게 질문해주세요.')

#####################
# 4. PDF 로드 및 전처리
#####################
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(model=model, input=[text])
    return response.data[0].embedding

@st.cache_data
def compute_chunk_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        embedding = get_embedding(chunk)
        embeddings.append(embedding)
    return embeddings

def get_relevant_context(query, chunks, chunk_embeddings, top_k=3):
    query_embedding = np.array(get_embedding(query))
    scores = []
    for emb in chunk_embeddings:
        score = cosine_similarity([query_embedding], [np.array(emb)])[0][0]
        scores.append(score)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    context = "\n\n".join([chunks[i] for i in top_indices])
    return context

def generate_answer(query, context):
    prompt = f"""아래의 학교 문서를 참고하여 질문에 답변해 주세요.
    약수초등학교에 관련된 질문만 답변을 해주세요.
    문서에 나와있지 않은 모르는 내용에 대해서는 죄송합니다. 정보를 찾을 수 없습니다. 라고 대답해주세요.

[문서 내용]
{context}

[질문]
{query}

[답변]
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


# PDF 파일 목록 (여기에 원하는 파일 추가)
pdf_files = ["학사일정.pdf", "학교정보.pdf"]

# 여러 개의 PDF에서 텍스트 추출
full_text = ""

for pdf_path in pdf_files:
    if os.path.exists(pdf_path):
        text = extract_text_from_pdf(pdf_path)
        full_text += text + "\n\n"  # 각 파일의 내용을 구분하여 추가
    else:
        st.warning(f"{pdf_path} 파일을 찾을 수 없습니다. 동일 폴더에 위치시켜 주세요.")

chunks = chunk_text(full_text, chunk_size=500)
chunk_embeddings = compute_chunk_embeddings(chunks)

#####################
# 6. 대화 세션 상태 관리
#####################
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 기존 대화 기록 출력 (각 메시지를 DeltaGenerator 객체의 .write()를 이용해 출력)
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message("assistant").write(msg["content"])

#####################
# 7. 사용자 입력 (화면 하단 입력창)
#####################
user_query = st.chat_input("약수초등학교에 대해 궁금한 점을 물어보세요...")
if user_query:
    # 사용자 메시지 추가 및 출력
    st.session_state["messages"].append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    
    # RAG 기반 문맥 추출 및 GPT-4 답변 생성
    context = get_relevant_context(user_query, chunks, chunk_embeddings)
    answer = generate_answer(user_query, context)
    
    # 챗봇 답변 추가 및 출력
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
