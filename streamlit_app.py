# app.py
import os
import io
import json
import time
import math
import numpy as np

import streamlit as st
import openai as openai_module
from openai import OpenAI

# PDF 텍스트 추출: PyPDF2가 없으면 TXT만 지원
try:
    import PyPDF2
    HAS_PYPDF2 = True
except Exception:
    HAS_PYPDF2 = False

# ----------------------------
# 기본 세팅
# ----------------------------
st.set_page_config(page_title="💬 Chatbot (All-in-One)", page_icon="💬", layout="wide")
st.title("💬 Chatbot (All-in-One)")

with st.expander("설명 보기", expanded=False):
    st.markdown(
        "- OpenAI 모델을 사용합니다. API 키는 세션에서만 쓰이고 서버에 저장하지 않습니다.\n"
        "- 배포 시 **환경변수** 또는 **Streamlit Secrets** 사용을 권장합니다.\n"
        "- 업로드 파일(PDF/TXT)은 세션 메모리에만 저장됩니다."
    )

# ----------------------------
# 사이드바: 설정
# ----------------------------
with st.sidebar:
    st.header("⚙️ 설정")
    default_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    openai_api_key = default_key or st.text_input("OpenAI API Key", type="password", help="환경변수/Secrets가 없으면 여기에 입력")
    model = st.selectbox(
        "Model",
        ["gpt-4o", "gpt-4o-mini"],
        index=0,
        help="일반 대화: gpt-4o / 비용절감: gpt-4o-mini"
    )
    temperature = st.slider("Temperature(창의성)", 0.0, 1.2, 0.7, 0.1)
    max_output_tokens = st.slider("Max output tokens(응답 길이)", 64, 4096, 1024, 64)
    stream_enable = st.toggle("실시간 스트리밍", value=True, help="끄면 응답 후 사용량(토큰)을 표시합니다.")
    st.divider()

    st.subheader("Assistant 스타일 프리셋")
    preset = st.selectbox(
        "말투/역할 프리셋",
        ["기본", "친절한 튜터", "초간단 요약봇", "문장 다듬기(교정)"],
        index=0
    )
    preset_map = {
        "기본": "You are a helpful, concise assistant.",
        "친절한 튜터": "You are a friendly tutor. Explain step-by-step with small, clear paragraphs and examples.",
        "초간단 요약봇": "You summarize any input into 3 bullet points with the most essential facts only.",
        "문장 다듬기(교정)": "Rewrite the user's text with improved clarity, grammar, and natural tone while preserving meaning."
    }
    system_prompt = st.text_area(
        "System prompt(세부 조정 가능)",
        value=preset_map.get(preset, preset_map["기본"]),
        height=100
    )

    st.subheader("대화 관리")
    max_turns_keep = st.slider("히스토리 보존 턴(질문/답변 쌍)", 5, 60, 30, 1)
    reset = st.button("🔄 새 대화 시작")
    st.caption("너무 길어지면 비용↑/속도↓ → 오래된 기록은 자동 트림")

# ----------------------------
# 세션 상태 초기화
# ----------------------------
if "messages" not in st.session_state or reset:
    st.session_state.messages = []  # [{"role": "system"/"user"/"assistant", "content": "..."}]
    st.session_state.has_system = False

# 간단 RAG용 세션 상태
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False
if "rag_chunks" not in st.session_state:
    st.session_state.rag_chunks = []
if "rag_embeds" not in st.session_state:
    st.session_state.rag_embeds = None  # np.array shape (N, D)
if "rag_model" not in st.session_state:
    st.session_state.rag_model = "text-embedding-3-small"

# ----------------------------
# 도우미 함수
# ----------------------------
def ensure_system_message(prompt_text: str):
    """시스템 프롬프트를 messages[0]에 보장/동기화"""
    if not st.session_state.has_system:
        st.session_state.messages.insert(0, {"role": "system", "content": prompt_text})
        st.session_state.has_system = True
    else:
        # 이미 system이 있다면 내용 업데이트
        st.session_state.messages[0]["content"] = prompt_text

def trim_history(max_turns: int):
    """system + 최근 max_turns*2 유지"""
    msgs = st.session_state.messages
    if not msgs:
        return
    sys = msgs[0] if msgs and msgs[0]["role"] == "system" else None
    body = msgs[1:] if sys else msgs[:]
    limit = max_turns * 2
    if len(body) > limit:
        body = body[-limit:]
    st.session_state.messages = ([sys] if sys else []) + body

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """PDF 텍스트 추출(PYPDF2 필요). 실패 시 빈 문자열."""
    if not HAS_PYPDF2:
        return ""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        texts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            texts.append(t)
        return "\n".join(texts)
    except Exception:
        return ""

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 200):
    """슬라이딩 윈도우로 텍스트 청크화 (간단 버전)"""
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def cosine_sim(A: np.ndarray, B: np.ndarray):
    """A: (n,d), B:(m,d) -> (n,m) 코사인 유사도"""
    # 정규화
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A_norm @ B_norm.T

def build_embeddings(client: OpenAI, chunks: list[str], embed_model: str) -> np.ndarray:
    """OpenAI 임베딩 생성 -> np.array (N, D)"""
    if not chunks:
        return np.zeros((0, 1536), dtype=np.float32)
    resp = client.embeddings.create(
        model=embed_model,
        input=chunks
    )
    vecs = [d.embedding for d in resp.data]
    return np.array(vecs, dtype=np.float32)

def retrieve_context(query: str, top_k: int = 4) -> str:
    """질의와 가장 유사한 청크 top_k를 이어붙여 컨텍스트 생성"""
    if not (st.session_state.rag_ready and st.session_state.rag_embeds is not None):
        return ""
    try:
        client = OpenAI(api_key=openai_api_key)
        q_embed = client.embeddings.create(
            model=st.session_state.rag_model, input=[query]
        ).data[0].embedding
        q_vec = np.array([q_embed], dtype=np.float32)  # (1, D)
        sims = cosine_sim(q_vec, st.session_state.rag_embeds).flatten()  # (N,)
        idx = np.argsort(-sims)[:top_k]
        selected = [st.session_state.rag_chunks[i] for i in idx]
        context = "\n\n".join(selected)
        return context
    except Exception:
        return ""

def export_chat_as_txt(messages: list[dict]) -> bytes:
    lines = []
    for m in messages:
        role = m.get("role", "")
        if role == "system":
            continue
        content = m.get("content", "")
        lines.append(f"[{role.upper()}]\n{content}\n")
    return "\n".join(lines).encode("utf-8")

def export_chat_as_json(messages: list[dict]) -> bytes:
    # system도 함께 내보내고 싶으면 필터 제거
    payload = [m for m in messages if m.get("role") in ("user", "assistant")]
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

# ----------------------------
# API 키 체크
# ----------------------------
if not openai_api_key:
    st.info("좌측 사이드바에서 OpenAI API Key를 입력하거나 환경변수/Secrets를 설정하세요. 🔐", icon="🗝️")
    st.stop()

client = OpenAI(api_key=openai_api_key)

# ----------------------------
# 파일 업로드 → RAG 인덱스 만들기
# ----------------------------
st.subheader("📎 파일 업로드 (선택) — PDF/TXT 지원, 질의 응답에 활용")
uploaded_files = st.file_uploader(
    "여기에 PDF나 TXT를 올리면, 내용 기반으로 답변 품질이 향상돼요. 여러 파일 가능.",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

build_btn_cols = st.columns([1, 1, 6])
with build_btn_cols[0]:
    use_rag = st.toggle("RAG 사용", value=False, help="켜면 업로드한 파일 내용이 답변 컨텍스트로 사용됩니다.")
with build_btn_cols[1]:
    rebuild = st.button("📚 인덱스 생성/재생성")

if rebuild and uploaded_files:
    all_text = []
    for f in uploaded_files:
        if f.type == "text/plain" or f.name.lower().endswith(".txt"):
            text = f.read().decode("utf-8", errors="ignore")
        elif f.type == "application/pdf" or f.name.lower().endswith(".pdf"):
            if not HAS_PYPDF2:
                st.warning(f"'{f.name}' → PyPDF2 미설치로 PDF 추출 불가(TXT만 지원).")
                text = ""
            else:
                text = extract_text_from_pdf(f.read())
        else:
            text = ""
        if text:
            all_text.append(text)

    full_text = "\n\n".join(all_text)
    chunks = chunk_text(full_text, chunk_size=900, overlap=200)

    if not chunks:
        st.warning("추출 가능한 텍스트가 없습니다. 스캔 PDF는 텍스트가 없을 수 있어요.")
    else:
        with st.spinner("임베딩 생성 중..."):
            vecs = build_embeddings(client, chunks, st.session_state.rag_model)
        st.session_state.rag_chunks = chunks
        st.session_state.rag_embeds = vecs
        st.session_state.rag_ready = True
        st.success(f"인덱스 생성 완료! 청크 {len(chunks)}개")

# ----------------------------
# 기존 히스토리 렌더링
# ----------------------------
for m in st.session_state.messages:
    if m["role"] in ("user", "assistant"):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

# ----------------------------
# 대화 입력 + 응답
# ----------------------------
user_input = st.chat_input("무엇을 도와드릴까요? (Shift+Enter 줄바꿈)")
if user_input:
    # 시스템 프롬프트 보장 & 히스토리 트림
    ensure_system_message(system_prompt)
    trim_history(max_turns_keep)

    # 사용자 발화 표시/저장
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # RAG 컨텍스트 구성
    additional_context = ""
    if use_rag:
        ctx = retrieve_context(user_input, top_k=4)
        if ctx:
            additional_context = (
                "You may use the following context extracted from the user's documents. "
                "If the context is relevant, ground your answer in it. If not, ignore it.\n\n"
                f"=== BEGIN CONTEXT ===\n{ctx}\n=== END CONTEXT ==="
            )

    # 모델 호출
    try:
        # 메시지 구성 (RAG 컨텍스트는 추가 user 메시지로 전달)
        call_messages = list(st.session_state.messages)
        if additional_context:
            call_messages.append({"role": "user", "content": additional_context})

        if stream_enable:
            # 스트리밍
            with st.chat_message("assistant"):
                stream = client.chat.completions.create(
                    model=model,
                    messages=[{"role": m["role"], "content": m["content"]} for m in call_messages],
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                    stream=True,
                )
                response_text = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        else:
            # 비스트리밍(usage 표시)
            with st.chat_message("assistant"):
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": m["role"], "content": m["content"]} for m in call_messages],
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                    stream=False,
                )
                response_text = resp.choices[0].message.content
                st.markdown(response_text)
                if getattr(resp, "usage", None):
                    in_tok = resp.usage.prompt_tokens
                    out_tok = resp.usage.completion_tokens
                    tot_tok = resp.usage.total_tokens
                    st.caption(f"🧮 tokens — prompt: {in_tok}, completion: {out_tok}, total: {tot_tok}")
            st.session_state.messages.append({"role": "assistant", "content": response_text})

    except openai_module.AuthenticationError:
        st.error("API 키 인증 오류입니다. 키를 확인해주세요. 🔑")
    except openai_module.RateLimitError:
        st.warning("요청이 많아 잠시 대기해야 합니다. ⏳")
    except openai_module.APIError as e:
        st.error(f"OpenAI API 오류: {e}")
    except Exception as e:
        st.exception(e)

# ----------------------------
# 내보내기(다운로드)
# ----------------------------
st.divider()
st.subheader("⬇️ 대화 내보내기")
col_txt, col_json = st.columns(2)
with col_txt:
    st.download_button(
        "TXT로 저장",
        data=export_chat_as_txt(st.session_state.messages),
        file_name="chat_export.txt",
        mime="text/plain",
        use_container_width=True
    )
with col_json:
    st.download_button(
        "JSON으로 저장",
        data=export_chat_as_json(st.session_state.messages),
        file_name="chat_export.json",
        mime="application/json",
        use_container_width=True
    )

# 푸터 도움말
st.caption(
    "Tip: 스트리밍을 끄면(사이드바) 응답 후 사용량(토큰)을 보여줍니다. "
    "PDF 텍스트 추출은 문서 유형에 따라 일부 제한될 수 있어요."
)
