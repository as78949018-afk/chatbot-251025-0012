import os
import streamlit as st
from openai import OpenAI
from openai import APIError, RateLimitError, AuthenticationError

st.set_page_config(page_title="💬 Chatbot", page_icon="💬")

# ===== 사이드바: 설정 패널 =====
with st.sidebar:
    st.header("⚙️ 설정")
    # 우선순위: 환경변수/Secrets → 입력창
    default_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    openai_api_key = default_key or st.text_input("OpenAI API Key", type="password")
    model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"], index=0)
    temperature = st.slider("Temperature(창의성)", 0.0, 1.2, 0.7, 0.1)
    max_output_tokens = st.slider("Max tokens(글자 예산)", 64, 4096, 1024, 64)
    system_prompt = st.text_area(
        "System prompt(말투/역할 가이드)",
        "You are a helpful, concise assistant.",
        height=90
    )
    reset = st.button("🔄 새 대화 시작")

# ===== 상단 영역 =====
st.title("💬 Chatbot")
with st.expander("설명 보기"):
    st.write("- API 키는 세션 동안만 사용되며 서버에 저장하지 않습니다.\n- 배포 시 환경변수/Secrets 사용을 권장합니다.")

# ===== 세션 상태 초기화 =====
if "messages" not in st.session_state or reset:
    st.session_state.messages = []

# ===== API 키 체크 =====
if not openai_api_key:
    st.info("좌측에서 OpenAI API Key를 입력하세요. 🔐", icon="🗝️")
    st.stop()

# ===== 시스템 프롬프트 보장 =====
def ensure_system():
    if not st.session_state.messages or st.session_state.messages[0]["role"] != "system":
        st.session_state.messages.insert(0, {"role": "system", "content": system_prompt})
    else:
        st.session_state.messages[0]["content"] = system_prompt

# ===== 히스토리 렌더링 =====
for m in st.session_state.messages:
    if m["role"] in ("user", "assistant"):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

# ===== 입력창 =====
prompt = st.chat_input("무엇을 도와드릴까요? (Shift+Enter 줄바꿈)")
if prompt:
    ensure_system()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        client = OpenAI(api_key=openai_api_key)
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            temperature=temperature,
            max_tokens=max_output_tokens,
            stream=True,
        )
        with st.chat_message("assistant"):
            response_text = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response_text})

    except AuthenticationError:
        st.error("API 키 인증 오류입니다. 키를 확인해주세요. 🔑")
    except RateLimitError:
        st.warning("요청이 많아 잠시 대기해야 합니다. ⏳")
    except APIError as e:
        st.error(f"OpenAI API 오류: {e}")
    except Exception as e:
        st.exception(e)

