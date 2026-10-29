import json
from pathlib import Path

import pandas as pd
import streamlit as st

TITLE = "뇌졸중 예측 실습실"
ICON = "🧠"
DATA_URL = "https://raw.githubusercontent.com/hsc-go/data/main/stroke.csv"
SAVE_DIR = Path(__file__).parent / "내_기록"
SAVE_FILE = SAVE_DIR / "기록.json"

st.set_page_config(page_title=TITLE, page_icon=ICON, layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 2.8rem; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- 데이터 불러오기 ----------
@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


# ---------- 기록 불러오기 / 저장하기 ----------
def load_records() -> dict:
    default = {"우리말_뜻": {}, "출처": ""}
    try:
        with open(SAVE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return {
            "우리말_뜻": dict(data.get("우리말_뜻", {})),
            "출처": str(data.get("출처", "")),
        }
    except (FileNotFoundError, json.JSONDecodeError, OSError, AttributeError, TypeError, ValueError):
        return default


def save_records(records: dict) -> bool:
    try:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        with open(SAVE_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        return True
    except OSError:
        return False


def describe_values(s: pd.Series) -> str:
    uniq = s.dropna().unique()
    n = len(uniq)
    if n <= 10:
        return ", ".join(str(v) for v in sorted(uniq, key=str)) + f" ({n}가지)"
    if pd.api.types.is_numeric_dtype(s):
        return f"숫자 {s.min():g} ~ {s.max():g} ({n}가지)"
    return f"글자 {n}가지"


# ---------- 화면 맨 위 ----------
st.title(f"{ICON} {TITLE}")
st.write(
    "이 데이터는 사람들의 성별·나이·질병 여부·생활 습관 등의 정보와, "
    "뇌졸중에 걸렸는지(`stroke`)를 한 줄에 한 명씩 적어 놓은 표입니다. "
    "앞으로 이 데이터로 뇌졸중을 예측하는 실습을 해 봅니다."
)

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다. 인터넷 연결을 확인한 뒤 새로고침해 주세요.\n\n({e})")
    st.stop()

# ---------- 저장된 기록 (세션 처음 한 번만 반영) ----------
if "init_meanings" not in st.session_state:
    saved = load_records()
    st.session_state["init_meanings"] = saved["우리말_뜻"]
    st.session_state["source_text"] = saved["출처"]

# ---------- 큰 숫자 카드 네 개 ----------
total = len(df)
n_cols = df.shape[1]
n_stroke = int((df["stroke"] == 1).sum())
ratio = n_stroke / total * 100 if total else 0.0

c1, c2, c3, c4 = st.columns(4)
with c1:
    with st.container(border=True):
        st.metric("전체 사람 수", f"{total:,}명")
with c2:
    with st.container(border=True):
        st.metric("열 개수", f"{n_cols}개")
with c3:
    with st.container(border=True):
        st.metric("stroke가 1인 사람 수", f"{n_stroke:,}명")
with c4:
    with st.container(border=True):
        st.metric("stroke가 1인 비율", f"{ratio:.2f}%")

# ---------- 열 설명 표 ----------
st.subheader("열 설명표")
st.caption("‘우리말 뜻’ 칸을 두 번 눌러 교재를 보고 직접 적어 보세요. 적은 내용은 자동으로 저장됩니다.")

init_meanings = st.session_state["init_meanings"]
info = pd.DataFrame(
    {
        "열 이름": df.columns,
        "우리말 뜻": [init_meanings.get(c, "") for c in df.columns],
        "값의 종류": [describe_values(df[c]) for c in df.columns],
        "빈 값 개수": [int(df[c].isna().sum()) for c in df.columns],
    }
)

edited = st.data_editor(
    info,
    hide_index=True,
    use_container_width=True,
    num_rows="fixed",
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름"),
        "우리말 뜻": st.column_config.TextColumn("우리말 뜻", help="교재를 보고 직접 적어 보세요."),
        "값의 종류": st.column_config.TextColumn("값의 종류", width="large"),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", format="%d"),
    },
    key="info_editor",
)

# ---------- 처음 다섯 줄 ----------
st.subheader("데이터 처음 다섯 줄")
st.dataframe(df.head(5), use_container_width=True)

# ---------- 데이터 출처 ----------
st.subheader("데이터 출처")
st.text_area(
    "출처를 적어 주세요",
    key="source_text",
    height=120,
    placeholder="교재에 나온 데이터 출처를 여기에 적어 보세요.",
)

# ---------- 자동 저장 ----------
current = {
    "우리말_뜻": {
        str(name): ("" if pd.isna(text) else str(text))
        for name, text in zip(edited["열 이름"], edited["우리말 뜻"])
    },
    "출처": st.session_state["source_text"],
}
stored = load_records()
file_missing = not SAVE_FILE.exists()
if file_missing and not any(current["우리말_뜻"].values()) and not current["출처"]:
    pass  # 아직 아무것도 적지 않았으면 파일을 만들지 않음
elif current != stored:
    if save_records(current):
        st.caption(f"💾 자동 저장됨 → {SAVE_DIR.name}/{SAVE_FILE.name}")
    else:
        st.warning("기록을 저장하지 못했습니다. 폴더 쓰기 권한을 확인해 주세요.")
else:
    st.caption(f"💾 저장된 상태입니다 ({SAVE_DIR.name}/{SAVE_FILE.name})")
