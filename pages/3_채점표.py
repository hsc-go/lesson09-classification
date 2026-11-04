import numpy as np
import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/hsc-go/data/main/stroke.csv"

LR_NAME = "로지스틱 회귀(확률로 답하는 모델)"
DT_NAME = "의사결정트리(질문으로 답하는 모델)"
BASE_NAME = "기준 모델(입력을 하나도 보지 않고 많은 쪽으로만 답하는 모델)"

KOR = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
# '분류 모델' 페이지에서 처음 골라져 있던 속성과 같다 (체질량지수 제외)
SELECTED = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

CANT = "계산할 수 없음"
MAX_ROWS = 20

st.set_page_config(page_title=f"채점표 · {TITLE}", page_icon="📝", layout="wide")


@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def confusion_cells(y_true: np.ndarray, y_pred: np.ndarray):
    """양성(뇌졸중=1) 기준 TP, FP, FN, TN 사람 수."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fp, fn, tn


def confusion_table(tp: int, fp: int, fn: int, tn: int) -> pd.DataFrame:
    """행은 실제, 열은 예측."""
    table = pd.DataFrame(
        {
            "예측: 뇌졸중": [f"TP: {tp:,}명", f"FP: {fp:,}명", f"{tp + fp:,}명"],
            "예측: 아님": [f"FN: {fn:,}명", f"TN: {tn:,}명", f"{fn + tn:,}명"],
            "합계": [f"{tp + fn:,}명", f"{fp + tn:,}명", f"{tp + fp + fn + tn:,}명"],
        },
        index=["실제: 뇌졸중", "실제: 아님", "합계"],
    )
    return table


def four_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    total = tp + fp + fn + tn
    accuracy = pct((tp + tn) / total) if total else CANT
    recall_val = tp / (tp + fn) if (tp + fn) else None  # 분모: 실제 뇌졸중인 사람 수
    precision_val = tp / (tp + fp) if (tp + fp) else None  # 분모: 뇌졸중이라 예측한 사람 수
    if precision_val is None or recall_val is None or (precision_val + recall_val) == 0:
        f1 = CANT
    else:
        f1 = pct(2 * precision_val * recall_val / (precision_val + recall_val))
    return {
        "정확도": accuracy,
        "재현율": pct(recall_val) if recall_val is not None else CANT,
        "정밀도": pct(precision_val) if precision_val is not None else CANT,
        "F1": f1,
    }


def error_list(rows: pd.DataFrame) -> pd.DataFrame:
    """나이가 많은 순서로 정렬한 목록 (나이가 같으면 번호 순)."""
    rows = rows.sort_values(["age", "id"], ascending=[False, True])
    return pd.DataFrame(
        {
            "번호": rows["id"].astype(int).to_numpy(),
            "나이": rows["age"].to_numpy(),
            "평균 혈당": rows["avg_glucose_level"].to_numpy(),
            "고혈압 여부": rows["hypertension"].map({1: "있음", 0: "없음"}).to_numpy(),
            "심장병 여부": rows["heart_disease"].map({1: "있음", 0: "없음"}).to_numpy(),
        }
    )


def show_list(title: str, note: str, full: pd.DataFrame) -> None:
    n = len(full)
    suffix = f" (앞 {MAX_ROWS}줄만 표시)" if n > MAX_ROWS else ""
    st.subheader(f"{title} — 전체 {n:,}명{suffix}")
    st.caption(note)
    if n == 0:
        st.write("해당하는 사람이 없어요.")
    else:
        st.dataframe(full.head(MAX_ROWS), hide_index=True)


# ======================================================================
st.title("📝 채점표")
st.write(
    "테스트 데이터로 두 모델의 답을 채점해요. 뇌졸중(`stroke`=1)을 양성으로 두고, "
    f"{LR_NAME}과 {DT_NAME}을 같은 방법으로 학습해 비교합니다."
)

try:
    raw = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다. 인터넷 연결을 확인한 뒤 새로고침해 주세요.\n\n({e})")
    st.stop()

# ---------- '분류 모델' 페이지와 같은 방법: 정렬 → 열 명씩 묶기 → 앞 세 명은 테스트용 ----------
df = raw.sort_values("id", kind="mergesort").reset_index(drop=True)
is_test = (np.arange(len(df)) % 10) < 3
train_df, test_df = df[~is_test].copy(), df[is_test].copy()

X_train_np = train_df[SELECTED].to_numpy(dtype=float)
X_test_np = test_df[SELECTED].to_numpy(dtype=float)
y_train = train_df["stroke"].to_numpy().astype(int)
y_test = test_df["stroke"].to_numpy().astype(int)

# 크기 맞추기는 훈련용으로만
scaler = StandardScaler().fit(X_train_np)
X_train_s, X_test_s = scaler.transform(X_train_np), scaler.transform(X_test_np)

lr = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(X_train_np, y_train)
base = DummyClassifier(strategy="most_frequent").fit(X_train_np, y_train)

preds = {
    LR_NAME: lr.predict(X_test_s),
    DT_NAME: tree.predict(X_test_np),
    BASE_NAME: base.predict(X_test_np),
}
cells = {name: confusion_cells(y_test, p) for name, p in preds.items()}

n_test, n_pos = len(test_df), int(y_test.sum())
st.write(f"테스트용 **{n_test:,}명** 가운데 뇌졸중인 사람은 **{n_pos:,}명**이에요.")
st.caption(
    "입력 속성은 '분류 모델' 페이지에서 처음 골라져 있던 "
    + ", ".join(KOR[f] for f in SELECTED)
    + " 넷이에요."
)

# ---------- 1. 혼동행렬 ----------
st.header("1. 혼동행렬")
st.caption("행은 실제, 열은 예측이에요. 사람 수는 모두 테스트 데이터 기준이에요.")
left, right = st.columns(2)
with left:
    st.subheader(LR_NAME)
    st.table(confusion_table(*cells[LR_NAME]))
with right:
    st.subheader(DT_NAME)
    st.table(confusion_table(*cells[DT_NAME]))
st.caption(
    "TP: 뇌졸중이라 예측했고 실제로 뇌졸중 · FP: 뇌졸중이라 예측했지만 실제로는 아님 · "
    "FN: 아니라고 예측했지만 실제로는 뇌졸중 · TN: 아니라고 예측했고 실제로도 아님"
)

# ---------- 2. 네 지표 ----------
st.header("2. 네 가지 지표")
metric_rows = []
for name in [LR_NAME, DT_NAME, BASE_NAME]:
    metric_rows.append({"모델": name, **four_metrics(*cells[name])})
st.dataframe(pd.DataFrame(metric_rows), hide_index=True)
st.caption(
    "정확도 = (TP+TN) ÷ 전체 · 재현율 = TP ÷ (TP+FN), 분모는 실제 뇌졸중인 사람 수 · "
    "정밀도 = TP ÷ (TP+FP), 분모는 뇌졸중이라 예측한 사람 수 · "
    "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율). "
    f"뇌졸중이라 예측한 사람이 0명이면 정밀도와 F1을 '{CANT}'으로 적었어요. "
    f"정밀도와 재현율이 둘 다 0이어서 F1을 나눌 수 없을 때도 같은 말을 적었어요."
)

# ---------- 3. 틀린 사람 목록 ----------
st.header("3. 틀린 사람 목록")
model_name = st.radio("어느 모델의 목록을 볼까요?", [LR_NAME, DT_NAME], horizontal=True)
pred = preds[model_name]
missed_full = error_list(test_df[(y_test == 1) & (pred == 0)])
wrong_full = error_list(test_df[(y_test == 0) & (pred == 1)])

col_a, col_b = st.columns(2)
with col_a:
    show_list("놓친 환자", "실제 뇌졸중인데 아니라고 예측한 사람", missed_full)
with col_b:
    show_list("헛짚은 사람", "아닌데 뇌졸중이라고 예측한 사람", wrong_full)

tp, fp, fn, tn = cells[model_name]
same = len(missed_full) == fn and len(wrong_full) == fp
st.write(
    f"{model_name}: 놓친 환자 {len(missed_full):,}명은 혼동행렬의 FN {fn:,}명과, "
    f"헛짚은 사람 {len(wrong_full):,}명은 FP {fp:,}명과 "
    + ("같아요." if same else "달라요.")
)
