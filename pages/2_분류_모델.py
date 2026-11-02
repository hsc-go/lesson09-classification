import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
FEATURES = list(KOR)
BINARY = {"hypertension", "heart_disease"}  # 0 또는 1 값만 가지는 속성
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

COLOR_NEG, COLOR_POS = "#2563eb", "#dc2626"  # 점 색 (아님 / 뇌졸중)
LIGHT_NEG, LIGHT_POS = "#93c5fd", "#fca5a5"  # 의사결정트리 칸 색
LINE_COLOR = "#7c3aed"

st.set_page_config(page_title=f"분류 모델 · {TITLE}", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
    .acc-card { border: 1px solid rgba(128,128,128,.4); border-radius: .6rem;
                padding: 1rem 1.1rem; height: 100%; }
    .acc-title { font-size: .95rem; opacity: .85; min-height: 2.6rem; }
    .acc-big { font-size: 2.9rem; font-weight: 700; line-height: 1.15; }
    .acc-cap { font-size: .85rem; opacity: .7; margin-bottom: .6rem; }
    .acc-row { display: flex; justify-content: space-between; font-size: .85rem;
               opacity: .8; border-top: 1px solid rgba(128,128,128,.3); padding-top: .5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def acc_card(title: str, train_acc: float, test_acc: float) -> str:
    return (
        '<div class="acc-card">'
        f'<div class="acc-title">{title}</div>'
        f'<div class="acc-big">{pct(test_acc)}</div>'
        '<div class="acc-cap">테스트 데이터 정확도</div>'
        '<div class="acc-row">'
        f"<span>훈련 {pct(train_acc)}</span><span>테스트 {pct(test_acc)}</span>"
        "</div></div>"
    )


def show_graphviz(dot: str) -> None:
    try:
        st.graphviz_chart(dot, width="stretch")
    except TypeError:  # 오래된 스트림릿
        st.graphviz_chart(dot, use_container_width=True)


def num(v: float) -> str:
    return f"{round(float(v), 2):g}"


def axis_range(feature: str, values: pd.Series) -> tuple[float, float]:
    if feature in BINARY:
        return -0.3, 1.3
    lo, hi = float(values.min()), float(values.max())
    pad = (hi - lo) * 0.05 or 1.0
    return lo - pad, hi + pad


def line_in_box(A: float, B: float, c: float, xlo: float, xhi: float, ylo: float, yhi: float):
    """A*x + B*y + c = 0 직선이 사각형 안을 지나는 두 끝점. 지나지 않으면 None."""
    ex, ey = (xhi - xlo) * 1e-9, (yhi - ylo) * 1e-9
    pts = []
    if abs(B) > 1e-12:
        for xe in (xlo, xhi):
            yv = -(A * xe + c) / B
            if ylo - ey <= yv <= yhi + ey:
                pts.append((xe, min(max(yv, ylo), yhi)))
    if abs(A) > 1e-12:
        for ye in (ylo, yhi):
            xv = -(B * ye + c) / A
            if xlo - ex <= xv <= xhi + ex:
                pts.append((min(max(xv, xlo), xhi), ye))
    if len(pts) < 2:
        return None
    best, best_d = None, -1.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2
            if d > best_d:
                best, best_d = (pts[i], pts[j]), d
    if best_d <= 0:
        return None
    return best


def build_tree_dot(tree: DecisionTreeClassifier, selected: list, X_train_np: np.ndarray, y_train_np: np.ndarray):
    """의사결정트리를 DOT 문자열로 바꾼다. (DOT, 잎 답 목록)을 돌려준다."""
    t = tree.tree_
    path = tree.decision_path(X_train_np)
    n_node = np.asarray(path.sum(axis=0)).ravel()
    n_pos = np.asarray(path.T.dot(y_train_np)).ravel()

    lines = [
        "digraph Tree {",
        'graph [bgcolor="white", pad="0.2", nodesep="0.35", ranksep="0.55"];',
        'node [shape=box, style="rounded,filled", fontname="sans-serif", fontsize=12, '
        'color="#6b7280", fillcolor="#f3f4f6", fontcolor="black", margin="0.15,0.08"];',
        'edge [fontname="sans-serif", fontsize=12, color="#374151", fontcolor="black"];',
    ]
    leaf_answers = []
    for node in range(t.node_count):
        n, p = int(n_node[node]), int(n_pos[node])
        rate = p / n * 100 if n else 0.0
        counts = f"사람 {n:,}명\\n뇌졸중 {p:,}명 ({rate:.1f}%)"
        if t.children_left[node] == -1:  # 더 묻지 않고 답을 내는 마디
            ans = int(tree.classes_[int(np.argmax(t.value[node][0]))])
            leaf_answers.append(ans)
            name = "뇌졸중" if ans == 1 else "뇌졸중 아님"
            fill, border = (LIGHT_POS, COLOR_POS) if ans == 1 else (LIGHT_NEG, COLOR_NEG)
            lines.append(
                f'n{node} [label="답: {name}\\n{counts}", fillcolor="{fill}", color="{border}", penwidth=2];'
            )
        else:
            feat = selected[int(t.feature[node])]
            thr = float(t.threshold[node])
            left, right = int(t.children_left[node]), int(t.children_right[node])
            if feat in BINARY:
                question = f"{KOR[feat]}이 있나요?"
                yes_child, no_child = right, left
            else:
                question = f"{KOR[feat]} ≤ {num(thr)} ?"
                yes_child, no_child = left, right
            lines.append(f'n{node} [label="{question}\\n{counts}"];')
            lines.append(f'n{node} -> n{yes_child} [label="예"];')
            lines.append(f'n{node} -> n{no_child} [label="아니요"];')
    lines.append("}")
    return "\n".join(lines), leaf_answers


# ======================================================================
st.title("🤖 분류 모델")
st.write(
    "`stroke`가 1이면 **뇌졸중**, 0이면 **아님**이에요. 뇌졸중을 양성(찾아내려는 쪽)으로 두고, "
    f"{LR_NAME}과 {DT_NAME}을 만들어 비교해 봅니다."
)

try:
    raw = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다. 인터넷 연결을 확인한 뒤 새로고침해 주세요.\n\n({e})")
    st.stop()

# ---------- 입력 속성 고르기 ----------
chosen = st.multiselect(
    "입력으로 사용할 속성을 고르세요 (두 개 이상)",
    options=FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda f: KOR[f],
)
selected = [f for f in FEATURES if f in chosen]  # 항상 같은 순서로
if len(selected) < 2:
    st.info("입력 속성을 두 개 이상 골라 주세요. 두 개 이상 고르면 모델이 만들어져요.")
    st.stop()

# ---------- 테스트용 고정: 번호 순으로 정렬 → 열 명씩 묶기 → 앞 세 명 ----------
df = raw.sort_values("id", kind="mergesort").reset_index(drop=True)
is_test = (np.arange(len(df)) % 10) < 3
train_df, test_df = df[~is_test].copy(), df[is_test].copy()

X_train = train_df[selected].copy()
X_test = test_df[selected].copy()
y_train = train_df["stroke"].to_numpy().astype(int)
y_test = test_df["stroke"].to_numpy().astype(int)

st.write(
    f"번호(id) 순으로 정렬해 열 명씩 묶고, 각 묶음의 앞 세 명을 테스트용으로 고정했어요. "
    f"테스트용 **{len(test_df):,}명**, 학습용 **{len(train_df):,}명**입니다."
)

# ---------- 체질량지수 빈 값: 고른 경우에만, 훈련용 중앙값으로 ----------
if "bmi" in selected:
    bmi_median = float(X_train["bmi"].median())
    miss_tr, miss_te = int(X_train["bmi"].isna().sum()), int(X_test["bmi"].isna().sum())
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)
    st.caption(
        f"체질량지수의 빈 값(훈련용 {miss_tr:,}명, 테스트용 {miss_te:,}명)은 "
        f"훈련용의 중앙값 {num(bmi_median)}로 채웠어요."
    )

X_train_np = X_train.to_numpy(dtype=float)
X_test_np = X_test.to_numpy(dtype=float)

# ---------- 크기 맞추기: 훈련용으로만 ----------
scaler = StandardScaler().fit(X_train_np)
X_train_s, X_test_s = scaler.transform(X_train_np), scaler.transform(X_test_np)

# ---------- 모델 만들기 ----------
lr = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(X_train_np, y_train)
base = DummyClassifier(strategy="most_frequent").fit(X_train_np, y_train)

acc = {
    "lr": (accuracy_score(y_train, lr.predict(X_train_s)), accuracy_score(y_test, lr.predict(X_test_s))),
    "tree": (accuracy_score(y_train, tree.predict(X_train_np)), accuracy_score(y_test, tree.predict(X_test_np))),
    "base": (accuracy_score(y_train, base.predict(X_train_np)), accuracy_score(y_test, base.predict(X_test_np))),
}

# ---------- 정확도 카드 ----------
st.header("1. 정확도 비교")
c1, c2, c3 = st.columns(3)
c1.markdown(acc_card(LR_NAME, *acc["lr"]), unsafe_allow_html=True)
c2.markdown(acc_card(DT_NAME, *acc["tree"]), unsafe_allow_html=True)
c3.markdown(acc_card(BASE_NAME, *acc["base"]), unsafe_allow_html=True)

base_answer = int(base.predict(X_test_np[:1])[0])
st.caption(
    f"기준 모델은 언제나 '{'뇌졸중' if base_answer == 1 else '뇌졸중 아님'}'이라고 답해요. "
    f"훈련용에서 뇌졸중인 사람은 {pct(y_train.mean())}예요. "
    f"{DT_NAME}은 질문을 최대 3번까지만 던지고(max_depth=3), "
    "마디에 사람이 5명 미만이 되는 나눔은 하지 않으며(min_samples_leaf=5), 난수는 42로 고정했어요."
)

# ---------- 산점도 ----------
st.header("2. 두 속성으로 본 테스트 데이터")
ax1, ax2 = st.columns(2)
x_feat = ax1.selectbox("가로축", selected, format_func=lambda f: KOR[f])
y_options = [f for f in selected if f != x_feat]
y_feat = ax2.selectbox("세로축", y_options, format_func=lambda f: KOR[f])

idx = {f: i for i, f in enumerate(selected)}
medians = {f: float(X_test[f].median()) for f in selected}
others = [f for f in selected if f not in (x_feat, y_feat)]

xlo, xhi = axis_range(x_feat, X_test[x_feat])
ylo, yhi = axis_range(y_feat, X_test[y_feat])

# 의사결정트리가 나눈 칸 (그림에 쓰지 않은 속성은 테스트 데이터의 중앙값에 세움)
gx = np.linspace(xlo, xhi, 220)
gy = np.linspace(ylo, yhi, 220)
GX, GY = np.meshgrid(gx, gy)
grid = np.empty((GX.size, len(selected)))
for f in selected:
    if f == x_feat:
        grid[:, idx[f]] = GX.ravel()
    elif f == y_feat:
        grid[:, idx[f]] = GY.ravel()
    else:
        grid[:, idx[f]] = medians[f]
tree_grid = tree.predict(grid).reshape(GX.shape)

# 로지스틱 회귀가 확률 0.5로 가르는 선: A*x + B*y + c = 0
w, b = lr.coef_[0], float(lr.intercept_[0])
mu, sc = scaler.mean_, scaler.scale_
ix, iy = idx[x_feat], idx[y_feat]
A, B = w[ix] / sc[ix], w[iy] / sc[iy]
c = b - w[ix] * mu[ix] / sc[ix] - w[iy] * mu[iy] / sc[iy]
for f in others:
    j = idx[f]
    c += w[j] * (medians[f] - mu[j]) / sc[j]
segment = line_in_box(A, B, c, xlo, xhi, ylo, yhi)

fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        x=gx, y=gy, z=tree_grid, zmin=0, zmax=1, zsmooth=False,
        colorscale=[[0, LIGHT_NEG], [1, LIGHT_POS]], showscale=False,
        opacity=0.45, hoverinfo="skip",
    )
)

rng = np.random.default_rng(0)
xv, yv = X_test[x_feat].to_numpy(dtype=float), X_test[y_feat].to_numpy(dtype=float)
xp = xv + rng.uniform(-0.12, 0.12, len(xv)) if x_feat in BINARY else xv  # 겹침 방지용 살짝 흩뿌리기
yp = yv + rng.uniform(-0.12, 0.12, len(yv)) if y_feat in BINARY else yv
for label, code, color, size, opacity in [
    ("뇌졸중 아님(실제)", 0, COLOR_NEG, 6, 0.6),
    ("뇌졸중(실제)", 1, COLOR_POS, 9, 0.95),
]:
    m = y_test == code
    fig.add_trace(
        go.Scatter(
            x=xp[m], y=yp[m], mode="markers", name=label,
            marker=dict(color=color, size=size, opacity=opacity, line=dict(width=0.6, color="white")),
            customdata=np.column_stack([xv[m], yv[m]]),
            hovertemplate=(
                f"{KOR[x_feat]}: %{{customdata[0]}}<br>{KOR[y_feat]}: %{{customdata[1]}}"
                f"<extra>{label}</extra>"
            ),
        )
    )
if segment is not None:
    (xa, ya), (xb, yb) = segment
    fig.add_trace(
        go.Scatter(
            x=[xa, xb], y=[ya, yb], mode="lines", name="로지스틱 회귀의 0.5 경계선",
            line=dict(color=LINE_COLOR, width=3), hoverinfo="skip",
        )
    )
for label, color in [("의사결정트리 칸: 뇌졸중 아님", LIGHT_NEG), ("의사결정트리 칸: 뇌졸중", LIGHT_POS)]:
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="markers", name=label,
            marker=dict(symbol="square", size=13, color=color),
        )
    )

fig.update_layout(
    xaxis=dict(title=KOR[x_feat], range=[xlo, xhi]),
    yaxis=dict(title=KOR[y_feat], range=[ylo, yhi]),
    height=620,
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0),
)
if x_feat in BINARY:
    fig.update_xaxes(tickvals=[0, 1], ticktext=["0 (없음)", "1 (있음)"])
if y_feat in BINARY:
    fig.update_yaxes(tickvals=[0, 1], ticktext=["0 (없음)", "1 (있음)"])
st.plotly_chart(fig)

if others:
    fixed_text = ", ".join(f"{KOR[f]} = {num(medians[f])}" for f in others)
    st.write(f"그림에 쓰지 않은 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: **{fixed_text}**")
else:
    st.write("고른 속성이 두 개뿐이라, 그림에 쓰지 않은 속성은 없어요.")

if segment is not None:
    st.write(f"보라색 선은 {LR_NAME}이 뇌졸중일 확률을 0.5로 가르는 선이에요.")
else:
    xc, yc = (xlo + xhi) / 2, (ylo + yhi) / 2
    side = "높은" if (A * xc + B * yc + c) > 0 else "낮은"
    answer = "뇌졸중" if side == "높은" else "뇌졸중 아님"
    top_prob = float(lr.predict_proba(X_test_s)[:, 1].max())
    st.write(
        f"{LR_NAME}이 확률 0.5로 가르는 선은 **이 그림 밖**에 있어요. "
        f"그림 안은 모두 뇌졸중일 확률이 0.5보다 {side} 쪽이라 '{answer}'이라고 답해요. "
        f"(참고: 고른 속성을 모두 실제 값으로 넣었을 때 이 모델이 테스트 데이터에서 낸 "
        f"가장 높은 뇌졸중 확률은 {pct(top_prob)}예요.)"
    )
if x_feat in BINARY or y_feat in BINARY:
    st.caption("고혈압·심장병은 값이 0 또는 1뿐이라, 점이 겹치지 않게 보기용으로만 살짝 흩뿌려 찍었어요.")

# ---------- 나무 그림 ----------
st.header(f"3. {DT_NAME}이 던진 질문")
dot, leaf_answers = build_tree_dot(tree, selected, X_train_np, y_train)
show_graphviz(dot)
st.caption(
    "마디 안의 숫자는 그 자리에 온 훈련용 사람 수와, 그중 실제 뇌졸중인 사람 수·비율이에요. "
    "파란 마디는 '뇌졸중 아님', 빨간 마디는 '뇌졸중'이라고 답을 내는 마디예요."
)

n_leaves = len(leaf_answers)
n_neg = sum(1 for a in leaf_answers if a == 0)
asked_idx = sorted({int(f) for f in tree.tree_.feature if f >= 0})
asked = ", ".join(KOR[selected[i]] for i in asked_idx) if asked_idx else "없음 (질문을 한 번도 하지 않았어요)"

st.write(f"- 답을 내는 마디는 모두 **{n_leaves}칸**이고, 그중 **{n_neg}칸**이 '뇌졸중 아님'이라고 답해요.")
st.write(f"- 고른 속성 {len(selected)}개 가운데 이 나무가 실제로 물은 것: **{asked}**")
