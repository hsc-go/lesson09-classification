import itertools

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/hsc-go/data/main/stroke.csv"

LR_NAME = "로지스틱 회귀(확률로 답하는 모델)"
DT_NAME = "의사결정트리(질문으로 답하는 모델)"

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

CANT = "계산할 수 없음"
COLOR_NEG, COLOR_POS = "#2563eb", "#dc2626"  # 점 색 (아님 / 뇌졸중)
COLOR_BEFORE, COLOR_AFTER = "#7c3aed", "#f59e0b"  # 맞추기 전 / 맞춘 뒤 경계

st.set_page_config(page_title=f"성능 높이기 · {TITLE}", page_icon="⚖️", layout="wide")


@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def num(v: float) -> str:
    return f"{round(float(v), 2):g}"


def confusion_cells(y_true: np.ndarray, y_pred: np.ndarray):
    """뇌졸중(1)을 양성으로 둔 TP, FP, FN, TN 사람 수."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fp, fn, tn


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


def before_after_table(before: tuple, after: tuple) -> pd.DataFrame:
    mb, ma = four_metrics(*before), four_metrics(*after)
    tp0, fp0, fn0, tn0 = before
    tp1, fp1, fn1, tn1 = after
    rows = {
        "찾아낸 환자 (TP)": (f"{tp0:,}명", f"{tp1:,}명"),
        "놓친 환자 (FN)": (f"{fn0:,}명", f"{fn1:,}명"),
        "헛짚은 사람 (FP)": (f"{fp0:,}명", f"{fp1:,}명"),
        "맞게 아니라고 한 사람 (TN)": (f"{tn0:,}명", f"{tn1:,}명"),
        "정확도": (mb["정확도"], ma["정확도"]),
        "재현율": (mb["재현율"], ma["재현율"]),
        "정밀도": (mb["정밀도"], ma["정밀도"]),
        "F1": (mb["F1"], ma["F1"]),
    }
    return pd.DataFrame(
        {"맞추기 전": [v[0] for v in rows.values()], "맞춘 뒤": [v[1] for v in rows.values()]},
        index=list(rows),
    )


def change_line(before: tuple, after: tuple) -> str:
    tp0, fp0, fn0, _ = before
    tp1, fp1, fn1, _ = after
    return (
        f"찾아낸 환자는 {tp0:,}명에서 {tp1:,}명으로, 놓친 환자는 {fn0:,}명에서 {fn1:,}명으로, "
        f"헛짚은 사람은 {fp0:,}명에서 {fp1:,}명으로 바뀌었어요."
    )


def axis_range(feature: str, values: pd.Series) -> tuple[float, float]:
    if feature in BINARY:
        return -0.3, 1.3
    lo, hi = float(values.min()), float(values.max())
    pad = (hi - lo) * 0.05 or 1.0
    return lo - pad, hi + pad


def boundary_coefs(lr: LogisticRegression, scaler: StandardScaler, selected: list, axes: list, medians: dict):
    """0.5 경계 위에서 coefs · (축 속성 값) + c = 0 이 되는 coefs, c.
    축이 아닌 속성은 medians의 값에 세워 둔다."""
    w, b = lr.coef_[0], float(lr.intercept_[0])
    mu, sc = scaler.mean_, scaler.scale_
    coefs = [w[selected.index(a)] / sc[selected.index(a)] for a in axes]
    c = b
    for i, f in enumerate(selected):
        if f in axes:
            c -= w[i] * mu[i] / sc[i]
        else:
            c += w[i] * (medians[f] - mu[i]) / sc[i]
    return coefs, c


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
    best, best_d = None, 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2
            if d > best_d:
                best, best_d = (pts[i], pts[j]), d
    return best


def plane_in_box(coefs, c: float, lo, hi):
    """a*x + b*y + d*z + c = 0 평면이 상자와 만나는 다각형의 꼭짓점(순서대로). 만나지 않으면 None."""
    n = np.array(coefs, dtype=float)
    if np.abs(n).max() < 1e-12:
        return None
    lo, hi = np.array(lo, dtype=float), np.array(hi, dtype=float)
    size = hi - lo
    corners = [np.array(p, dtype=float) for p in itertools.product(*[(lo[k], hi[k]) for k in range(3)])]
    pts = []
    for p0, p1 in itertools.combinations(corners, 2):
        if np.count_nonzero(p0 != p1) != 1:  # 상자의 모서리만
            continue
        f0, f1 = float(n @ p0 + c), float(n @ p1 + c)
        if f0 == 0 and f1 == 0:
            pts += [p0, p1]
        elif f0 * f1 <= 0 and f0 != f1:
            pts.append(p0 + f0 / (f0 - f1) * (p1 - p0))
    uniq = {}
    for p in pts:
        uniq[tuple(np.round((p - lo) / size, 9))] = p
    pts = list(uniq.values())
    if len(pts) < 3:
        return None
    center = np.mean(pts, axis=0)
    helper = np.zeros(3)
    helper[int(np.argmin(np.abs(n)))] = 1.0
    u = np.cross(n, helper)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    pts.sort(key=lambda p: np.arctan2((p - center) @ v, (p - center) @ u))
    return np.array(pts)


def side_text(value_at_center: float) -> tuple[str, str]:
    """그림(상자) 가운데에서의 판단값으로, 확률이 0.5보다 높은/낮은 쪽과 그 답."""
    if value_at_center > 0:
        return "높은", "뇌졸중"
    return "낮은", "뇌졸중 아님"


# ======================================================================
st.title("⚖️ 성능 높이기")

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

try:
    raw = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다. 인터넷 연결을 확인한 뒤 새로고침해 주세요.\n\n({e})")
    st.stop()

# ---------- '채점표'·'분류 모델' 페이지와 같은 방법으로 나누기 ----------
df = raw.sort_values("id", kind="mergesort").reset_index(drop=True)
is_test = (np.arange(len(df)) % 10) < 3
train_df, test_df = df[~is_test].copy(), df[is_test].copy()

X_train = train_df[selected].copy()
X_test = test_df[selected].copy()
y_train = train_df["stroke"].to_numpy().astype(int)
y_test = test_df["stroke"].to_numpy().astype(int)

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

scaler = StandardScaler().fit(X_train_np)  # 크기 맞추기는 훈련용으로만
X_train_s, X_test_s = scaler.transform(X_train_np), scaler.transform(X_test_np)

n_train_pos = int(y_train.sum())
n_train_neg = len(y_train) - n_train_pos
n_test, n_test_pos = len(test_df), int(y_test.sum())

st.write(
    f"뇌졸중을 겪은 사람이 아주 드물어요(훈련용 뇌졸중 {n_train_pos:,}명, 아님 {n_train_neg:,}명). "
    f"그래서 뇌졸중 한 명을 아님 한 명의 약 {n_train_neg / max(n_train_pos, 1):.1f}배로 셈해 "
    "양쪽의 가중치를 맞춘 뒤(class_weight=\"balanced\")와 맞추기 전(기본)의 결과를 나란히 비교해요."
)
st.write(f"테스트용 **{n_test:,}명** 가운데 뇌졸중인 사람은 **{n_test_pos:,}명**이에요.")


# ---------- 모델 학습: 맞추기 전(None) / 맞춘 뒤("balanced") ----------
def fit_pair(class_weight):
    lr = LogisticRegression(max_iter=1000, class_weight=class_weight).fit(X_train_s, y_train)
    tree = DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=5, random_state=42, class_weight=class_weight
    ).fit(X_train_np, y_train)
    return lr, tree


lr_before, tree_before = fit_pair(None)
lr_after, tree_after = fit_pair("balanced")

cells = {
    LR_NAME: (
        confusion_cells(y_test, lr_before.predict(X_test_s)),
        confusion_cells(y_test, lr_after.predict(X_test_s)),
    ),
    DT_NAME: (
        confusion_cells(y_test, tree_before.predict(X_test_np)),
        confusion_cells(y_test, tree_after.predict(X_test_np)),
    ),
}

# ---------- 1. 전·후 표 ----------
st.header("1. 가중치를 맞추기 전과 맞춘 뒤")
st.caption(
    "찾아낸 환자(TP)는 뇌졸중이라 예측한 실제 환자, 놓친 환자(FN)는 아니라고 예측한 실제 환자, "
    "헛짚은 사람(FP)은 뇌졸중이라 예측했지만 실제로는 아닌 사람이에요. 모두 테스트 데이터 기준이에요."
)
col_lr, col_dt = st.columns(2)
for col, name in [(col_lr, LR_NAME), (col_dt, DT_NAME)]:
    before, after = cells[name]
    with col:
        st.subheader(name)
        st.table(before_after_table(before, after))
        st.write(change_line(before, after))

# ---------- 2. 두 속성 산점도 ----------
st.header("2. 두 속성으로 본 경계선")
st.write(
    f"점은 테스트 데이터이고, 색은 실제 뇌졸중 여부예요. 선은 {LR_NAME}이 뇌졸중일 확률을 "
    "0.5로 가르는 선이고, 맞추기 전과 맞춘 뒤를 다른 색으로 겹쳐 그렸어요."
)
ax1, ax2 = st.columns(2)
x_feat = ax1.selectbox("가로축", selected, format_func=lambda f: KOR[f])
y_feat = ax2.selectbox("세로축", [f for f in selected if f != x_feat], format_func=lambda f: KOR[f])

medians = {f: float(X_test[f].median()) for f in selected}
others2 = [f for f in selected if f not in (x_feat, y_feat)]
xlo, xhi = axis_range(x_feat, X_test[x_feat])
ylo, yhi = axis_range(y_feat, X_test[y_feat])

rng = np.random.default_rng(0)
xv, yv = X_test[x_feat].to_numpy(dtype=float), X_test[y_feat].to_numpy(dtype=float)
xp = xv + rng.uniform(-0.12, 0.12, len(xv)) if x_feat in BINARY else xv  # 겹침 방지용
yp = yv + rng.uniform(-0.12, 0.12, len(yv)) if y_feat in BINARY else yv

fig2 = go.Figure()
for label, code, color, size, opacity in [
    ("뇌졸중 아님(실제)", 0, COLOR_NEG, 6, 0.6),
    ("뇌졸중(실제)", 1, COLOR_POS, 9, 0.95),
]:
    m = y_test == code
    fig2.add_trace(
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

line_notes = []
for tag, model, color in [("맞추기 전", lr_before, COLOR_BEFORE), ("맞춘 뒤", lr_after, COLOR_AFTER)]:
    (A, B), c = boundary_coefs(model, scaler, selected, [x_feat, y_feat], medians)
    seg = line_in_box(A, B, c, xlo, xhi, ylo, yhi)
    if seg is not None:
        (xa, ya), (xb, yb) = seg
        fig2.add_trace(
            go.Scatter(
                x=[xa, xb], y=[ya, yb], mode="lines", name=f"{tag} 0.5 경계선",
                line=dict(color=color, width=3), hoverinfo="skip",
            )
        )
    else:
        side, answer = side_text(A * (xlo + xhi) / 2 + B * (ylo + yhi) / 2 + c)
        line_notes.append(
            f"**{tag}** 0.5 경계선은 **이 그림 밖**에 있어요. 그림 안은 모두 뇌졸중일 확률이 "
            f"0.5보다 {side} 쪽이라 '{answer}'이라고 답해요."
        )

fig2.update_layout(
    xaxis=dict(title=KOR[x_feat], range=[xlo, xhi]),
    yaxis=dict(title=KOR[y_feat], range=[ylo, yhi]),
    height=600,
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0),
)
if x_feat in BINARY:
    fig2.update_xaxes(tickvals=[0, 1], ticktext=["0 (없음)", "1 (있음)"])
if y_feat in BINARY:
    fig2.update_yaxes(tickvals=[0, 1], ticktext=["0 (없음)", "1 (있음)"])
st.plotly_chart(fig2)

if others2:
    st.write(
        "그림에 쓰지 않은 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: **"
        + ", ".join(f"{KOR[f]} = {num(medians[f])}" for f in others2)
        + "**"
    )
else:
    st.write("고른 속성이 두 개뿐이라, 그림에 쓰지 않은 속성은 없어요.")
for note in line_notes:
    st.write(note)
if x_feat in BINARY or y_feat in BINARY:
    st.caption("고혈압·심장병은 값이 0 또는 1뿐이라, 점이 겹치지 않게 보기용으로만 살짝 흩뿌려 찍었어요.")

# ---------- 3. 세 속성 3차원 산점도 ----------
st.header("3. 세 속성으로 본 경계 평면")
if len(selected) < 3:
    st.info("속성을 세 개 이상 고르면 3차원 그림도 볼 수 있어요.")
else:
    st.write("마우스로 끌어서 돌려 보고, 휠로 크기를 바꿔 보세요. 평면은 확률 0.5로 가르는 면이에요.")
    a1, a2, a3 = st.columns(3)
    x3 = a1.selectbox("3차원 그림의 X축", selected, format_func=lambda f: KOR[f])
    y3 = a2.selectbox("3차원 그림의 Y축", [f for f in selected if f != x3], format_func=lambda f: KOR[f])
    z3 = a3.selectbox(
        "3차원 그림의 Z축", [f for f in selected if f not in (x3, y3)], format_func=lambda f: KOR[f]
    )
    axes3 = [x3, y3, z3]
    others3 = [f for f in selected if f not in axes3]
    lo3 = [axis_range(f, X_test[f])[0] for f in axes3]
    hi3 = [axis_range(f, X_test[f])[1] for f in axes3]

    rng3 = np.random.default_rng(1)
    vals = [X_test[f].to_numpy(dtype=float) for f in axes3]
    shown = [v + rng3.uniform(-0.12, 0.12, len(v)) if f in BINARY else v for f, v in zip(axes3, vals)]

    fig3 = go.Figure()
    for label, code, color, size, opacity in [
        ("뇌졸중 아님(실제)", 0, COLOR_NEG, 2.5, 0.45),
        ("뇌졸중(실제)", 1, COLOR_POS, 5, 0.95),
    ]:
        m = y_test == code
        fig3.add_trace(
            go.Scatter3d(
                x=shown[0][m], y=shown[1][m], z=shown[2][m], mode="markers", name=label,
                marker=dict(color=color, size=size, opacity=opacity),
                customdata=np.column_stack([vals[0][m], vals[1][m], vals[2][m]]),
                hovertemplate=(
                    f"{KOR[x3]}: %{{customdata[0]}}<br>{KOR[y3]}: %{{customdata[1]}}<br>"
                    f"{KOR[z3]}: %{{customdata[2]}}<extra>{label}</extra>"
                ),
            )
        )

    plane_notes = []
    for tag, model, color in [("맞추기 전", lr_before, COLOR_BEFORE), ("맞춘 뒤", lr_after, COLOR_AFTER)]:
        coefs, c = boundary_coefs(model, scaler, selected, axes3, medians)
        poly = plane_in_box(coefs, c, lo3, hi3)
        if poly is not None:
            m_pts = len(poly)
            fig3.add_trace(
                go.Mesh3d(
                    x=poly[:, 0], y=poly[:, 1], z=poly[:, 2],
                    i=[0] * (m_pts - 2), j=list(range(1, m_pts - 1)), k=list(range(2, m_pts)),
                    color=color, opacity=0.45, name=f"{tag} 0.5 평면",
                    showlegend=True, hoverinfo="skip",
                )
            )
        else:
            center = [(lo3[k] + hi3[k]) / 2 for k in range(3)]
            side, answer = side_text(sum(cf * ct for cf, ct in zip(coefs, center)) + c)
            plane_notes.append(
                f"**{tag}** 0.5 평면은 **이 그림 밖**에 있어요. 그림 안은 모두 뇌졸중일 확률이 "
                f"0.5보다 {side} 쪽이라 '{answer}'이라고 답해요."
            )

    scene = {"aspectmode": "cube"}
    for key, f, lo, hi in zip(["xaxis", "yaxis", "zaxis"], axes3, lo3, hi3):
        axis = {"title": KOR[f], "range": [lo, hi]}
        if f in BINARY:
            axis.update(tickvals=[0, 1], ticktext=["0 (없음)", "1 (있음)"])
        scene[key] = axis
    fig3.update_layout(
        scene=scene,
        height=680,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="top", y=0.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig3)

    if others3:
        st.write(
            "그림에 쓰지 않은 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: **"
            + ", ".join(f"{KOR[f]} = {num(medians[f])}" for f in others3)
            + "**"
        )
    else:
        st.write("고른 속성이 세 개뿐이라, 그림에 쓰지 않은 속성은 없어요.")
    for note in plane_notes:
        st.write(note)
    if any(f in BINARY for f in axes3):
        st.caption("고혈압·심장병은 값이 0 또는 1뿐이라, 점이 겹치지 않게 보기용으로만 살짝 흩뿌려 찍었어요.")
