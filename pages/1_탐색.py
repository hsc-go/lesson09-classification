import pandas as pd
import plotly.express as px
import streamlit as st

TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/hsc-go/data/main/stroke.csv"

st.set_page_config(page_title=f"탐색 · {TITLE}", page_icon="🔍", layout="wide")


@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🔍 탐색")
st.write("데이터를 여러 방향에서 살펴보고, 뇌졸중과 관련 있어 보이는 단서를 찾아봅니다.")

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다. 인터넷 연결을 확인한 뒤 새로고침해 주세요.\n\n({e})")
    st.stop()

GROUP_ORDER = ["뇌졸중 없음", "뇌졸중 있음"]
df = df.copy()
df["뇌졸중 여부"] = df["stroke"].map({0: "뇌졸중 없음", 1: "뇌졸중 있음"})

# ---------- 1. 분포 (히스토그램) ----------
st.header("1. 나이와 평균 혈당의 분포")
col1, col2 = st.columns(2)
with col1:
    fig = px.histogram(
        df, x="age", nbins=30, title="나이(age)의 분포",
        labels={"age": "나이", "count": "사람 수"},
    )
    fig.update_layout(yaxis_title="사람 수", bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.histogram(
        df, x="avg_glucose_level", nbins=40, title="평균 혈당(avg_glucose_level)의 분포",
        labels={"avg_glucose_level": "평균 혈당"},
    )
    fig.update_layout(yaxis_title="사람 수", bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)

# ---------- 2. 뇌졸중 여부별 비교 (상자그림) ----------
st.header("2. 뇌졸중 여부에 따른 나이와 평균 혈당")
col1, col2 = st.columns(2)
with col1:
    fig = px.box(
        df, x="뇌졸중 여부", y="age", color="뇌졸중 여부",
        category_orders={"뇌졸중 여부": GROUP_ORDER},
        title="나이 비교", labels={"age": "나이"},
    )
    fig.update_layout(showlegend=False, xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.box(
        df, x="뇌졸중 여부", y="avg_glucose_level", color="뇌졸중 여부",
        category_orders={"뇌졸중 여부": GROUP_ORDER},
        title="평균 혈당 비교", labels={"avg_glucose_level": "평균 혈당"},
    )
    fig.update_layout(showlegend=False, xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

means = (
    df.groupby("뇌졸중 여부")
    .agg(
        **{
            "사람 수": ("stroke", "size"),
            "나이 평균": ("age", "mean"),
            "평균 혈당 평균": ("avg_glucose_level", "mean"),
        }
    )
    .reindex(GROUP_ORDER)
    .round(2)
    .reset_index()
)
st.subheader("두 그룹의 평균값")
st.dataframe(means, hide_index=True, use_container_width=True)

# ---------- 3. 고혈압·심장병별 뇌졸중 비율 (막대그래프) ----------
st.header("3. 고혈압·심장병 여부에 따른 뇌졸중 비율")


def stroke_rate_by(column: str, name: str) -> pd.DataFrame:
    rate = (
        df.groupby(column)["stroke"]
        .agg(["mean", "size"])
        .rename(columns={"mean": "뇌졸중 비율(%)", "size": "사람 수"})
        .reset_index()
    )
    rate["뇌졸중 비율(%)"] = (rate["뇌졸중 비율(%)"] * 100).round(2)
    rate[name] = rate[column].map({0: "없음", 1: "있음"})
    return rate


col1, col2 = st.columns(2)
for col, column, name in [(col1, "hypertension", "고혈압"), (col2, "heart_disease", "심장병")]:
    rate = stroke_rate_by(column, name)
    fig = px.bar(
        rate, x=name, y="뇌졸중 비율(%)", text="뇌졸중 비율(%)", color=name,
        category_orders={name: ["없음", "있음"]},
        hover_data={"사람 수": True},
        title=f"{name} 여부별 뇌졸중 비율",
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(showlegend=False, xaxis_title=f"{name} 여부")
    fig.update_yaxes(range=[0, max(rate["뇌졸중 비율(%)"].max() * 1.25, 1)])
    with col:
        st.plotly_chart(fig, use_container_width=True)

# ---------- 4. bmi 빈 값 ----------
st.header("4. 체질량지수(bmi)가 비어 있는 사람들")
bmi_missing = df[df["bmi"].isna()]
summary = pd.DataFrame(
    {
        "구분": ["bmi가 빈 사람", "전체"],
        "사람 수": [len(bmi_missing), len(df)],
        "뇌졸중 사람 수": [int(bmi_missing["stroke"].sum()), int(df["stroke"].sum())],
    }
)
summary["뇌졸중 비율(%)"] = (summary["뇌졸중 사람 수"] / summary["사람 수"] * 100).round(2)
st.dataframe(summary, hide_index=True, use_container_width=True)

# ---------- 5. 흡연 상태 ----------
st.header("5. 흡연 상태별 사람 수")
smoking = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태(smoking_status)")
    .reset_index(name="사람 수")
)
st.dataframe(smoking, hide_index=True, use_container_width=True)
