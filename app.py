"""
공동주택 세대당 가스 사용량 분석 대시보드 (Streamlit + BigQuery)

실행:  streamlit run app.py
필요:  pip install streamlit pandas "altair>=5" google-cloud-bigquery db-dtypes pyarrow

인증(둘 중 하나)
  1) .streamlit/secrets.toml (또는 Streamlit Cloud의 Secrets)에 [gcp_service_account] 입력
  2) 또는 로컬에서  gcloud auth application-default login  후 실행
"""

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account

# ──────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────
PROJECT = "dse-marketing"
DATASET = f"{PROJECT}.sales"

AREA_BINS = [0, 40, 60, 85, 102, 135, np.inf]  # (a, b] 구간 (DB의 면적대와 동일)
AREA_LABELS = ["40㎡ 이하", "40~60㎡", "60~85㎡", "85~102㎡", "102~135㎡", "135㎡ 초과"]
AREA_NONE = "면적 미등록"
BUILT_NONE = "미상"
TTM = "최근12개월"

W_AVG = "세대수 가중평균"
S_AVG = "단지 단순평균"
S_MED = "단지 중앙값"

st.set_page_config(page_title="공동주택 가스 사용량", page_icon="🔥", layout="wide")

# ──────────────────────────────────────────────────────────────
# 디자인 (색·차트 스타일)
# ──────────────────────────────────────────────────────────────
# charts-begin
INK = "#191919"      # 본문 글자
INK2 = "#666666"     # 보조 글자 (축, 범례, 부제)
GRID = "#e7e7e7"     # 눈금선
AXIS = "#c2c2c2"     # 기준선
SURFACE = "#ffffff"  # 차트 바탕
# 범주형 색: 순서가 고정된 8색 (색각이상 검증 통과 순서, 9번째부터는 쓰지 않음)
SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
FONT = 'system-ui, -apple-system, "Segoe UI", "Malgun Gothic", "Apple SD Gothic Neo", sans-serif'


def style(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    """모든 차트에 같은 모양(얇은 눈금, 조용한 축, 위쪽 범례)을 입힌다."""
    return (
        chart.configure(background=SURFACE)
        .configure_view(stroke=None)
        .configure_axis(
            domainColor=AXIS, tickColor=AXIS, gridColor=GRID, gridWidth=1,
            labelColor=INK2, titleColor=INK2, labelFont=FONT, titleFont=FONT,
            labelFontSize=12, titleFontSize=12, titleFontWeight="normal", labelPadding=6,
        )
        .configure_legend(
            orient="top", direction="horizontal", title=None, labelColor=INK, labelFont=FONT,
            labelFontSize=12, symbolStrokeWidth=3, symbolSize=120, columnPadding=14, padding=0,
        )
    )


def bar_chart(df: pd.DataFrame, unit_label: str, ref=None, height=320):
    """단일 계열 막대 (열: 그룹, 값). 그룹이 8개 초과면 가로 막대."""
    order = df["그룹"].tolist()
    n = len(order)
    horizontal = n > 8
    vmax = float(df["값"].abs().max()) if n else 0
    fmt = ",.0f" if vmax >= 100 else ",.1f"
    tip = [alt.Tooltip("그룹:N", title="구분"), alt.Tooltip("값:Q", title=unit_label, format=",.1f")]

    if horizontal:
        enc = dict(
            y=alt.Y("그룹:N", sort=order, title=None, axis=alt.Axis(grid=False, ticks=False, labelLimit=180)),
            x=alt.X("값:Q", title=unit_label, axis=alt.Axis(format=",.0f", grid=True)),
            tooltip=tip,
        )
        bars = alt.Chart(df).mark_bar(size=16, color=SERIES[0], cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(**enc)
        layers = [bars]
        if n <= 14:
            layers.append(
                alt.Chart(df).mark_text(align="left", dx=6, color=INK, font=FONT, fontSize=12)
                .encode(y=alt.Y("그룹:N", sort=order), x="값:Q", text=alt.Text("값:Q", format=fmt))
            )
        if ref is not None:
            layers.append(alt.Chart(pd.DataFrame({"r": [ref]})).mark_rule(color=INK2, strokeWidth=1).encode(x="r:Q"))
        h = max(height, n * 26 + 60)
    else:
        enc = dict(
            x=alt.X("그룹:N", sort=order, title=None, axis=alt.Axis(grid=False, ticks=False, labelAngle=0, labelLimit=140)),
            y=alt.Y("값:Q", title=unit_label, axis=alt.Axis(format=",.0f")),
            tooltip=tip,
        )
        bars = alt.Chart(df).mark_bar(size=24, color=SERIES[0], cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(**enc)
        layers = [bars]
        layers.append(
            alt.Chart(df).mark_text(baseline="bottom", dy=-6, color=INK, font=FONT, fontSize=12)
            .encode(x=alt.X("그룹:N", sort=order), y="값:Q", text=alt.Text("값:Q", format=fmt))
        )
        if ref is not None:
            layers.append(alt.Chart(pd.DataFrame({"r": [ref]})).mark_rule(color=INK2, strokeWidth=1).encode(y="r:Q"))
        h = height
    return style(alt.layer(*layers).properties(height=h))


def line_chart(long_df: pd.DataFrame, unit_label: str, slots: dict, x_title=None, height=320):
    """연도별 추이 (열: 연도, 그룹, 값). slots: {그룹: 색}. 선 2px, 점 8px + 흰 테두리."""
    groups = list(slots.keys())
    scale = alt.Scale(domain=groups, range=[slots[g] for g in groups])
    color = alt.Color("그룹:N", scale=scale, legend=alt.Legend(title=None) if len(groups) > 1 else None)
    base = alt.Chart(long_df).encode(
        x=alt.X("연도:O", title=x_title, axis=alt.Axis(grid=False, labelAngle=0)),
        y=alt.Y("값:Q", title=unit_label, scale=alt.Scale(zero=False, nice=True), axis=alt.Axis(format=",.0f")),
        color=color,
    )
    tip = [alt.Tooltip("그룹:N", title="구분"), alt.Tooltip("연도:O"), alt.Tooltip("값:Q", title=unit_label, format=",.1f")]
    lines = base.mark_line(strokeWidth=2, strokeJoin="round", strokeCap="round")
    dots = base.mark_circle(size=64, opacity=1, stroke=SURFACE, strokeWidth=2)
    hit = base.mark_circle(size=420, opacity=0).encode(tooltip=tip)  # 호버 영역을 크게
    return style(alt.layer(lines, dots, hit).properties(height=height))


def grouped_bar_chart(long_df: pd.DataFrame, unit_label: str, slots: dict, height=320):
    """여러 단지의 연도별 막대 (열: 연도, 단지, 값)."""
    names = list(slots.keys())
    scale = alt.Scale(domain=names, range=[slots[g] for g in names])
    ch = (
        alt.Chart(long_df)
        .mark_bar(size=14, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("연도:O", title=None, axis=alt.Axis(grid=False, ticks=False, labelAngle=0)),
            xOffset=alt.XOffset("단지:N", sort=names),
            y=alt.Y("값:Q", title=unit_label, axis=alt.Axis(format=",.0f")),
            color=alt.Color("단지:N", scale=scale, legend=alt.Legend(title=None)),
            tooltip=[alt.Tooltip("단지:N"), alt.Tooltip("연도:O"), alt.Tooltip("값:Q", title=unit_label, format=",.1f")],
        )
        .properties(height=height)
    )
    return style(ch)


def monthly_chart(long_df: pd.DataFrame, unit_label: str, slots: dict, height=320):
    """월별 추이 (열: 월(날짜), 단지, 값). 점이 많아 마우스를 올린 점만 표시."""
    names = list(slots.keys())
    scale = alt.Scale(domain=names, range=[slots[g] for g in names])
    hover = alt.selection_point(fields=["월", "단지"], on="mouseover", clear="mouseout", empty=False)
    base = alt.Chart(long_df).encode(
        x=alt.X("월:T", title=None, axis=alt.Axis(grid=False, format="%Y-%m", labelAngle=0, tickCount=8)),
        y=alt.Y("값:Q", title=unit_label, scale=alt.Scale(zero=False, nice=True), axis=alt.Axis(format=",.0f")),
        color=alt.Color("단지:N", scale=scale, legend=alt.Legend(title=None) if len(names) > 1 else None),
    )
    lines = base.mark_line(strokeWidth=2, strokeJoin="round", strokeCap="round")
    dots = (
        base.mark_circle(size=90, stroke=SURFACE, strokeWidth=2)
        .encode(
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            tooltip=[alt.Tooltip("단지:N"), alt.Tooltip("월:T", format="%Y-%m"),
                     alt.Tooltip("값:Q", title=unit_label, format=",.1f")],
        )
        .add_params(hover)
    )
    return style(alt.layer(lines, dots).properties(height=height))


def color_slots(selected: list, universe: list) -> dict:
    """색은 '대상(그룹)'을 따라간다.
    전체 그룹이 8개 이하이면 전체 목록 순서대로 고정(필터를 바꿔도 색이 안 바뀜),
    8개를 넘으면 선택한 순서대로 앞의 색부터 배정한다(색을 돌려쓰지 않음)."""
    if len(universe) <= len(SERIES):
        return {g: SERIES[universe.index(g)] for g in selected}
    return {g: SERIES[i] for i, g in enumerate(selected[: len(SERIES)])}
# charts-end

CSS = """
<style>
.block-container {padding-top: 2.2rem;}
.ct {font-size: 15px; font-weight: 600; color: inherit; margin: 2px 0 0 0;}
.cs {font-size: 12.5px; opacity: .7; margin: 0 0 6px 0;}
.kpi-row {display: grid; grid-template-columns: 1.5fr 1fr 1fr 1fr; gap: 12px; margin: 6px 0 22px 0;}
.kpi {background: rgba(128,128,128,0.08); border: 1px solid rgba(128,128,128,0.25); border-radius: 12px; padding: 14px 18px;}
.kpi .lab {opacity: .7; font-size: 13px; margin-bottom: 2px;}
.kpi .val {color: inherit; font-size: 28px; font-weight: 600; line-height: 1.25;}
.kpi.hero .val {font-size: 48px; line-height: 1.15;}
.kpi .unit {opacity: .7; font-size: 14px; font-weight: 400; margin-left: 6px;}
.kpi .sub {opacity: .7; font-size: 12.5px; margin-top: 2px;}
@media (max-width: 900px) {.kpi-row {grid-template-columns: 1fr 1fr;}}
</style>
"""


def chart_title(title: str, sub: str = ""):
    st.markdown(f"<div class='ct'>{title}</div><div class='cs'>{sub}</div>", unsafe_allow_html=True)


def kpi_row(items: list):
    """items: [(라벨, 값, 단위, 보조설명, hero 여부)]"""
    cells = []
    for lab, val, unit, sub, hero in items:
        u = f"<span class='unit'>{unit}</span>" if unit else ""
        cells.append(
            f"<div class='kpi{' hero' if hero else ''}'><div class='lab'>{lab}</div>"
            f"<div class='val'>{val}{u}</div><div class='sub'>{sub}</div></div>"
        )
    st.markdown(CSS + "<div class='kpi-row'>" + "".join(cells) + "</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# BigQuery 로드
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_client() -> bigquery.Client:
    try:
        if "gcp_service_account" in st.secrets:
            creds = service_account.Credentials.from_service_account_info(
                dict(st.secrets["gcp_service_account"])
            )
            return bigquery.Client(credentials=creds, project=PROJECT)
    except Exception:
        pass  # secrets.toml 이 없으면 기본 인증(ADC) 사용
    return bigquery.Client(project=PROJECT)


@st.cache_data(ttl=3600, show_spinner="평형 구성을 불러오는 중...")
def load_pyeong() -> pd.DataFrame:
    sql = f"""
    SELECT apartment_code,
      CASE WHEN pyeong IS NULL THEN '미등록'
           WHEN pyeong < 30 THEN '20평형'
           WHEN pyeong < 40 THEN '30평형'
           WHEN pyeong < 50 THEN '40평형'
           ELSE '50평형 이상' END AS band,
      COUNT(*) AS hh
    FROM `{DATASET}.pbi_ref_apt_unit_area`
    GROUP BY 1, 2
    """
    return get_client().query(sql).to_dataframe()


@st.cache_data(ttl=3600, show_spinner="BigQuery에서 사용량 데이터를 불러오는 중...")
def load_fact() -> pd.DataFrame:
    sql = f"""
    SELECT sales_ym, apartment_code, category, product_name,
           metered_row_cnt, usage_m3, usage_mj
    FROM `{DATASET}.pbi_fact_residential_apt`
    WHERE metered_row_cnt > 0 AND usage_m3 IS NOT NULL
    """
    df = get_client().query(sql).to_dataframe()
    df["sales_ym"] = pd.to_datetime(df["sales_ym"])
    df["product"] = df["category"].astype(str) + "/" + df["product_name"].astype(str)
    return df


@st.cache_data(ttl=3600, show_spinner="BigQuery에서 단지 정보를 불러오는 중...")
def load_master() -> pd.DataFrame:
    sql = f"""
    SELECT d.apartment_code, d.apartment_name, d.sigungu,
           b.built_year, a.avg_area, a.units, i.households, i.address, i.lat, i.lon
    FROM `{DATASET}.pbi_dim_apartment` d
    LEFT JOIN (
        SELECT apartment_code, ANY_VALUE(built_year) AS built_year
        FROM `{DATASET}.pbi_ref_apt_built_year` GROUP BY 1
    ) b USING (apartment_code)
    LEFT JOIN (
        SELECT apartment_code, AVG(area_m2) AS avg_area, COUNT(*) AS units
        FROM `{DATASET}.pbi_ref_apt_unit_area`
        WHERE area_m2 IS NOT NULL GROUP BY 1
    ) a USING (apartment_code)
    LEFT JOIN (
        SELECT apartment_code, ANY_VALUE(households) AS households,
               ANY_VALUE(address) AS address, ANY_VALUE(lat) AS lat, ANY_VALUE(lon) AS lon
        FROM `{DATASET}.pbi_ref_apt_info` GROUP BY 1
    ) i USING (apartment_code)
    """
    m = get_client().query(sql).to_dataframe()
    m = m.drop_duplicates("apartment_code").reset_index(drop=True)
    m["built_year"] = pd.to_numeric(m["built_year"], errors="coerce")
    m["avg_area"] = pd.to_numeric(m["avg_area"], errors="coerce")
    m["lat"] = pd.to_numeric(m["lat"], errors="coerce")
    m["lon"] = pd.to_numeric(m["lon"], errors="coerce")
    band = pd.cut(m["avg_area"], bins=AREA_BINS, labels=AREA_LABELS, right=True)
    m["area_band"] = band.astype(object).where(m["avg_area"].notna(), AREA_NONE)
    return m


# ──────────────────────────────────────────────────────────────
# 계산 함수
# ──────────────────────────────────────────────────────────────
def build_apt_period(fact: pd.DataFrame, val_col: str, per_month: bool) -> pd.DataFrame:
    """단지 × 기간(연도 / 최근12개월) 별 세대당 사용량.

    - 월별로 (선택한 상품들의) 사용량·계량기 수를 단지 단위로 합산
    - 기간 내 12개월이 모두 있는 단지만 사용 (신축·중도 개통 단지의 부분연도 왜곡 방지)
    - hh   = 월평균 계량기 수(≈ 세대수),  per_hh = 사용량 / hh
    """
    if fact.empty:
        return pd.DataFrame(columns=["apartment_code", "period", "usage", "hh", "per_hh"])

    m = fact.groupby(["apartment_code", "sales_ym"], as_index=False).agg(
        usage=(val_col, "sum"), cnt=("metered_row_cnt", "sum")
    )
    m["year"] = m["sales_ym"].dt.year

    def agg(frame: pd.DataFrame, period_col) -> pd.DataFrame:
        g = frame.groupby(["apartment_code", period_col], as_index=False).agg(
            usage=("usage", "sum"), cnt=("cnt", "sum"), n=("sales_ym", "nunique")
        )
        g = g[g["n"] >= 12].copy()
        g["hh"] = g["cnt"] / g["n"]
        g = g.rename(columns={period_col: "period"})
        return g[["apartment_code", "period", "usage", "hh"]]

    cal = agg(m, "year")
    cal["period"] = cal["period"].astype(int).astype(str)

    last = m["sales_ym"].max()
    start = last - pd.DateOffset(months=11)
    ttm_frame = m[m["sales_ym"] >= start].copy()
    ttm_frame["p"] = TTM
    ttm = agg(ttm_frame, "p")

    out = pd.concat([cal, ttm], ignore_index=True)
    if per_month:
        out["usage"] = out["usage"] / 12
    out["per_hh"] = out["usage"] / out["hh"]
    return out


def group_period_table(ap: pd.DataFrame, group_col: str, how: str) -> pd.DataFrame:
    keys = [group_col, "period"]
    if how == W_AVG:
        g = ap.groupby(keys).agg(u=("usage", "sum"), h=("hh", "sum"))
        s = g["u"] / g["h"]
    elif how == S_AVG:
        s = ap.groupby(keys)["per_hh"].mean()
    else:
        s = ap.groupby(keys)["per_hh"].median()
    return s.unstack("period")


def calendar_cols(cols) -> list:
    return sorted(c for c in cols if str(c).isdigit())


def sort_labels(labels, order=None):
    if order:
        rank = {l: i for i, l in enumerate(order)}
        return sorted(labels, key=lambda x: (rank.get(x, 999), str(x)))
    return sorted(labels, key=lambda x: (x == BUILT_NONE, str(x)))


def make_built_group(year: pd.Series, mode: str) -> pd.Series:
    out = pd.Series(BUILT_NONE, index=year.index, dtype=object)
    ok = year.notna()
    y = year[ok].astype(int)
    if mode == "연도별":
        out[ok] = y.astype(str) + "년"
    else:
        step = 10 if mode.startswith("10") else 5
        s = (y // step) * step
        out[ok] = s.astype(str) + "~" + (s + step - 1).astype(str)
    return out


def summarize(ap: pd.DataFrame, group_col: str, how: str, n_recent: int, order=None):
    """그룹별 연도 표 + 최근 N개년 평균 + 3개년 이동평균."""
    tbl = group_period_table(ap, group_col, how)
    cal = calendar_cols(tbl.columns)
    if not cal:
        return None
    recent = cal[-n_recent:]
    roll = tbl[cal].T.rolling(3, min_periods=3).mean().T

    meta = (
        ap[ap["period"] == cal[-1]]
        .groupby(group_col)
        .agg(단지수=("apartment_code", "nunique"), 세대수=("hh", "sum"))
    )
    out = tbl[cal + ([TTM] if TTM in tbl.columns else [])].copy()
    out[f"최근{len(recent)}개년 평균({recent[0]}~{recent[-1]})"] = tbl[recent].mean(axis=1)
    out[f"3개년 이동평균({cal[-1]})"] = roll[cal[-1]] if len(cal) >= 3 else np.nan
    out = out.join(meta)
    out = out.reindex(sort_labels(out.index, order))
    return out, roll, cal


def show_df(data, **kw):
    """Streamlit 버전에 따라 width 인자가 다르므로 호환 처리."""
    try:
        st.dataframe(data, width="stretch", **kw)
    except TypeError:
        st.dataframe(data, use_container_width=True, **kw)


def show_alt(chart):
    try:
        st.altair_chart(chart, theme=None, width="stretch")
    except TypeError:
        st.altair_chart(chart, theme=None, use_container_width=True)


def fmt_table(df: pd.DataFrame):
    count_cols = [c for c in ["단지수", "세대수"] if c in df.columns]
    sty = df.style.format("{:,.1f}", na_rep="-")
    if count_cols:
        sty = sty.format("{:,.0f}", na_rep="-", subset=count_cols)
    return sty


def to_long(wide: pd.DataFrame, value_cols: list, groups: list) -> pd.DataFrame:
    """행=그룹, 열=연도 표를 (연도, 그룹, 값) 세로 형태로."""
    w = wide.loc[groups, value_cols].copy()
    w.index.name = "그룹"
    long = w.reset_index().melt(id_vars="그룹", var_name="연도", value_name="값").dropna()
    long["연도"] = long["연도"].astype(str)
    return long


# ──────────────────────────────────────────────────────────────
# 화면 공통 컴포넌트
# ──────────────────────────────────────────────────────────────
def render_group_tab(ap, group_col, how, n_recent, unit_label, order=None, key="",
                     sort_by_value=False, overall=None):
    res = summarize(ap, group_col, how, n_recent, order)
    if res is None:
        st.warning("12개월이 모두 채워진 연도 데이터가 없습니다. 필터를 확인하세요.")
        return
    tbl, roll, cal = res
    avg_col = [c for c in tbl.columns if c.startswith("최근") and "개년 평균" in c][0]

    left, right = st.columns([1, 1], gap="large")
    with left:
        bar_df = tbl[[avg_col]].dropna().reset_index()
        bar_df.columns = ["그룹", "값"]
        if sort_by_value:
            bar_df = bar_df.sort_values("값", ascending=False)
        sub = unit_label + (f" · 회색 선 = 전체 평균 {overall:,.1f}" if overall is not None else "")
        chart_title(avg_col, sub)
        show_alt(bar_chart(bar_df, unit_label, ref=overall))
    with right:
        groups_all = list(tbl.index)
        default = groups_all if len(groups_all) <= len(SERIES) else groups_all[:5]
        chart_title("연도별 추이", unit_label)
        picked = st.multiselect(
            "표시할 그룹 (최대 8개)", groups_all, default=default, max_selections=len(SERIES),
            key=f"trend_{key}_{abs(hash(tuple(groups_all))) % 10**8}",
        )
        if picked:
            slots = color_slots(picked, groups_all)
            show_alt(line_chart(to_long(tbl, cal, picked), unit_label, slots))
        else:
            st.info("표시할 그룹을 선택하세요.")

    st.markdown("**연도별 값 / 최근 평균 / 3개년 이동평균**")
    show_df(fmt_table(tbl))

    with st.expander("3개년 이동평균 추이 (각 연도 기준 직전 3년 평균)"):
        roll_cols = [c for c in cal if roll[c].notna().any()]
        if roll_cols and picked:
            slots = color_slots(picked, groups_all)
            show_alt(line_chart(to_long(roll, roll_cols, picked), unit_label, slots))
        show_df(roll.style.format("{:,.1f}", na_rep="-"))

    csv = tbl.to_csv().encode("utf-8-sig")
    st.download_button("표 CSV 다운로드", csv, file_name=f"{group_col}_summary.csv", key=f"dl_{key}")


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────
st.title("🔥 공동주택 세대당 가스 사용량")

try:
    fact_all = load_fact()
    master = load_master()
except Exception as e:  # 인증/권한 오류를 화면에 표시
    st.error(f"BigQuery 로드 실패: {e}")
    st.stop()

# ── 사이드바 ─────────────────────────────────────────────────
with st.sidebar:
    st.header("조건")
    unit = st.radio("사용량 단위", ["m³", "MJ"], horizontal=True)
    basis = st.radio("표시 기준", ["연간 세대당", "월평균 세대당"], horizontal=True)
    how = st.radio("그룹 평균 방식", [W_AVG, S_AVG, S_MED])
    n_recent = st.slider("‘최근 N개년 평균’의 N", 2, 5, 3)

    usages = [u for u in ["개별난방용", "취사난방용", "취사용", "중앙난방용"]
              if u in set(fact_all["product_name"])]
    usages += sorted(set(fact_all["product_name"]) - set(usages))
    sel_usages = st.multiselect(
        "용도", usages, default=[u for u in ["개별난방용", "취사난방용"] if u in usages],
        help="개별난방용·취사난방용: 세대별 난방 포함. 취사용은 난방이 없어 사용량이 훨씬 작고, "
             "중앙난방용은 단지 단위 계량이라 세대당 값이 크게 달라 기본에서 제외했습니다.",
    )
    cats = sorted(fact_all["category"].dropna().unique())
    sel_cats = st.multiselect(
        "요금 구분", cats, default=[c for c in ["가스요금"] if c in cats],
        help="가스요금: 일반 요금 적용분 / 우선사용: 우선사용 적용분",
    )
    sel_products = [p for p in fact_all["product"].unique()
                    if p.split("/", 1)[-1] in sel_usages and p.split("/", 1)[0] in sel_cats]
    st.divider()
    st.subheader("세부 조건")
    st.caption("여러 조건을 함께 고르면 모두 만족하는 단지만 집계합니다. (예: 수성구 · 준공 최근 10년 · 85㎡ 이상)")
    sigungus = sorted(master["sigungu"].dropna().unique())
    sel_sigungu = st.multiselect("구·군 (비우면 전체)", sigungus)

    BY_MIN = 1980
    BY_MAX = max(pd.Timestamp.today().year, int(pd.to_numeric(master["built_year"], errors="coerce").max() or 0))
    by_lo, by_hi = st.slider(
        "준공연도 (바를 좌우로 조정)", BY_MIN, BY_MAX, (BY_MIN, BY_MAX), step=1,
        help="기본(1980~현재 전체)에서는 준공연도를 모르는 단지와 1980년 이전 단지까지 모두 포함합니다. "
             "바를 움직이면 그 범위에 준공된 단지만 집계합니다.",
    )
    if (by_lo, by_hi) == (BY_MIN, BY_MAX):
        by_lo, by_hi = None, None

    area_opts = AREA_LABELS + [AREA_NONE]
    sel_areas = st.multiselect(
        "면적대 (전용면적 기준)", area_opts,
        help="공급면적이 아니라 ‘전용면적’(㎡) 기준입니다. 단지 내 세대별 전용면적의 평균으로 단지를 분류합니다. "
             "비우면 전체. 공급평형으로 대략 환산하면 40㎡ 이하≈20평형 미만, 40~60㎡≈20평형대, "
             "60~85㎡≈25~34평형, 85~102㎡≈34~40평형, 102~135㎡≈40평형대, 135㎡ 초과≈50평형 이상입니다 "
             "(단지마다 차이가 있는 대략값).",
    )
    st.caption("면적대는 전용면적(㎡) 기준 · 평형은 공급평형 기준(아파트 검색의 ‘평형별 세대 구성’)")

    hh_vals = pd.to_numeric(master["households"], errors="coerce").dropna()
    hh_cap = int(min(3000, max(500, (hh_vals.max() // 100) * 100))) if len(hh_vals) else 2000
    min_hh = st.slider("세대기준 (이상)", 0, hh_cap, 0, step=100, format="%d세대 이상",
                       help="단지 세대수가 이 값 이상인 단지만 집계합니다. 0이면 전체 "
                            "(세대수 정보가 없는 단지는 0일 때만 포함).")
    st.divider()

    st.caption(
        "연간 값은 해당 연도 12개월이 모두 있는 단지만 계산합니다. "
        "‘최근12개월’은 데이터의 마지막 월까지의 12개월입니다."
    )
    if st.button("데이터 새로고침"):
        st.cache_data.clear()
        st.rerun()

if not sel_products:
    st.info("사이드바에서 용도와 요금 구분을 하나 이상 선택하세요.")
    st.stop()

val_col = "usage_m3" if unit == "m³" else "usage_mj"
per_month = basis.startswith("월평균")
unit_label = f"{unit}/세대/{'월' if per_month else '년'}"

fact = fact_all[fact_all["product"].isin(sel_products)]
ap_all = build_apt_period(fact, val_col, per_month)
ap_all = ap_all.merge(master, on="apartment_code", how="inner")
if min_hh > 0:
    ap_all = ap_all[pd.to_numeric(ap_all["households"], errors="coerce") >= min_hh]
if by_lo is not None:
    _by = pd.to_numeric(ap_all["built_year"], errors="coerce")
    ap_all = ap_all[(_by >= by_lo) & (_by <= by_hi)]
if sel_areas:
    ap_all = ap_all[ap_all["area_band"].isin(sel_areas)]
ap = ap_all[ap_all["sigungu"].isin(sel_sigungu)] if sel_sigungu else ap_all

if ap.empty:
    st.warning("조건에 맞는 데이터가 없습니다. 세부 조건(구·군, 준공 시기, 면적대, 세대기준)을 확인하세요.")
    st.stop()

# ── 요약 카드 ────────────────────────────────────────────────
last_month = fact_all["sales_ym"].max().strftime("%Y-%m")
ov_tbl = group_period_table(ap.assign(_all="전체"), "_all", how)
ov_cal = calendar_cols(ov_tbl.columns)
overall_recent = None
if ov_cal:
    rec = ov_cal[-n_recent:]
    overall_recent = float(ov_tbl.loc["전체", rec].mean())
    latest = ov_cal[-1]
    latest_ap = ap[ap["period"] == latest]
    delta_txt, delta_sub = "-", "비교할 전년 데이터 없음"
    if len(ov_cal) >= 2:
        prev = ov_cal[-2]
        pct = (ov_tbl.loc["전체", latest] / ov_tbl.loc["전체", prev] - 1) * 100
        delta_txt = f"{'▲' if pct >= 0 else '▼'} {abs(pct):.1f}%"
        delta_sub = f"{prev}년 → {latest}년"
    kpi_row([
        (f"전체 세대당 사용량 · 최근 {len(rec)}개년 평균", f"{overall_recent:,.0f}", unit_label,
         f"{rec[0]}~{rec[-1]} · {how}", True),
        ("직전 연도 대비 증감", delta_txt, "", delta_sub, False),
        (f"분석 단지 ({latest}년 기준)", f"{latest_ap['apartment_code'].nunique():,}", "개",
         f"세대수 {latest_ap['hh'].sum():,.0f}", False),
        ("데이터 최종 월", last_month, "", f"‘{TTM}’은 이 달까지 12개월", False),
    ])

    # 분석 단지 전체 목록 (다운로드)
    lst_codes = latest_ap["apartment_code"].unique()
    pv_all = ap[ap["apartment_code"].isin(lst_codes)].pivot_table(
        index="apartment_code", columns="period", values="per_hh", aggfunc="first")
    cal_cols = [c for c in ov_cal if c in pv_all.columns]
    out = pv_all[cal_cols + ([TTM] if TTM in pv_all.columns else [])].copy()
    out.columns = [f"{c}년 {unit_label}" if c in cal_cols else f"{TTM} {unit_label}" for c in out.columns]
    out[f"최근{len(rec)}개년 평균({rec[0]}~{rec[-1]})"] = pv_all[rec].mean(axis=1)
    base = master[master["apartment_code"].isin(lst_codes)].set_index("apartment_code")[
        ["apartment_name", "sigungu", "address", "built_year", "households", "avg_area", "area_band"]
    ].rename(columns={"apartment_name": "단지명", "sigungu": "구·군", "address": "주소", "built_year": "준공연도",
                      "households": "세대수(정보)", "avg_area": "평균 전용면적(㎡)", "area_band": "면적대"})
    hh_last = latest_ap.set_index("apartment_code")["hh"].rename(f"{latest}년 검침 세대수(평균)")
    listing = base.join(hh_last).join(out).reset_index().rename(columns={"apartment_code": "단지코드"})
    listing = listing.sort_values(["구·군", "단지명"]).reset_index(drop=True)
    d1, d2 = st.columns([1, 3])
    with d1:
        st.download_button(
            f"⬇ 분석 단지 {len(listing):,}개 목록 (CSV)",
            listing.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"분석단지목록_{last_month}.csv", mime="text/csv",
        )
    with d2:
        with st.expander(f"분석 단지 {len(listing):,}개 목록 미리보기"):
            show_df(listing.style.format(
                {c: "{:,.1f}" for c in listing.columns if "m³" in c or "MJ" in c or "평균" in c}, na_rep="-"),
                hide_index=True)
else:
    st.markdown(CSS, unsafe_allow_html=True)
    st.warning("12개월이 모두 채워진 연도 데이터가 없습니다.")



# ── 단지 상세 (검색 탭·지도 탭 공용) ─────────────────────────
def render_apartments(codes):
    names = master.set_index("apartment_code")["apartment_name"]
    disp = {c: f"{names[c]} ({c})" for c in codes}
    slots = {disp[c]: SERIES[i] for i, c in enumerate(codes)}  # 선택 순서대로 고정 배정
    sel = ap_all[ap_all["apartment_code"].isin(codes)].copy()
    sel["단지"] = sel["apartment_code"].map(disp)

    # 연도별 표
    pv = sel.pivot_table(index="단지", columns="period", values="per_hh", aggfunc="first")
    cal = calendar_cols(pv.columns)
    tbl = pv[cal + ([TTM] if TTM in pv.columns else [])].copy()
    if cal:
        recent = cal[-n_recent:]
        tbl[f"최근{len(recent)}개년 평균({recent[0]}~{recent[-1]})"] = pv[recent].mean(axis=1)
        if len(cal) >= 3:
            tbl[f"3개년 이동평균({cal[-1]})"] = pv[cal].T.rolling(3, min_periods=3).mean().T[cal[-1]]

    chart_title("연도별 세대당 사용량", unit_label)
    if tbl.empty or not cal:
        st.warning("선택한 단지는 12개월이 모두 채워진 연도가 없습니다.")
    else:
        if len(codes) == 1:
            one = pv.loc[disp[codes[0]], cal].dropna()
            bdf = pd.DataFrame({"그룹": [f"{c}년" for c in one.index], "값": one.values})
            show_alt(bar_chart(bdf, unit_label))
        else:
            gl = pv[cal].reset_index().melt(id_vars="단지", var_name="연도", value_name="값").dropna()
            show_alt(grouped_bar_chart(gl, unit_label, slots))
        show_df(fmt_table(tbl))

    # 비교 기준(같은 구 / 같은 면적대 / 같은 준공 연대)
    if cal:
        recent = cal[-n_recent:]
        st.markdown(f"**비교 기준 대비** · 최근 {len(recent)}개년 평균({recent[0]}~{recent[-1]}), {how}")
        rows = []
        ap_c = ap_all.copy()
        ap_c["decade"] = make_built_group(ap_c["built_year"], "10년 단위(연대)")
        tabs_by = {
            "sigungu": group_period_table(ap_c, "sigungu", how),
            "area_band": group_period_table(ap_c, "area_band", how),
            "decade": group_period_table(ap_c, "decade", how),
        }
        mi = master.set_index("apartment_code")
        for code in codes:
            r = mi.loc[code]
            dec = make_built_group(pd.Series([r["built_year"]]), "10년 단위(연대)").iloc[0]
            own = pv.loc[disp[code], recent].mean() if disp[code] in pv.index else np.nan

            def ref(key, grp):
                t = tabs_by[key]
                cols = [c for c in recent if c in t.columns]
                return t.loc[grp, cols].mean() if grp in t.index and cols else np.nan

            g_gu, g_ar, g_de = ref("sigungu", r["sigungu"]), ref("area_band", r["area_band"]), ref("decade", dec)
            rows.append({
                "단지": disp[code],
                "이 단지": own,
                f"{r['sigungu']} 평균": g_gu,
                f"면적대 평균({r['area_band']})": g_ar,
                f"준공 연대 평균({dec})": g_de,
                "구 평균 대비(%)": (own / g_gu - 1) * 100 if pd.notna(own) and pd.notna(g_gu) else np.nan,
            })
        cmp_df = pd.DataFrame(rows).set_index("단지")
        show_df(cmp_df.style.format("{:,.1f}", na_rep="-"))

    # 월별 추이
    chart_title("월별 세대당 사용량", f"{unit}/세대/월 · 선택한 상품 합산 · 점에 마우스를 올리면 값이 보입니다")
    mf = fact[fact["apartment_code"].isin(codes)]
    mm = mf.groupby(["apartment_code", "sales_ym"], as_index=False).agg(
        u=(val_col, "sum"), c=("metered_row_cnt", "sum")
    )
    mm["값"] = mm["u"] / mm["c"]
    mm["단지"] = mm["apartment_code"].map(disp)
    mm = mm.rename(columns={"sales_ym": "월"})[["월", "단지", "값"]]
    show_alt(monthly_chart(mm, f"{unit}/세대/월", slots))

    # 평형별 구성
    PY = ["20평형", "30평형", "40평형", "50평형 이상", "미등록"]
    try:
        pyd = load_pyeong()
    except Exception:
        pyd = pd.DataFrame(columns=["apartment_code", "band", "hh"])
    pyd = pyd[pyd["apartment_code"].isin(codes)]
    if not pyd.empty:
        chart_title("평형별 세대 구성", "공급평형 기준(20평형=30평 미만, 30평형=30~39평, 40평형=40~49평) · 사용량이 아니라 세대수입니다")
        pick_py = st.segmented_control("평형 선택", ["전체"] + PY[:4], default="전체", key="py_pick") \
            if hasattr(st, "segmented_control") else "전체"
        pick_py = pick_py or "전체"
        pt = pyd.pivot_table(index="band", columns="apartment_code", values="hh", aggfunc="sum").reindex(PY).fillna(0)
        pt.columns = [disp[c] for c in pt.columns]
        tot = pt.sum()
        shown = pt if pick_py == "전체" else pt.loc[[pick_py]]
        out = shown.copy().astype(int)
        for c in pt.columns:
            out[f"{c} 비중(%)"] = (shown[c] / tot[c] * 100).round(1) if tot[c] else np.nan
        out.index.name = "평형"
        show_df(out.style.format("{:,.0f}", subset=[c for c in out.columns if "비중" not in c]).format("{:.1f}", subset=[c for c in out.columns if "비중" in c], na_rep="-"))
        long = shown.reset_index().melt(id_vars="band", var_name="단지", value_name="세대수")
        long = long[long["세대수"] > 0]
        if not long.empty:
            show_alt(
                alt.Chart(long).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=24).encode(
                    x=alt.X("band:N", sort=PY, title=None, axis=alt.Axis(labelAngle=0)),
                    xOffset="단지:N", y=alt.Y("세대수:Q", title="세대수"),
                    color=alt.Color("단지:N", scale=alt.Scale(domain=list(slots), range=list(slots.values())), legend=alt.Legend(orient="bottom", title=None)),
                    tooltip=["단지:N", "band:N", alt.Tooltip("세대수:Q", format=",")],
                ).properties(height=240, background=SURFACE).configure_view(stroke=None))
        st.caption("※ 가스 사용량은 단지 단위로만 집계되어 평형별 사용량은 계산할 수 없습니다. "
                   "평형별 사용량이 필요하면 호(세대)별 검침 데이터가 BigQuery에 추가되어야 합니다.")

    info = master[master["apartment_code"].isin(codes)][
        ["apartment_code", "apartment_name", "sigungu", "address", "built_year", "households", "units", "avg_area", "area_band"]
    ].rename(columns={
        "apartment_code": "코드", "apartment_name": "단지명", "sigungu": "구·군", "address": "주소",
        "built_year": "준공연도", "households": "세대수(정보)", "units": "세대수(면적등록)",
        "avg_area": "평균 전용면적(㎡)", "area_band": "면적대",
    })
    with st.expander("단지 기본 정보"):
        show_df(info, hide_index=True)


# ── 지도 ─────────────────────────────────────────────────────
MAP_RAMP = ["#84b9f3", "#519ced", "#257de0", "#1b64be", "#0f4a96"]  # 한 가지 파랑, 연함 → 진함
MAP_STYLE = "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json"


def hex_rgb(h: str, a: int = 230) -> list:
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)] + [a]


def selected_map_code():
    """지도에서 클릭한 단지의 코드 (없으면 None). 위젯 상태에서 읽는다."""
    state = st.session_state.get("apt_map")
    try:
        objs = (state["selection"]["objects"] or {}).get("apts") or []
        return objs[0].get("apartment_code") if objs else None
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def load_road(name: str = "달구벌대로") -> pd.DataFrame:
    """도로명주소 좌표에서 도로 중심선(서→동)을 근사해 만든다."""
    sql = f"""
    SELECT lon, lat FROM `{DATASET}.pbi_ref_addr_geo`
    WHERE REGEXP_CONTAINS(address, r'{name} [0-9]+(-[0-9]+)?$') AND lat IS NOT NULL AND lon IS NOT NULL
    """
    d = get_client().query(sql).to_dataframe()
    if d.empty:
        return d
    d["lon"] = pd.to_numeric(d["lon"]); d["lat"] = pd.to_numeric(d["lat"])
    d["bin"] = (d["lon"] / 0.004).round()
    line = d.groupby("bin").agg(lon=("lon", "median"), lat=("lat", "median"), n=("lon", "size"))
    line = line[line["n"] >= 2].sort_values("lon").reset_index(drop=True)
    return line[["lon", "lat"]]


def load_road_safe():
    try:
        return load_road()
    except Exception:
        return pd.DataFrame(columns=["lon", "lat"])


def render_map_tab():
    cal_all = calendar_cols(ap["period"].unique())
    if not cal_all:
        st.warning("12개월이 모두 채워진 연도 데이터가 없습니다.")
        return
    recent = cal_all[-n_recent:]
    val = (ap[ap["period"].isin(recent)].groupby("apartment_code")["per_hh"].mean()
           .rename("값").reset_index())
    m = master.merge(val, on="apartment_code", how="inner").dropna(subset=["lat", "lon"]).copy()
    if m.empty:
        st.info("지도에 표시할 단지가 없습니다. 상품·구·군 선택을 확인하세요.")
        return

    # 값이 비슷한 단지끼리 5단계로 묶어 한 가지 파랑의 농도로 표시
    codes_q = pd.qcut(m["값"], 5, labels=False, duplicates="drop")
    nb = int(codes_q.max()) + 1
    picks = np.linspace(0, len(MAP_RAMP) - 1, nb).round().astype(int)
    ramp = [MAP_RAMP[i] for i in picks]
    m["bin"] = codes_q.astype(int)
    m["color"] = m["bin"].map(lambda b: hex_rgb(ramp[b]))
    m["단지"] = m["apartment_name"].fillna("-")
    m["구"] = m["sigungu"].fillna("-")
    m["준공"] = m["built_year"].map(lambda y: f"{int(y)}년" if pd.notna(y) else "미상")
    m["값_txt"] = m["값"].map(lambda v: f"{v:,.1f} {unit_label}")

    chart_title(f"단지별 세대당 사용량 지도 · 최근 {len(recent)}개년 평균({recent[0]}~{recent[-1]})",
                "점을 클릭하면 아래에 그 단지의 사용량이 표시됩니다 · 색이 진할수록 사용량이 많음 · 주황색 선은 달구벌대로")

    sel_code = selected_map_code()
    if sel_code not in set(m["apartment_code"]):
        sel_code = None

    layers = []
    road = load_road_safe()
    if len(road) >= 2:
        path = [[float(x), float(y)] for x, y in zip(road["lon"], road["lat"])]
        mid = path[len(path) // 2]
        layers.append(pdk.Layer("PathLayer", id="road", data=[{"path": path}], get_path="path",
                                get_color=[244, 162, 97, 170], width_min_pixels=4, get_width=30,
                                pickable=False))
        labels = [{"pos": path[int(len(path) * f)], "t": "달구벌대로"} for f in (0.15, 0.5, 0.85)]
        layers.append(pdk.Layer("TextLayer", id="road_label", data=labels, get_position="pos",
                                get_text="t", get_size=14, get_color=[214, 122, 52, 255],
                                get_text_anchor="'middle'", get_alignment_baseline="'bottom'",
                                font_family="'Malgun Gothic','Noto Sans KR',sans-serif",
                                font_weight=700, outline_width=3, outline_color=[255, 255, 255, 230],
                                background=False, billboard=True))
    layers.append(pdk.Layer(
        "ScatterplotLayer", id="apts",
        data=m[["apartment_code", "단지", "구", "준공", "값_txt", "lat", "lon", "color"]],
        get_position="[lon, lat]", get_fill_color="color", get_radius=70,
        radius_min_pixels=4, radius_max_pixels=14, pickable=True, auto_highlight=True,
        highlight_color=[25, 25, 25, 90],
    ))
    if sel_code:
        layers.append(pdk.Layer(
            "ScatterplotLayer", id="picked", data=m[m["apartment_code"] == sel_code][["lat", "lon"]],
            get_position="[lon, lat]", filled=False, stroked=True, get_line_color=[25, 25, 25, 255],
            line_width_min_pixels=3, get_radius=70, radius_min_pixels=11, radius_max_pixels=20,
            pickable=False,
        ))
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=float(m["lat"].mean()), longitude=float(m["lon"].mean()),
                                         zoom=10.5, pitch=0),
        tooltip={"html": "<b>{단지}</b><br>{구} · 준공 {준공}<br>세대당 {값_txt}",
                 "style": {"backgroundColor": "#191919", "color": "#ffffff", "fontSize": "13px",
                           "padding": "8px 10px", "borderRadius": "6px"}},
        map_style=MAP_STYLE,
    )
    try:
        st.pydeck_chart(deck, on_select="rerun", selection_mode="single-object", key="apt_map",
                        width="stretch", height=520)
    except TypeError:  # 구버전 Streamlit: 클릭 선택 없이 지도만 표시
        st.pydeck_chart(deck)
        st.caption("이 Streamlit 버전에서는 지도 클릭 선택을 지원하지 않습니다. 아래 검색을 이용하세요.")

    # 범례 (단계별 값 범위)
    parts = []
    for b in range(nb):
        sub = m[m["bin"] == b]["값"]
        parts.append(
            f"<span style='display:inline-flex;align-items:center;margin-right:16px;font-size:12.5px'>"
            f"<span style='width:12px;height:12px;border-radius:50%;background:{ramp[b]};margin-right:6px'></span>"
            f"{sub.min():,.0f} ~ {sub.max():,.0f}</span>"
        )
    st.markdown("".join(parts) + f"<span style='font-size:12.5px;opacity:.7'>({unit_label})</span>",
                unsafe_allow_html=True)
    st.caption(f"좌표가 있고 12개월 데이터가 있는 단지 {len(m):,}개만 표시됩니다.")

    # 이름으로 찾기 (지도 클릭 대신)
    labels = {f"{r.단지} · {r.구} ({r.apartment_code})": r.apartment_code for r in m.sort_values(["구", "단지"]).itertuples()}
    pick = st.selectbox("또는 이름으로 찾기", ["(지도에서 선택)"] + list(labels.keys()), key="map_pick")
    code = labels.get(pick, sel_code)
    if code:
        st.divider()
        row = m[m["apartment_code"] == code].iloc[0]
        st.subheader(f"🏢 {row['단지']}")
        st.caption(f"{row['구']} · 준공 {row['준공']} · {row['값_txt']} (최근 {len(recent)}개년 평균)")
        render_apartments([code])
    else:
        st.info("지도에서 점을 클릭하거나 위에서 이름으로 찾으세요.")


# 탭 대신 '보기 선택'을 쓴다: 선택한 화면만 그려서, 숨겨진 탭 안에서 지도가 빈 화면으로
# 나오는 문제를 피하고 화면 전환도 빨라진다.
VIEWS = ["구청별", "면적별", "준공연도별", "아파트 검색"]
try:
    view = st.segmented_control("보기", VIEWS, default=VIEWS[0], key="view", label_visibility="collapsed")
except AttributeError:  # 구버전 Streamlit
    view = st.radio("보기", VIEWS, horizontal=True, key="view", label_visibility="collapsed")
view = view or VIEWS[0]
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ── 구청별 ───────────────────────────────────────────────────
if view == "구청별":
    render_group_tab(ap, "sigungu", how, n_recent, unit_label, key="gu",
                     sort_by_value=True, overall=overall_recent)
    st.divider()
    render_map_tab()

# ── 면적별 ───────────────────────────────────────────────────
if view == "면적별":
    st.caption(
        "면적대는 ‘전용면적’(공급면적 아님) 기준으로, 단지 내 세대별 전용면적의 평균으로 단지를 분류한 값입니다 "
        "(사용량 테이블이 단지 단위라 세대별 면적과 직접 연결할 수 없음)."
    )
    render_group_tab(ap, "area_band", how, n_recent, unit_label,
                     order=AREA_LABELS + [AREA_NONE], key="area", overall=overall_recent)

# ── 준공연도별 ───────────────────────────────────────────────
if view == "준공연도별":
    mode = st.radio("묶음 단위", ["10년 단위(연대)", "5년 단위", "연도별"], horizontal=True)
    ap_b = ap.copy()
    ap_b["built_grp"] = make_built_group(ap_b["built_year"], mode)
    render_group_tab(ap_b, "built_grp", how, n_recent, unit_label, key="built", overall=overall_recent)

# ── 아파트 검색 ──────────────────────────────────────────────
if view == "아파트 검색":
    kw = st.text_input("아파트명 검색 (일부만 입력해도 됨)", placeholder="예: 범어, 센트럴, 푸르지오")
    cand = master[master["apartment_code"].isin(ap_all["apartment_code"].unique())]
    if kw.strip():
        cand = cand[cand["apartment_name"].fillna("").str.contains(kw.strip(), case=False, regex=False)]
    else:
        cand = cand.iloc[0:0]

    if not kw.strip():
        st.info("아파트명을 입력하면 검색 결과가 나옵니다.")
    elif cand.empty:
        st.warning("검색 결과가 없습니다. (선택한 상품·기간 기준으로 12개월 데이터가 있는 단지만 표시)")
    else:
        cand = cand.sort_values(["sigungu", "apartment_name"]).copy()

        def label(r):
            yr = f"{int(r.built_year)}년" if pd.notna(r.built_year) else "준공 미상"
            hh = f"{int(r.households):,}세대" if pd.notna(r.households) else "세대수 미상"
            return f"{r.apartment_name} · {r.sigungu} · {yr} · {hh} ({r.apartment_code})"

        cand["label"] = cand.apply(label, axis=1)
        picked = st.multiselect(
            f"검색 결과 {len(cand)}건 — 비교할 단지 선택 (최대 8개)", cand["label"].tolist(),
            default=cand["label"].tolist()[:1], max_selections=len(SERIES),
        )
        codes = cand[cand["label"].isin(picked)]["apartment_code"].tolist()
        if codes:
            render_apartments(codes)
