"""
공동주택 세대당 가스 사용량 분석 대시보드 (Streamlit + BigQuery)

실행:  streamlit run app.py
필요:  pip install streamlit pandas altair google-cloud-bigquery db-dtypes pyarrow

인증(둘 중 하나)
  1) .streamlit/secrets.toml 에 [gcp_service_account] 로 서비스계정 JSON 내용을 넣기
  2) 또는 로컬에서  gcloud auth application-default login  후 실행
"""

import altair as alt
import numpy as np
import pandas as pd
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
           b.built_year, a.avg_area, a.units, i.households, i.address
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
               ANY_VALUE(address) AS address
        FROM `{DATASET}.pbi_ref_apt_info` GROUP BY 1
    ) i USING (apartment_code)
    """
    m = get_client().query(sql).to_dataframe()
    m = m.drop_duplicates("apartment_code").reset_index(drop=True)
    m["built_year"] = pd.to_numeric(m["built_year"], errors="coerce")
    m["avg_area"] = pd.to_numeric(m["avg_area"], errors="coerce")
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
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)


def fmt_table(df: pd.DataFrame):
    count_cols = [c for c in ["단지수", "세대수"] if c in df.columns]
    sty = df.style.format("{:,.1f}", na_rep="-")
    if count_cols:
        sty = sty.format("{:,.0f}", na_rep="-", subset=count_cols)
    return sty


# ──────────────────────────────────────────────────────────────
# 화면 공통 컴포넌트
# ──────────────────────────────────────────────────────────────
def render_group_tab(ap, group_col, how, n_recent, unit_label, order=None, key=""):
    res = summarize(ap, group_col, how, n_recent, order)
    if res is None:
        st.warning("12개월이 모두 채워진 연도 데이터가 없습니다. 필터를 확인하세요.")
        return
    tbl, roll, cal = res
    avg_col = [c for c in tbl.columns if c.startswith("최근") and "개년 평균" in c][0]

    left, right = st.columns([1, 1])
    with left:
        st.markdown(f"**{avg_col}** · {unit_label}")
        chart_df = tbl[[avg_col]].reset_index()
        chart_df.columns = ["그룹", "값"]
        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("그룹:N", sort=list(tbl.index), title=None),
                y=alt.Y("값:Q", title=unit_label),
                tooltip=["그룹", alt.Tooltip("값:Q", format=",.1f")],
            )
            .properties(height=320)
        )
        show_alt(chart)
    with right:
        st.markdown(f"**연도별 추이** · {unit_label}")
        yearly = tbl[cal].T
        yearly.index.name = "연도"
        st.line_chart(yearly, height=320)

    st.markdown("**연도별 값 / 최근 평균 / 3개년 이동평균**")
    show_df(fmt_table(tbl))

    with st.expander("3개년 이동평균 추이 (각 연도 기준 직전 3년 평균)"):
        r = roll.T
        r.index.name = "연도"
        st.line_chart(r.dropna(how="all"), height=300)
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

    products = sorted(fact_all["product"].unique())
    default_products = [p for p in products if p.endswith(("/개별난방용", "/취사난방용"))
                        and p.startswith("가스요금")]
    sel_products = st.multiselect(
        "상품(용도)", products, default=default_products,
        help="취사용은 난방이 없어 사용량이 훨씬 작고, 중앙난방용은 단지 단위 계량이라 "
             "세대당 값이 크게 달라서 기본에서 제외했습니다.",
    )
    sigungus = sorted(master["sigungu"].dropna().unique())
    sel_sigungu = st.multiselect("구·군 (비우면 전체)", sigungus)

    st.caption(
        "연간 값은 해당 연도 12개월이 모두 있는 단지만 계산합니다. "
        "‘최근12개월’은 데이터의 마지막 월까지의 12개월입니다."
    )
    if st.button("데이터 새로고침"):
        st.cache_data.clear()
        st.rerun()

if not sel_products:
    st.info("사이드바에서 상품(용도)을 하나 이상 선택하세요.")
    st.stop()

val_col = "usage_m3" if unit == "m³" else "usage_mj"
per_month = basis.startswith("월평균")
unit_label = f"{unit}/세대/{'월' if per_month else '년'}"

fact = fact_all[fact_all["product"].isin(sel_products)]
ap_all = build_apt_period(fact, val_col, per_month)
ap_all = ap_all.merge(master, on="apartment_code", how="inner")
ap = ap_all[ap_all["sigungu"].isin(sel_sigungu)] if sel_sigungu else ap_all

last_month = fact_all["sales_ym"].max().strftime("%Y-%m")
c1, c2, c3 = st.columns(3)
c1.metric("분석 단지 수", f"{ap['apartment_code'].nunique():,}")
c2.metric("데이터 최종 월", last_month)
c3.metric("단위", unit_label)

tab_gu, tab_area, tab_built, tab_search = st.tabs(
    ["구청별", "면적별", "준공연도별", "아파트 검색"]
)

# ── 구청별 ───────────────────────────────────────────────────
with tab_gu:
    render_group_tab(ap, "sigungu", how, n_recent, unit_label, key="gu")

# ── 면적별 ───────────────────────────────────────────────────
with tab_area:
    st.caption(
        "면적대는 단지 내 세대별 전용면적의 평균으로 단지를 분류한 값입니다 "
        "(사용량 테이블이 단지 단위라 세대별 면적과 직접 연결할 수 없음)."
    )
    render_group_tab(ap, "area_band", how, n_recent, unit_label,
                     order=AREA_LABELS + [AREA_NONE], key="area")

# ── 준공연도별 ───────────────────────────────────────────────
with tab_built:
    mode = st.radio("묶음 단위", ["10년 단위(연대)", "5년 단위", "연도별"], horizontal=True)
    ap_b = ap.copy()
    ap_b["built_grp"] = make_built_group(ap_b["built_year"], mode)
    render_group_tab(ap_b, "built_grp", how, n_recent, unit_label, key="built")

# ── 아파트 검색 ──────────────────────────────────────────────
with tab_search:
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
            f"검색 결과 {len(cand)}건 — 비교할 단지 선택", cand["label"].tolist(),
            default=cand["label"].tolist()[:1],
        )
        codes = cand[cand["label"].isin(picked)]["apartment_code"].tolist()

        if codes:
            names = master.set_index("apartment_code")["apartment_name"]
            sel = ap_all[ap_all["apartment_code"].isin(codes)].copy()
            sel["단지"] = sel["apartment_code"].map(names) + " (" + sel["apartment_code"] + ")"

            # 연도별 표
            pv = sel.pivot_table(index="단지", columns="period", values="per_hh", aggfunc="first")
            cal = calendar_cols(pv.columns)
            tbl = pv[cal + ([TTM] if TTM in pv.columns else [])].copy()
            if cal:
                recent = cal[-n_recent:]
                tbl[f"최근{len(recent)}개년 평균({recent[0]}~{recent[-1]})"] = pv[recent].mean(axis=1)
                if len(cal) >= 3:
                    tbl[f"3개년 이동평균({cal[-1]})"] = pv[cal].T.rolling(3, min_periods=3).mean().T[cal[-1]]

            st.markdown(f"**연도별 세대당 사용량** · {unit_label}")
            if tbl.empty or not cal:
                st.warning("선택한 단지는 12개월이 모두 채워진 연도가 없습니다.")
            else:
                show_df(fmt_table(tbl))
                yearly = pv[cal].T
                yearly.index.name = "연도"
                st.bar_chart(yearly, height=300)

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
                    own = pv.loc[names[code] + " (" + code + ")", recent].mean() if (
                        names[code] + " (" + code + ")") in pv.index else np.nan

                    def ref(key, grp):
                        t = tabs_by[key]
                        cols = [c for c in recent if c in t.columns]
                        return t.loc[grp, cols].mean() if grp in t.index and cols else np.nan

                    g_gu, g_ar, g_de = ref("sigungu", r["sigungu"]), ref("area_band", r["area_band"]), ref("decade", dec)
                    rows.append({
                        "단지": f"{names[code]} ({code})",
                        "이 단지": own,
                        f"{r['sigungu']} 평균": g_gu,
                        f"면적대 평균({r['area_band']})": g_ar,
                        f"준공 연대 평균({dec})": g_de,
                        "구 평균 대비(%)": (own / g_gu - 1) * 100 if pd.notna(own) and pd.notna(g_gu) else np.nan,
                    })
                cmp_df = pd.DataFrame(rows).set_index("단지")
                show_df(cmp_df.style.format("{:,.1f}", na_rep="-"))

            # 월별 추이
            st.markdown(f"**월별 세대당 사용량** · {unit}/세대/월 (선택한 상품 합산)")
            mf = fact[fact["apartment_code"].isin(codes)]
            mm = mf.groupby(["apartment_code", "sales_ym"], as_index=False).agg(
                u=(val_col, "sum"), c=("metered_row_cnt", "sum")
            )
            mm["per_hh"] = mm["u"] / mm["c"]
            mm["단지"] = mm["apartment_code"].map(names) + " (" + mm["apartment_code"] + ")"
            st.line_chart(mm.pivot_table(index="sales_ym", columns="단지", values="per_hh"), height=300)

            info = master[master["apartment_code"].isin(codes)][
                ["apartment_code", "apartment_name", "sigungu", "address", "built_year", "households", "units", "avg_area", "area_band"]
            ].rename(columns={
                "apartment_code": "코드", "apartment_name": "단지명", "sigungu": "구·군", "address": "주소",
                "built_year": "준공연도", "households": "세대수(정보)", "units": "세대수(면적등록)",
                "avg_area": "평균 전용면적(㎡)", "area_band": "면적대",
            })
            with st.expander("단지 기본 정보"):
                show_df(info, hide_index=True)
