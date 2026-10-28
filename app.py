from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ml_model import load_data, train_and_score

st.set_page_config(page_title="ChurnVision AI", page_icon="📉", layout="wide")

ACCENT, CYAN, RED, AMBER, GREEN = "#7C5CFF", "#22D3EE", "#F43F5E", "#F59E0B", "#10B981"
RISK_COLORS = {"Düşük": GREEN, "Orta": AMBER, "Yüksek": RED}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 3rem; max-width: 1280px; }
.stApp { background: radial-gradient(1200px 500px at 85% -10%, rgba(124,92,255,.14), transparent 60%), #0B0F1A; }

/* Sidebar */
[data-testid="stSidebar"] { background: #0E1424; border-right: 1px solid #1B2438; }
[data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
.brand { display:flex; align-items:center; gap:12px; margin: 0 0 6px 4px; }
.brand-logo { width:38px; height:38px; border-radius:11px; display:grid; place-items:center; font-size:19px;
  background: linear-gradient(135deg,#7C5CFF,#22D3EE); box-shadow: 0 6px 20px rgba(124,92,255,.45); }
.brand-name { font-weight:800; font-size:1.15rem; letter-spacing:-.02em; color:#fff; line-height:1.1; }
.brand-sub { font-size:.68rem; letter-spacing:.14em; color:#6B7794; font-weight:600; }
.nav-label { font-size:.68rem; letter-spacing:.12em; color:#56617D; font-weight:700; margin:22px 0 6px 6px; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: 4px; }
[data-testid="stSidebar"] [role="radiogroup"] label { padding: 10px 14px; border-radius: 10px; width:100%; cursor:pointer;
  border: 1px solid transparent; transition: all .15s ease; }
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display:none; }
[data-testid="stSidebar"] [role="radiogroup"] label p { font-size:.93rem; font-weight:500; color:#9AA6C3; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background:#151D33; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background: linear-gradient(90deg, rgba(124,92,255,.22), rgba(124,92,255,.05));
  border-color: rgba(124,92,255,.45); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p { color:#fff; font-weight:600; }
.side-card { margin-top:28px; padding:14px; border-radius:12px; background:#121A2E; border:1px solid #1D2842; font-size:.78rem; color:#8D99B8; line-height:1.5; }
.side-card b { color:#E5E7EB; }

/* Page header */
.page-title { font-size:1.9rem; font-weight:800; letter-spacing:-.03em; color:#fff; margin:0; }
.page-sub { color:#8D99B8; margin: 2px 0 22px; font-size:.95rem; }
.pill { display:inline-block; font-size:.7rem; font-weight:600; padding:3px 10px; border-radius:99px; margin-left:10px; vertical-align:middle;
  background:rgba(16,185,129,.12); color:#34D399; border:1px solid rgba(16,185,129,.3); }

/* KPI cards */
.kpi { background: linear-gradient(180deg,#131B30,#0F1626); border:1px solid #1D2842; border-radius:16px; padding:18px 20px; height:100%; }
.kpi-label { font-size:.76rem; color:#8D99B8; font-weight:600; letter-spacing:.04em; text-transform:uppercase; }
.kpi-value { font-size:2rem; font-weight:800; color:#fff; letter-spacing:-.03em; margin-top:6px; line-height:1.1; }
.kpi-note { font-size:.78rem; color:#6B7794; margin-top:6px; }
.kpi-bar { height:3px; border-radius:3px; margin-bottom:14px; width:36px; }

/* Cards / sections */
.section { font-size:1.05rem; font-weight:700; color:#fff; margin: 26px 0 10px; letter-spacing:-.01em; }
[data-testid="stDataFrame"] { border:1px solid #1D2842; border-radius:14px; overflow:hidden; }
[data-testid="stPlotlyChart"] { background:#0F1626; border:1px solid #1D2842; border-radius:16px; padding:8px; }
div[data-testid="stMetric"] { background:#0F1626; border:1px solid #1D2842; border-radius:14px; padding:14px 16px; }
.stButton button, .stDownloadButton button { border-radius:10px; font-weight:600; border:1px solid #2A3757; }
.callout { padding:14px 16px; border-radius:12px; background:rgba(124,92,255,.08); border:1px solid rgba(124,92,255,.28); color:#C9CFE4; font-size:.88rem; line-height:1.55; }
.chip { display:inline-block; padding:4px 11px; border-radius:99px; font-size:.78rem; font-weight:600; margin:0 6px 6px 0; background:#151D33; border:1px solid #243051; color:#C9CFE4; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Model eğitiliyor…")
def get_results():
    df = load_data()
    return train_and_score(df)


def kpi(label: str, value: str, note: str = "", color: str = ACCENT) -> str:
    return (f"<div class='kpi'><div class='kpi-bar' style='background:{color}'></div><div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value'>{value}</div><div class='kpi-note'>{note}</div></div>")


def style(fig: go.Figure, height: int = 330) -> go.Figure:
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=height,
                      margin=dict(l=12, r=12, t=44, b=12), font=dict(family="Inter", color="#C9CFE4"),
                      title=dict(font=dict(size=14, color="#fff")), legend_title_text="")
    fig.update_xaxes(gridcolor="#1B2438", zeroline=False)
    fig.update_yaxes(gridcolor="#1B2438", zeroline=False)
    return fig


def header(title: str, sub: str, pill: str = "") -> None:
    st.markdown(f"<h1 class='page-title'>{title}{f'<span class=pill>{pill}</span>' if pill else ''}</h1><p class='page-sub'>{sub}</p>", unsafe_allow_html=True)


scored, metrics, importance = get_results()
high = scored[scored["risk_level"] == "Yüksek"]

with st.sidebar:
    st.markdown("<div class='brand'><div class='brand-logo'>📉</div><div><div class='brand-name'>ChurnVision AI</div>"
                "<div class='brand-sub'>CUSTOMER INTELLIGENCE</div></div></div><div class='nav-label'>MENÜ</div>", unsafe_allow_html=True)
    page = st.radio("Menü", ["📊  Genel Bakış", "👥  Müşteri Riski", "🔎  Müşteri Detayı", "🧠  Model & Veri"], label_visibility="collapsed")
    st.markdown(f"<div class='side-card'><b>Veri seti</b><br>IBM Telco Customer Churn<br>{len(scored):,} müşteri · {scored.shape[1]-4} değişken<br><br>"
                f"<b>Model</b><br>Gradient Boosting · AUC {metrics['auc']:.2f}</div>", unsafe_allow_html=True)

if page.endswith("Genel Bakış"):
    header("Genel Bakış", "Kayıp riski taşıyan müşterileri erken tespit et, gelir kaybını önle.", "Gerçek veri")
    revenue_risk = float(high["MonthlyCharges"].sum())
    c = st.columns(4)
    c[0].markdown(kpi("Toplam müşteri", f"{len(scored):,}", "Telekom abonesi", ACCENT), unsafe_allow_html=True)
    c[1].markdown(kpi("Gerçek churn oranı", f"%{metrics['base_rate']*100:.1f}", "Geçmiş kayıp müşteriler", CYAN), unsafe_allow_html=True)
    c[2].markdown(kpi("Yüksek riskli müşteri", f"{len(high):,}", "Risk puanı ≥ %60", RED), unsafe_allow_html=True)
    c[3].markdown(kpi("Risk altındaki aylık gelir", f"${revenue_risk:,.0f}", "Yüksek riskli grubun faturası", AMBER), unsafe_allow_html=True)

    st.markdown("<div class='section'>Churn nerede yoğunlaşıyor?</div>", unsafe_allow_html=True)
    a, b = st.columns(2)
    by_contract = scored.groupby("Contract")["ChurnFlag"].mean().mul(100).reset_index(name="churn")
    f = px.bar(by_contract, x="Contract", y="churn", title="Sözleşme tipine göre churn (%)", text=by_contract["churn"].round(1).astype(str) + "%")
    f.update_traces(marker_color=ACCENT, marker_cornerradius=8, textposition="outside")
    a.plotly_chart(style(f).update_layout(xaxis_title=None, yaxis_title=None, yaxis_range=[0, 52]), width="stretch")
    by_tenure = scored.groupby("TenureGroup", observed=True)["ChurnFlag"].mean().mul(100).reset_index(name="churn")
    f = px.bar(by_tenure, x="TenureGroup", y="churn", title="Müşteri yaşına göre churn (%)", text=by_tenure["churn"].round(1).astype(str) + "%")
    f.update_traces(marker_color=CYAN, marker_cornerradius=8, textposition="outside")
    b.plotly_chart(style(f).update_layout(xaxis_title=None, yaxis_title=None, yaxis_range=[0, 52]), width="stretch")

    a, b = st.columns([1, 1.5])
    counts = scored["risk_level"].value_counts().reindex(["Düşük", "Orta", "Yüksek"]).reset_index()
    counts.columns = ["Risk", "Müşteri"]
    f = px.pie(counts, names="Risk", values="Müşteri", hole=0.65, color="Risk", color_discrete_map=RISK_COLORS, title="Risk dağılımı")
    f.update_traces(textinfo="percent", marker=dict(line=dict(color="#0F1626", width=3)))
    a.plotly_chart(style(f, 360), width="stretch")
    with b:
        st.markdown("<div style='height:6px'></div><b style='color:#fff'>En riskli 8 müşteri</b>", unsafe_allow_html=True)
        top = scored.head(8)[[ "customerID", "Contract", "tenure", "MonthlyCharges", "risk"]]
        st.dataframe(top, hide_index=True, width="stretch", column_config={
            "customerID": "Müşteri", "Contract": "Sözleşme", "tenure": st.column_config.NumberColumn("Ay", format="%d"),
            "MonthlyCharges": st.column_config.NumberColumn("Aylık ücret", format="$%.2f"),
            "risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=1, format="percent")})

elif page.endswith("Müşteri Riski"):
    header("Müşteri Riski", "Filtrele, önceliklendir, kampanya listesini CSV olarak indir.")
    f1, f2, f3 = st.columns([1, 1.2, 1])
    levels = f1.multiselect("Risk seviyesi", ["Düşük", "Orta", "Yüksek"], default=["Yüksek", "Orta"])
    contracts = f2.multiselect("Sözleşme", sorted(scored["Contract"].unique()), default=sorted(scored["Contract"].unique()))
    min_risk = f3.slider("Minimum risk (%)", 0, 100, 0, 5)
    view = scored[scored["risk_level"].astype(str).isin(levels) & scored["Contract"].isin(contracts) & (scored["risk"] * 100 >= min_risk)]
    st.caption(f"{len(view):,} müşteri · toplam aylık gelir ${view['MonthlyCharges'].sum():,.0f}")
    cols = ["customerID", "risk", "risk_level", "Contract", "tenure", "MonthlyCharges", "InternetService", "PaymentMethod", "TechSupport"]
    st.dataframe(view[cols], hide_index=True, width="stretch", height=460, column_config={
        "customerID": "Müşteri", "risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=1, format="percent"),
        "risk_level": "Seviye", "Contract": "Sözleşme", "tenure": "Ay", "MonthlyCharges": st.column_config.NumberColumn("Aylık ücret", format="$%.2f"),
        "InternetService": "İnternet", "PaymentMethod": "Ödeme", "TechSupport": "Teknik destek"})
    st.download_button("⬇️  Listeyi CSV indir", view[cols].to_csv(index=False).encode("utf-8-sig"), "churn_risk_listesi.csv", "text/csv")

elif page.endswith("Müşteri Detayı"):
    header("Müşteri Detayı", "Tek bir müşterinin risk puanını ve profilini incele.")
    cid = st.selectbox("Müşteri seç (en riskliler üstte)", scored["customerID"].tolist())
    row = scored[scored["customerID"] == cid].iloc[0]
    a, b = st.columns([1, 1.4])
    color = RISK_COLORS[str(row["risk_level"])]
    g = go.Figure(go.Indicator(mode="gauge+number", value=row["risk"] * 100, number=dict(suffix="%", font=dict(size=44, color="#fff")),
        title=dict(text=f"Kayıp riski · {row['risk_level']}", font=dict(size=14)),
        gauge=dict(axis=dict(range=[0, 100], tickcolor="#56617D"), bar=dict(color=color, thickness=0.28), bgcolor="#151D33", borderwidth=0,
                   steps=[dict(range=[0, 30], color="#12261F"), dict(range=[30, 60], color="#2A2314"), dict(range=[60, 100], color="#2B1620")])))
    a.plotly_chart(style(g, 320), width="stretch")
    with b:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        m = st.columns(3)
        m[0].metric("Müşteri yaşı", f"{int(row['tenure'])} ay")
        m[1].metric("Aylık ücret", f"${row['MonthlyCharges']:.2f}")
        m[2].metric("Toplam harcama", f"${row['TotalCharges']:,.0f}")
        chips = [f"📄 {row['Contract']}", f"🌐 {row['InternetService']}", f"💳 {row['PaymentMethod']}", f"🛠 Teknik destek: {row['TechSupport']}", f"🔒 Online güvenlik: {row['OnlineSecurity']}"]
        st.markdown("".join(f"<span class='chip'>{c}</span>" for c in chips), unsafe_allow_html=True)
        tips = []
        if row["Contract"] == "Month-to-month": tips.append("Aylık sözleşmeli: yıllık plana geçiş için indirim teklif et.")
        if row["TechSupport"] == "No" and row["InternetService"] != "No": tips.append("Teknik destek yok: ücretsiz deneme paketi sun.")
        if row["PaymentMethod"] == "Electronic check": tips.append("Elektronik çek: otomatik ödemeye geçişi teşvik et.")
        if row["tenure"] <= 12: tips.append("İlk yıl müşterisi: onboarding ve memnuniyet araması planla.")
        st.markdown("<div class='callout'><b>Önerilen aksiyonlar</b><br>" + ("<br>".join("• " + t for t in tips) or "• Belirgin bir risk faktörü yok; standart iletişime devam et.") + "</div>", unsafe_allow_html=True)

else:
    header("Model & Veri", "Şeffaflık: model neyi, ne kadar iyi tahmin ediyor?")
    c = st.columns(4)
    c[0].markdown(kpi("ROC-AUC", f"{metrics['auc']:.3f}", "Hold-out test seti", ACCENT), unsafe_allow_html=True)
    c[1].markdown(kpi("Doğruluk", f"%{metrics['accuracy']*100:.1f}", f"{metrics['n_test']:,} test müşterisi", CYAN), unsafe_allow_html=True)
    c[2].markdown(kpi("Precision", f"%{metrics['precision']*100:.1f}", "'Ayrılacak' dediklerinin isabeti", GREEN), unsafe_allow_html=True)
    c[3].markdown(kpi("Recall", f"%{metrics['recall']*100:.1f}", "Ayrılanların yakalanma oranı", AMBER), unsafe_allow_html=True)
    a, b = st.columns(2)
    imp = importance.head(8).iloc[::-1]
    f = px.bar(imp, x="importance", y="feature", orientation="h", title="En etkili değişkenler (permutation importance)")
    f.update_traces(marker_color=ACCENT, marker_cornerradius=6)
    a.plotly_chart(style(f, 360).update_layout(xaxis_title="AUC kaybı", yaxis_title=None), width="stretch")
    cm = metrics["cm"]
    f = px.imshow(cm, text_auto=True, x=["Kaldı (tahmin)", "Ayrıldı (tahmin)"], y=["Kaldı (gerçek)", "Ayrıldı (gerçek)"],
                  color_continuous_scale=[[0, "#131B30"], [1, ACCENT]], title="Karışıklık matrisi")
    f.update_coloraxes(showscale=False)
    b.plotly_chart(style(f, 360), width="stretch")
    st.markdown(
        "<div class='callout'><b>Veri kaynağı:</b> IBM Telco Customer Churn örnek veri seti (7.043 müşteri, 21 sütun). "
        "<b>Yöntem:</b> %80/%20 stratified bölme; metrikler hiç görülmememiş test setinden. Tablodaki risk puanları 5-fold out-of-fold tahminlerdir "
        "(model hiçbir müşteriyi kendi puanını hesaplarken görmez).<br><b>Sınırlar:</b> Veri tek bir zaman kesitidir; zaman bazlı doğrulama yapılmamıştır. "
        "Kampanya etkisi ölçülmedi, nedensellik iddia edilmez.</div>", unsafe_allow_html=True)
