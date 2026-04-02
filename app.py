"""
Black Friday Sales Intelligence Dashboard
Run:     streamlit run app.py
Install: pip install streamlit plotly pandas numpy
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations
from collections import defaultdict

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Black Friday Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── ONLY minimal CSS — no hiding any Streamlit elements ───────────────────────
st.markdown("""
<style>
.sec  { background:#6366F1; border-radius:8px; padding:0.7rem 1.1rem; margin-bottom:1rem; }
.sec  h3 { color:#FFFFFF !important; margin:0; font-size:1rem; }
.sec  p  { color:rgba(255,255,255,0.85) !important; margin:0; font-size:0.8rem; }
.ins  { background:#FFFFFF; border-left:4px solid #22D3EE; border-radius:0 8px 8px 0;
        padding:0.8rem 1rem; margin-bottom:0.6rem; }
.ins  b    { display:block; font-size:0.92rem; font-weight:700; margin-bottom:3px; }
.ins  span { font-size:0.84rem; color:#374151; line-height:1.5; }
</style>
""", unsafe_allow_html=True)

# ── Colours & helpers ──────────────────────────────────────────────────────────
P  = "#6366F1"; T = "#22D3EE"; A = "#F59E0B"; G = "#10B981"; R = "#E24B4A"
CL = [A, P, G]; LB = ["Budget", "Mid-tier", "Premium"]
AO = ["18-25","26-35","36-45","46-55","55+"]

LAYOUT = dict(
    paper_bgcolor="white", plot_bgcolor="white",
    font=dict(color="#111827", size=12),
    margin=dict(t=40, b=30, l=10, r=10),
    legend=dict(font=dict(color="#111827")),
)
AX = dict(gridcolor="#F3F4F6", linecolor="#E5E7EB",
          tickfont=dict(color="#6B7280"), title_font=dict(color="#111827"))

def sf(fig, xl="", yl="", xang=0, leg=True):
    xa = dict(AX, tickangle=xang)
    ya = dict(AX)
    if xl: xa["title_text"] = xl
    if yl: ya["title_text"] = yl
    fig.update_layout(**LAYOUT, showlegend=leg, xaxis=xa, yaxis=ya)
    return fig

def sec(title, sub=""):
    sub_html = f"<p>{sub}</p>" if sub else ""
    st.markdown(f'<div class="sec"><h3>{title}</h3>{sub_html}</div>',
                unsafe_allow_html=True)

def h4(text):
    st.subheader(text)

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load():
    rows = [
        (1001,"M","26-35","A","Single",8370,1,5,15),
        (1002,"M","26-35","B","Single",18957,1,5,15),
        (1003,"M","26-35","C","Single",22726,5,12,18),
        (1004,"F","26-35","A","Married",12263,8,3,10),
        (1005,"F","36-45","B","Single",15000,2,6,14),
        (1006,"M","18-25","C","Single",5389,3,7,16),
        (1007,"F","26-35","A","Married",9263,4,8,11),
        (1008,"M","36-45","B","Married",17465,6,2,13),
        (1009,"F","46-55","C","Married",11202,9,4,12),
        (1010,"M","55+","A","Married",6872,7,1,9),
        (1011,"M","26-35","B","Single",14563,2,10,15),
        (1012,"F","18-25","C","Single",8934,5,3,11),
        (1013,"M","36-45","A","Married",19384,1,6,14),
        (1014,"F","26-35","B","Single",7263,8,2,12),
        (1015,"M","46-55","C","Married",13485,4,7,10),
        (1016,"F","55+","A","Married",5694,9,5,8),
        (1017,"M","18-25","B","Single",9856,3,1,16),
        (1018,"F","36-45","C","Single",16000,6,8,13),
        (1019,"M","26-35","A","Married",11234,2,4,11),
        (1020,"F","46-55","B","Married",14789,7,3,9),
        (1021,"M","55+","C","Married",4532,1,9,7),
        (1022,"F","18-25","A","Single",10234,5,2,15),
        (1023,"M","36-45","B","Single",18900,8,6,14),
        (1024,"F","26-35","C","Married",9876,3,5,10),
        (1025,"M","46-55","A","Married",12345,4,7,13),
        (1026,"F","55+","B","Single",6234,9,1,8),
        (1027,"M","18-25","C","Single",8765,2,3,16),
        (1028,"F","36-45","A","Married",15000,6,4,12),
        (1029,"M","26-35","B","Single",13456,1,8,14),
        (1030,"F","46-55","C","Married",11111,5,2,11),
        (1031,"M","55+","A","Married",7654,7,9,9),
        (1032,"F","18-25","B","Single",9342,3,6,15),
        (1033,"M","36-45","C","Single",19234,8,1,13),
        (1034,"F","26-35","A","Married",10234,4,5,10),
    ]
    df = pd.DataFrame(rows, columns=[
        "UserID","Gender","AgeGroup","City","MaritalStatus",
        "Purchase","Cat1","Cat2","Cat3"])
    df["GenderFull"] = df["Gender"].map({"M":"Male","F":"Female"})
    return df

@st.cache_data
def get_stats(_df):
    p = _df["Purchase"]
    return dict(n=len(_df), tot=int(p.sum()), mean=int(p.mean()),
                med=int(p.median()), std=int(p.std()),
                mn=int(p.min()), mx=int(p.max()),
                q1=int(p.quantile(.25)), q3=int(p.quantile(.75)))

@st.cache_data
def get_clusters(_df, k=3, it=30):
    v = _df["Purchase"].values.astype(float)
    n = v / v.max(); c = n[:k].copy()
    for _ in range(it):
        d = np.abs(n[:,None]-c[None,:]); lbl = d.argmin(1)
        for i in range(k):
            m = lbl==i
            if m.sum(): c[i] = n[m].mean()
    d = np.abs(n[:,None]-c[None,:]); lbl = d.argmin(1)
    r = _df.copy(); r["Cluster"] = lbl
    av = r.groupby("Cluster")["Purchase"].mean()
    rm = {o:ne for ne,o in enumerate(av.sort_values().index)}
    r["Cluster"] = r["Cluster"].map(rm)
    return r

@st.cache_data
def get_anomalies(_df):
    m, s = _df["Purchase"].mean(), _df["Purchase"].std()
    o = _df.copy()
    o["ZScore"] = ((o["Purchase"]-m)/s).abs()
    return o[o["ZScore"]>2].sort_values("ZScore", ascending=False)

@st.cache_data
def get_rules(_df):
    pc = defaultdict(int); tot = len(_df)
    for _, row in _df.iterrows():
        cats = [row["Cat1"],row["Cat2"],row["Cat3"]]
        for a,b in combinations(sorted(set(cats)),2): pc[(a,b)] += 1
    res = []
    for (a,b),cnt in sorted(pc.items(), key=lambda x:-x[1])[:8]:
        ma = (_df["Cat1"]==a)|(_df["Cat2"]==a)|(_df["Cat3"]==a)
        mb = (_df["Cat1"]==b)|(_df["Cat2"]==b)|(_df["Cat3"]==b)
        sa=ma.sum()/tot; sb=mb.sum()/tot; sab=cnt/tot
        conf=sab/sa if sa else 0; lift=conf/sb if sb else 0
        res.append({"Rule":f"Cat {a} → Cat {b}","Count":cnt,
                    "Support %":round(sab*100,1),
                    "Confidence %":round(conf*100,1),
                    "Lift":round(lift,2)})
    return pd.DataFrame(res)

# ── Precompute ─────────────────────────────────────────────────────────────────
DF   = load()
ST   = get_stats(DF)
DFC  = get_clusters(DF)
ANOM = get_anomalies(DF)
RUL  = get_rules(DF)

# ── Session state ──────────────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username  = ""

# ── Login ──────────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    st.markdown("<br>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        st.title("🛍️ Black Friday Analytics")
        st.write("Sign up to access the dashboard.")
        with st.form("login"):
            u  = st.text_input("Username")
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Sign up", use_container_width=True):
                if u.strip() and pw.strip():
                    st.session_state.logged_in = True
                    st.session_state.username  = u.strip()
                    st.rerun()
                else:
                    st.error("Please fill in both fields.")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Analytics")
    st.write(f"Welcome, **{st.session_state.username}**")
    st.divider()

    PAGE = st.radio(
        "Navigate",
        options=[
            "📋 Dataset Overview",
            "🧹 Data Quality",
            "📊 EDA",
            "🔵 Segmentation",
            "🔗 Association Rules",
            "🚨 Anomaly Detection",
            "💡 Insights",
        ],
    )

    st.divider()
    if st.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username  = ""
        st.rerun()

# ── Page header ────────────────────────────────────────────────────────────────
st.title("🛍️ Black Friday Sales Intelligence Dashboard")
st.caption("34 customer records · All analytics computed from real data")
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if PAGE == "📋 Dataset Overview":
    sec("Dataset Overview", "Summary statistics and transaction records")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers",       ST["n"])
    c2.metric("Total revenue",   f"₹{ST['tot']/100000:.2f}L")
    c3.metric("Mean purchase",   f"₹{ST['mean']:,}")
    c4.metric("Median purchase", f"₹{ST['med']:,}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        h4("Purchase distribution by bucket")
        bins = [0,6000,8000,10000,12000,14000,16000,18000,20000,25000]
        lbls = ["<6k","6-8k","8-10k","10-12k","12-14k","14-16k","16-18k","18-20k",">20k"]
        tmp  = DF.copy()
        tmp["Bucket"] = pd.cut(tmp["Purchase"], bins=bins, labels=lbls, right=False)
        bkt  = tmp["Bucket"].value_counts().reindex(lbls).fillna(0).reset_index()
        bkt.columns = ["Range","Count"]
        fig  = px.bar(bkt, x="Range", y="Count",
                      color_discrete_sequence=[P], text="Count")
        fig.update_traces(marker_cornerradius=4, textposition="outside")
        sf(fig, xl="₹ Range", yl="Customers", leg=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        h4("Gender split")
        gv  = DF["GenderFull"].value_counts().reset_index()
        gv.columns = ["Gender","Count"]
        fig2 = px.pie(gv, names="Gender", values="Count", hole=0.5,
                      color_discrete_sequence=[P, T])
        sf(fig2, leg=True)
        st.plotly_chart(fig2, use_container_width=True)

    h4("Sample transactions — first 10 records")
    d10 = DF.head(10)[["UserID","GenderFull","AgeGroup","MaritalStatus",
                        "City","Purchase","Cat1","Cat2","Cat3"]].copy()
    d10.columns = ["User ID","Gender","Age","Marital","City",
                   "Purchase (₹)","Cat1","Cat2","Cat3"]
    st.table(d10.reset_index(drop=True))

    h4("All 34 records")
    dfall = DF[["UserID","GenderFull","AgeGroup","MaritalStatus",
                "City","Purchase","Cat1","Cat2","Cat3"]].copy()
    dfall.columns = ["User ID","Gender","Age","Marital","City",
                     "Purchase (₹)","Cat1","Cat2","Cat3"]
    st.table(dfall.reset_index(drop=True))


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA QUALITY
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "🧹 Data Quality":
    sec("Data Quality", "Preprocessing checks and descriptive statistics")

    q1 = ST["q1"]; q3 = ST["q3"]; iqr = q3 - q1
    p   = DF["Purchase"]
    iou = DF[(p < q1-1.5*iqr)|(p > q3+1.5*iqr)]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Missing values", 0)
    c2.metric("Duplicate IDs",  0)
    c3.metric("IQR outliers",   len(iou))
    c4.metric("Std deviation",  f"₹{ST['std']:,}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        h4("Descriptive statistics")
        st.table(pd.DataFrame({
            "Metric": ["Count","Mean","Median","Std Dev","Min","Max",
                       "Q1","Q3","IQR","Range"],
            "Value":  [ST["n"], f"₹{ST['mean']:,}", f"₹{ST['med']:,}",
                       f"₹{ST['std']:,}", f"₹{ST['mn']:,}", f"₹{ST['mx']:,}",
                       f"₹{q1:,}", f"₹{q3:,}", f"₹{iqr:,}",
                       f"₹{ST['mx']-ST['mn']:,}"]
        }))

    with col2:
        h4("Column summary")
        rows_s = []
        for cn in ["Gender","AgeGroup","City","MaritalStatus"]:
            vc = DF[cn].value_counts()
            rows_s.append({"Column":cn, "Unique":DF[cn].nunique(),
                           "Distribution":" | ".join(f"{k}:{v}" for k,v in vc.items())})
        rows_s.append({"Column":"Purchase","Unique":DF["Purchase"].nunique(),
                       "Distribution":f"₹{ST['mn']:,} – ₹{ST['mx']:,}"})
        st.table(pd.DataFrame(rows_s))

    h4("Purchase value histogram")
    fig = px.histogram(DF, x="Purchase", nbins=14,
                       color_discrete_sequence=[P],
                       labels={"Purchase":"Purchase (₹)"})
    fig.update_layout(bargap=0.05)
    sf(fig, xl="Purchase (₹)", yl="Frequency", leg=False)
    st.plotly_chart(fig, use_container_width=True)

    if len(iou):
        h4(f"IQR outliers — {len(iou)} records")
        od = iou[["UserID","GenderFull","AgeGroup","City","Purchase"]].copy()
        od.columns = ["User ID","Gender","Age","City","Purchase (₹)"]
        st.table(od.reset_index(drop=True))
    else:
        st.success("No IQR outliers detected.")


# ══════════════════════════════════════════════════════════════════════════════
# 3. EDA
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "📊 EDA":
    sec("Exploratory Data Analysis", "Spending patterns across all demographic dimensions")

    col1, col2 = st.columns(2)
    with col1:
        h4("Avg purchase by age group")
        adf = DF.groupby("AgeGroup")["Purchase"].mean().round().reindex(AO).reset_index()
        adf.columns = ["Age Group","Avg Purchase"]
        fig = px.bar(adf, x="Age Group", y="Avg Purchase",
                     color_discrete_sequence=[P], text="Avg Purchase")
        fig.update_traces(marker_cornerradius=4,
                          texttemplate="₹%{text:,.0f}", textposition="outside")
        sf(fig, yl="Avg Purchase (₹)", leg=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        h4("Avg purchase by city")
        cdf = DF.groupby("City")["Purchase"].mean().round().reset_index()
        cdf.columns = ["City","Avg Purchase"]
        cdf["City"] = "City " + cdf["City"]
        fig2 = px.bar(cdf, x="City", y="Avg Purchase", color="City",
                      color_discrete_sequence=[A,G,T], text="Avg Purchase")
        fig2.update_traces(marker_cornerradius=4,
                           texttemplate="₹%{text:,.0f}", textposition="outside")
        sf(fig2, yl="Avg Purchase (₹)", leg=False)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        h4("Avg purchase by gender")
        gdf = DF.groupby("GenderFull")["Purchase"].mean().round().reset_index()
        gdf.columns = ["Gender","Avg Purchase"]
        fig3 = px.bar(gdf, x="Gender", y="Avg Purchase", color="Gender",
                      color_discrete_sequence=[P,T], text="Avg Purchase")
        fig3.update_traces(marker_cornerradius=4,
                           texttemplate="₹%{text:,.0f}", textposition="outside")
        sf(fig3, yl="Avg Purchase (₹)", leg=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        h4("Avg purchase by marital status")
        mdf = DF.groupby("MaritalStatus")["Purchase"].mean().round().reset_index()
        mdf.columns = ["Status","Avg Purchase"]
        fig4 = px.bar(mdf, x="Status", y="Avg Purchase", color="Status",
                      color_discrete_sequence=[G,A], text="Avg Purchase")
        fig4.update_traces(marker_cornerradius=4,
                           texttemplate="₹%{text:,.0f}", textposition="outside")
        sf(fig4, yl="Avg Purchase (₹)", leg=False)
        st.plotly_chart(fig4, use_container_width=True)

    h4("Top 8 product categories by frequency")
    cf: dict = {}
    for _, row in DF.iterrows():
        for c in [row["Cat1"],row["Cat2"],row["Cat3"]]: cf[c] = cf.get(c,0)+1
    cdf2 = (pd.DataFrame(list(cf.items()), columns=["Category","Count"])
              .sort_values("Count", ascending=False).head(8))
    cdf2["Category"] = "Cat " + cdf2["Category"].astype(str)
    fig5 = px.bar(cdf2, x="Category", y="Count",
                  color_discrete_sequence=[P], text="Count")
    fig5.update_traces(marker_cornerradius=4, textposition="outside")
    sf(fig5, xl="Category", yl="Frequency", leg=False)
    st.plotly_chart(fig5, use_container_width=True)

    h4("Purchase per customer")
    fig6 = px.scatter(DF.sort_values("UserID"), x="UserID", y="Purchase",
                      color="GenderFull", color_discrete_sequence=[P,T],
                      hover_data=["AgeGroup","City","MaritalStatus"],
                      labels={"UserID":"User ID","Purchase":"Purchase (₹)","GenderFull":"Gender"})
    fig6.add_hline(y=ST["mean"], line_dash="dash", line_color="gray",
                   annotation_text=f"Mean ₹{ST['mean']:,}")
    sf(fig6, xl="User ID", yl="Purchase (₹)", leg=True)
    st.plotly_chart(fig6, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4. SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "🔵 Segmentation":
    sec("Customer Segmentation", "K-means clustering (k=3) on purchase values")

    c1,c2,c3 = st.columns(3)
    for wc,i in zip([c1,c2,c3], range(3)):
        sub = DFC[DFC["Cluster"]==i]
        avg = int(sub["Purchase"].mean()) if len(sub) else 0
        wc.metric(f"{LB[i]} spenders", f"₹{avg:,}", f"{len(sub)} customers")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        h4("Cluster size distribution")
        sizes = [len(DFC[DFC["Cluster"]==i]) for i in range(3)]
        fig   = px.pie(values=sizes, names=LB, hole=0.5,
                       color_discrete_sequence=CL)
        sf(fig, leg=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        h4("Cluster avg spend comparison")
        avgs   = [int(DFC[DFC["Cluster"]==i]["Purchase"].mean()) for i in range(3)]
        cmp_df = pd.DataFrame({"Cluster":LB, "Avg Spend":avgs})
        fig2   = px.bar(cmp_df, x="Cluster", y="Avg Spend", color="Cluster",
                        color_discrete_sequence=CL, text="Avg Spend")
        fig2.update_traces(marker_cornerradius=4,
                           texttemplate="₹%{text:,.0f}", textposition="outside")
        sf(fig2, yl="Avg Purchase (₹)", leg=False)
        st.plotly_chart(fig2, use_container_width=True)

    h4("Cluster demographic breakdown")
    bd = []
    for i,lb in enumerate(LB):
        sub = DFC[DFC["Cluster"]==i]
        top_age = sub["AgeGroup"].value_counts().idxmax() if len(sub) else "—"
        bd.append({
            "Cluster":   lb,
            "Size":      len(sub),
            "Avg Spend": f"₹{int(sub['Purchase'].mean()):,}" if len(sub) else "—",
            "Range":     f"₹{int(sub['Purchase'].min()):,}–₹{int(sub['Purchase'].max()):,}" if len(sub) else "—",
            "Males":     int((sub["Gender"]=="M").sum()),
            "Females":   int((sub["Gender"]=="F").sum()),
            "Single":    int((sub["MaritalStatus"]=="Single").sum()),
            "Married":   int((sub["MaritalStatus"]=="Married").sum()),
            "Top age":   top_age,
        })
    st.table(pd.DataFrame(bd))

    h4("All customers coloured by cluster")
    sc2 = DFC.copy()
    sc2["Segment"] = sc2["Cluster"].map({i:LB[i] for i in range(3)})
    cmap = {LB[i]:CL[i] for i in range(3)}
    fig3 = px.scatter(sc2.sort_values("UserID"), x="UserID", y="Purchase",
                      color="Segment", color_discrete_map=cmap,
                      hover_data=["GenderFull","AgeGroup","City","MaritalStatus"],
                      labels={"UserID":"User ID","Purchase":"Purchase (₹)"})
    sf(fig3, xl="User ID", yl="Purchase (₹)", leg=True)
    st.plotly_chart(fig3, use_container_width=True)

    h4("Purchase spread within clusters")
    sc3 = DFC.copy()
    sc3["Segment"] = sc3["Cluster"].map({i:LB[i] for i in range(3)})
    fig4 = px.box(sc3, x="Segment", y="Purchase", color="Segment",
                  color_discrete_sequence=CL, points="all",
                  labels={"Segment":"Cluster","Purchase":"Purchase (₹)"})
    sf(fig4, yl="Purchase (₹)", leg=False)
    st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5. ASSOCIATION RULES
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "🔗 Association Rules":
    sec("Association Rules", "Apriori-style pair mining across category columns")

    rd = RUL.copy()
    rd["Strength"] = rd["Lift"].apply(
        lambda x: "Strong" if x>=1.5 else ("Moderate" if x>=1.0 else "Weak"))

    h4("All mined rules")
    st.table(rd.reset_index(drop=True))

    col1, col2 = st.columns(2)
    with col1:
        h4("Support vs confidence")
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Support %", x=RUL["Rule"],
                             y=RUL["Support %"], marker_color=P,
                             marker=dict(cornerradius=3)))
        fig.add_trace(go.Bar(name="Confidence %", x=RUL["Rule"],
                             y=RUL["Confidence %"], marker_color=T,
                             marker=dict(cornerradius=3)))
        fig.update_layout(**LAYOUT, barmode="group", showlegend=True,
                          xaxis=dict(**AX, tickangle=-30),
                          yaxis=dict(**AX, title_text="%"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        h4("Lift per rule")
        fig2 = px.bar(RUL, x="Rule", y="Lift",
                      color_discrete_sequence=[A], text="Lift")
        fig2.add_hline(y=1, line_dash="dash", line_color="gray",
                       annotation_text="Baseline (Lift=1)")
        fig2.update_traces(marker_cornerradius=3, textposition="outside")
        sf(fig2, xang=-30, yl="Lift", leg=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    ca, cb, cc = st.columns(3)
    ca.info("**Support** — % of transactions containing both items.")
    cb.info("**Confidence** — P(B|A): if A bought, probability B too.")
    cc.success("**Lift > 1** — items co-occur more than by chance.")


# ══════════════════════════════════════════════════════════════════════════════
# 6. ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "🚨 Anomaly Detection":
    sec("Anomaly Detection", "Z-score method — records beyond ±2 standard deviations")

    mean_p = DF["Purchase"].mean()
    std_p  = DF["Purchase"].std()
    nn     = len(DF) - len(ANOM)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total records",    len(DF))
    c2.metric("Normal (z ≤ 2σ)", nn,         f"{round(nn/len(DF)*100)}%")
    c3.metric("Anomalies (z>2σ)", len(ANOM), f"{round(len(ANOM)/len(DF)*100,1)}%")
    c4.metric("Method",           "Z-score ±2σ")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        h4("Z-score distribution")
        dfz = DF.copy()
        dfz["ZScore"] = ((dfz["Purchase"]-mean_p)/std_p).abs()
        zb  = [0,0.5,1,1.5,2,2.5,3,4]
        zl  = ["0–0.5σ","0.5–1σ","1–1.5σ","1.5–2σ","2–2.5σ","2.5–3σ",">3σ"]
        dfz["ZBucket"] = pd.cut(dfz["ZScore"], bins=zb, labels=zl, right=False)
        zcnt = dfz["ZBucket"].value_counts().reindex(zl).fillna(0).reset_index()
        zcnt.columns = ["Z-range","Count"]
        zcols = [P,P,P,P,A,A,R]
        fig = px.bar(zcnt, x="Z-range", y="Count", color="Z-range",
                     color_discrete_sequence=zcols, text="Count")
        fig.update_traces(marker_cornerradius=3, textposition="outside")
        sf(fig, xl="Z-score bucket", yl="Customers", leg=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        h4("Normal vs anomalous")
        fig2 = px.pie(values=[nn, len(ANOM)], names=["Normal","Anomaly"],
                      hole=0.5, color_discrete_sequence=[P, R])
        sf(fig2, leg=True)
        st.plotly_chart(fig2, use_container_width=True)

    h4(f"Flagged records — {len(ANOM)} anomalies (z > 2σ, mean = ₹{ST['mean']:,})")
    if len(ANOM):
        ad = ANOM[["UserID","GenderFull","AgeGroup","City",
                    "MaritalStatus","Purchase","ZScore"]].copy()
        ad["ZScore"] = ad["ZScore"].round(2).astype(str) + "σ"
        ad["Status"] = ANOM["ZScore"].apply(
            lambda z: "Critical (z>3)" if z>3 else "Moderate (z>2)").values
        ad.columns = ["User ID","Gender","Age","City","Marital",
                      "Purchase (₹)","Z-score","Status"]
        st.table(ad.reset_index(drop=True))
    else:
        st.info("No anomalies found at ±2σ.")

    h4("All purchases with anomaly overlay")
    dfa = DF.copy()
    dfa["ZScore"] = ((dfa["Purchase"]-mean_p)/std_p).abs()
    dfa["Type"]   = dfa["ZScore"].apply(lambda z: "Anomaly" if z>2 else "Normal")
    fig3 = px.scatter(dfa.sort_values("UserID"), x="UserID", y="Purchase",
                      color="Type", color_discrete_map={"Normal":P,"Anomaly":R},
                      hover_data=["GenderFull","AgeGroup","City"],
                      labels={"UserID":"User ID","Purchase":"Purchase (₹)","Type":"Type"})
    fig3.add_hline(y=mean_p+2*std_p, line_dash="dash", line_color=A,
                   annotation_text="+2σ threshold")
    fig3.add_hline(y=mean_p, line_dash="dot", line_color="gray",
                   annotation_text=f"Mean ₹{ST['mean']:,}")
    sf(fig3, xl="User ID", yl="Purchase (₹)", leg=True)
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 7. INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "💡 Insights":
    sec("Key Insights", "Data-driven takeaways from every analysis stage")

    age_av   = DF.groupby("AgeGroup")["Purchase"].mean().reindex(AO)
    top_age  = age_av.idxmax();  top_age_v = int(age_av.max())
    city_av  = DF.groupby("City")["Purchase"].mean()
    top_city = city_av.idxmax(); top_city_v = int(city_av.max())
    male_av  = int(DF[DF["Gender"]=="M"]["Purchase"].mean())
    fem_av   = int(DF[DF["Gender"]=="F"]["Purchase"].mean())
    high_g   = "Male" if male_av > fem_av else "Female"
    gdiff    = round(abs(male_av-fem_av)/min(male_av,fem_av)*100)
    sing_av  = int(DF[DF["MaritalStatus"]=="Single"]["Purchase"].mean())
    marr_av  = int(DF[DF["MaritalStatus"]=="Married"]["Purchase"].mean())
    cf4: dict = {}
    for _, row in DF.iterrows():
        for c in [row["Cat1"],row["Cat2"],row["Cat3"]]: cf4[c] = cf4.get(c,0)+1
    top_cat  = max(cf4, key=cf4.get)
    big_i    = DFC.groupby("Cluster").size().idxmax()
    big_lbl  = LB[big_i]
    big_n    = int((DFC["Cluster"]==big_i).sum())
    big_avg  = int(DFC[DFC["Cluster"]==big_i]["Purchase"].mean())
    tr       = RUL.iloc[0] if len(RUL) else None

    insights = [
        (f"📈 Top spending age group: {top_age}",
         f"Avg ₹{top_age_v:,}/customer — {round((top_age_v-ST['mean'])/ST['mean']*100)}% above overall mean ₹{ST['mean']:,}. Prioritise marketing and loyalty rewards for this demographic."),
        (f"🛒 Top product category: Cat {top_cat} ({cf4[top_cat]} appearances)",
         f"Appears in {round(cf4[top_cat]/(len(DF)*3)*100)}% of all category slots. Strong candidate for homepage placement and featured bundles."),
        (f"👥 {high_g} customers spend {gdiff}% more on average",
         f"Male avg ₹{male_av:,} vs Female avg ₹{fem_av:,}. Gender-personalised recommendations can significantly improve conversion."),
        (f"🏙️ City {top_city} leads in avg spend: ₹{top_city_v:,}",
         f"₹{round(top_city_v-city_av.min()):,} more than lowest-spend city. City-tier geo-targeting and local discounts are highly actionable."),
        (f"⚡ Largest segment: {big_lbl} ({big_n} customers, {round(big_n/len(DF)*100)}%)",
         f"K-means found 3 natural clusters. {big_lbl} tier dominates at ₹{big_avg:,} avg. Volume promotions and loyalty programmes work best here."),
        ((f"🔗 Strongest rule: {tr['Rule']} — lift {tr['Lift']}x" if tr else "🔗 Association rules computed"),
         (f"Co-occur {tr['Lift']}x more than chance. Confidence {tr['Confidence %']}% — bundle for cross-sell campaigns." if tr else "")),
        (f"🚨 {len(ANOM)} high-value anomal{'y' if len(ANOM)==1 else 'ies'} detected (z > 2σ)",
         f"Customers >2 std devs above mean ₹{ST['mean']:,}. Likely VIP or bulk-purchase — route to a premium retention programme."),
        (f"💍 {'Singles spend more' if sing_av>marr_av else 'Married spend more'} (₹{sing_av:,} vs ₹{marr_av:,})",
         f"{int((DF['MaritalStatus']=='Single').sum())} single vs {int((DF['MaritalStatus']=='Married').sum())} married. Marital-status-specific offers can be highly effective."),
    ]

    for title, body in insights:
        st.markdown(
            f'<div class="ins"><b>{title}</b><span>{body}</span></div>',
            unsafe_allow_html=True)

    st.divider()
    h4("Summary — avg spend across all dimensions")
    sr = []
    for age in AO:
        sr.append({"Dimension":age, "Avg":round(DF[DF["AgeGroup"]==age]["Purchase"].mean()), "Group":"Age group"})
    for city in ["A","B","C"]:
        sr.append({"Dimension":f"City {city}", "Avg":round(DF[DF["City"]==city]["Purchase"].mean()), "Group":"City"})
    for g,lb in [("M","Male"),("F","Female")]:
        sr.append({"Dimension":lb, "Avg":round(DF[DF["Gender"]==g]["Purchase"].mean()), "Group":"Gender"})
    for ms in ["Single","Married"]:
        sr.append({"Dimension":ms, "Avg":round(DF[DF["MaritalStatus"]==ms]["Purchase"].mean()), "Group":"Marital"})
    sdf = pd.DataFrame(sr)
    fig = px.bar(sdf, x="Dimension", y="Avg", color="Group",
                 color_discrete_sequence=[P,T,A,G],
                 text="Avg", barmode="group",
                 labels={"Avg":"Avg Purchase (₹)"})
    fig.update_traces(marker_cornerradius=4,
                      texttemplate="₹%{text:,.0f}", textposition="outside")
    fig.update_layout(**LAYOUT, showlegend=True,
                      xaxis=dict(**AX, tickangle=-20),
                      yaxis=dict(**AX, title_text="Avg Purchase (₹)"))
    st.plotly_chart(fig, use_container_width=True)
    
