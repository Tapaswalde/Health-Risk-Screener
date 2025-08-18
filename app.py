import streamlit as st
import sqlite3
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# -----------------------------
# DB Helpers
# -----------------------------
DB_PATH = "health_agent.sqlite3"

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def exec_sql(conn, sql, params=()):
    cur = conn.cursor()
    cur.execute(sql, params)
    conn.commit()
    return cur

def query_df(conn, sql, params=()):
    return pd.read_sql_query(sql, conn, params=params)

# -----------------------------
# Initialize DB
# -----------------------------
def init_db():
    conn = get_conn()
    exec_sql(conn, """
    CREATE TABLE IF NOT EXISTS records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT,
        created_at TEXT,
        age INTEGER,
        height_cm REAL,
        weight_kg REAL,
        bmi REAL,
        dizzy_headache INTEGER,
        tiredness INTEGER,
        smoker INTEGER,
        diabetic INTEGER,
        cholesterol_high INTEGER,
        physically_active INTEGER,
        rule_score REAL,
        rule_band TEXT,
        ml_band TEXT,
        ml_confidence REAL,
        report TEXT
    )
    """)
    return conn

# -----------------------------
# Risk Score (Rule Based)
# -----------------------------
def rule_based_score(age,bmi,dizzy,tired,smoker,diabetic,chol,active):
    score=0
    if age>=60: score+=3
    elif age>=45: score+=2
    if bmi>=35: score+=3
    elif bmi>=30: score+=2
    elif bmi>=25: score+=1
    if dizzy: score+=2
    if tired: score+=1
    if smoker: score+=2
    if diabetic: score+=2
    if chol: score+=2
    if not active: score+=1
    band="Low"
    if score>=8: band="High"
    elif score>=4: band="Moderate"
    return score, band

def risk_badge(band:str)->str:
    colors={"Low":"🟢 Low","Moderate":"🟡 Moderate","High":"🔴 High"}
    return colors.get(band,"❓ Unknown")

# -----------------------------
# ML Model (KNN on SQLite Data)
# -----------------------------
def knn_predict(df, features):
    if len(df)<5:   # not enough data to train
        return None, None

    X=df[["age","bmi","dizzy_headache","tiredness","smoker",
          "diabetic","cholesterol_high","physically_active"]]
    y=df["rule_band"]  # use rule-based band as labels

    le=LabelEncoder()
    y_enc=le.fit_transform(y)

    knn=KNeighborsClassifier(n_neighbors=min(5,len(df)))
    knn.fit(X,y_enc)

    pred=knn.predict([features])[0]
    proba=knn.predict_proba([features])[0]

    predicted_band=le.inverse_transform([pred])[0]
    confidence=float(np.max(proba))

    return predicted_band, confidence

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config("🧬 Health Risk Agent","🧬",layout="wide")
conn=init_db()

# Sidebar navigation
menu=st.sidebar.radio("📌 Navigation",["Home","Form","Dashboard","Reports"])

# -----------------------------
# Home Page
# -----------------------------
if menu=="Home":
    st.title("🌿 Health Risk Screener (Rule + ML)")
    st.write("""
    👉 Answer simple questions.  
    👉 We calculate risk using **Rule-based logic** + **KNN ML** from past data.  
    👉 If ML confidence ≥70%, we use ML result. Otherwise, fallback to rule-based.  
    """)

# -----------------------------
# Form Page
# -----------------------------
elif menu=="Form":
    st.title("📝 Health Checkup Form")

    with st.form("health_form"):
        name=st.text_input("👤 Your Name")
        email=st.text_input("📧 Your Email")

        age=st.number_input("🎂 Age",18,100,30)
        height=st.number_input("📏 Height (cm)",100.0,220.0,165.0,step=0.5)
        weight=st.number_input("⚖️ Weight (kg)",30.0,150.0,60.0,step=0.5)

        st.markdown("### 🩺 Health Questions (Yes/No)")
        dizzy=st.radio("Do you often feel dizzy or get headaches?",["No","Yes"])=="Yes"
        tired=st.radio("Do you feel tired most of the time?",["No","Yes"])=="Yes"
        smoker=st.radio("Do you smoke?",["No","Yes"])=="Yes"
        diabetic=st.radio("Do you have diabetes?",["No","Yes"])=="Yes"
        chol=st.radio("Have you been told you have high cholesterol?",["No","Yes"])=="Yes"
        active=st.radio("Do you do physical work / exercise regularly?",["Yes","No"])=="Yes"

        submitted = st.form_submit_button("🔍 Analyze Health")

        if submitted:
            bmi=weight/((height/100)**2)
            score,rule_band=rule_based_score(age,bmi,dizzy,tired,smoker,diabetic,chol,active)

            # Get historical data
            df_all=query_df(conn,"SELECT * FROM records")
            features=[age,bmi,int(dizzy),int(tired),int(smoker),int(diabetic),int(chol),int(active)]
            ml_band,confidence=knn_predict(df_all,features)

            # Decide final
            final_band=rule_band
            source="Rule-based"
            if ml_band is not None and confidence>=0.7:
                final_band=ml_band
                source=f"ML (KNN, {confidence*100:.1f}% confident)"

            report=f"""
Health Risk Report ({datetime.utcnow().isoformat()} UTC)

👤 Name: {name}  
📧 Email: {email}  

Age: {age}, Height: {height} cm, Weight: {weight} kg, BMI: {bmi:.1f}  

➡️ Rule-based Risk: {rule_band}  
➡️ ML Predicted Risk: {ml_band if ml_band else 'Not enough data'}  
➡️ Final Risk Level: {final_band} ({source})

⚠️ This is a screening tool, not a medical diagnosis.
"""

            exec_sql(conn,"""INSERT INTO records 
            (name,email,created_at,age,height_cm,weight_kg,bmi,
            dizzy_headache,tiredness,smoker,diabetic,cholesterol_high,physically_active,
            rule_score,rule_band,ml_band,ml_confidence,report)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (name,email,datetime.utcnow().isoformat(),age,height,weight,bmi,
             int(dizzy),int(tired),int(smoker),int(diabetic),int(chol),int(active),
             score,rule_band,ml_band,float(confidence) if confidence else None,report))

            st.success("✅ Report Generated!")
            st.markdown(f"### Final Risk Level: {risk_badge(final_band)}")
            st.code(report,language="text")

# -----------------------------
# Dashboard
# -----------------------------
elif menu=="Dashboard":
    st.title("📊 Dashboard")

    df=query_df(conn,"SELECT * FROM records ORDER BY id DESC")
    if df.empty:
        st.info("No records yet.")
    else:
        # Show latest entry
        latest=df.iloc[0]
        st.markdown(f"### Latest Final Risk: {risk_badge(latest['rule_band'])}")

        st.dataframe(df[["created_at","name","email","age","bmi","rule_band","ml_band","ml_confidence"]])

# -----------------------------
# Reports
# -----------------------------
elif menu=="Reports":
    st.title("🗂 Reports")
    df=query_df(conn,"SELECT report FROM records ORDER BY id DESC LIMIT 1")
    if not df.empty:
        st.code(df.iloc[0]["report"],language="text")
    else:
        st.info("No reports yet.")
