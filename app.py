import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
import cv2
from ultralytics import YOLO
import time

# Paths
ATTENDANCE_FOLDER = Path("Attendance")
ATTENDANCE_FILE = ATTENDANCE_FOLDER / "attendance.csv"
WEIGHTS_FILE = Path("weights/best.pt")

# Ensure folder exists
ATTENDANCE_FOLDER.mkdir(exist_ok=True)

# Columns
COLUMNS = ["Roll No", "Name", "Section", "Role", "Date", "Time", "Status"]

# Ensure CSV exists
if not ATTENDANCE_FILE.exists():
    pd.DataFrame(columns=COLUMNS).to_csv(ATTENDANCE_FILE, index=False)

# Load attendance
def load_attendance():
    return pd.read_csv(ATTENDANCE_FILE)

# Save attendance
def save_attendance(roll_no, name, section, role, status):
    df = load_attendance()
    now = datetime.now()
    new_entry = pd.DataFrame([{
        "Roll No": roll_no,
        "Name": name,
        "Section": section,
        "Role": role,
        "Date": now.strftime("%Y-%m-%d"),
        "Time": now.strftime("%H:%M:%S"),
        "Status": status
    }])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(ATTENDANCE_FILE, index=False)

# Run YOLO live detection for 3 seconds in Streamlit
def run_live_attendance():
    if not WEIGHTS_FILE.exists():
        st.error(f"Model file not found: {WEIGHTS_FILE}")
        return

    model = YOLO(str(WEIGHTS_FILE))
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Could not open webcam.")
        return

    st_frame = st.empty()
    detected_names = set()

    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                name = model.names[cls_id]

                if name not in detected_names:
                    roll_no = cls_id + 1
                    section = "A"
                    role = "Student"
                    save_attendance(roll_no, name, section, role, "Present")
                    detected_names.add(name)

        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        st_frame.image(annotated_frame, channels="RGB", use_container_width=True)

        if time.time() - start_time > 3:
            break

    cap.release()

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Automated Attendance System", page_icon="🗒", layout="centered")
st.title("🗒 Automated Attendance System")

menu = st.sidebar.radio("📌 Navigation", ["Mark Attendance", "View Attendance", "Clear Attendance"])

if menu == "Mark Attendance":
    st.subheader("✅ Mark Attendance")
    with st.form("attendance_form"):
        roll_no = st.text_input("Roll Number")
        name = st.text_input("Full Name")
        section = st.text_input("Section")
        role = st.selectbox("Role", ["Student", "Teacher", "Employee", "Guest"])
        status = st.selectbox("Attendance Status", ["Present", "Absent", "Late"])
        submitted = st.form_submit_button("Submit")

        if submitted:
            if not roll_no.strip() or not name.strip() or not section.strip():
                st.error("⚠️ Please fill in all fields.")
            else:
                save_attendance(roll_no, name, section, role, status)
                st.success(f"✅ Attendance marked for {name} ({roll_no}) as {status}.")

    if st.button("📷 Mark Live Attendance"):
        run_live_attendance()

elif menu == "View Attendance":
    st.subheader("📋 Attendance Records")
    df = load_attendance()

    if df.empty:
        st.warning("No attendance records found.")
    else:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        with st.expander("🔍 Filter Options"):
            section_options = ["All"] + sorted(df["Section"].dropna().unique().tolist())
            selected_section = st.selectbox("Select Section", section_options)

            min_date = df["Date"].min()
            max_date = df["Date"].max()
            start_date, end_date = st.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )

            if selected_section != "All":
                df = df[df["Section"] == selected_section]

            df = df[
                (df["Date"] >= pd.to_datetime(start_date)) &
                (df["Date"] <= pd.to_datetime(end_date))
            ]

        st.dataframe(df.style.hide(axis="index"), use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Attendance CSV", csv, "attendance.csv", "text/csv")

elif menu == "Clear Attendance":
    st.subheader("⚠️ Clear All Attendance Data")
    if st.button("🗑 Clear Attendance Data"):
        pd.DataFrame(columns=COLUMNS).to_csv(ATTENDANCE_FILE, index=False)
        st.success("✅ All attendance data cleared.")
