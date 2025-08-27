"""
Schema-Locked Single-Pass Diabetes Report System
Main Streamlit application with forms-first intake and consultant-grade reporting
"""
import json
import uuid
import os
from datetime import datetime, date
from pathlib import Path

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# ────────────────────────────────────────────────────────────────────────────────
# 1) MUST be first Streamlit call
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Consultant Helper AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────────────────────
# 2) Secrets / env handling (supports Cloud + local .env)
# ────────────────────────────────────────────────────────────────────────────────
load_dotenv()  # local dev only; Streamlit Cloud ignores .env

def get_secret(name: str, default=None):
    # Prefer Streamlit Secrets on Cloud; fall back to OS env for local dev
    try:
        val = st.secrets.get(name)
    except Exception:
        val = None
    return val if val is not None else os.getenv(name, default)

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
CHROMA_PERSIST_DIRECTORY = get_secret("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
NHS_THEME_PRIMARY = get_secret("NHS_THEME_PRIMARY", "#005EB8")
NHS_THEME_SECONDARY = get_secret("NHS_THEME_SECONDARY", "#41B6E6")

if not OPENAI_API_KEY:
    st.error(
        "⚠️ **OPENAI_API_KEY** not configured.\n\n"
        "Add it in **Settings → Advanced settings → Secrets** as:\n\n"
        "```toml\nOPENAI_API_KEY = \"sk-...\"\n```\n"
        "Then rerun the app."
    )
    st.stop()

# ────────────────────────────────────────────────────────────────────────────────
# 3) App imports (modules from your project)
#    NOTE: We will lazy-import PDFParser later to avoid hard-crash if PyMuPDF is missing.
# ────────────────────────────────────────────────────────────────────────────────
from rules.schemas.report import PatientIntake, ReportOut, PDFExtraction
from agents.report_orchestrator import ReportOrchestrator
from rag.index_builder import RAGIndexBuilder
from rag.retriever import RAGRetriever
from rules import load_rules
from utils.formatters import render_text_report, create_clinical_snapshot
from utils.pdf import PDFGenerator
from ui.components import (
    create_conflict_resolver, create_clinical_snapshot_cards,
    create_download_button, create_urgent_banner, create_report_tabs,
    create_progress_tracker, create_validation_feedback, display_validation_warnings
)

# ────────────────────────────────────────────────────────────────────────────────
# 4) Session state
# ────────────────────────────────────────────────────────────────────────────────
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
for k, v in {
    'patient_data': {},
    'pdf_data': {},
    'conflicts': {},
    'generated_report': None,
    'report_text': None,
}.items():
    st.session_state.setdefault(k, v)

# ────────────────────────────────────────────────────────────────────────────────
# 5) Helpers
# ────────────────────────────────────────────────────────────────────────────────
def ensure_rag_index() -> bool:
    """Ensure RAG index exists, build if needed."""
    try:
        api_key = OPENAI_API_KEY
        if not api_key:
            st.error("❌ Missing `OPENAI_API_KEY`. Set it in Streamlit Secrets.")
            return False

        index_path = Path("data/rag")
        index_path.mkdir(parents=True, exist_ok=True)
        index_file = index_path / "index.faiss"
        if index_file.exists():
            st.sidebar.success("✅ Using existing knowledge base")
            return True

        with st.spinner("🔧 Building knowledge base (first time only)..."):
            # Quick connectivity check (OpenAI Python SDK v1.x)
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                _ = client.models.list()
            except Exception as e:
                msg = str(e)
                if "Incorrect API key" in msg:
                    raise ValueError("The provided API key is invalid or has been revoked.")
                elif "Rate limit" in msg:
                    raise ValueError("API rate limit exceeded. Please try again later.")
                else:
                    raise ValueError(f"API connection failed: {msg}")

            builder = RAGIndexBuilder(api_key=api_key)
            st.sidebar.info("📚 Building knowledge base index...")
            builder.build_index()

            if index_file.exists():
                st.sidebar.success("✅ Knowledge base built successfully")
                return True
            else:
                raise RuntimeError("Failed to create knowledge base index file")

    except Exception as e:
        # Avoid leaking the API key in UI
        err = str(e).replace(OPENAI_API_KEY, f"{OPENAI_API_KEY[:4]}...{OPENAI_API_KEY[-4:]}")
        st.error(f"### ❌ Failed to Build Knowledge Base\n\n{err}")
        return False

def render_patient_intake_form():
    """Render the main patient intake form"""
    st.header("📋 Patient Information")

    with st.form("patient_intake_form"):
        # Basic Demographics
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Full Name*", value=st.session_state.patient_data.get('name', ''))
            dob = st.date_input("Date of Birth*", value=date(1980, 1, 1))
            sex = st.selectbox("Sex*", ["Male", "Female", "Other"])
            diabetes_type = st.selectbox("Diabetes Type*", ["T1DM", "T2DM"])

        with col2:
            height_cm = st.number_input("Height (cm)*", min_value=100, max_value=250, value=175)
            weight_kg = st.number_input("Weight (kg)*", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
            diagnosis_date = st.text_input("Year of Diagnosis", placeholder="e.g. 2015")
            ethnicity = st.text_input("Ethnicity", placeholder="Optional")

        # Clinical Measurements
        st.subheader("Current Vital Signs")
        col3, col4, col5 = st.columns(3)

        with col3:
            bp_sys = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=250, value=120)
            bp_dia = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=80)

        with col4:
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=72)
            waist_cm = st.number_input("Waist Circumference (cm)", min_value=50, max_value=200, value=85)

        with col5:
            hypos_90d = st.number_input("Hypoglycemic episodes (last 90 days)", min_value=0, max_value=100, value=0)
            severe_hypos_90d = st.number_input("Severe hypoglycemic episodes (last 90 days)", min_value=0, max_value=20, value=0)
            dka_12m = st.number_input("DKA episodes (last 12 months)", min_value=0, max_value=10, value=0)

        # Laboratory Values
        st.subheader("Recent Laboratory Results")
        col6, col7 = st.columns(2)

        with col6:
            hba1c_pct = st.number_input("HbA1c (%)", min_value=4.0, max_value=20.0, value=7.5, step=0.1)
            fpg_mmol = st.number_input("Fasting Glucose (mmol/L)", min_value=2.0, max_value=30.0, value=5.5, step=0.1)
            ppg2h_mmol = st.number_input("2h Post-meal Glucose (mmol/L)", min_value=3.0, max_value=40.0, value=8.0, step=0.1)

        with col7:
            st.write("**Lipid Profile**")
            tc = st.number_input("Total Cholesterol (mmol/L)", min_value=2.0, max_value=15.0, value=5.0, step=0.1)
            ldl = st.number_input("LDL Cholesterol (mmol/L)", min_value=1.0, max_value=10.0, value=2.5, step=0.1)
            hdl = st.number_input("HDL Cholesterol (mmol/L)", min_value=0.5, max_value=3.0, value=1.2, step=0.1)
            tg = st.number_input("Triglycerides (mmol/L)", min_value=0.5, max_value=20.0, value=1.5, step=0.1)

        # Kidney function
        col8, col9 = st.columns(2)
        with col8:
            egfr = st.number_input("eGFR (mL/min/1.73m²)", min_value=5, max_value=150, value=90)
            creatinine_umol = st.number_input("Creatinine (μmol/L)", min_value=40, max_value=800, value=80)
        with col9:
            acr_mgmmol = st.number_input("ACR (mg/mmol)", min_value=0.0, max_value=100.0, value=2.0, step=0.1)

        # Medications
        st.subheader("Current Medications")
        medications_text = st.text_area(
            "List current diabetes medications (one per line)",
            placeholder="e.g.\nInsulin aspart - 1:10 ratio with meals\nMetformin - 500mg twice daily",
            height=100
        )

        # Lifestyle
        st.subheader("Lifestyle Factors")
        col10, col11 = st.columns(2)
        with col10:
            alcohol_units = st.number_input("Alcohol units per week", min_value=0, max_value=50, value=0)
            smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
            sleep_hours = st.number_input("Average sleep hours per night", min_value=3, max_value=12, value=7)
        with col11:
            activity_level = st.selectbox("Activity Level", ["Sedentary", "Light", "Moderate", "Very Active"])
            diet_pattern = st.selectbox("Diet Pattern", ["Standard", "Low Carb", "Mediterranean", "Vegetarian", "Other"])

        # Goals
        goals = st.text_area("Patient Goals", placeholder="e.g. Improve HbA1c, reduce hypoglycemia, lose weight")

        # Submit
        submitted = st.form_submit_button("💾 Save Patient Data", type="primary")
        if submitted:
            # Parse medications
            meds_list = []
            if medications_text:
                for line in medications_text.strip().split('\n'):
                    if line.strip():
                        parts = line.split(' - ')
                        med_data = {
                            'name': parts[0].strip(),
                            'dose': parts[1].strip() if len(parts) > 1 else '',
                            'schedule': '',
                            'notes': ''
                        }
                        meds_list.append(med_data)

            patient_data = {
                'uuid': st.session_state.session_id,
                'name': name,
                'dob': dob.isoformat(),
                'sex': sex,
                'diabetes_type': diabetes_type,
                'diagnosis_date': diagnosis_date or None,
                'ethnicity': ethnicity or None,
                'height_cm': float(height_cm),
                'weight_kg': float(weight_kg),
                'waist_cm': float(waist_cm) if waist_cm else None,
                'bp_sys': float(bp_sys) if bp_sys else None,
                'bp_dia': float(bp_dia) if bp_dia else None,
                'heart_rate': float(heart_rate) if heart_rate else None,
                'hypos_90d': int(hypos_90d),
                'severe_hypos_90d': int(severe_hypos_90d),
                'dka_12m': int(dka_12m),
                'meds': meds_list,
                'lifestyle': {
                    'alcohol_units': int(alcohol_units),
                    'smoking': smoking,
                    'sleep_hours': int(sleep_hours),
                    'activity_level': activity_level,
                    'diet_pattern': diet_pattern
                },
                'labs': {
                    'hba1c_pct': float(hba1c_pct),
                    'fpg_mmol': float(fpg_mmol),
                    'ppg2h_mmol': float(ppg2h_mmol),
                    'egfr': int(egfr),
                    'creatinine_umol': int(creatinine_umol),
                    'acr_mgmmol': float(acr_mgmmol),
                    'lipids': {
                        'tc': float(tc),
                        'ldl': float(ldl),
                        'hdl': float(hdl),
                        'tg': float(tg)
                    }
                },
                'goals': goals or None,
                'devices': [],
                'comorbidities': [],
                'screenings': {},
                'allergies': None,
                'notes': None
            }

            st.session_state.patient_data = patient_data
            st.success("✅ Patient data saved successfully!")
            st.rerun()

def render_pdf_import():
    """Render PDF import and conflict resolution"""
    st.header("📄 Lab Report Import (Optional)")

    uploaded_file = st.file_uploader(
        "Upload recent lab report (PDF)",
        type=['pdf'],
        help="Upload a PDF lab report to auto-fill lab values"
    )

    if uploaded_file is not None:
        if st.button("🔍 Process PDF"):
            with st.spinner("Processing PDF report..."):
                try:
                    # Lazy import so missing PyMuPDF doesn't crash the app
                    try:
                        from agents.pdf_parser import PDFParser
                    except Exception as e:
                        st.error(
                            "PyMuPDF (fitz) is not available. "
                            "Add `PyMuPDF==1.26.3` to requirements.txt and redeploy."
                        )
                        return

                    parser = PDFParser()
                    extraction = parser.parse_pdf(uploaded_file)

                    if getattr(extraction, "warnings", None):
                        for warning in extraction.warnings:
                            st.warning(f"⚠️ {warning}")

                    if getattr(extraction, "confidence", 0) > 0.1:
                        st.success(f"✅ PDF processed (confidence: {extraction.confidence:.1%})")
                        st.session_state.pdf_data = {
                            **getattr(extraction, "labs", {}),
                            **getattr(extraction, "vitals", {}),
                            '_confidence': {k: extraction.confidence for k in {**getattr(extraction, "labs", {}), **getattr(extraction, "vitals", {})}}
                        }
                        with st.expander("📊 Extracted Data", expanded=True):
                            if extraction.labs:
                                st.write("**Laboratory Values:**")
                                st.json(extraction.labs)
                            if extraction.vitals:
                                st.write("**Vital Signs:**")
                                st.json(extraction.vitals)
                        st.rerun()
                    else:
                        st.error("❌ Could not extract meaningful data from PDF")

                except Exception as e:
                    st.error(f"PDF processing failed: {str(e)}")

    # Show conflict resolution if needed
    if st.session_state.pdf_data and st.session_state.patient_data:
        st.session_state.conflicts = create_conflict_resolver(
            st.session_state.patient_data,
            st.session_state.pdf_data,
            st.session_state.pdf_data.get('_confidence', {})
        )

def render_report_generation():
    """Render report generation section"""
    st.header("🤖 Generate AI Report")

    if not st.session_state.patient_data:
        st.info("👆 Please complete patient information first")
        return

    # Show validation warnings
    warnings = create_validation_feedback(st.session_state.patient_data)
    if warnings:
        display_validation_warnings(warnings)

    # Clinical snapshot cards
    rules = load_rules()
    create_clinical_snapshot_cards(st.session_state.patient_data, rules)

    # Generate report button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Generate Consultant Report", type="primary", use_container_width=True):

            if not ensure_rag_index():
                st.error("Cannot generate report without knowledge base")
                return

            # Progress tracking
            create_progress_tracker("Collecting Data")

            with st.spinner("🧠 Generating AI report with clinical evidence..."):
                try:
                    orchestrator = ReportOrchestrator()

                    # Merge data sources
                    create_progress_tracker("Processing PDF")
                    merged_data = orchestrator.merge_data_sources(
                        st.session_state.patient_data,
                        st.session_state.pdf_data,
                        st.session_state.conflicts
                    )

                    # Generate report
                    create_progress_tracker("Generating Report")
                    report, errors = orchestrator.generate_report(merged_data)

                    if report:
                        create_progress_tracker("Finalizing")

                        st.session_state.generated_report = report
                        st.session_state.report_text = render_text_report(
                            report.dict(),
                            merged_data.dict(),
                            rules
                        )

                        save_dir = orchestrator.save_report(merged_data.uuid, report, merged_data)

                        st.success("✅ Report generated successfully!")
                        st.info(f"📁 Saved to: {save_dir}")
                        st.rerun()
                    else:
                        st.error("❌ Failed to generate report")
                        for error in errors:
                            st.error(f"• {error}")

                except Exception as e:
                    st.error(f"Report generation failed: {str(e)}")
                    st.exception(e)

def render_report_display():
    """Render the generated report"""
    if not st.session_state.generated_report:
        return

    st.header("📊 Generated Report")

    create_report_tabs(
        st.session_state.generated_report.dict(),
        st.session_state.patient_data,
        load_rules()
    )

    st.markdown("---")
    st.subheader("📥 Download Options")

    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.report_text:
            st.download_button(
                label="📄 Download Text Report",
                data=st.session_state.report_text,
                file_name=f"diabetes_report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )

    with col2:
        if st.button("📄 Generate PDF", use_container_width=True):
            with st.spinner("🔄 Creating PDF..."):
                try:
                    pdf_generator = PDFGenerator()
                    pdf_content = pdf_generator.generate_pdf_report(
                        st.session_state.patient_data,
                        st.session_state.generated_report.dict(),
                        load_rules()
                    )

                    st.download_button(
                        label="💾 Download PDF Report",
                        data=pdf_content,
                        file_name=f"diabetes_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

                except Exception as e:
                    st.error(f"PDF generation failed: {str(e)}")

def main():
    """Main application"""

    # Header
    st.title("🩺  Diabetes Consultant Helper AI")
    st.markdown("### Schema-Locked Single-Pass Clinical Report Generator")

    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        st.info(f"Session: {st.session_state.session_id[:8]}...")

        steps_completed = []
        steps_completed.append("✅ Patient Data" if st.session_state.patient_data else "⭕ Patient Data")
        steps_completed.append("✅ PDF Import" if st.session_state.pdf_data else "⭕ PDF Import")
        steps_completed.append("✅ Report Generated" if st.session_state.generated_report else "⭕ Report Generated")
        for step in steps_completed:
            st.write(step)

        if st.button("🔄 Start New Report"):
            for key in ['patient_data', 'pdf_data', 'conflicts', 'generated_report', 'report_text']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    # Main content
    tab1, tab2, tab3 = st.tabs(["📋 Patient Intake", "📄 PDF Import", "📊 Generate Report"])

    with tab1:
        render_patient_intake_form()

    with tab2:
        render_pdf_import()

    with tab3:
        render_report_generation()
        render_report_display()

    # Footer
    st.markdown("---")
    st.markdown(
        "🔬 **Powered by:** GPT-4o-mini + NICE/BDA Guidelines • "
        "🏥 **For:**  Digital Health • "
        f"⚡ **Built:** {datetime.now().strftime('%Y')}"
    )

if __name__ == "__main__":
    main()
