from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import OUTPUT_DIR, TEMP_DIR, ProcessingSettings, SUPPORTED_VIDEO_EXTENSIONS
from modules.file_manager import OutputManager
from modules.report_generator import ReportGenerator
from modules.utils import configure_logging, ensure_dir, format_bytes, get_system_info, timestamp
from modules.video_processor import AnalysisResult, VideoProcessor
from modules.visualization import VisualizationBuilder


st.set_page_config(page_title="AI Shooting Range Scoring System", page_icon="🎯", layout="wide")


@st.cache_resource(show_spinner=False)
def get_logger() -> Any:
    return configure_logging(OUTPUT_DIR / "logs")


def initialize_session_state() -> None:
    st.session_state.setdefault("processing", False)
    st.session_state.setdefault("analysis_result", None)
    st.session_state.setdefault("reports", {})
    st.session_state.setdefault("video_path", None)
    st.session_state.setdefault("run_dir", None)
    st.session_state.setdefault("settings", ProcessingSettings())
    st.session_state.setdefault("dark_mode", True)
    st.session_state.setdefault("current_page", "🏠 Dashboard")
    st.session_state.setdefault("processing_thread", None)
    st.session_state.setdefault("processing_progress", 0.0)
    st.session_state.setdefault("processing_status", "Idle")
    st.session_state.setdefault("processing_elapsed", 0)
    st.session_state.setdefault("processing_frame", 0)
    st.session_state.setdefault("processing_total_frames", 0)
    st.session_state.setdefault("processing_fps", 0.0)
    st.session_state.setdefault("processing_queue", None)
    st.session_state.setdefault("processing_error", None)
    st.session_state.setdefault("banner_last_rerun", 0.0)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        .stApp { background: linear-gradient(135deg, #07111f 0%, #0b1120 45%, #111827 100%); }
        .block-container { padding-top: 1rem; padding-bottom: 3rem; max-width: 1550px; }
        [data-testid="stSidebar"] { background: rgba(11, 17, 32, 0.95); border-right: 1px solid rgba(148, 163, 184, 0.16); }
        [data-testid="stSidebarContent"] { padding-top: 0.8rem; }
        .glass-card {
            background: rgba(30, 41, 59, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            box-shadow: 0 20px 45px rgba(2, 8, 23, 0.28);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
        }
        .hero-card {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.28), rgba(34, 211, 238, 0.18));
            border: 1px solid rgba(96, 165, 250, 0.3);
            border-radius: 24px;
            padding: 1.35rem 1.4rem;
            box-shadow: 0 24px 60px rgba(2, 8, 23, 0.35);
        }
        .kpi-card {
            background: linear-gradient(135deg, rgba(17, 24, 39, 0.95), rgba(30, 41, 59, 0.92));
            border-radius: 20px;
            padding: 1rem 1rem 0.85rem;
            border: 1px solid rgba(148, 163, 184, 0.16);
            box-shadow: 0 18px 40px rgba(2, 8, 23, 0.22);
            min-height: 122px;
            transition: transform 180ms ease, border-color 180ms ease;
        }
        .kpi-card:hover { transform: translateY(-2px); border-color: rgba(96, 165, 250, 0.4); }
        .chip {
            display: inline-flex; align-items: center; gap: 0.45rem;
            border-radius: 999px; padding: 0.32rem 0.7rem; font-size: 0.8rem; font-weight: 600;
            background: rgba(34, 197, 94, 0.18); color: #bbf7d0; border: 1px solid rgba(34, 197, 94, 0.24);
        }
        .warning-chip { background: rgba(245, 158, 11, 0.16); color: #fde68a; border-color: rgba(245, 158, 11, 0.24); }
        .danger-chip { background: rgba(239, 68, 68, 0.16); color: #fecaca; border-color: rgba(239, 68, 68, 0.24); }
        .topbar {
            position: sticky; top: 0; z-index: 10; background: linear-gradient(135deg, rgba(7, 17, 31, 0.98), rgba(15, 23, 42, 0.92));
            backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
            padding: 0.8rem 1rem; margin-bottom: 0.95rem; border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 20px; box-shadow: 0 18px 44px rgba(2, 8, 23, 0.26);
        }
        .topbar-icon {
            width: 44px; height: 44px; border-radius: 14px; display: flex; align-items: center; justify-content: center;
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.28), rgba(34, 211, 238, 0.2));
            border: 1px solid rgba(125, 211, 252, 0.24); font-size: 1.08rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
        }
        .topbar-pill {
            display: inline-flex; align-items: center; gap: 0.45rem; border-radius: 999px; padding: 0.38rem 0.72rem;
            background: rgba(15, 23, 42, 0.92); border: 1px solid rgba(148, 163, 184, 0.16); color: #e2e8f0;
            font-size: 0.84rem; font-weight: 600; letter-spacing: 0.01em;
        }
        .topbar-title {
            font-size: 1.2rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.01em;
            margin-bottom: 0.12rem;
        }
        .topbar-subtitle {
            color: #94a3b8; font-size: 0.9rem;
        }
        .stButton > button {
            border-radius: 999px !important; border: none !important;
            background: linear-gradient(90deg, #3b82f6 0%, #22d3ee 100%) !important;
            color: white !important; font-weight: 700 !important; padding: 0.55rem 1rem !important;
            box-shadow: 0 12px 30px rgba(59, 130, 246, 0.22) !important;
        }
        [data-testid="stSidebar"] .stButton > button {
            justify-content: flex-start !important;
            width: 100% !important;
            border-radius: 14px !important;
            padding: 0.7rem 0.8rem !important;
            background: rgba(15, 23, 42, 0.8) !important;
            border: 1px solid rgba(148, 163, 184, 0.16) !important;
            color: #e2e8f0 !important;
            box-shadow: none !important;
            font-weight: 600 !important;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(59, 130, 246, 0.16) !important;
            border-color: rgba(96, 165, 250, 0.24) !important;
            transform: translateX(2px) !important;
        }
        [data-testid="stSidebar"] .stButton > button:focus {
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.25) !important;
        }
        .stButton > button:hover { transform: translateY(-1px); }
        .stDownloadButton > button, .stDownloadButton button {
            border-radius: 999px !important; border: 1px solid rgba(96, 165, 250, 0.28) !important;
            background: rgba(15, 23, 42, 0.82) !important; color: #f8fafc !important; padding: 0.55rem 0.9rem !important;
        }
        .stExpander > div[role="button"] {
            border-radius: 14px !important; border: 1px solid rgba(148, 163, 184, 0.16) !important; background: rgba(15, 23, 42, 0.7) !important;
        }
        .stTextInput > div > div > input, .stSelectbox > div > div > select, .stNumberInput input {
            background-color: rgba(15, 23, 42, 0.9) !important; color: #f8fafc !important; border-radius: 12px !important;
        }
        .stDataFrame, .stDataFrame > div { border-radius: 16px; overflow: hidden; }
        .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
        .stTabs [data-baseweb="tab"] { border-radius: 999px; padding: 0.45rem 0.85rem; }
        .stTabs [aria-selected="true"] { background: rgba(59, 130, 246, 0.2); border: 1px solid rgba(96, 165, 250, 0.28); }
        .sidebar-logo { font-size: 1.08rem; font-weight: 800; color: #f8fafc; letter-spacing: 0.02em; }
        .sidebar-subtitle { color: #94a3b8; font-size: 0.9rem; margin-top: -0.2rem; }
        .modern-nav { display: flex; flex-direction: column; gap: 0.45rem; margin-top: 0.85rem; }
        .nav-item { display: flex; align-items: center; gap: 0.65rem; padding: 0.7rem 0.8rem; border-radius: 14px; color: #e2e8f0; font-weight: 600; transition: all 180ms ease; border: 1px solid transparent; }
        .nav-item:hover { background: rgba(59, 130, 246, 0.12); border-color: rgba(96, 165, 250, 0.2); transform: translateX(2px); }
        .nav-item.active { background: linear-gradient(90deg, rgba(59, 130, 246, 0.2), rgba(34, 211, 238, 0.12)); border-color: rgba(96, 165, 250, 0.28); color: white; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03); }
        .nav-dot { width: 8px; height: 8px; border-radius: 999px; background: #60a5fa; box-shadow: 0 0 0 0 rgba(96, 165, 250, 0.45); animation: pulse 1.8s infinite; }
        @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(96, 165, 250, 0.4); } 70% { box-shadow: 0 0 0 7px rgba(96, 165, 250, 0); } 100% { box-shadow: 0 0 0 0 rgba(96, 165, 250, 0); } }
        .preview-shell { min-height: 300px; border-radius: 20px; border: 1px solid rgba(148, 163, 184, 0.16); background: radial-gradient(circle at top, rgba(59, 130, 246, 0.16), rgba(15, 23, 42, 0.9)); display: flex; align-items: center; justify-content: center; overflow: hidden; }
        .preview-placeholder { text-align: center; color: #cbd5e1; padding: 1rem; }
        .preview-ring { width: 70px; height: 70px; border-radius: 50%; border: 3px solid rgba(96, 165, 250, 0.22); border-top-color: #22d3ee; animation: spin 1.1s linear infinite; margin: 0 auto 0.8rem; }
        .status-pill { display: inline-flex; align-items: center; gap: 0.45rem; border-radius: 999px; padding: 0.32rem 0.68rem; background: rgba(15, 23, 42, 0.92); border: 1px solid rgba(148, 163, 184, 0.16); color: #e2e8f0; }
        .analyzing-pill { color: #a7f3d0; border-color: rgba(74, 222, 128, 0.22); background: rgba(6, 78, 59, 0.42); animation: pulse-soft 1.2s ease-in-out infinite; }
        .analyzing-dot { width: 8px; height: 8px; border-radius: 999px; background: #34d399; box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.45); animation: pulse-dot 1.2s infinite; }
        .progress-track { position: relative; height: 10px; border-radius: 999px; background: rgba(148, 163, 184, 0.16); overflow: hidden; border: 1px solid rgba(148, 163, 184, 0.12); }
        .progress-fill { position: absolute; inset: 0 auto 0 0; border-radius: inherit; background: linear-gradient(90deg, #3b82f6 0%, #22d3ee 50%, #8b5cf6 100%); transition: width 220ms ease; box-shadow: 0 0 20px rgba(34, 211, 238, 0.28); }
        .progress-shine { position: absolute; inset: 0; background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.28) 45%, transparent 100%); animation: shimmer 1.7s linear infinite; transform: translateX(-100%); }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes shimmer { 100% { transform: translateX(100%); } }
        @keyframes pulse-soft { 0%, 100% { transform: scale(1); opacity: 0.9; } 50% { transform: scale(1.02); opacity: 1; } }
        @keyframes pulse-dot { 0% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.45); } 70% { box-shadow: 0 0 0 7px rgba(52, 211, 153, 0); } 100% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); } }
        ::-webkit-scrollbar { width: 9px; }
        ::-webkit-scrollbar-thumb { background: rgba(148, 163, 184, 0.28); border-radius: 999px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_dashboard_stats(result: AnalysisResult | None) -> dict[str, Any]:
    if result is None:
        return {
            "videos_processed": 0,
            "targets_detected": 0,
            "shots_detected": 0,
            "average_accuracy": 0.0,
            "fps": 24.0,
            "processing_time": 0.0,
        }

    summary = result.summary or {}
    return {
        "videos_processed": 1,
        "targets_detected": int(summary.get("total_targets", result.total_targets)),
        "shots_detected": int(summary.get("total_shots", result.total_shots)),
        "average_accuracy": float(summary.get("accuracy", result.accuracy)),
        "fps": 24.0,
        "processing_time": round(float(summary.get("frame_count", result.frame_count)) / 24.0, 1),
    }


def start_background_analysis(video_path: Path, run_dir: Path, settings: ProcessingSettings) -> None:
    st.session_state["processing"] = True
    st.session_state["processing_progress"] = 0.0
    st.session_state["processing_status"] = "Preparing analysis"
    st.session_state["processing_elapsed"] = 0
    st.session_state["processing_frame"] = 0
    st.session_state["processing_total_frames"] = 0
    st.session_state["processing_fps"] = 0.0
    st.session_state["processing_error"] = None
    progress_queue: Queue = Queue()
    st.session_state["processing_queue"] = progress_queue

    def worker() -> None:
        logger = get_logger()
        processor = VideoProcessor(settings=settings)
        try:
            logger.info("Starting analysis for %s", video_path)

            def progress_callback(frame_idx: int, total: int, elapsed: float) -> None:
                progress_queue.put(
                    {
                        "processing_status": f"Analyzing frame {frame_idx}/{total}",
                        "processing_progress": round(min(1.0, frame_idx / max(total, 1)), 3),
                        "processing_elapsed": int(elapsed),
                        "processing_frame": frame_idx,
                        "processing_total_frames": total,
                        "processing_fps": round(frame_idx / max(elapsed, 1), 1),
                    }
                )

            result = processor.process_video(video_path, run_dir, progress_callback=progress_callback)
            report_generator = ReportGenerator(run_dir)
            reports = report_generator.generate(result.video_path, result.summary, result.shot_table, result.target_table)
            progress_queue.put(
                {
                    "analysis_result": result,
                    "reports": reports,
                    "processing_status": "Completed",
                    "processing_progress": 1.0,
                    "processing_elapsed": int(time.time() - time.time()),
                }
            )
            st.session_state["analysis_result"] = result
            st.session_state["reports"] = reports
            st.session_state["processing_status"] = "Completed"
            st.session_state["processing_progress"] = 1.0
        except Exception as exc:  # pragma: no cover - runtime error path
            progress_queue.put({"processing_status": "Failed", "processing_error": str(exc)})
            st.session_state["processing_status"] = "Failed"
            st.session_state["processing_error"] = str(exc)
        finally:
            progress_queue.put({"processing": False, "processing_thread": None})
            st.session_state["processing"] = False
            st.session_state["processing_thread"] = None

    thread = threading.Thread(target=worker, daemon=True)
    st.session_state["processing_thread"] = thread
    thread.start()


@st.fragment
def render_processing_banner() -> None:
    if not st.session_state.get("processing"):
        return

    progress_queue: Queue | None = st.session_state.get("processing_queue")
    received_update = False
    if progress_queue is not None:
        while True:
            try:
                update = progress_queue.get_nowait()
                received_update = True
            except Empty:
                break
            for key, value in update.items():
                if key == "processing":
                    st.session_state[key] = value
                elif key == "processing_thread":
                    st.session_state[key] = value
                else:
                    st.session_state[key] = value

    progress = st.session_state.get("processing_progress", 0.0)
    status = st.session_state.get("processing_status", "Preparing analysis")
    elapsed = st.session_state.get("processing_elapsed", 0)
    frame = st.session_state.get("processing_frame", 0)
    total_frames = st.session_state.get("processing_total_frames", 0)
    fps = st.session_state.get("processing_fps", 0.0)
    percent = int(round(min(100.0, max(0.0, progress * 100))))
    frame_text = f"{frame}/{total_frames}" if total_frames else str(frame)
    st.markdown(
        f"""
        <div class='glass-card' style='margin-bottom:0.8rem; padding:1rem 1.05rem;'>
            <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.8rem;'>
                <div>
                    <div class='status-pill analyzing-pill'><span class='analyzing-dot'></span>Analyzing</div>
                    <div style='margin-top:0.45rem; color:#f8fafc; font-weight:700;'>Status: {status}</div>
                    <div style='color:#94a3b8; font-size:0.92rem;'>Elapsed: {elapsed}s • {fps:.1f} fps</div>
                </div>
                <div style='min-width:280px; flex:1;'>
                    <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.45rem; color:#cbd5e1; font-size:0.86rem;'>
                        <span>Progress</span>
                        <span>{percent}%</span>
                    </div>
                    <div class='progress-track'>
                        <div class='progress-fill' style='width:{percent}%;'></div>
                        <div class='progress-shine'></div>
                    </div>
                    <div style='margin-top:0.4rem; color:#94a3b8; font-size:0.9rem;'>Frame {frame_text}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.get("processing"):
        now = time.monotonic()
        last_rerun = st.session_state.get("banner_last_rerun", 0.0)
        if received_update or (now - last_rerun) >= 0.6:
            st.session_state["banner_last_rerun"] = now
            st.rerun()


def render_topbar(page_name: str) -> None:
    status_text = "Processing in background" if st.session_state.get("processing") else "Ready"
    status_class = "warning-chip" if st.session_state.get("processing") else "chip"
    st.markdown(
        f"""
        <div class='topbar'>
            <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.7rem;'>
                <div style='display:flex; align-items:center; gap:0.75rem;'>
                    <div class='topbar-icon'>🎯</div>
                    <div>
                        <div class='topbar-title'>{page_name}</div>
                        <div class='topbar-subtitle'>Premium AI video analytics workspace</div>
                    </div>
                </div>
                <div style='display:flex; gap:0.6rem; align-items:center; flex-wrap:wrap;'>
                    <div class='{status_class}'>● {status_text}</div>
                    <div class='topbar-pill'>⚡ {st.session_state.get('settings').use_gpu and 'CUDA' or 'CPU'} Mode</div>
                    <div class='topbar-pill'><strong>Run:</strong> {st.session_state.get('run_dir').name if st.session_state.get('run_dir') else 'Idle'}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("<div class='glass-card' style='padding:1rem 1rem 0.85rem; margin-bottom:0.8rem;'>"
                    "<div class='sidebar-logo'>🎯 AI Shooting Range</div>"
                    "<div class='sidebar-subtitle'>Scoring System</div></div>", unsafe_allow_html=True)

        pages = ["🏠 Dashboard", "🎥 Upload Video", "📊 Results", "📂 Output Folder", "📝 Logs"]
        # Ensure navigation click state exists
        nav_clicked = st.session_state.get("nav_clicked")
        current_page = st.session_state.get("current_page", "🏠 Dashboard")
        # If a nav was clicked in a previous run, prefer it
        if nav_clicked:
            current_page = nav_clicked
            st.session_state["current_page"] = current_page

        st.markdown("<div class='modern-nav'>", unsafe_allow_html=True)
        for page in pages:
            is_active = current_page == page
            if st.button(page, key=f"nav_{page}", use_container_width=True, type="primary" if is_active else "secondary"):
                # update shared session state and mark click
                st.session_state["current_page"] = page
                st.session_state["nav_clicked"] = page
                # trigger a rerun so main() picks up the new page on next cycle
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card' style='margin-top:1rem; padding:0.8rem;'>"
                    "<div style='font-size:0.78rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.12em;'>System</div>"
                    "<div style='margin-top:0.4rem; display:flex; flex-direction:column; gap:0.4rem;'>"
                    "<div class='chip'>🖥 GPU Status: <span style='color:#bbf7d0;'>Ready</span></div>"
                    "<div class='chip warning-chip'>Version 2.1.0</div>"
                    "</div></div>", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1rem;'><small style='color:#64748b;'>Built for tactical training analytics and high-precision reporting.</small></div>", unsafe_allow_html=True)
    return current_page


def render_dashboard() -> None:
    render_topbar("Dashboard")
    render_processing_banner()

    result = st.session_state.get("analysis_result")
    stats = get_dashboard_stats(result)

    st.markdown(
        """
        <div class='hero-card'>
            <div style='display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:1rem;'>
                <div>
                    <div class='chip'>AI Powered</div>
                    <h1 style='font-size:2rem; margin:0.3rem 0 0.35rem; color:white;'>AI Shooting Range Scoring System</h1>
                    <p style='margin:0; color:#e2e8f0; max-width:700px;'>Real-time detection, accuracy scoring, and executive-ready reporting for modern training ranges.</p>
                </div>
                <div class='glass-card' style='padding:0.85rem 1rem; min-width:240px;'>
                    <div style='font-size:0.8rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.12em;'>System Status</div>
                    <div style='font-size:1.25rem; font-weight:700; margin-top:0.25rem;'>● Online · Monitoring</div>
                    <div style='margin-top:0.45rem; color:#cbd5e1;'>Latest run: {latest_run}</div>
                </div>
            </div>
        </div>
        """.format(latest_run=st.session_state.get("run_dir").name if st.session_state.get("run_dir") else "No run yet"),
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    left, right = st.columns([2.1, 1])
    with left:
        with st.container():
            st.markdown("<div class='glass-card'><h3 style='margin-top:0;'>Live Camera Preview</h3></div>", unsafe_allow_html=True)
            if result is not None and result.processed_path.exists():
                st.markdown("<div class='preview-shell' style='margin-top:0.6rem;'>", unsafe_allow_html=True)
                st.video(str(result.processed_path))
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div class='preview-shell' style='margin-top:0.6rem;'><div class='preview-placeholder'><div class='preview-ring'></div><div style='font-size:1rem; font-weight:600;'>Awaiting live feed</div><div style='margin-top:0.35rem; color:#94a3b8;'>Upload a video to generate the first preview pass.</div></div></div>",
                    unsafe_allow_html=True,
                )

    with right:
        st.markdown(
            "<div class='glass-card' style='padding:1rem;'>"
            "<h4 style='margin-top:0;'>Ready to run</h4>"
            "<p style='color:#94a3b8; margin-bottom:0;'>Upload a video, tune the settings, and let the analysis run in the background while you explore the workspace.</p>"
            "</div>",
            unsafe_allow_html=True,
        )


def render_upload() -> None:
    render_topbar("Upload & Process")
    render_processing_banner()
    st.markdown("<div class='glass-card'><h3 style='margin-top:0;'>Upload a training video</h3><p style='color:#94a3b8; margin-bottom:0;'>The pipeline will detect targets, score hits, and generate exports in one workflow.</p></div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a video", type=list(SUPPORTED_VIDEO_EXTENSIONS), accept_multiple_files=False)
    is_processing = st.session_state.get("processing", False)
    if uploaded_file is not None:
        st.markdown("<div class='glass-card' style='margin-top:0.8rem;'>", unsafe_allow_html=True)
        col_a, col_b = st.columns([1.6, 1])
        with col_a:
            st.video(uploaded_file)
        with col_b:
            st.markdown(f"<div style='padding:1rem; border-radius:18px; background:rgba(15,23,42,0.8);'><div style='font-size:1.1rem; font-weight:700;'>File Details</div>"
                        f"<div style='margin-top:0.6rem; color:#94a3b8;'>Name: {uploaded_file.name}</div>"
                        f"<div style='margin-top:0.35rem; color:#94a3b8;'>Size: {format_bytes(uploaded_file.size)}</div></div>", unsafe_allow_html=True)
            if is_processing:
                st.button("Processing...", type="primary", use_container_width=True, disabled=True)
            else:
                if st.button("Start Analysis", type="primary", use_container_width=True):
                    temp_dir = ensure_dir(TEMP_DIR / timestamp())
                    video_path = temp_dir / uploaded_file.name
                    with open(video_path, "wb") as out_file:
                        out_file.write(uploaded_file.getbuffer())
                    st.session_state["video_path"] = video_path
                    st.session_state["run_dir"] = OUTPUT_DIR / f"run_{timestamp()}"
                    st.session_state["processing"] = True
                    st.session_state["processing_status"] = "Starting analysis"
                    st.session_state["processing_progress"] = 0.0
                    st.session_state["processing_elapsed"] = 0
                    st.session_state["processing_error"] = None
                    ensure_dir(st.session_state["run_dir"])
                    start_background_analysis(video_path, st.session_state["run_dir"], st.session_state.get("settings") or ProcessingSettings())
                    st.toast("Analysis started. You can switch pages while it runs.")
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get("processing"):
        st.markdown("<div class='glass-card' style='margin-top:0.8rem;'>", unsafe_allow_html=True)
        st.progress(st.session_state.get("processing_progress", 0.0))
        st.caption(f"Status: {st.session_state.get('processing_status', 'Working')}"
                   f" | Elapsed: {st.session_state.get('processing_elapsed', 0)}s")
        if st.session_state.get("processing_error"):
            st.error(st.session_state["processing_error"])
        st.markdown("</div>", unsafe_allow_html=True)
    elif st.session_state.get("analysis_result") is not None:
        st.success("Analysis finished. Results are ready to review.")


def render_settings() -> None:
    render_topbar("AI Settings")
    render_processing_banner()
    st.markdown("<div class='glass-card'><h3 style='margin-top:0;'>Model & inference controls</h3><p style='color:#94a3b8; margin-bottom:0;'>Tune the detector behavior and output quality for each run.</p></div>", unsafe_allow_html=True)

    settings = st.session_state.get("settings") or ProcessingSettings()
    with st.form("settings_form"):
        col1, col2 = st.columns(2)
        with col1:
            selected_model = st.selectbox("YOLO Model", ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"], index=0)
            confidence = st.slider("Confidence Threshold", 0.25, 0.95, round(settings.confidence_threshold, 2), 0.01)
            iou = st.slider("IoU Threshold", 0.25, 0.95, 0.45, 0.01)
            device = st.selectbox("Device", ["CPU", "CUDA"], index=0 if not settings.use_gpu else 1)
            frame_skip = st.slider("Frame Skip", 0, 8, settings.frame_skip, 1)
        with col2:
            resolution = st.selectbox("Resolution", ["Original", "720p", "1080p"], index=0)
            output_quality = st.selectbox("Output Quality", ["Balanced", "High", "Ultra"], index=0)
            save_frames = st.toggle("Save Frames", value=settings.save_frames)
            save_hits = st.toggle("Save Hits", value=settings.enable_heatmap)
            max_targets = st.slider("Max Targets", 1, 16, settings.max_targets, 1)
            hit_radius = st.slider("Hit Radius", 6, 40, settings.hit_radius, 1)

        submitted = st.form_submit_button("Save Settings", use_container_width=True)
        if submitted:
            st.session_state["settings"] = ProcessingSettings(
                frame_skip=frame_skip,
                confidence_threshold=confidence,
                max_targets=max_targets,
                hit_radius=hit_radius,
                use_gpu=device == "CUDA",
                enable_heatmap=save_hits,
                save_frames=save_frames,
            )
            st.success("Settings updated successfully.")


def render_results() -> None:
    render_topbar("Results")
    render_processing_banner()
    result: AnalysisResult | None = st.session_state.get("analysis_result")
    if result is None:
        st.markdown("<div class='glass-card' style='min-height:260px; display:flex; align-items:center; justify-content:center;'><div style='text-align:center; color:#cbd5e1;'><div style='font-size:2.4rem;'>📊</div><div style='font-size:1rem; font-weight:600;'>No results available yet. Run an analysis to unlock the reporting workspace.</div></div></div>", unsafe_allow_html=True)
        return

    st.markdown("<div class='glass-card'><h3 style='margin-top:0;'>Analysis Summary</h3></div>", unsafe_allow_html=True)
    reports = st.session_state.get("reports", {})
    export_col, info_col = st.columns([1.3, 1])
    with export_col:
        st.markdown("<div class='glass-card' style='margin-top:0.7rem;'><h4 style='margin-top:0;'>Export Suite</h4><p style='color:#94a3b8; margin-bottom:0.4rem;'>Download polished report assets for review or handoff.</p></div>", unsafe_allow_html=True)
        if reports:
            for name, path in [("PDF", reports.get("pdf_path")), ("CSV", reports.get("csv_path")), ("JSON", reports.get("json_path"))]:
                if path and Path(path).exists():
                    with open(path, "rb") as handle:
                        st.download_button(f"Download {name}", handle.read(), file_name=Path(path).name, mime="application/pdf" if name == "PDF" else "text/csv" if name == "CSV" else "application/json")
        else:
            st.info("Exports will appear after a run is processed.")
    with info_col:
        stats = get_dashboard_stats(result)
        st.markdown("<div class='glass-card' style='margin-top:0.7rem;'><div class='status-pill'>● Processing Complete</div><div style='margin-top:0.7rem; color:#94a3b8;'>Accuracy: {accuracy}%</div><div style='margin-top:0.3rem; color:#94a3b8;'>Targets: {targets}</div><div style='margin-top:0.3rem; color:#94a3b8;'>Shots: {shots}</div></div>".format(accuracy=f"{stats['average_accuracy']:.1f}", targets=stats['targets_detected'], shots=stats['shots_detected']), unsafe_allow_html=True)
    stat_cols = st.columns(4)
    for col, (label, value) in zip(stat_cols, [("Accuracy", f"{stats['average_accuracy']:.1f}%"), ("Shots", stats["shots_detected"]), ("Targets", stats["targets_detected"]), ("Frames", result.frame_count)]):
        with col:
            st.markdown(f"<div class='kpi-card'><div style='color:#94a3b8;'>{label}</div><div style='font-size:1.4rem; font-weight:800; margin-top:0.35rem;'>{value}</div></div>", unsafe_allow_html=True)

    if result.processed_path.exists():
        st.markdown("<div class='glass-card' style='margin-top:0.9rem;'><h4 style='margin-top:0;'>Processed Video</h4></div>", unsafe_allow_html=True)
        st.video(str(result.processed_path))

    tabs = st.tabs(["Overview", "Targets", "Shots", "Timeline", "Gallery", "Reports"])
    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            if not result.shot_table.empty:
                fig = px.line(result.shot_table, x=result.shot_table.index, y="score", markers=True)
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#e2e8f0"})
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.write("No shot data available.")
        with c2:
            if not result.target_table.empty:
                fig = px.bar(result.target_table, x="target_id", y="score", color="status")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#e2e8f0"})
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.write("No target data available.")
    with tabs[1]:
        st.dataframe(result.target_table.reset_index(drop=True), use_container_width=True)
    with tabs[2]:
        st.dataframe(result.shot_table.reset_index(drop=True), use_container_width=True)
    with tabs[3]:
        if not result.shot_table.empty:
            timeline = px.scatter(result.shot_table, x=result.shot_table.index, y="score", size="score", color="hit")
            timeline.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#e2e8f0"})
            st.plotly_chart(timeline, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Timeline will populate once hits are detected.")
    with tabs[4]:
        run_dir = result.run_dir
        gallery_files = [p for p in run_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
        if gallery_files:
            cols = st.columns(3)
            for idx, image_path in enumerate(gallery_files[:6]):
                with cols[idx % 3]:
                    st.image(str(image_path), caption=image_path.name)
        else:
            st.info("Frame gallery will appear when saved output images are available for the run.")
    with tabs[5]:
        reports = st.session_state.get("reports", {})
        if reports:
            report_files = [
                ("PDF", reports.get("pdf_path")),
                ("CSV", reports.get("csv_path")),
                ("JSON", reports.get("json_path")),
            ]
            for name, path in report_files:
                if path and Path(path).exists():
                    with open(path, "rb") as handle:
                        st.download_button(f"Download {name}", handle.read(), file_name=Path(path).name, mime="application/pdf" if name == "PDF" else "text/csv" if name == "CSV" else "application/json")
        else:
            st.info("No report files were generated for this run.")


def render_output_folder() -> None:
    render_topbar("Output Folder")
    render_processing_banner()
    manager = OutputManager(OUTPUT_DIR)
    runs = manager.list_runs()
    if not runs:
        st.markdown("<div class='glass-card' style='min-height:220px; display:flex; align-items:center; justify-content:center;'><div style='text-align:center; color:#cbd5e1;'><div style='font-size:2.2rem;'>📁</div><div style='font-size:1rem; font-weight:600;'>No runs created yet. Process a video to create your first output bundle.</div></div></div>", unsafe_allow_html=True)
        return

    st.markdown("<div class='glass-card'><h3 style='margin-top:0;'>Run Explorer</h3><p style='color:#94a3b8; margin-bottom:0;'>Browse generated folders, preview outputs, and download reports.</p></div>", unsafe_allow_html=True)
    for run in runs:
        path = run["path"]
        summary = manager.get_folder_summary(path)
        with st.expander(f"📂 {path.name} · {summary['file_count']} files · {summary['folder_size']}"):
            st.markdown(f"<div class='glass-card'>Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run['created']))}</div>", unsafe_allow_html=True)
            files = sorted([p for p in path.rglob("*") if p.is_file()])
            if files:
                for file in files[:12]:
                    rel = file.relative_to(path)
                    st.write(f"- {rel}")
            if (path / "processed.mp4").exists():
                st.video(str(path / "processed.mp4"))
            cols = st.columns(4)
            if (path / "reports" / "report.pdf").exists():
                with cols[0]:
                    st.download_button("Download PDF", (path / "reports" / "report.pdf").read_bytes(), file_name="report.pdf", mime="application/pdf")
            if (path / "reports" / "report.csv").exists():
                with cols[1]:
                    st.download_button("Download CSV", (path / "reports" / "report.csv").read_bytes(), file_name="report.csv", mime="text/csv")
            if (path / "reports" / "report.json").exists():
                with cols[2]:
                    st.download_button("Download JSON", (path / "reports" / "report.json").read_bytes(), file_name="report.json", mime="application/json")
            with cols[3]:
                if st.button(f"Delete {path.name}", key=f"delete_{path.name}", use_container_width=True):
                    manager.delete_run(path)
                    st.rerun()


def render_logs() -> None:
    render_topbar("Logs")
    render_processing_banner()
    log_path = OUTPUT_DIR / "logs" / "app.log"
    if not log_path.exists():
        st.info("No log file has been generated yet.")
        return
    st.markdown("<div class='glass-card'><h3 style='margin-top:0;'>Event Log</h3></div>", unsafe_allow_html=True)
    with open(log_path, "r", encoding="utf-8") as handle:
        content = handle.read().splitlines()[-80:]
    st.code("\n".join(content), language="text")


def render_about() -> None:
    render_topbar("About")
    render_processing_banner()
    st.markdown("<div class='glass-card'><h3 style='margin-top:0;'>About this experience</h3><p style='color:#94a3b8;'>This interface combines a premium desktop-inspired UI with an AI processing backbone for shooting-range scoring, target detection, and report generation.</p></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='glass-card'><h4 style='margin-top:0;'>Architecture</h4><p style='color:#94a3b8;'>Modular detectors, trackers, scoring engines, and report generation work together to deliver explainable AI outputs.</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='glass-card'><h4 style='margin-top:0;'>Exports</h4><p style='color:#94a3b8;'>Generate PDF, CSV, and JSON reports for review, auditing, and operational analysis.</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='glass-card'><h4 style='margin-top:0;'>Next Steps</h4><p style='color:#94a3b8;'>Extend the UI with live camera streams, richer charts, and deeper workflow automation.</p></div>", unsafe_allow_html=True)


def main() -> None:
    initialize_session_state()
    ensure_dir(OUTPUT_DIR)
    ensure_dir(TEMP_DIR)
    inject_css()
    page = render_sidebar()

    if page == "🏠 Dashboard":
        render_dashboard()
    elif page == "🎥 Upload Video":
        render_upload()
    elif page == "⚙ AI Settings":
        render_settings()
    elif page == "📊 Results":
        render_results()
    elif page == "📂 Output Folder":
        render_output_folder()
    elif page == "📝 Logs":
        render_logs()
    else:
        render_about()


if __name__ == "__main__":
    main()
