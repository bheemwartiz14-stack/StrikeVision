# AI Shooting Range Scoring System

A production-ready Streamlit application for analyzing fixed-camera shooting range videos.
It detects shooters, targets, bullet impacts, shot timing, hit location, and score, then exports annotated video, images, CSV, JSON, and PDF reports.

## Features

- Upload MP4 videos and run AI-based analysis
- Detect targets, shooters, and bullet impacts
- Track shots and associate hits with targets
- Generate reports and export artifacts
- Explore output folders inside the app

## Installation

```bash
cd shooting_ai
pip install -r requirements.txt
 uv run streamlit run app.py
```

## Project Structure

- `app.py` – Streamlit UI entry point
- `config.py` – Global settings and paths
- `modules/` – Processing modules for detection, tracking, scoring, reporting, and utilities
