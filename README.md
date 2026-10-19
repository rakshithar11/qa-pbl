# Optical Screw Thread Measurement (DM PBL)

Streamlit app that measures an ISO metric screw thread from a silhouette image:
detects crests/roots, predicts pitch with an ML classifier, then auto-calibrates
the scale (mm/px) and reports major/minor diameter.

## Structure
```
code/thread_features.py   # shared image -> geometry extraction
code/train_model.py       # trains the pitch classifier
code/app.py               # Streamlit UI
iso_thread_dataset/       # pitch_<mm>/ folders of thread images (+ SolidWorks parts)
cad/                      # M8 bolt CAD
models/                   # generated model (.pkl) goes here
```

## Setup
```bash
pip install -r requirements.txt
python code/train_model.py        # creates models/thread_pitch_classifier.pkl
streamlit run code/app.py
```

## Method
1. CLAHE -> Canny -> largest contour -> upper/lower envelope.
2. Peak detection gives crest and root positions -> `pitch_px`, `depth_px`.
3. Random Forest predicts pitch (mm) from the features.
4. Scale = pitch_mm / pitch_px; diameters = envelope height x scale.
5. Theoretical minor diameter = major - 2 x 0.6134 x pitch (ISO 60 deg thread).

## Known limitations
- Small dataset (~32 images; only 1-2 usable for 1.5 mm pitch; no 2.0 mm class).
- Features are in pixels, so accuracy depends on consistent image size/zoom.
- Peak detection parameters (`PEAK_DISTANCE`, `PEAK_PROMINENCE`) are tuned to this dataset's resolution.
- Intended for backlit silhouette images of external threads.
