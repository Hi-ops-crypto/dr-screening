---
title: DR Screening Prototype
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# DR Screening Prototype

Triages a retinal fundus photograph for diabetic retinopathy: a quality gate,
an ICDR grade with a Grad-CAM heatmap and screening guidance, plus a
screening-priority calculator from clinical risk factors.

**This is a screening aid, not a medical diagnosis.** It is a research
prototype: it is not a medical device, has not been clinically validated and
must not be used for patient care.

Uploaded photographs are never written to the audit log, which records only
the time, the quality outcome and the grade. Gradio holds each upload in a
temporary cache that is cleared within about two minutes.
