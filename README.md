# Stampede Identification and Alert System

## Problem Statement

Crowd stampedes in public spaces pose serious safety risks, often leading to injuries or fatalities. Early identification of overcrowding and potential stampede situations is crucial for timely intervention by authorities.

This project aims to develop a real-time prediction and alert system that analyzes video feeds from multiple sectors of a venue, detects human density and movement patterns, and identifies conditions likely to lead to a stampede. By continuously monitoring crowd behavior and triggering alerts when critical thresholds are exceeded, the system helps authorities take proactive measures to ensure public safety and prevent disasters.

---

## Features

- Real-time person detection and tracking using YOLOv8 and DeepSORT.
- Division of video frames into grid sectors to monitor crowd density.
- Visual and textual alerts when overcrowding is detected.
- Trajectory visualization of tracked individuals.
- Support for simultaneous processing of 6 sector video feeds via a Gradio web interface.

---
## Demo 

Demo link:  https://huggingface.co/spaces/nikisded/LaunchHacks

yt video link: https://github.com/stargalax/LaunchHacks-IV
## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/stargalax/LaunchHacks-IV.git
   cd LaunchHacks-IV
   ```
2.Create and activate a Python virtual environment:

  ```bash
  python -m venv venv
  ```
3.Activate the virtual environment:
  ```bash
  .\venv\Scripts\activate
  ```
Install dependencies:
  ```bash
      pip install -r requirements.txt
  ```
## Usage
Run the Gradio app:
```bash
python app.py
```
Open the URL shown in the terminal (usually http://127.0.0.1:7860/) to access the interface.

Upload videos for each sector and click Start Processing to see the analyzed video feed and alerts.

## About Us
Meet the dynamic duo behind the scenes! 🎉

I'm the Machine Learning & UI wizard, laser-focused on crafting smart solutions that put people first. Whether it’s building intuitive interfaces or training models to solve real-world problems, I’m all about making tech work for humans.

Meanwhile, my friend(Nirmal) is the Master of All Trades — a true tech polymath who knows the ins and outs of every gadget, language, and framework out there. From backend magic to cloud wizardry, there’s no tech puzzle too big or small for him.

Together, we’re unstoppable: his broad knowledge fuels our projects with versatility, while my deep focus ensures we deliver impactful, user-centered innovations. When these forces combine, creativity and technical prowess collide — and that’s when the real magic happens! 🚀✨


> Notes
>Tested with Python 3.8 and above.
> Looking for mentors who can guide us on how to improve the current model accuracy!
> For prototype, we have submitted the webapp that can take mp4 files, will be implementing streaming feature in near future.
