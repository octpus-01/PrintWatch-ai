# PrintWatch AI: Real-time 3D Printing Defect Detection

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**PrintWatch AI** is an intelligent monitoring system designed to detect common defects in Fused Deposition Modeling (FDM) 3D prints in real-time. By leveraging computer vision and deep learning on a Raspberry Pi, it aims to save time and filament. 
**THIS PROJECT IS UNDER CONSTRUCTION!**

## 📌 Problem Statement

3D printing, while powerful, is prone to various failures like warping, stringing, layer shifts, and under-extrusion. These issues often go unnoticed until the print is complete, leading to wasted materials and time. Traditional manual inspection is inefficient and not scalable.

## 🚀 Solution Overview

This project provides an end-to-end solution:
1.  **Capture**: A camera module (e.g., Raspberry Pi Camera) continuously monitors the printing process.
2.  **Detect**: A lightweight model, trained on a custom dataset of 3D printing defects, runs directly on the Raspberry Pi for on-device inference.
3.  **Alert & Log**: When a defect is detected, the system can send an alert via MQTT and log the image with its annotation for further analysis and reinforce learning.

## 🛠️ Tech Stack

*   **Core Framework**: haven't decided yet
*   **Edge Device**: Raspberry Pi 4/5
*   **Camera**: Raspberry Pi Camera Module or any compatible USB webcam
*   **Communication**: [Eclipse Paho MQTT](https://pypi.org/project/paho-mqtt/) for sending images and annotations to a central server or dashboard.
*   **Language**: Python 3

## 📂 Project Structure


## 🚦 Getting Started

### Prerequisites

1.  A Raspberry Pi (4 or 5 recommended) with Raspberry Pi OS installed.
2.  A camera module connected and enabled (`sudo raspi-config`).
3.  An MQTT Broker (e.g.Mosquitto) running on your local network or a cloud service.
4.  A powerful computer to finetune the models.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/octpus-01/PrintWatch-ai.git
    cd PrintWatch-ai
    ```

2.  **Set up a virtual environment ( uv recommended):**
    ```bash
    uv venv
    source venv/bin/activate # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```

### Usage

1.  **(Optional) Train your model:** Place your dataset in the `data/finetune` folder and follow the YOLO training instructions.
2.  **Configure the scripts:** Update `publisher.py` and `detect.py` with your MQTT broker's IP address, port, and topic.
3.  **Run the detection script on your Raspberry Pi: NOT FINISHED YET**
    ```bash
    python src/detect.py --weights models/best.pt --source 0 # '0' for default camera
    ```
    The system will now monitor the print and send alerts via MQTT upon detecting a defect.

## 🤝 Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request for bug fixes, new features, or documentation improvements.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements


*   This project was inspired by the need for accessible and reliable quality control in desktop 3D printing.
