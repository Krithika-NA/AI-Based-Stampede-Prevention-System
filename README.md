# AI-Based Stampede Prevention System

An intelligent real-time crowd monitoring system that uses Computer Vision and AI to detect high crowd density and trigger alerts to help prevent stampede-like situations in public places.

## 📌 Problem Statement
Stampedes in crowded areas such as temples, railway stations, concerts, and festivals often occur due to uncontrolled crowd density and lack of real-time monitoring. Manual supervision is slow and error-prone.

This project aims to automatically detect dangerous crowd conditions and issue instant alerts, enabling timely preventive action.

## 💡 Solution Overview
The system analyzes live or recorded video feeds to:
- Detect people using a YOLO-based deep learning model
- Estimate crowd density in real time
- Trigger audio and SMS alerts when crowd density crosses a safe threshold

## 🛠 Tech Stack
- Programming Language: Python  
- Computer Vision: OpenCV  
- Deep Learning Model: YOLO (pre-trained)  
- Backend / Web App: Flask  
- Frontend: HTML, CSS  
- Alert System: Twilio (SMS alerts)  
- Database: SQLite  

## ⚙️ How It Works
1. Video frames are captured from a camera or video file  
2. YOLO detects and counts people in each frame  
3. Crowd density is calculated dynamically  
4. If density exceeds a predefined limit:
   - Alarm sound is triggered  
   - SMS alert is sent to authorities  
5. Live status is displayed on the web interface  


