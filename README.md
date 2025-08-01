# 🧠 Automated Underwriting Platform

An AI-powered underwriting system that extracts appraisal data, analyzes risk from reports and property images, and classifies insurance risk level using **Gemini 1.5 Pro Vision**, **spaCy NLP**, and **FastAPI**.

---

## ⚙️ Features

- Upload property **appraisal report** (PDF/Image)
- Upload **property photos** (multiple JPG/PNG)
- AI analyzes for fire, cracks, smoke, water damage, etc.
- Returns **risk score**, **risk level**, and **summary**
- Built with FastAPI backend + Gemini 2.5 Pro + HTML/CSS/JS frontend

---

## 🚀 Getting Started

### 1. Clone this Repository

```bash
git clone https://github.com/Avi112005/Automated-Underwriting-Platform_HiDevs.git
cd Automated-Underwriting-Platform_HiDevs
```

### 2. Create and Activate Virtual Environment
On Windows (PowerShell/VS Code):
```bash
python -m venv venv
venv\Scripts\activate
```
On Mac/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
Using spaCy, download the model:
```bash
python -m spacy download en_core_web_sm
```
### 4. Set Up Environment Variables
Create a .env file in the root directory:
```bash
GEMINI_API_KEY=AIzaSyChXOYoFcK-aSpqAMzw0tC-xjYcurFloeA
```
### 5. Run the Backend (FastAPI)
```bash
python -m uvicorn main:app --reload
```