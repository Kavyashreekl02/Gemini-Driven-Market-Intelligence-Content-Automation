# Gemini-Driven-Market-Intelligence-Content-Automation

This repository implements a modular AI-based system designed to unify app-market datasets, generate structured analytical insights, and automate creative content generation. The solution integrates datasets from the Google Play Store, Apple App Store, and Direct-to-Consumer (D2C) marketing environments, with Google Gemini serving as the central reasoning engine.

The project operationalizes the methodology presented in the accompanying research paper included in the `docs/` directory.

## 1. Introduction

Modern app ecosystems produce large and heterogeneous datasets across multiple platforms. Extracting meaningful insights from these sources typically requires extensive manual processing, interpretation, and domain understanding.

This project addresses these challenges through an end-to-end pipeline that:

- Consolidates platform-specific datasets into a unified schema  
- Performs cross-market analytics  
- Generates structured insights using Google Gemini  
- Produces automated reports and data-driven marketing content  

The system is modular, reproducible, and suitable for academic research, prototyping, and decision-support applications.

## 2. Methodology

The workflow follows five structured phases:

### 2.1 Data Ingestion and Cleaning
- Load Google Play Store dataset (Kaggle)  
- Retrieve Apple App Store dataset through API or controlled mock data  
- Standardize heterogeneous attributes across platforms    

### 2.2 Unified Schema Construction and Sentiment Integration
- Integrate sentiment polarity (where available)  
- Produce a harmonized dataset suitable for cross-platform analysis  

### 2.3 Insight Generation Using Large Language Model  
- Generate structured insights using prompt-engineered instructions  
- Enforce a predefined JSON schema for consistency  
- Assign confidence scores for each analytical insight  
- Employ retry mechanisms to ensure output validity  

### 2.4 Automated Report Generation  
- Transform validated insights into a formal Markdown/PDF report  
- Include executive summary, metrics overview, and recommendation set  

### 2.5 Creative Content Generation  
- Generate marketing materials (ad copy, headlines, SEO descriptions, etc.)  
- Align creative assets with analytical findings

## 3. Quickstart

### 3.1 Clone the project
git clone https://github.com/Kavyashreekl02/Gemini-Driven-Market-Intelligence-Content-Automation.git <br>
cd Gemini-Driven-Market-Intelligence-Content-Automation

### 3.2 Create virtual environment
#### Windows
python -m venv .venv
.venv\Scripts\activate

#### 3.3 Mac/Linux
python3 -m venv .venv
source .venv/bin/activate

### 3.4 Install dependencies
pip install -r requirements.txt

### 3.5 Run the pipeline
python src/data_pipeline.py <br>
python src/ai_insights.py <br>
python src/report_generator.py <br>
python src/d2c_data_generator.py <br> 
python src/d2c_metrics_analysis.py <br> 
python src/ai_creative.py <br>

### 3.6 UI Component
The project includes a browser-based interface:

You can open the UI in two ways:

#### **1. Using VS Code Live Server (Recommended)**
1. Install the **Live Server** extension in VS Code.  
2. Right-click on `query_interface.html`.  
3. Select **“Open with Live Server.”**

This will launch the UI in your default browser.

#### **2. Opening Manually**
Alternatively:
- Navigate to `src/`
- Double-click `query_interface.html`  
- It will open directly in your browser












