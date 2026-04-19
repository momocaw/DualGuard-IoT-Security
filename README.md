DualGuard: A Two-Stage IoT Anomaly Detection Framework

DualGuard is a hybrid security framework designed for IoT environments. It combines the efficiency of Machine Learning (Isolation Forest) with the deep semantic interpretability of Large Language Models (LLMs).

This project is part of the course assignment for Academic Communication Activities at Beijing University of Posts and Telecommunications (BUPT) for the MIC 2026 conference theme: Connecting the Intelligent Future.

🌟 Research Highlights

Efficiency: Reduces LLM API calls by 55% through a lightweight ML filtering stage.

Interpretability: Bridges the gap between numerical anomalies and human-readable security reports.

Scalability: Optimized for edge computing scenarios where real-time detection and deep analysis are both required.

🏗 System Architecture

DualGuard operates in two distinct stages:

Stage 1 (Detection Layer): An unsupervised Isolation Forest model scans all incoming IoT traffic (CICIoT2023 dataset) to identify potential threats with high precision and low latency.

Stage 2 (Explanation Layer): Only flagged anomalies are sent to Alibaba Qwen-Plus (LLM) to generate natural language explanations and actionable mitigation strategies.

📂 Project Structure

config.py: Configuration file for API keys and model parameters.

step2\_iforest.py: Stage 1 - Machine Learning detection using Isolation Forest.

step3\_llm\_explain.py: Stage 2 - Semantic explanation generation using LLM API.

step4\_evaluate.py: Evaluation script to generate performance metrics and visualizations.

api\_reduction.png: Experimental result showing the reduction in API calls.

🛠 Installation \& Setup

1\. Requirements

Python 3.12+

Libraries: openai, scikit-learn, matplotlib, pandas, numpy

code

Bash

pip install openai scikit-learn matplotlib pandas numpy

2\. Configuration

⚠️ IMPORTANT: For security reasons, the API\_KEY in config.py has been removed.

Obtain your API Key from Alibaba ModelStudio (DashScope).

Fill in your key in config.py:

code

Python

API\_KEY = "your\_real\_api\_key\_here"

🚀 Usage Guide

To reproduce the experimental results:

Detection: Run Stage 1 to filter anomalies.

code

Bash

python step2\_iforest.py

Explanation: Generate security reports for detected threats.

code

Bash

python step3\_llm\_explain.py

Evaluation: Generate the comparison charts.

code

Bash

python step4\_evaluate.py

