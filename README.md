
# Log Parser

An advanced AI-powered solution for parsing and analyzing logs to identify patterns and anomalies. This tool provides actionable insights for diagnosing and resolving issues efficiently, simplifying log analysis for quicker and more accurate problem detection and resolution.

![Log Parser](https://miro.medium.com/v2/resize:fit:1400/1*iGdFJTHMIG79N2HChWaooQ.gif)

## Tools Used
<div align="center">
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">&nbsp;
  <img src="https://img.shields.io/badge/Llama 3-0467DF?style=for-the-badge&logo=meta&logoColor=white"> &nbsp;
  <img src="https://custom-icon-badges.demolab.com/badge/embedding 001-FFFFFF?style=for-the-badge&logo=google"> &nbsp;
  <img src="https://custom-icon-badges.demolab.com/badge/Langchain-FBEEE9?style=for-the-badge&logo=ln"> &nbsp;
  <img src="https://custom-icon-badges.demolab.com/badge/FAISS DB-999999?style=for-the-badge&logo=faiss"> &nbsp;
  <img src="https://custom-icon-badges.demolab.com/badge/GROQ Cloud-FFFFFF?style=for-the-badge&logo=groq"> &nbsp;
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">&nbsp;
  <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white"> &nbsp;
</div>


## Table of Contents
- [Demo](#demo)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies](#technologies)
- [Contributing](#contributing)

## Demo
You can see the live demo of the application [here](https://log-parsing-tool.streamlit.app).

## Features
- Parse various logs to identify patterns and anomalies.
- Provides actionable insights for diagnosing issues.
- Simplifies log analysis for quicker problem detection.
- Currently able to parse OVS, Kernel, Sys Logs & DMESG Logs.

## Installation
To run this project locally, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/YashasJKumar/Log_Parser.git
    cd Log_Parser
    ```

2. **Set up a virtual environment:**
    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows use `env\Scriptsctivate`
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Place your respective API KEYS:**

   Replace st.secrets['GROQ_API_KEY'] & st.secrets['GOOGLE_API_KEY'] with your respective API Keys in "main.py".
   
   Link for API Keys :
   1. [GROQ_API_KEY](https://console.groq.com/keys)
   2. [GOOGLE_API_KEY](https://aistudio.google.com/app/apikey)


## Usage
1. **Run the application:**
    ```bash
    streamlit run main.py
    ```

2. **Navigate to the application:**
    Open your browser and go to `http://localhost:8501` to view the Streamlit interface.

## Technologies
- **Python** - For scripting and backend logic.
- **Streamlit** - For creating an interactive web interface.
- **Machine Learning** - For log analysis and anomaly detection.

## Contributing
Contributions are welcome! If you have any suggestions or improvements, please create an issue or submit a pull request.

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

