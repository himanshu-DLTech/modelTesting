
# Model Testing Framework

A lightweight Python utility to **test and benchmark locally hosted LLMs** for Python code generation.  
The script evaluates the model’s responses, prints results to the console, and also saves detailed scores in `results.txt`.

---

## 🚀 Features
- Test any locally available **Ollama-hosted LLM** for Python code generation.
- Console output with evaluation metrics.
- Generates `results.txt` with additional details.
- Simple setup and easy to extend for other models/datasets.

---

## 📂 Repository Structure
```
modelTesting/
│── churn.csv
│── cities.csv
│── sales.csv
│── transactions.csv
│── scores.csv
│── requirements.txt
│── testModel.py
│── testModel0.1.py
│── testModel0.2.py
│── results.txt   # generated after running
``

---

## ⚙️ Prerequisites
- Python 3.9+ recommended
- [Ollama](https://ollama.ai/) installed and configured
- The model(s) you want to test must be **downloaded in Ollama** beforehand

---

## 🛠 Installation

Clone the repo:
```bash
git clone https://github.com/himanshu-DLTech/modelTesting.git
cd modelTesting
```

(Optional) Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate    # Linux / macOS
.venv\Scripts\activate       # Windows
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the testing script with the model name as an argument:

```bash
python testModel.py <ollama_model_name>
```

Example:
```bash
python testModel.py llama2
```

This will:
- Query the specified model for Python code generation tasks.
- Print results in the console.
- Save detailed scores and evaluation in `results.txt`.

---

## 📑 Sample Run

**Command:**
```bash
python testModel.py llama2
```

**Console Output (example):**
```
Running model: llama2
Test Case 1: PASS
Test Case 2: FAIL
Test Case 3: PASS
Overall Score: 66.7%
Results saved to results.txt
```

**results.txt (snippet):**
```
Model Tested: llama2
Date: 2025-09-08
--------------------------------
Test Case 1: PASS
Prompt: Write a Python function to add two numbers
Expected: def add(a, b): return a + b
Output: def add(a, b): return a+b
--------------------------------
Test Case 2: FAIL
Prompt: Generate the Fibonacci sequence up to n
Expected: Recursive or iterative implementation
Output: Incorrect formatting
--------------------------------
Final Score: 66.7%
```

---

## 🔧 Notes
- Ensure the model is available in Ollama before running:
  ```bash
  ollama pull llama2
  ```
- If you face environment issues, use a virtual environment (`venv`) as shown above.
- You can modify or extend `testModel.py` to test with different datasets.

---

## 🤝 Contributing
Feel free to open issues or submit pull requests for improvements.

