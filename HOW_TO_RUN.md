# 🚀 How to Run the RAG System

This document outlines exactly how to run your RAG system, what each command does, and when to use them for your assignment and viva.

---

## 🏗️ 1. Setup & Pre-requisites

Make sure your terminal is inside the project directory:
```bash
cd "/Users/vaibhav/Study/Sem 6/Gen AI/Main Assignment/Assignment 1"
```

*Note: All dependencies (like `transformers`, `torch`, `chromadb`, etc.) are already installed in your environment.*

---

## 🏃 2. Execution Modes

You run the system using `main.py` with different `--mode` flags. Here is your game plan:

### Option A: The "Viva/Teacher Demo" Mode 🎤
**Command:**
```bash
python main.py --mode demo
```
**When to use:** Use this **live during your assignment viva or presentation.** 
**What it does:** 
1. Loads a fast configuration (512 chunks + MiniLM + FAISS + TinyLlama).
2. Gives you an interactive prompt `📝 Your question: ` in the terminal.
3. You (or the teacher) can type any custom question.
4. It will show exactly which chunks it retrieved from the papers, and then generate the answer live.
5. *(Type "quit" or "exit" to stop).*

### Option B: The "Full Experiment" Mode ⏳ (For your Report)
**Command:**
```bash
python main.py --mode full
```
**When to use:** Run this **ONCE overnight or when you have a few hours free** to generate the final data for your PDFs/report.
**What it does:** 
1. It runs all **36 experimental configurations** (different chunk sizes, different vector DBs, different LLMs) systematically over 100 research papers.
2. It evaluates every answer using ROUGE and BLEU scores.
3. It saves the raw data to `results/evaluation_results.csv` and the clean summary to `results/summary_results.csv`.
4. It prints a giant, formatted table in the terminal at the very end.

> [!WARNING]
> Because it is processing 15,000+ chunks through large Language Models without a GPU, this command takes a few hours. Grab a coffee!

### Option C: The "Quick Test" Mode 🔬
**Command:**
```bash
python main.py --mode quick
```
**When to use:** Use this when you are **testing the code or making small changes** and want to ensure it still works without waiting hours.
**What it does:** 
1. Uses a tiny subset of documents (only 20 papers).
2. Runs only ONE configuration (512 chunk size, MiniLM, FAISS, TinyLlama).
3. Saves a mini result file to `results/quick_results.csv`.
4. Takes about 2–5 minutes.

### Option D: The "Pipeline Check" Mode ⚙️
**Command:**
```bash
python main.py --mode pipeline
```
**When to use:** Use this to **debug the retrieval and generation quality** without having to type queries manually.
**What it does:** 
1. Automatically runs 3 hard-coded sample questions through the pipeline.
2. Prints the retrieved contexts, the generated answer, and the metrics (ROUGE/BLEU) for those 3 questions directly to the terminal.

---

## ⏱️ 3. How to Make the Full Experiment Faster

If `python main.py --mode full` is taking *too long* and your deadline is approaching, you can safely scale down the experiment:

1. Open `config.py`
2. Change **Line 15** from `100` to a smaller number, like `50` or `30`:
   ```python
   NUM_DOCUMENTS = 30  # Processing fewer papers drastically speeds up the run
   ```
3. Save the file and re-run `--mode full`. The experiment will still perform all 36 configuration combinations and generate high-quality academic data for your report, it will just process a smaller corpus of papers!

---

## 📂 4. Where are my results?

After running `--mode full` or `--mode quick`, a new `results/` folder will appear in your project.

Inside, you will find:
1. `evaluation_results.csv` (Every single query tested, great for detailed analysis)
2. `summary_results.csv` (The averaged scores for each configuration, perfect for pasting into Word/Docs tables)

Copy the numbers from `summary_results.csv` into your final report!
