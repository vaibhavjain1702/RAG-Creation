# 📖 Complete Project Explanation — RAG System From Scratch

This document explains every single thing about this project, from the very beginning concept to the final output. Written as if you know nothing about any of it.

---

# PART 0: What Problem Are We Solving?

Imagine you are a student and you have 100 research papers on your desk. A teacher walks in and asks:

> *"What are additive models and why are they useful in regression?"*

You have two options:
1. **Read all 100 papers** right now to find the answer (impossibly slow)
2. **Flip through them quickly**, find the 2-3 pages that actually talk about additive models, read just those, and then give your answer (smart and fast)

**Option 2 is exactly what RAG does.** But for a computer.

RAG stands for **Retrieval-Augmented Generation**. It has two parts:
- **Retrieval** → Find the relevant paragraphs from the papers
- **Generation** → Use an AI model to read those paragraphs and write a proper answer

Without RAG, an AI model (like ChatGPT or TinyLlama) only knows what it was trained on. It cannot look at *your specific 100 papers*. With RAG, it can.

---

# PART 1: The Dataset — Where Does the Data Come From?

**File responsible:** `data_loader.py`

## What is the dataset?

We use a dataset called **`ccdv/arxiv-summarization`** from a website called **HuggingFace** (think of it as GitHub but for AI datasets and models).

This dataset contains thousands of real research papers published on **arXiv** (a website where scientists post their research papers). Each paper in the dataset has two important fields:

| Field | What it contains | How we use it |
|-------|-----------------|---------------|
| `article` | The **full body text** of the research paper | This is our **knowledge base** — the text the system searches through |
| `abstract` | A short 1-paragraph **summary** of the paper | We use this as a **reference answer** to measure how good our generated answers are |

## How many papers do we load?

From `config.py`, line 15:
```python
NUM_DOCUMENTS = 100
```
We load the **first 100 papers** from the training split of the dataset.

## The problem with raw arXiv papers

When you download an arXiv paper programmatically, the raw text looks horrible. It's full of **LaTeX code** — the formatting language scientists use to write papers. For example:

```
the probability is $\alpha \geq \beta$ where
\begin{equation}
f(x) = \sum_{i=1}^{n} w_i x_i
\end{equation}
as shown in @xcite and @xmath2.
```

An AI model cannot understand `\begin{equation}`, `@xcite`, or `$\alpha \geq \beta$`. So we need to **clean the text first**.

## How does `data_loader.py` clean the text?

The `clean_text()` function in `data_loader.py` runs 7 cleaning steps using **Regular Expressions** (pattern-matching tools):

| Step | What it removes | Example |
|------|----------------|---------|
| 1 | Full LaTeX math blocks | `\begin{equation}...\end{equation}` |
| 2 | Inline math | `$x^2 + y^2$` |
| 3 | Display math | `$$E = mc^2$$` |
| 4 | Citation commands | `\cite{lewis2020retrieval}` |
| 5 | Citation markers | `@xcite`, `@xref`, `@xmath2` |
| 6 | Bold/italic commands | `\textbf{important}` → `important` |
| 7 | All remaining LaTeX commands | `\alpha`, `\newcommand`, etc. |
| 8 | Normalizes whitespace | Multiple spaces → single space |

**Before cleaning:**
```
additive models @xcite provide $f(x) = \sum w_i$ for semiparametric regression
```

**After cleaning:**
```
additive models provide for semiparametric regression
```

Now the text is clean and readable by both humans and AI models.

---

# PART 2: Chunking — Why We Cut the Papers Into Pieces

**File responsible:** `chunking.py`

## The core problem

Each research paper, after cleaning, is roughly **15,000 characters** long (about 4,000 words). But an AI model has a **context window limit** — the maximum amount of text it can read at once.

TinyLlama, for example, can only handle about **2,048 tokens** (~1,500 words) at a time. If you gave it an entire 4,000-word paper, it would either crash or ignore the extra text.

Also, if you sent the whole 15,000 character paper to the search step, the search would match on the paper as a whole — not on the specific paragraph that actually answers the question. This makes the search imprecise.

## The solution: Chunking

We cut each paper into smaller, overlapping pieces called **chunks**. Think of it like cutting a long rope into shorter sections — but with a slight overlap at each cut so you don't lose any part of the rope.

## Code explanation

In `chunking.py`, we use a tool called `RecursiveCharacterTextSplitter` from the LangChain library.

The word "**Recursive**" is important. It tries to cut at natural boundaries in this order:
1. First tries to cut at **paragraph breaks** (`\n\n`)
2. If not possible, tries **line breaks** (`\n`)
3. If not possible, tries **sentence endings** (`. `)
4. If not possible, tries **commas** (`, `)
5. If not possible, tries **word boundaries** (` `)
6. Only as a last resort, cuts **mid-character**

This means the chunks are always as readable and coherent as possible.

## The three chunk sizes we test

From `config.py`:
```python
CHUNK_SIZES = [256, 512, 1024]
CHUNK_OVERLAP_RATIO = 0.2  # 20% overlap
```

| Chunk Size | Overlap | # Chunks from 100 papers | What it looks like |
|-----------|---------|--------------------------|-------------------|
| 256 chars | 51 chars | ~2,100 chunks | About 40-50 words — one short paragraph |
| 512 chars | 102 chars | ~1,100 chunks | About 80-100 words — one full paragraph |
| 1024 chars | 204 chars | ~580 chunks | About 160-200 words — multiple paragraphs |

## What does each chunk look like?

Each chunk is stored as a Python dictionary with 4 pieces of information:
```python
{
    "text": "additive models provide an important family of models for semiparametric regression...",
    "doc_id": 0,          # Which paper this came from (0 = first paper)
    "chunk_id": 5,        # The unique ID of this chunk (5th chunk overall)
    "chunk_size_config": 512   # What chunk size setting was used
}
```

## Why overlap?

Without overlap, you might cut a sentence in half:

```
[Chunk 1]: "Additive models are useful for regression. The key advantage is their"
[Chunk 2]: "increased flexibility compared to linear models."
```

The second chunk starts mid-sentence. The first chunk ends mid-sentence. Either chunk alone is misleading. With 20% overlap (for size 512, that's 102 characters), both chunks slightly re-include the boundary text, so neither chunk loses the complete sentence.

---

# PART 3: Embeddings — Turning Text Into Numbers

**File responsible:** `embeddings.py`

## The fundamental problem

Computers cannot compare two sentences for meaning. They can only compare numbers. So we need to convert every text chunk into a list of numbers — called a **vector** or **embedding** — that somehow captures its *meaning*.

## What is an embedding?

An embedding is a list of numbers (a vector) of fixed length. In our case, each chunk becomes a list of **384 numbers**.

For example:
```
"Additive models are flexible" → [0.23, -0.51, 0.88, 0.12, ..., -0.34]  (384 numbers total)
```

The magic is that **similar-meaning text produces similar numbers**. So:
```
"Additive models are flexible"   → [0.23, -0.51, 0.88, ...]
"Additive models offer flexibility" → [0.24, -0.49, 0.86, ...]  ← very similar!
"The weather is sunny today"     → [0.78,  0.32, -0.12, ...]  ← very different!
```

This is not random. These models were trained on billions of text pairs to learn what "similar meaning" looks like numerically.

## The two embedding models we compare

From `config.py`:
```python
EMBEDDING_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge":    "BAAI/bge-small-en-v1.5",
}
```

| Model | Full Name | Size | Trained For | Speed |
|-------|----------|------|------------|-------|
| **MiniLM** | all-MiniLM-L6-v2 | 22M parameters | General sentence similarity | Fast |
| **BGE** | BAAI/bge-small-en-v1.5 | 33M parameters | Specifically for information retrieval | Slightly slower |

Both produce **384-dimensional vectors**. BGE tends to produce better results because it was specifically trained to match questions to relevant paragraphs.

## How does the embedding code work?

In `embeddings.py`, the `embed_chunks()` function:
1. Extracts the `"text"` field from every chunk
2. Passes all texts to the model in **batches of 32** (to be memory efficient)
3. Gets back a numpy array of shape `(number_of_chunks, 384)`
4. **Normalizes** each vector to length 1 (so we can use a simple dot product for similarity later)

The result is a giant table where:
- Each **row** = one chunk
- Each **column** = one of the 384 numbers describing its meaning

---

# PART 4: Vector Stores — The Search Index

**File responsible:** `vector_store.py`

## The problem

We now have ~1,100 embeddings (for chunk size 512). When a user asks a question, we need to find the 5 most similar embeddings out of those 1,100 — as fast as possible. Scanning them one by one is too slow at scale.

A **vector store** is a special database that is built for exactly this: storing embeddings and searching them by similarity, fast.

## What is "similarity"?

We use **cosine similarity**. Since all our vectors are normalized to length 1, this simplifies to a **dot product** (multiply corresponding numbers and add them up).

- Score = **1.0** → Identical meaning
- Score = **0.7-0.9** → Very similar meaning
- Score = **0.4-0.6** → Somewhat related
- Score = **< 0.3** → Likely unrelated

## The two vector databases we compare

### FAISS (Facebook AI Similarity Search)

`FAISSStore` in `vector_store.py`

- Created by Facebook/Meta
- Lives entirely **in RAM** (in-memory)
- Uses `IndexFlatIP` — "IP" stands for Inner Product. With normalized vectors, inner product = cosine similarity.
- Does **exact** search: checks every single vector and returns the true top-k
- Blazing fast: can search 1,000 vectors in under 1 millisecond
- Limitation: **no metadata filtering** — it only stores and searches the numbers, not any text labels

### ChromaDB

`ChromaStore` in `vector_store.py`

- A modern **vector database** (like a full database, not just a library)
- Also in-memory in our setup (no disk persistence needed)
- Uses **HNSW** internally — an approximate search algorithm that builds a graph structure for faster search on very large collections
- Stores texts AND embeddings AND metadata together, so you could filter by `doc_id` if you wanted
- Slightly slower than FAISS at our scale

### The `create_vector_store()` factory function

Instead of writing `if faiss... else chroma...` everywhere, we have one function that creates either type:
```python
store = create_vector_store("faiss", dimension=384)
store = create_vector_store("chroma", dimension=384)
# Both return an object with .add() and .search() methods
```

This is called the **Factory Pattern** in software engineering — it's a clean design principle.

---

# PART 5: Retrieval — Finding the Right Paragraphs

**File responsible:** `retriever.py`

This is the "lookup" step. When the user types a question, we need to find the most relevant chunks.

## How retrieval works (step by step)

1. **The user types a question:** `"What are additive models?"`

2. **We embed the question** using the same embedding model we used for the chunks:
   ```python
   query_embedding = embed_query("What are additive models?", model)
   # → [0.31, -0.42, 0.77, ...] (384 numbers)
   ```

3. **We search the vector store** for the 5 chunks whose embeddings are most similar (closest in 384-dimensional space) to the query embedding:
   ```python
   results = vector_store.search(query_embedding, top_k=5)
   ```

4. **We get back the top 5 chunks** with their similarity scores:
   ```
   [1] Score: 0.6395 | Chunk: "additive models provide an important family..."
   [2] Score: 0.4659 | Chunk: "and for parametric quantile regression..."
   [3] Score: 0.4156 | Chunk: "additive structure with each component..."
   [4] Score: 0.3912 | Chunk: "nonparametric regression methods..."
   [5] Score: 0.3701 | Chunk: "semiparametric estimation approaches..."
   ```

This is called **top-k retrieval** where k=5 (set in `config.py`).

## Why is the same embedding model used for queries AND chunks?

This is critical and often misunderstood. The embedding model must be the same for both, because the "coordinate system" (what each of the 384 numbers means) is determined by the model.

It's like asking someone for directions in English when they only speak French. The coordinate system (language) must match.

---

# PART 6: Prompt Building — Packaging the Context + Question for the AI

**File responsible:** `prompt_builder.py`

## The problem: an LLM needs everything in one text block

The LLM (the AI that generates answers) doesn't have separate inputs for "context" and "question". It just reads one big string of text and continues writing from where it ends. So we need to cleverly format the retrieved chunks and the question into a single, well-structured text called a **prompt**.

## The three prompt templates

### Template 1: Basic
```
Based on the following research paper excerpts, answer the question.

Context:
[Excerpt 1]: additive models provide an important family...
[Excerpt 2]: and for parametric quantile regression...

Question: What are additive models?

Answer:
```
Simple. No instructions. The LLM might make things up (hallucinate) because nothing tells it not to.

### Template 2: Structured ← (This is the default we use)
```
You are a scientific research assistant. Answer the question using ONLY
the information provided in the context below. If the context does not
contain enough information, say "I cannot answer this based on the
provided context."

Context from research papers:
---
[Excerpt 1]: additive models provide an important family...
[Excerpt 2]: and for parametric quantile regression...
---

Question: What are additive models?

Provide a concise, factual answer with specific details from the context:
```
Much better. It explicitly:
- Assigns a role ("scientific research assistant")
- Constrains the answer to only use the given context ("ONLY")
- Provides a fallback ("I cannot answer") so the model doesn't make things up

### Template 3: Chain-of-Thought
```
You are an expert AI research assistant...

[Context...]

Question: What are additive models?

Instructions:
1. First, identify which parts of the context are relevant.
2. Then, synthesize the information to form a complete answer.
3. If the context is insufficient, explicitly state what is missing.
4. Do NOT add information that is not in the context.

Step-by-step reasoning and answer:
```
Makes the model reason step-by-step before answering. Best for complex questions but produces longer, more verbose answers.

## What is the context length limit?

In `build_context()`, there is a limit of **2,000 characters** for the total context. This is because even with chunking, 5 chunks combined might exceed the LLM's context window. We add chunks one by one and stop when we hit 2,000 characters.

---

# PART 7: LLM Generation — The AI Writes the Answer

**File responsible:** `generator.py`

## What is an LLM?

An LLM (Large Language Model) is an AI model trained on enormous amounts of text. It learned to predict "what word comes next" billions of times, and through doing that, it learned grammar, facts, reasoning, and writing style.

When we give it a prompt (our formatted context + question), it predicts what the most natural continuation would be — which is the answer.

## The two LLMs we compare

### TinyLlama (1.1B parameters)
- **Tiny** because it's small (1.1 billion parameters, which is actually tiny for an LLM — GPT-4 has ~1 trillion)
- Trained on 3 trillion tokens of text
- Fast: ~8-10 seconds per answer on an M-series Mac
- Less capable: sometimes gives short or incomplete answers

### Microsoft Phi-2 (2.7B parameters)
- 2.7 billion parameters — about 2.5x larger than TinyLlama
- Trained on high-quality "textbook" and "synthetic" data — specifically designed to teach reasoning
- Slower: ~25-28 seconds per answer
- Much more capable: gives structured, complete answers

## How does the loading work?

In `generator.py`, `load_llm()` does this:

```python
# Step 1: Detect device (MPS = Apple Silicon GPU, CPU = Intel)
if torch.backends.mps.is_available():
    device_map = "mps"       # Use Apple GPU
    torch_dtype = torch.float16  # Half precision (faster, saves memory)
else:
    device_map = "cpu"       # Fall back to CPU
    torch_dtype = torch.float32  # Full precision

# Step 2: Load tokenizer (converts text ↔ numbers)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 3: Load the model weights (the actual "brain")
model = AutoModelForCausalLM.from_pretrained(model_name, ...)

# Step 4: Wrap in a pipeline (makes it easy to call)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
```

Models are **cached** in `_pipeline_cache` — so if you run two experiments with TinyLlama back-to-back, it only loads the model once instead of twice.

## Generation parameters

```python
temperature = 0.1  # How "creative" the model is (0 = robotic, 1 = wild)
top_p = 0.9        # "Nucleus sampling" — only considers the top 90% of likely words
max_new_tokens = 256  # Maximum length of the answer (in tokens, roughly 180 words)
```

A temperature of **0.1** means the model is nearly deterministic — it almost always picks the most likely word. This is good for factual Q&A because we want consistency, not creativity.

## Post-processing the answer

After generation, the model sometimes keeps going and generates a fake second question + answer. We cut this off:

```python
stop_markers = ["\n---", "\nQuestion:", "\n\nQuestion", "---\n", "\n\n\n"]
for marker in stop_markers:
    if marker in answer:
        answer = answer[:answer.index(marker)].strip()
```

---

# PART 8: Evaluation — Measuring How Good the Answers Are

**File responsible:** `evaluator.py`

## The problem: how do you grade an AI's answer?

Unlike a maths test, there's no single correct answer. A good answer can be phrased in many ways. So we use multiple metrics that measure different aspects.

## Metric 1: ROUGE Score

**ROUGE** = Recall-Oriented Understudy for Gisting Evaluation

It was originally created for evaluating **text summarization**. It measures how much of the reference text (the paper's abstract) appears in the generated answer.

We compute 3 variants:

| Metric | What it counts | Example |
|--------|----------------|---------|
| ROUGE-1 | Overlap of **individual words** (unigrams) | How many single words match |
| ROUGE-2 | Overlap of **word pairs** (bigrams) | How many 2-word sequences match |
| ROUGE-L | **Longest matching sequence** of words in order | Measures structural similarity |

**Example:**
- Reference: `"additive models provide flexibility for semiparametric regression"`
- Generated: `"additive models offer increased flexibility in regression"`
- ROUGE-1 F1 = 0.55 (words "additive", "models", "flexibility", "regression" match)
- ROUGE-L = 0.45 (longest common subsequence is "additive models ... flexibility ... regression")

## Metric 2: BLEU Score

**BLEU** = Bilingual Evaluation Understudy (originally for machine translation)

Measures **precision**: of all the words/phrases the model generated, how many also appear in the reference? ROUGE measures recall; BLEU measures precision.

We use **smoothing** (SmoothingFunction.method1) to avoid getting 0 for short sentences.

## Metric 3: Retrieval Relevance

This measures whether the **retrieved chunks** are actually relevant to the question — not the final answer, but the retrieval step. It's computed as the **cosine similarity** between the query embedding and each retrieved chunk embedding, averaged.

A high retrieval relevance means the system found the right paragraphs. A low score means it retrieved wrong/irrelevant chunks, which will lead to a bad answer.

## Metric 4: Answer Relevance

Similar to retrieval relevance, but measures the **cosine similarity** between the generated answer and the original query. If the answer is on-topic, this will be high. If the model went off on a tangent or said "I cannot answer", this will be low.

## Metric 5: Latency

Simply measured with `time.time()`:
```python
start_time = time.time()
answer = generate_answer(pipe, prompt)
latency = time.time() - start_time  # seconds
```

This measures how many seconds the generation took. Important because fast answers = better user experience.

---

# PART 9: The Experiment Orchestrator — Running Everything Automatically

**File responsible:** `experiment.py`

## The big picture

We want to compare 36 different system configurations to find which one works best. Doing this by hand would take forever. `experiment.py` automates the entire thing.

## The nested loop structure

```
FOR each chunk_size in [256, 512, 1024]:
    → Chunk all 100 documents
    
    FOR each embedding model in [MiniLM, BGE]:
        → Embed all chunks
        
        FOR each vector database in [FAISS, Chroma]:
            → Build the search index
            
            FOR each LLM in [TinyLlama, Phi-2]:
                → Load the LLM
                
                FOR each of the 10 test queries:
                    1. Retrieve 5 chunks
                    2. Build prompt
                    3. Generate answer (timed)
                    4. Evaluate (ROUGE, BLEU, relevance)
                    5. Save result to the list
```

**Total:** 3 × 2 × 2 × 2 = **24 configurations** × 10 queries = **240 data points**

## Smart design decisions

**Embeddings are computed once per chunk_size × model combination.** When we switch from FAISS to Chroma but keep the same chunk_size and same model, we reuse the same embeddings. We don't re-compute them.

**LLMs are cached.** Once TinyLlama is loaded, it stays in memory. Reloading it for every configuration would waste 30+ seconds each time.

**Errors don't crash the experiment.** If one LLM fails to load, the code catches the error, records a "0" for that configuration, and continues. This way you don't lose all your results because of one failure.

## After all experiments finish

The results are saved to 3 files in the `results/` folder:

| File | What it contains | Usage |
|------|-----------------|-------|
| `evaluation_results.csv` | All 240 individual rows | Detailed analysis, debugging |
| `evaluation_results.json` | Same data as JSON | Programmatic inspection |
| `summary_results.csv` | 24 rows (one per config), averaged | **Paste into your report!** |

---

# PART 10: The Main Entry Point — Putting It All Together

**File responsible:** `main.py`

This is the file you actually run. It reads the `--mode` argument you pass in the terminal and then calls the right function.

```
python main.py --mode demo
      ↓
main() in main.py
      ↓
if mode == "demo": run_demo()
if mode == "full": run_full_experiment() from experiment.py
if mode == "quick": run_quick_experiment() from experiment.py
if mode == "pipeline": run_pipeline_demo()
```

## The 4 modes in plain English

| Mode | What runs | Time | Output |
|------|-----------|------|--------|
| `pipeline` | Runs 3 fixed questions through 1 config | ~2 min | Terminal printout |
| `quick` | Runs 3 questions × 1 config | ~3-5 min | `quick_results.csv` |
| `full` | Runs 10 questions × 24 configs | Several hours | `evaluation_results.csv` + `summary_results.csv` |
| `demo` | Interactive: you type questions | Runs until you quit | Terminal answers |

---

# PART 11: Config File — The Control Panel

**File responsible:** `config.py`

This file is the **single source of truth** for all settings. Instead of having numbers scattered across 12 different files, every important number is defined here:

```python
NUM_DOCUMENTS = 100       # How many papers to load
CHUNK_SIZES = [256, 512, 1024]  # What chunk sizes to test
CHUNK_OVERLAP_RATIO = 0.2  # How much overlap between chunks (20%)
TOP_K = 5                 # How many chunks to retrieve per query
MAX_NEW_TOKENS = 256      # Max length of generated answers
TEMPERATURE = 0.1         # How deterministic the LLM should be (low = deterministic)
```

If you want to change ANYTHING about the experiment, you change it here — and it automatically affects every module that imports from `config.py`.

---

# PART 12: The Complete Data Flow — Everything Connected

Here is the **entire project as one single flow**, start to finish:

```
STEP 1: DATA LOADING (data_loader.py)
────────────────────────────────────
HuggingFace Dataset
→ Download 100 arXiv papers
→ Extract "article" field from each
→ Clean LaTeX with clean_text()
→ Store: documents[] (100 clean strings) + abstracts[]


STEP 2: CHUNKING (chunking.py)
──────────────────────────────
documents[]
→ RecursiveCharacterTextSplitter
→ Split each paper into 256 / 512 / 1024 char overlapping chunks
→ Store: chunks[] (~580 to ~2100 dicts with text + doc_id + chunk_id)


STEP 3: EMBEDDING (embeddings.py)
──────────────────────────────────
chunks[]
→ SentenceTransformer (MiniLM or BGE)
→ Each chunk text → 384 numbers
→ Store: embeddings[] (NumPy array of shape [num_chunks, 384])


STEP 4: INDEXING (vector_store.py)
────────────────────────────────────
embeddings[] + chunks[]
→ FAISS: load into IndexFlatIP (a fast similarity search structure)
  OR
→ Chroma: load into a cosine-similarity collection


STEP 5: USER ASKS A QUESTION
──────────────────────────────
query = "What are additive models?"


STEP 6: RETRIEVAL (retriever.py + embeddings.py)
──────────────────────────────────────────────────
query
→ Same SentenceTransformer → 384 numbers (query embedding)
→ Search index for top-5 most similar chunk embeddings
→ Return: top 5 chunks with similarity scores


STEP 7: PROMPT BUILDING (prompt_builder.py)
────────────────────────────────────────────
query + top-5 chunks
→ Inject into "Structured" template
→ Build one complete text string (the prompt)


STEP 8: ANSWER GENERATION (generator.py)
──────────────────────────────────────────
prompt
→ TinyLlama or Phi-2
→ Predict the next ~256 tokens
→ Post-process (remove trailing junk)
→ Return: answer string


STEP 9: EVALUATION (evaluator.py)
────────────────────────────────────
answer + query + reference (abstract) + retrieved chunks
→ ROUGE-1, ROUGE-2, ROUGE-L (vs abstract)
→ BLEU (vs abstract)
→ Retrieval Relevance (cosine sim: chunks vs query)
→ Answer Relevance (cosine sim: answer vs query)
→ Latency (seconds)
→ Return: metrics dict


STEP 10: SAVE RESULTS (utils.py)
─────────────────────────────────
All metrics from all 240 experiments
→ results/evaluation_results.csv (raw)
→ results/summary_results.csv (averaged per config)
→ Print comparison table to terminal
```

---

# PART 13: The 12 Files and What Each One Does

| File | Role | Key functions |
|------|------|---------------|
| `config.py` | Control panel for all settings | — |
| `utils.py` | Shared tools used everywhere | `log()`, `save_results_csv()`, `timer()` |
| `data_loader.py` | Load + clean 100 papers | `load_arxiv_data()`, `clean_text()` |
| `chunking.py` | Split papers into chunks | `chunk_documents()` |
| `embeddings.py` | Convert text → numbers | `get_embedding_model()`, `embed_chunks()`, `embed_query()` |
| `vector_store.py` | Store + search embeddings | `FAISSStore`, `ChromaStore`, `create_vector_store()` |
| `retriever.py` | Find relevant chunks | `retrieve()`, `retrieve_and_display()` |
| `prompt_builder.py` | Build prompts | `build_prompt()`, `build_context()` |
| `generator.py` | Generate answers with LLM | `load_llm()`, `generate_answer()` |
| `evaluator.py` | Measure answer quality | `compute_rouge()`, `compute_bleu()`, `evaluate_single()` |
| `experiment.py` | Run all 24 configs automatically | `run_experiment()`, `run_full_experiment()`, `run_quick_experiment()` |
| `main.py` | Entry point, mode switcher | `main()`, `run_demo()`, `run_pipeline_demo()` |
| `app.py` | Streamlit interactive GUI | `load_data()`, `get_chunks()`, Streamlit layouts |

---

# PART 14: The Streamlit Interactive UI

**File responsible:** `app.py`

While scripts are great, an interactive **Graphical User Interface (GUI)** is the best way to present a project to teachers and test assumptions instantly.

You can launch the UI by running:
```bash
streamlit run app.py
```

## How the UI works:
1. **Dynamic Configuration Settings Box:** The left sidebar allows you to hot-swap components on-the-fly without editing code. You can choose:
   - Chunk Size: (256, 512, 1024)
   - Embedding Model: (MiniLM or BGE)
   - Vector Store: (FAISS or ChromaDB)
   - LLM Generator: (TinyLlama or Phi-2)
2. **"Live Demo" Tab:** After selecting your preferred configuration, type any question. The system will build your entire custom RAG pipeline in seconds, retrieve the chunks, generate the answer, and print the evaluation metrics automatically.
3. **"Comparison Dashboard" Tab:** This reads the pre-computed `summary_results.csv` generated by the full experiment. It displays automatic bar charts comparing all 24 configurations visually—allowing you to instantly prove to examiners *why* one configuration is better than another.

---

# PART 15: Scaling with Google Colab (GPU Acceleration)

Large language models like `Phi-2` (2.7 Billion parameters) are mathematically intense to run on a standard CPU. Generating an answer can take 60-90 seconds.

To make the experiment feasible, the code in `generator.py` features an **hardware auto-detection framework**.

If the code is run on **Google Colab**:
1. It detects `torch.cuda.is_available()`.
2. It automatically switches the models to **GPU (cuda)** and uses `float16` precision parameters.
3. This slashes the answer generation time of Phi-2 from ~60 seconds to ~2 seconds per question.
4. An experiment that takes 4 hours locally will finish in 10-15 minutes seamlessly.

---

# PART 16: Why This Is a Good Assignment (What Makes It Impressive)
