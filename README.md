# 📌 Generative AI – Video 1: Simple Notes

## 🎥 Introduction

GenAI changes very fast — every day new tools, new models, and new research appear. So he spent almost 3 months researching, planning, and designing a proper curriculum before teaching it.

This video explains:

* Why GenAI matters
* What GenAI actually is
* Why learning GenAI is important today
* The journey and thought process behind designing the GenAI curriculum

## 🚀 What is Generative AI?

Generative AI is a type of AI that can create new content such as text, images, music, videos, and even code. It learns patterns from existing data and mimics human creativity.

### Brief History

AI has existed for 60–70 years. Multiple approaches were developed:

* Symbolic AI (Expert systems)
* Fuzzy Logic
* Evolutionary Algorithms
* NLP
* Computer Vision
* Machine Learning (ML)

ML made AI useful for tasks like prediction, classification, and ranking. But ML couldn’t generate human-like creative output — until Generative AI arrived.

### The Biggest Power of GenAI

Generative AI can **generate new content**, not just predictions. This changed everything:

* Earlier belief: “AI can never replace human creativity”
* Now: GenAI can write articles, create videos, generate images, and even build software

## 🧭 Where Does GenAI Fit in AI Landscape?

Imagine nested circles:
**AI → Machine Learning → Deep Learning → Generative AI**

* **AI** – broad umbrella
* **ML** – learns patterns from data
* **DL** – neural networks (especially transformers)
* **GenAI** – evolved from Deep Learning + Transformers

## 🌍 GenAI Impact Areas

Generative AI has transformed several industries:

### 1️⃣ Customer Support

AI chatbots now handle the first level of queries, reducing cost and improving efficiency.

### 2️⃣ Content Creation

AI can generate blogs, videos, images, and professional-level creative content.

### 3️⃣ Education

Tools like ChatGPT act as personal tutors for instant explanations, doubt-solving, and study planning.

### 4️⃣ Software Development

GenAI can write production-ready code, reducing workload and improving developer efficiency.

## ❓ Is Generative AI a Successful Technology?

To judge success, compare tech using 5 questions:

| Question              | Internet | Crypto  | GenAI              |
| --------------------- | -------- | ------- | ------------------ |
| Solves real problems? | Yes      | ?       | Yes                |
| Useful daily?         | Yes      | No      | Yes                |
| Economic impact?      | Huge     | Small   | Huge               |
| Creates jobs?         | Yes      | Limited | Yes (AI engineers) |
| Accessible to all?    | Yes      | No      | Yes                |

**Conclusion:** GenAI follows the path of the **Internet**, not **Crypto**. It is here to stay.

## 🧠 The Core Mental Model of GenAI

Everything in GenAI revolves around **Foundation Models**.

### What are Foundation Models?

* Large AI models trained on massive datasets
* Have billions of parameters
* Perform multiple tasks
* Example: **LLMs (Large Language Models)**

**Foundation Models = Learn once → Perform many tasks**
They can answer questions, summarize text, write code, and generate content.

## 🪄 Two Sides of Generative AI

The GenAI world splits into two roles:
**GEN AI = USING Foundation Models + BUILDING Foundation Models**

### 1️⃣ User Perspective (Using Models)

Suitable for developers who want to build applications.
You learn:

* LLM APIs
* LangChain
* RAG
* Prompt Engineering
* Vector Databases
* AI Agents
* LLM Ops
* Basic Fine-tuning

### 2️⃣ Builder Perspective (Creating Models)

For those who want to build new AI models.
You learn:

* Transformer architecture
* Tokenization
* Pre-training
* Optimization
* Quantization
* Distributed training
* Advanced Fine-tuning
* Deployment

## 🎯 Who Should Learn What?

| Role                   | Suggested Path |
| ---------------------- | -------------- |
| Software Developer     | User Side      |
| Research / ML Engineer | Builder Side   |
| AI Engineer            | Both Sides     |

Knowing both sides increases salary and career opportunities.

## 📚 Curriculum Plan (Roadmap)

### Builder Track

* Transformers
* Types of Transformers (BERT, GPT, Encoder–Decoder)
* Pre-training
* Optimization
* Fine-tuning (Advanced)
* Evaluation
* Deployment

### User Track

* Build basic LLM apps
* Prompt engineering
* RAG
* Fine-tuning (Basic)
* AI Agents
* LLM Ops
* Multimodal GenAI

Both sides will be covered in small playlists.

## ❗ Why No Paid Course Yet?

* GenAI evolves rapidly
* He is still mastering it
* Doesn't want to deliver incomplete knowledge
* YouTube enables free learning and community feedback

A paid version may come later.

## 🕒 Timeline

* 2–3 videos weekly
* Approx. 1 year to cover full curriculum

This is worth it because GenAI is new, powerful, and expanding.

## 🎉 Final Thoughts

GenAI is transforming industries and global economies. Learning it now gives you a head start. Whether you're a developer or student, this is the perfect time to join the AI revolution.

## 💬 Summary in One Line

**Generative AI is the future of creativity, automation, and intelligence — learn it now before it becomes a basic requirement.**


# 📌 Generative AI – Video 2: LangChain Playlist Notes

## 🎥 Introduction

This video starts the **User Side** journey of the Generative AI curriculum.

In the previous video, Nitesh explained that GenAI has two major parts:

* **Builder Side** – Creating foundation models
* **User Side** – Using these models to build applications

The LangChain playlist belongs to the **User Side** of the curriculum.

## 🔁 Recap of Previous Video

Generative AI is divided into two tracks:

### **Builder Side**

Includes concepts like:

* Transformers
* Tokenization
* Pre-training
* Fine-tuning
* Optimization
* Deployment

### **User Side**

Focuses on:

* Building LLM-based applications
* Prompt Engineering
* RAG (Retrieval Augmented Generation)
* AI Agents
* LLM Ops and more

LangChain helps in building LLM-powered apps, so it is the first step in the user-side journey.

## ❓ What is LangChain?

LangChain is an open-source framework that makes it easy to build applications powered by Large Language Models (LLMs).

Using LangChain, you can build:

* Chatbots
* Question-answering systems
* RAG-based applications
* Autonomous AI Agents
* Many other GenAI-powered applications

LangChain provides:

* Ready-to-use components
* End-to-end development tools
* Integrations with multiple LLMs

## ⭐ Why is LangChain Popular?

### 1️⃣ Supports Almost Every LLM

Works with both open-source and closed-source models:

* OpenAI GPT models
* Anthropic Claude
* Google Gemini
* Hugging Face models
* Ollama etc.

### 2️⃣ Simplifies LLM App Development

* Removes complex boilerplate code
* Provides **Chains** to combine multiple steps easily

### 3️⃣ Easy Integrations

Connects effortlessly with:

* Databases
* APIs
* Data sources
* Deployment services

LangChain includes wrappers to integrate tools without writing everything manually.

### 4️⃣ Free and Open Source

* 100% free
* Very active developer community
* Multiple versions released within 1–2 years

### 5️⃣ Supports All Major GenAI Use Cases

Works for building:

* Chatbots
* RAG-based systems
* Autonomous AI agents
* Memory-based conversational apps

LangChain is like a **Swiss Army Knife** for LLM app development.

## 🎯 Why Start with LangChain First?

LangChain overlaps with almost every topic of the **User Side** of GenAI.

By learning LangChain, you get exposure to:

* LLM APIs
* Hugging Face and Ollama
* Prompt engineering basics
* RAG workflows
* AI agents
* Parts of LLM Ops

Once LangChain is complete, learning other concepts becomes easier.

## 🗂 LangChain Playlist Structure

The complete playlist is divided into **three major parts**:

### 📍 Part 1 — Fundamentals

You will learn:

* What is LangChain and why it is needed
* LangChain components
* Integrating LLM models
* Working with prompts
* Parsing LLM outputs
* Runnables and LCEL (LangChain Expression Language)
* Understanding and using Chains
* Memory in chat-based applications

### 📍 Part 2 — RAG (Retrieval Augmented Generation)

You will learn:

* Document loaders
* Text splitters
* Embeddings
* Vector databases
* Retrievers
* Building a complete RAG application from scratch

### 📍 Part 3 — AI Agents

Topics include:

* Tools and toolkits
* Tool calling
* Agent workflows
* Building a full-fledged AI Agent

**Total planned videos:** ~17 (may increase if required)

## 🎯 Focus Areas for This Playlist

Nitesh’s teaching goals:

### 🔹 Updated Content

* Teach using the latest **LangChain v3** version
* Earlier versions were different, so updated learning is essential

### 🔹 Clarity over Copy-Paste

* Not just writing code
* Understanding how LangChain works internally

### 🔹 Conceptual Understanding

Learn core ideas like:

* Chains
* LCEL
* Memory
* Runnables

Concepts stay relevant even if new versions release.

### 🔹 Covering the Most Important 80%

* Only the most practical LangChain features used in real projects

## 🕒 Timeline

* Playlist starts in **1–2 days**
* **2 videos per week**
* Around **8 weeks** to complete all videos
* Runs along with other courses like PyTorch and the Builder-side curriculum

## 🔚 Final Thoughts

LangChain is a great starting point if you want to build Generative AI applications in the real world.
Once you master it, you can create:

* Chatbots
* RAG systems
* AI agents
* Many other GenAI workflows

This playlist is designed to be practical, clear, and future-proof.

## 💬 One-Line Summary

**LangChain is the best starting point to build real-world Generative AI applications using LLMs.**


# 📌 LangChain – Video 3 Notes

## What is LangChain?

LangChain is an open-source framework used to build applications powered by Large Language Models (LLMs) like GPT. If you want to create apps that use AI models for text understanding and generation, LangChain makes the whole process easier.

---

## ❓ Why do we need LangChain?

Before LangChain, building an LLM-based application was very complex. You needed to:

* Upload documents
* Split documents into chunks
* Generate embeddings
* Store embeddings in a database
* Retrieve relevant content
* Send it to an LLM
* Handle responses properly

All these tasks required a lot of engineering and complicated code. LangChain simplifies this and handles all these complex parts for you.

---

## 🧠 How an App Works Without LangChain (Example)

Imagine an app where you upload a PDF and chat with it. You can ask questions like:

* "Explain page 5 like I am 5 years old"
* "Create true/false questions for Linear Regression"
* "Give summary of Decision Trees"

To build this app, you need to:

1. Upload PDF → store it in cloud storage
2. Split PDF → into multiple pages/paragraphs
3. Generate embeddings → convert each page into a number vector
4. Save embeddings → in a special database
5. User asks a question → convert the question into embeddings
6. Semantic Search → find the most relevant pages
7. LLM Brain → read only those pages, understand the question, generate an answer

This architecture is powerful but very difficult to implement manually.

---

## 🔍 What is Semantic Search?

Semantic Search performs search based on meaning instead of keyword matching.

**Example:**
Question: "How many runs has Virat Kohli scored?"
The system compares the meaning of the question with embeddings of each paragraph and picks the most relevant paragraph automatically.

---

## 🚧 Main Challenges Without LangChain

| Challenge             | Explanation                                |
| --------------------- | ------------------------------------------ |
| Understanding queries | LLM must understand natural language       |
| Generating answers    | It should produce context-aware text       |
| Infrastructure        | Running LLMs on your servers is expensive  |
| Orchestration         | Connecting 5–6 components manually is hard |

LangChain solves all of these.

---

## 🎯 Benefits of LangChain

### 1️⃣ Chains

You can create pipelines where output of one step becomes input of another.

```
Load PDF → Split → Embeddings → Store → Search → LLM → Answer
```

No manual wiring needed.

### 2️⃣ Model Agnostic

Use any model (OpenAI, Google Gemini, LLaMA, etc.)
Switching requires only 1–2 lines of code.

### 3️⃣ Huge Ecosystem

* Many document loaders
* Many text splitters
* Many embedding models
* Many vector databases

Everything is plug-and-play.

### 4️⃣ Memory Support

LangChain remembers past conversation context.

If user asks:

```
What are the parts of Linear Regression?
```

Then asks:

```
Give interview questions on this algorithm.
```

The system still knows "this" refers to Linear Regression.

---

## 🏗️ What Can You Build With LangChain?

| Use Case                   | Explanation                                    |
| -------------------------- | ---------------------------------------------- |
| 🤖 Chatbots                | Customer support bots like Swiggy, Zomato      |
| 🎓 AI Knowledge Assistants | Ask doubts directly from lecture notes/books   |
| 🧭 AI Agents               | Bots that perform actions (e.g., book tickets) |
| 🔁 Workflow Automation     | Automate repeated business tasks               |
| 📚 Research Summaries      | Summarize PDFs, research papers, books         |

LangChain makes building these apps easy and scalable.

---

## 🔁 Alternatives to LangChain

Other frameworks you may hear about:

* **LlamaIndex**
* **Haystack**

These also help build LLM applications, but LangChain is currently the most popular.

---

## ✅ Conclusion

LangChain helps us build AI apps powered by LLMs easily. It handles the messy engineering so you can focus on your idea. It provides tools, memory, chains, and integrations that make LLM apps production-ready.

This is why LangChain is becoming a very important technology in the world of AI.


# 📌 LangChain – Video 4 

## 🔷 Why this video is important

The speaker explains that before writing code in LangChain, we must first understand how the framework is organized and what components it provides. This video builds a roadmap for the upcoming tutorials.

## 🔁 Quick Recap of Previous Video

In the last video, we learned:

* LangChain is an open-source framework that helps build LLM-powered apps.
* It becomes difficult to create apps that:

  * Read documents
  * Split them into chunks
  * Store embeddings
  * Search relevant content
  * Send it to an LLM
  * Get and format the answer

LangChain solves this by connecting all parts efficiently using minimal code.

We also learned:

* Chains help link components together like a pipeline.
* LangChain is model-agnostic — meaning we can switch models like OpenAI, Gemini, Mistral, etc. without rewriting the whole code.
* LangChain is used today for:

  * Chatbots
  * Knowledge assistants
  * AI agents

## 📦 LangChain Components

LangChain has six major components:

1. Models
2. Prompts
3. Chains
4. Memory
5. Indexes
6. Agents

If you understand these six, you understand most of LangChain.

## 1️⃣ Models

Models are the core interface that communicates with AI models (LLMs).

### Why models are needed:

Different companies give different APIs for communication.

Without LangChain, you need different code for:

* OpenAI’s GPT
* Anthropic’s Claude
* Google Gemini

Each API behaves differently.

LangChain standardizes this by giving a single interface that works for all models.
Changing model provider requires just **1–2 lines of code**.

### Types of Models

* **Language Models** → text in, text out
  *Example: ChatGPT replies to a question*

* **Embedding Models** → text in, vector out
  *Used for semantic search*

## 2️⃣ Prompts

Prompts are the input we give to LLMs.

LLM output depends heavily on how we write prompts. Changing even one word can change the answer.

### Types of prompts in LangChain:

* **Dynamic prompts** — Fill values later using placeholders
  *Example: "Summarize {topic} in a {tone}"*

* **Role-based prompts** — Tell the model who it is
  *Example: "You are an experienced doctor"*

* **Few-shot prompts** — Provide examples first, then ask a question

Prompt engineering is now a real job profile because LLMs are very sensitive to input wording.

## 3️⃣ Chains

Chains let you create pipelines.

They automate the process where:

* Output of one step becomes input to the next
* You don’t manually pass values

### Examples:

* English text → Translate to Hindi → Summarize it → Return final result
* **Parallel chains**: multiple models run at the same time
* **Conditional chains**: different execution based on rules

Chains remove repeated coding and handle flow automatically.

## 4️⃣ Indexes

Indexes allow your LLM to access external knowledge, such as:

* PDFs
* Websites
* Company databases

### Indexes are made of four parts:

| Component       | Meaning                              |
| --------------- | ------------------------------------ |
| Document Loader | Load PDFs, pages, files              |
| Text Splitter   | Break large text into chunks         |
| Vector Store    | Store embeddings (special DB)        |
| Retriever       | Find relevant chunks and return them |

Indexes enable private and custom knowledge search — something ChatGPT cannot do alone.

## 5️⃣ Memory

LLMs are stateless, meaning they forget previous messages unless we send history again.

Memory solves this by storing previous conversation context.

### Types of memory:

* **Conversation buffer** → stores full history
* **Window memory** → stores recent messages
* **Summary memory** → stores summarized history
* **Custom memory** → stores special info like preferences

Memory makes chatbots feel continuous and human-like.

## 6️⃣ Agents

Agents are advanced chatbots that can think and take actions.

### Difference:

| Chatbot    | Agent                  |
| ---------- | ---------------------- |
| Only talks | Talks + performs tasks |

### Agents have:

* **Reasoning ability** → Break tasks into steps
* **Tools access** → They can call APIs, search the web, calculate, etc.

#### Example:

User: *Multiply today's temperature of Delhi by 3*

Agent:

* Gets weather from API
* Uses calculator
* Returns result

This is why agents are considered the next big thing in AI.

## 🎯 Conclusion

* LangChain has six core components.
* Understanding them gives a complete foundation.
* Next videos will dive deeper into each component, starting with Models.
* No code yet — this video builds conceptual understanding first.

# 📌 LangChain Models – Video 3

👋 **Introduction**

In this section, we explore one of the most important components of LangChain: **Models**. By the end of this part, you will understand:

* What models are in LangChain
* Types of models
* How to use them with code examples
* Differences between **LLMs** and **Chat Models**

This part is practical and code-driven, making it easier to understand.

---

## 🔁 **Recap of Previous Videos**

### **Video-1 Covered:**

* What LangChain is
* Why LangChain is needed
* Types of applications built using LangChain
* LangChain alternatives

### **Video-2 Covered:**

LangChain core components:

* Models
* Prompts
* Chains
* Agents
* How and where each component is used

---

## 🤖 **What Are Models in LangChain?**

LangChain provides a common interface to interact with different AI models without worrying about how each model behaves internally.

### **Types of Models**

| Model Type                 | Input | Output            | Use-case                          |
| -------------------------- | ----- | ----------------- | --------------------------------- |
| **Language Models (LLMs)** | Text  | Text              | Chatbots, summarization, Q/A      |
| **Embedding Models**       | Text  | Numbers (Vectors) | Semantic search, RAG applications |

### **Simple Definition**

> **Models in LangChain act as a bridge between your code and various AI models.**

---

## 🧠 **Language Models**

A **Language Model** takes text as input and gives text as output.

**Example:**

```
Input: "What is the capital of India?"
Output: "New Delhi"
```

### **Types of Language Models**

| Type           | Name                 | Purpose                              |
| -------------- | -------------------- | ------------------------------------ |
| **LLM**        | Large Language Model | General text generation              |
| **Chat Model** | Chat-based model     | Used to build assistants or chatbots |

---

## 🔄 **LLM vs Chat Model**

| Feature        | LLM                     | Chat Model                    |
| -------------- | ----------------------- | ----------------------------- |
| Training       | Trained on generic text | Trained on chat conversations |
| Input          | Single text string      | List of chat messages         |
| Output         | Plain text              | Structured output             |
| Role awareness | ❌ No                    | ✅ Yes                         |
| Memory support | ❌ No                    | ✅ Yes                         |
| Use case       | Summarization, coding   | Assistants, chatbots          |

### 💡 Important

LangChain is gradually moving away from **LLMs** in favor of **Chat Models**.

---

## 🛠 **Setup for Coding**

### 1️⃣ Create a Project Folder

```
LangChain-Models/
```

### 2️⃣ Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Required Libraries

Create a file `requirements.txt` and add the required packages.
Then run:

```
pip install -r requirements.txt
```

### 4️⃣ Test LangChain Installation

```python
import langchain
print(langchain.__version__)
```

---

## 📁 **Project Structure**

```
LangChain-Models/
│── requirements.txt
│── .env
│── llms/
│── chat_models/
│── embedding_models/
```

---

## 💻 **LLM Code Demo Using OpenAI**

### Step-1: Add API Key in `.env`

```
OPENAI_API_KEY="your_api_key_here"
```

### Step-2: Write Code (llms/llm_demo.py)

```python
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model="gpt-3.5-turbo-instruct")
result = llm.invoke("What is the capital of India?")
print(result)
```

**Output:**

```
New Delhi
```

---

## 💬 **Chat Model Demo Using GPT-4o**

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o")
result = model.invoke("What is the capital of India?")
print(result.content)
```

**Output:**

```
New Delhi
```

> Chat Models return structured responses, so we use `.content`.

---

## 🎛 **Important Parameters**

### **Temperature** (Controls creativity)

| Value     | Behavior              |
| --------- | --------------------- |
| 0 – 0.3   | Accurate, predictable |
| 0.5 – 0.7 | Balanced              |
| 0.9 – 1.5 | Creative              |

### Example

```python
model = ChatOpenAI(model="gpt-4o", temperature=1.2)
```

### **Max Tokens** (Output length limit)

```python
model = ChatOpenAI(model="gpt-4o", max_completion_tokens=20)
```

---

## ✨ **Why Chat Models Are Better**

✔ Handle conversation history
✔ Understand roles
✔ Ideal for chatbots & AI assistants

Industry adoption is shifting toward Chat Models.

---

## 🚀 **What’s Next?**

Upcoming topics:

* Anthropic Claude Chat Model
* Google Gemini Chat Model
* Open-source models (LLaMA, Mistral, DeepSeek)
* Embedding Models + RAG Demo

# 🔍 Open-Source Models –

Open-source models are freely available models that developers can download and run on their own machines. They offer flexibility and control, but also come with challenges.

---

## ✅ Advantages of Open-Source Models

* You can modify and fine-tune them.
* You can run them locally without depending on a third party.
* Good for privacy-sensitive applications.

---

## ❌ Disadvantages of Open-Source Models

| Problem                    | Explanation                                                                                                                        |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Strong hardware required   | Running models on your machine needs a powerful GPU, lots of RAM, and good storage. Weak machines hang or crash.                   |
| Complex setup              | Installing dependencies, downloading models, and configuration can be difficult.                                                   |
| Less refined responses     | Open-source models are not fine-tuned with human feedback (RLHF), so answers may feel less polished than OpenAI / Gemini / Claude. |
| Limited multimodal support | Most open-source models currently handle text only, not images or audio.                                                           |

---

## 🚀 Working With Open-Source Models

We use two approaches:

### 1️⃣ Using Hugging Face Inference API (Online)

* The model stays on Hugging Face servers.
* We call it using an API key.
* No need to download the model.

**Steps:**

1. Create a Hugging Face account.
2. Go to **Access Tokens** and create a token.
3. Save the token in `.env` file:

```
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

4. Import classes in Python:

```python
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
```

5. Provide model repo ID (example model used in lecture):

```
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

6. Invoke the model and print the result.

---

### 2️⃣ Download the Model Locally

Here we download the model to our computer and run it without API calls.

**Important Notes**

* First-time execution downloads:

  * Model weights
  * Tokenizer
  * Config files
* These are stored in Hugging Face cache
* On weak hardware, the model may take 10+ minutes to run, may hang the PC

**To change download location:**

```python
import os
os.environ["HF_HOME"] = "D:/huggingface_cache"
```

**Then run the model using:**

```python
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
```

---

## 🧩 Embedding Models

Embedding Models convert text into numeric vectors. These vectors represent meaning, so we can compare texts based on similarity.

### Why embeddings?

Used for:

* Semantic search
* RAG (Retrieval Augmented Generation)
* Clustering
* Document similarity checking

---

## 🔡 Embeddings Using OpenAI

```python
from langchain_openai import OpenAIEmbeddings
emb = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)
```

**Output** → A 32-dimensional vector representing the meaning of the text.

* More dimensions = More context
* Less dimensions = Cheaper & faster

### 📚 Embedding Multiple Documents

```python
emb.embed_documents(["Paris is the capital of France", "Delhi is the capital of India"])
```

This returns a list of embedding vectors.

---

## 🔥 Using Open-Source Embedding Model (Local)

**Model used:** `sentence-transformers/all-MiniLM-L6-v2`

* Size: ~90 MB
* Output: 384-dim vectors
* Good for semantic search

```python
from langchain_huggingface import HuggingFaceEmbeddings
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

---

## 🔍 Document Similarity Search (Mini Project)

**Goal:** Given a user question, find which document is most related.

**Steps:**

1. Convert all documents into vectors (embeddings)
2. Convert query into a vector
3. Compare query vector with document vectors using cosine similarity
4. Highest score = most relevant document

```python
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(query_vector, document_vectors)
```

This is the foundation of RAG-based AI systems.

---

## 💡 Why Store Embeddings?

Generating embeddings again and again is costly.

**Better approach:**

| Need                        | Solution                                              |
| --------------------------- | ----------------------------------------------------- |
| Store embeddings            | Use a Vector Database (like Pinecone, Chroma, Qdrant) |
| Retrieve nearest embeddings | Use similarity search                                 |

This is how modern chatbots search your documents efficiently.

---

🚀 LangChain Playlist – Video 6

🧠 What We Learned So Far (Recap)

* **Video 1** – Introduction to LangChain and why we need it as a framework.
* **Video 2** – The 6 major components of LangChain.
* **Video 3** – Deep dive into the Models component.
* **Video 4 (current video)** – Understanding Prompts in LangChain.

---

🔥 **What Are Prompts?**

A prompt is the message you send to an LLM (like GPT-4) asking it to perform a task.

**Example:**

```
model.invoke("Write a five-line poem on cricket")
```

The text `"Write a five-line poem on cricket"` → **Prompt**

### Prompts can be:

| Type  | Example                                         |
| ----- | ----------------------------------------------- |
| Text  | "Explain transformers in simple words"          |
| Image | Upload an image → ask "Identify objects inside" |
| Audio | Upload a song → "Who is the singer?"            |
| Video | Upload video → "Summarize this"                 |

In this video, we focus only on **text prompts**, because 99% of real-world apps use them today.

---

💡 **Why Are Prompts Important?**

* The output of LLMs depends heavily on the prompt.
* A slight change in prompt → completely different answer.
* That’s why **Prompt Engineering** is a job profile now.

---

❄️ **Static vs. Dynamic Prompts**

### ❌ Static Prompt

User types the complete prompt manually.

```
[ Enter Prompt Here ] → "Summarize the paper Attention Is All You Need"
```

**Problems:**

* Users can type wrong text, spelling mistakes, unclear instructions
* Inconsistent results
* No control over structure

### ✔️ Dynamic Prompt

We create a prompt template and fill only necessary user inputs.

**Example template:**

```
Please summarize the research paper titled {paper_input}
using {style_input} explanation in {length_input} format.

Make sure the summary is accurate and simple.
```

User only selects:

| Paper                       | Style                            | Length                |
| --------------------------- | -------------------------------- | --------------------- |
| "Attention is All You Need" | Code-heavy / Math-heavy / Simple | Short / Medium / Long |

**Benefits:**

* Consistent responses
* Controlled structure
* Better UX

---

🛠️ **Building a Dynamic Prompt UI (Streamlit)**

### Install Dependencies

```
pip install streamlit langchain openai python-dotenv
```

### Import & Load Model

```
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
model = ChatOpenAI()
```

### UI for Dynamic Inputs

```
paper_input = st.selectbox("Select Research Paper", [
    "Attention Is All You Need",
    "Word2Vec",
    "BERT",
    "Transformer"
])

style_input = st.selectbox("Select Style", [
    "Simple", "Math-heavy", "Code-oriented"
])

length_input = st.selectbox("Summary Length", [
    "Short", "Medium", "Long"
])
```

---

🧱 **Creating a Prompt Template**

```
from langchain_core.prompts import PromptTemplate

template = """
Please summarize the research paper titled {paper_input}
using {style_input} explanation in {length_input} length.

Include mathematical equations if present and explain concepts clearly.
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["paper_input", "style_input", "length_input"]
)
```

### Invoke the prompt

```
filled_prompt = prompt.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})

result = model.invoke(filled_prompt)
st.write(result.content)
```

---

❓ **Why Use PromptTemplate Instead of f-strings?**

| Feature            | f-string | PromptTemplate              |
| ------------------ | -------- | --------------------------- |
| Validation         | ❌ No     | ✅ Yes                       |
| Reusable templates | ❌ Hard   | ✅ Easy (can save/load JSON) |
| Works with Chains  | ❌ No     | ✅ Yes                       |

**Validation example:**

```
prompt = PromptTemplate(
    template=template,
    input_variables=["paper_input", "style_input"],
    validate_template=True
)
```

If a variable is missing → error immediately.

---

💾 **Saving Prompt Template to JSON**

```
prompt.save("template.json")
```

### Load later

```
from langchain_core.prompts import load_prompt
prompt = load_prompt("template.json")
```

---

🔗 **Using PromptTemplate with Chains**

```
chain = prompt | model
result = chain.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})
```

Only **one invoke** is needed now.

---

🤖 **Building a Simple Chatbot**

```
model = ChatOpenAI()

while True:
    msg = input("You: ")
    if msg == "exit":
        break
    reply = model.invoke(msg)
    print("AI:", reply.content)
```

**Problem** → AI forgets previous messages.

---

🧠 **Add Chat History**

```
chat_history = []

while True:
    user_msg = input("You: ")
    if user_msg == "exit":
        break

    chat_history.append(user_msg)
    result = model.invoke(chat_history)
    chat_history.append(result.content)

    print("AI:", result.content)
```

Still missing: who said what.

---

🏷️ **Using Message Types**

LangChain supports **3 types of messages**:

| Message Type  | Meaning               |
| ------------- | --------------------- |
| SystemMessage | Sets AI role/behavior |
| HumanMessage  | User input            |
| AIMessage     | Model response        |

```
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage(content="You are a helpful assistant.")
]

while True:
    user_msg = input("You: ")
    if user_msg == "exit":
        break

    messages.append(HumanMessage(content=user_msg))
    result = model.invoke(messages)
    messages.append(AIMessage(content=result.content))

    print("AI:", result.content)
```

Now each message is labeled → AI understands context properly.

---

🎯 **Summary**

| Concept        | Why Important                                |
| -------------- | -------------------------------------------- |
| Prompts        | Control model output                         |
| Static Prompt  | Bad for real apps                            |
| Dynamic Prompt | Better customisation + structure             |
| PromptTemplate | Validation + Reusability + Works with Chains |
| Message Types  | Enable memory + context-aware chatbots       |

---

🟢 Quick Recap of invoke()

You can use `model.invoke()` in two ways:

---

### **1️⃣ Send a Single Message**

Used for **one-time queries** like:

* Summarizing a paper
* Translating text
* Asking a standalone question

You can:
✔️ send a **static prompt**
✔️ or a **dynamic prompt** using `PromptTemplate`

---

### **2️⃣ Send a List of Messages**

Used for **multi-turn conversations** (*chatbots*)

Messages can be:

* **SystemMessage** → defines AI behavior
* **HumanMessage** → user input
* **AIMessage** → model responses

You maintain a **chat history** list and pass it each time.

---

## 🆕 CHAT PROMPT TEMPLATE

So far we used `PromptTemplate` for single prompts.
For multiple messages, LangChain gives us:

### **`ChatPromptTemplate`**

Use it when you want **dynamic values inside multiple messages** in a conversation.

---

### 📌 Why do we need `ChatPromptTemplate`?

Consider this prompt:

```
System Message → You are a helpful {domain} expert
Human Message → Explain about {topic}
```

Both `{domain}` and `{topic}` are **dynamic** → filled at runtime.

---

## 🧱 Creating a Chat Prompt Template

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful {domain} expert."),
    HumanMessage(content="Explain in simple terms, what is {topic}?")
])

prompt = chat_template.invoke({
    "domain": "cricket",
    "topic": "What is doosra?"
})

print(prompt)
```

### ❗ Issue

The placeholders **won’t fill** using this syntax.
LangChain treats message classes differently here.

---

### ✔️ Correct Syntax (**Recommended**)

Use **tuples** instead of message classes:

```python
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {domain} expert."),
    ("human", "Explain in simple terms, what is {topic}?")
])

prompt = chat_template.invoke({
    "domain": "cricket",
    "topic": "What is doosra?"
})

print(prompt)
```

### ✅ Output

```
System: You are a helpful cricket expert.
Human: Explain in simple terms, what is doosra?
```

---

## 🧩 Difference Summary

| Feature        | PromptTemplate | ChatPromptTemplate |
| -------------- | -------------- | ------------------ |
| Use Case       | Single prompt  | Multiple messages  |
| Dynamic fields | Yes            | Yes                |
| Messages       | No             | Yes                |
| Best for       | Summaries, QA  | Chatbots, agents   |

---

## 🟣 MESSAGE PLACEHOLDER

### ❓ What is it?

A **Message Placeholder** inserts an **entire list of messages** (chat history) into a `ChatPromptTemplate` dynamically.

Used when:
✔️ Chat history is stored somewhere (DB / file)
✔️ You want new messages to continue previous context

---

### 🧠 Real Use Case

A user chatted earlier:

```
User: I want a refund for order 12345
Bot: Refund initiated
```

Stored this chat.

Next day user asks:

```
Where is my refund?
```

The bot must understand previous context → load chat history.

---

## 🛠️ Code Example – Using MessagePlaceholder

### Step 1: Import

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
```

### Step 2: Create Chat Template

```python
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support agent."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])
```

`MessagesPlaceholder("chat_history")` → placeholder for past messages.

---

### Step 3: Load Previous Chat History

```python
chat_history = []
with open("chat_history.txt") as f:
    for line in f.readlines():
        chat_history.append(line.strip())
```

### Step 4: Invoke Template

```python
prompt = chat_template.invoke({
    "chat_history": chat_history,
    "query": "Where is my refund?"
})

print(prompt)
```

### 🟢 Output

```
System: You are a helpful customer support agent.
Human: I want a refund for order 12345
AI: Your refund is initiated...
Human: Where is my refund?
```

Now the LLM understands context from previous chats.

---

## 🎯 Final Takeaways

| Concept                   | Purpose                               |
| ------------------------- | ------------------------------------- |
| PromptTemplate            | Create dynamic single prompts         |
| ChatPromptTemplate        | Create dynamic conversational prompts |
| MessagesPlaceholder       | Insert past chat history dynamically  |
| invoke() single message   | One-time tasks                        |
| invoke() list of messages | Chatbots / multi-turn dialogue        |

---

🎉 **End of Prompts Component**

You now understand:

* Static vs Dynamic prompts
* PromptTemplate
* ChatPromptTemplate
* Message types (System/Human/AI)
* MessagePlaceholder
* Why prompts are critical in LangChain


# 📌 Structured Output in LangChain — Video 7

## 🟢 Recap

* In the previous lesson, we learned how to give inputs (prompts) to LLMs.
* Today, we focus on the output generated by LLMs and how to process it.
* Normally, LLMs give unstructured text output.
* Our goal: Convert LLM output into structured formats (like JSON) so machines and APIs can use it easily.

## ❓ What is Structured Output?

### 🔴 Unstructured Output

LLMs generally reply in plain text.

**Example:**

```
Q: What is the capital of India?
A: New Delhi is the capital of India.
```

➡️ This is text-only, no structure.

### 🟢 Structured Output

Output is returned in a specific data format like JSON:

```
[
  { "time": "morning", "activity": "Visit Eiffel Tower" },
  { "time": "afternoon", "activity": "Visit Louvre Museum" },
  { "time": "evening", "activity": "Dinner at a café" }
]
```

➡️ Machines can easily read, store, and process this.

## 💡 Why Do We Need Structured Output?

| Use Case        | Explanation                                                  |
| --------------- | ------------------------------------------------------------ |
| Data Extraction | Extract structured details from resumes → store in DB        |
| APIs            | Convert messy reviews into structured info → expose API      |
| Agents          | Agents need structured info to call tools (e.g., calculator) |

🔥 Without structured data, LLMs can talk to humans, but not to machines.

## ⚙️ Two Ways LLMs Provide Structured Output

1. **Models that directly support structured output** → use `with_structured_output()` in LangChain
2. **Models that don't support it** → use Output Parsers (next video)

## 🧠 `with_structured_output()` Function

You attach a schema before invoking the model, telling it:
✔️ What keys you want
✔️ What data types you expect

---

## 🥇 Method 1 — Using TypedDict

### 🔸 What is TypedDict?

A Python way to define dictionaries with expected keys and value types.

```python
from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int
```

⚠️ No validation — wrong data types won't cause errors.

### 🟢 Example

```python
from langchain_openai import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI()

class Review(TypedDict):
    summary: str
    sentiment: str

structured_model = model.with_structured_output(Review)

review = """
The phone has great battery life and camera quality.
However, it heats up while gaming.
"""

result = structured_model.invoke(review)
print(result["summary"])
print(result["sentiment"])
```

---

## 🥈 Method 2 — Using Pydantic (Recommended)

### 🔸 What is Pydantic?

A library for data validation in Python. It stops wrong data from entering the system.

```bash
pip install pydantic
```

### 🟢 Example Schema with Validation

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class Review(BaseModel):
    themes: List[str] = Field(description="Key topics")
    summary: str = Field(description="Brief summary")
    sentiment: Literal["POS", "NEG"] = Field(description="Overall sentiment")
    pros: Optional[List[str]] = None
    cons: Optional[List[str]] = None
    name: Optional[str] = None
```

### Use with structured output

```python
structured_model = model.with_structured_output(Review)
result = structured_model.invoke(review_text)
print(result.sentiment)
```

🔁 Pydantic converts result into an **object**, not a plain dict.

---

## 🥉 Method 3 — JSON Schema

Use when working with multiple programming languages (Python + JavaScript).

```json
{
  "title": "Review",
  "type": "object",
  "properties": {
    "summary": { "type": "string" },
    "sentiment": { "type": "string", "enum": ["POS", "NEG"] }
  },
  "required": ["summary", "sentiment"]
}
```

---

## 🆚 When to Use What?

| Method      | Use Case                       |
| ----------- | ------------------------------ |
| TypedDict   | Only Python + no validation    |
| Pydantic 👑 | Python + validation + defaults |
| JSON Schema | Multi-language projects        |

---

## 🚧 Important Notes

`with_structured_output()` supports two modes:

| Mode             | When?                       |
| ---------------- | --------------------------- |
| function_calling | For OpenAI models (default) |
| json_mode        | For Gemini, Claude, Groq    |

❗ Some models (like TinyLlama) do not support structured output → require output parsers.

---

## 🎯 Final Summary

* LLMs normally output text → humans understand it but machines can't.
* Structured output provides a defined format (mostly JSON).
* LangChain makes it easy using `with_structured_output()`.
* Best approach in real-world projects: **Pydantic schema**.

---

📌 **LangChain Output Parsers – Video 8**

## 🚀 Why Do We Need Structured Output?

When you ask any LLM (like GPT, LLaMA, etc.) a question, it replies in plain text. This plain text is:

❌ unstructured
❌ hard to send to APIs or databases
❌ difficult to extract specific values from

To solve this, we use **Structured Output**. We instruct the model to return output in a fixed format such as:

* JSON
* key–value pairs
* lists
* objects with schema

---

## 🧠 What Are Output Parsers?

Output Parsers help convert raw LLM text responses into structured formats like:

✔ JSON
✔ CSV
✔ Python dict
✔ Pydantic models

They ensure:

* Consistent output
* Easy integration with other systems
* Cleaner parsing of responses

---

## 🔥 Four Most Important Output Parsers

LangChain provides many parsers, but you mainly use these 4:

| Parser                     | Purpose                              | Schema Enforced? | Data-Type Validation? |
| -------------------------- | ------------------------------------ | ---------------- | --------------------- |
| **StringOutputParser**     | Convert response into plain string   | ❌                | ❌                     |
| **JsonOutputParser**       | Return JSON output                   | ❌                | ❌                     |
| **StructuredOutputParser** | Return JSON with a defined structure | ✔                | ❌                     |
| **PydanticOutputParser**   | JSON + validation using Pydantic     | ✔                | ✔                     |

---

## 1️⃣ **StringOutputParser**

### 📍 When to use?

Use when you simply want the **text output** and want to pass it to another step in a chain.

### 🧩 Example Use Case

Ask for a detailed report → summarize it again using the model.

### ✅ CODE

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StringOutputParser
from langchain_core.runnables import RunnableSequence

model = ChatOpenAI(model="gpt-4o-mini")
parser = StringOutputParser()

template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Summarize the following text in 5 lines:\n{text}",
    input_variables=["text"]
)

chain = RunnableSequence(
    template1 | model | parser | template2 | model | parser
)

result = chain.invoke({"topic": "Black Hole"})
print(result)
```

### 📌 Why useful?

It extracts only the **text** from model output and ignores metadata like token usage.

---

## 2️⃣ **JsonOutputParser**

### 📍 When to use?

If your model should return a **JSON object**.

❌ No schema enforcement
❌ LLM decides JSON structure

### ✨ CODE

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

parser = JsonOutputParser()
model = ChatOpenAI(model="gpt-4o-mini")

template = PromptTemplate(
    template="Give me the name, age, and city of a fictional person.\n{format_instructions}",
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt = template.format()
response = model.invoke(prompt)
final = parser.parse(response.content)

print(final)
print(type(final))  # dict
```

### 📌 Limitation

JSON format is returned, but you **cannot enforce** which keys or types must appear.

---

## 3️⃣ **StructuredOutputParser**

### 📍 Why this?

You can **force the model** to return JSON in a **predefined structure**.

✔ schema enforced
❌ no validation of values

### 🧩 Example

```python
from langchain.schema import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")

schemas = [
    ResponseSchema(name="fact_one", description="Fact one about topic"),
    ResponseSchema(name="fact_two", description="Fact two about topic"),
    ResponseSchema(name="fact_three", description="Fact three about topic"),
]

parser = StructuredOutputParser.from_response_schemas(schemas)

template = PromptTemplate(
    template="Give 3 facts about {topic}\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

response = model.invoke(template.format(topic="Black Hole"))
final = parser.parse(response.content)
print(final)
```

---

## 4️⃣ **PydanticOutputParser – THE BEST**

### 📍 Why best?

✔ Enforces structure
✔ Validates data types
✔ Rejects wrong formats

### 🧩 Example

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age must be > 18")
    city: str = Field(description="City name")

parser = PydanticOutputParser(pydantic_object=Person)
model = ChatOpenAI(model="gpt-4o-mini")

template = PromptTemplate(
    template="Generate details of a fictional {place} person\n{format_instructions}",
    input_variables=["place"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt = template.format(place="Indian")
response = model.invoke(prompt)
final = parser.parse(response.content)
print(final)
```

### 📌 Note

If the model gives `"age": "35 years"` → **Pydantic will throw an error** because age must be an integer.

---

## 🏁 Summary Table

| Parser                     | JSON? | Structure? | Validation? | Use Case              |
| -------------------------- | ----- | ---------- | ----------- | --------------------- |
| **StringOutputParser**     | ❌     | ❌          | ❌           | Just get text         |
| **JsonOutputParser**       | ✔     | ❌          | ❌           | Quick JSON            |
| **StructuredOutputParser** | ✔     | ✔          | ❌           | Fixed schema          |
| **PydanticOutputParser**   | ✔     | ✔          | ✔           | Production grade apps |

---

## 🎯 Conclusion

If you're building:

| App Type                         | Recommended Parser         |
| -------------------------------- | -------------------------- |
| Simple chatbots                  | **StringOutputParser**     |
| JSON API responses               | **JsonOutputParser**       |
| Integrations with backend DB     | **StructuredOutputParser** |
| Real apps with strict validation | **PydanticOutputParser** ✔ |


# 📌 LangChain Runnables Video 9

## ❓ Why do Runnables exist?

### Background

When ChatGPT was launched (2022), companies started creating LLM-based applications. LangChain was created to make building such apps easier.

### Early LangChain Structure

LangChain initially offered many components:

| Component        | Purpose                      |
| ---------------- | ---------------------------- |
| LLMs             | Talk to language models      |
| Prompt Templates | Build prompts dynamically    |
| Document Loaders | Load files (PDF, text, etc.) |
| Text Splitters   | Break large text into chunks |
| Embeddings       | Convert text into vectors    |
| Vector Stores    | Store embeddings             |
| Retrievers       | Search relevant chunks       |
| Output Parsers   | Format final answer          |

### Problem

These components were not standardized. Each one had different methods:

| Component      | Method                     |
| -------------- | -------------------------- |
| LLM            | `predict()`                |
| PromptTemplate | `format()`                 |
| Retriever      | `get_relevant_documents()` |
| Parser         | `parse()`                  |

⚠ Because of different function names, the LangChain team had to build custom chains for every use-case.

This created:

* Too many chain classes (LLMChain, SequentialChain, RetrievalQAChain, etc.)
* Huge codebase
* High learning curve → users confused

---

## 🧠 What are Runnables?

**Runnable = a standard unit of work**

Think of runnables like **LEGO blocks**:

* ✔ Takes an input
* ✔ Does one specific task
* ✔ Produces an output
* ✔ Can connect with another runnable

### Common Interface for all runnables

```
invoke(input)    # one input → one output
batch(list)      # many inputs → many outputs
stream(input)    # stream output chunks
```

Because they share the same interface, **all runnables can connect to each other**.

---

## 🟢 Why Runnables solve everything?

### Before runnables

```
Prompt → format → LLM → predict → parse → output
```

Every component had its own method — chaos.

### After runnables

```
Prompt.invoke → LLM.invoke → Parser.invoke
```

One universal interface → **Composable workflows**.

---

## 🔧 Code Examples

### 1️⃣ Dummy LLM

```python
import random

class FakeLLM:
    def invoke(self, prompt):
        responses = [
            "Delhi is the capital of India",
            "AI stands for Artificial Intelligence",
            "IPL is a cricket league"
        ]
        return {"response": random.choice(responses)}
```

### 2️⃣ Dummy Prompt Template

```python
class FakePromptTemplate:
    def __init__(self, template):
        self.template = template

    def invoke(self, inputs):
        return self.template.format(**inputs)
```

**Usage:**

```python
prompt = FakePromptTemplate("Write a poem about {topic}")
print(prompt.invoke({"topic": "India"}))
```

### 3️⃣ Runnable Connector (like a chain)

```python
class RunnableConnector:
    def __init__(self, runnables):
        self.runnables = runnables

    def invoke(self, input_data):
        for runnable in self.runnables:
            input_data = runnable.invoke(input_data)
        return input_data
```

### 4️⃣ Build a simple chain

```python
prompt = FakePromptTemplate("Write a poem about {topic}")
llm = FakeLLM()

chain = RunnableConnector([prompt, llm])
print(chain.invoke({"topic": "Cricket"}))
```

### 5️⃣ Add a Parser Runnable

```python
class FakeParser:
    def invoke(self, llm_output):
        return llm_output["response"]

parser_chain = RunnableConnector([prompt, llm, FakeParser()])
print(parser_chain.invoke({"topic": "India"}))
```

---

## 🧩 Composing Chains (Chain inside Chain)

```python
# Chain 1 : Generate Joke
joke_template = FakePromptTemplate("Tell a joke about {topic}")
joke_chain = RunnableConnector([joke_template, llm])

# Chain 2 : Explain the Joke
explain_template = FakePromptTemplate("Explain this joke: {response}")
explain_chain = RunnableConnector([explain_template, llm, FakeParser()])

# Final Chain
final_chain = RunnableConnector([joke_chain, explain_chain])
print(final_chain.invoke({"topic": "Programming"}))
```

---

## 🎯 Summary (Exam Revision Style)

| Concept     | Meaning                                     |
| ----------- | ------------------------------------------- |
| Runnables   | Standard building blocks in LangChain       |
| Why needed  | Too many inconsistent Chain classes earlier |
| Benefit     | Single interface → easy composition         |
| invoke()    | Universal method to run work                |
| Composition | Runnable → Chain → Chain of Chains          |

---

## 🌟 One-line Understanding

**Runnables turned LangChain into LEGO** — every component clicks together because all have one common interface: `invoke()`

# LangChain Runnables  Video 10

## 1️⃣ What are Runnables?

Earlier, LangChain had many components like:

* **PromptTemplate**
* **LLMs**
* **Parsers**
* **Retrievers**

### The Problem

Each component used different methods, such as:

* `format()` for prompts
* `predict()` for LLM
* `parse()` for parser
* `get_relevant_docs()` for retriever

👉 Because of this, connecting components was difficult.

### The Solution

LangChain introduced a common interface called **Runnable**.
Every component now uses one method:

```python
component.invoke(input)
```

This makes it easy to connect different parts together.

---

## 2️⃣ Types of Runnables

There are **2 major types**:

### A. Task-Specific Runnables

These represent core LangChain components:
✔️ PromptTemplate
✔️ ChatOpenAI
✔️ StrOutputParser
✔️ Retriever

Each has a specific job.

### B. Runnable Primitives

These are building blocks that help combine components:

| Primitive           | Purpose                               |
| ------------------- | ------------------------------------- |
| RunnableSequence    | Connect runnables in order            |
| RunnableParallel    | Run multiple runnables at once        |
| RunnablePassthrough | Return input unchanged                |
| RunnableLambda      | Convert Python function into runnable |
| RunnableBranch      | Conditional logic (if-else)           |

---

## 3️⃣ RunnableSequence

Used to connect runnables step-by-step.

### Example: Write and parse a joke

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

prompt = PromptTemplate.from_template("Write a joke about {topic}")
model = ChatOpenAI()
parser = StrOutputParser()

chain = RunnableSequence(prompt, model, parser)
print(chain.invoke({"topic": "AI"}))
```

✔️ prompt → model → parser

### Longer Sequence Example

```python
prompt1 = PromptTemplate.from_template("Write a joke about {topic}")
prompt2 = PromptTemplate.from_template("Explain the joke: {text}")

chain = RunnableSequence(
    prompt1, model, parser,
    prompt2, model, parser
)
print(chain.invoke({"topic": "AI"}))
```

---

## 4️⃣ RunnableParallel

Run multiple runnables at the same time.

```python
from langchain.schema.runnable import RunnableParallel

tweet_prompt = PromptTemplate.from_template("Write a tweet on {topic}")
post_prompt  = PromptTemplate.from_template("Write a LinkedIn post on {topic}")

parallel = RunnableParallel({
    "tweet": RunnableSequence(tweet_prompt, model, parser),
    "linkedin": RunnableSequence(post_prompt, model, parser)
})

result = parallel.invoke({"topic": "AI"})
print(result["tweet"])
print(result["linkedin"])
```

📌 Output is a dictionary like:

```json
{ "tweet": "...", "linkedin": "..." }
```

---

## 5️⃣ RunnablePassthrough

Returns the **same input** back.

```python
from langchain.schema.runnable import RunnablePassthrough

pt = RunnablePassthrough()
print(pt.invoke(10))  # 10
```

Useful when you want one path to modify data and another to keep original.

---

## 6️⃣ RunnableLambda

Convert any Python function into a runnable.

```python
from langchain.schema.runnable import RunnableLambda

def word_count(text):
    return len(text.split())

counter = RunnableLambda(word_count)
print(counter.invoke("AI is great"))  # 3
```

Used to add custom logic inside chains.

---

## 7️⃣ RunnableBranch

Used for **if-else** logic.

```python
from langchain.schema.runnable import RunnableBranch

branch = RunnableBranch(
    (lambda x: len(x.split()) > 500,
        RunnableSequence(summary_prompt, model, parser)
    ),
    RunnablePassthrough()  # default
)
```

If report length > 500 words → summarize, else return as-is.

---

## 8️⃣ LCEL – LangChain Expression Language

Short syntax for `RunnableSequence`.

### Old Way

```python
RunnableSequence(prompt, model, parser)
```

### New LCEL Way

```python
prompt | model | parser
```

✔️ Cleaner
✔️ Easier to read
✔️ Declarative

Future operator like `&` for parallel expected.

---

## 🟢 Summary

| Concept       | Why it exists                  |
| ------------- | ------------------------------ |
| Runnables     | Standard way to run components |
| Task-specific | Core LangChain units           |
| Sequence      | Step-by-step workflow          |
| Parallel      | Multi-path execution           |
| Passthrough   | Keep original data             |
| Lambda        | Add Python logic               |
| Branch        | Conditional chain              |
| LCEL          | Short syntax with pipes        |

---

## Final Takeaway

Master **Runnable Primitives** to build any AI workflow easily.
Going forward, write chains using:

```python
prompt | model | parser
```

This is the future direction of LangChain.

# LangChain – Document Loaders (Simple Notes)

## Why this Video?

* The creator planned to teach Memory in LangChain.
* But LangChain is moving the Memory feature into LangGraph.
* So, Memory will now be taught later with LangGraph.
* From this video onwards, a new topic begins:
  **RAG-based (Retrieval Augmented Generation) applications using LangChain**.

## What Did We Learn So Far? (Recap)

### Previous videos covered:

#### Core LangChain Components

* Models
* Prompts
* Chains
* Explained with hands-on code.

#### Important LangChain Concepts

* Especially **Runnables**.

At this point, fundamentals of LangChain are clear, so now we are ready to build **LLM-based applications**.

---

## What is RAG?

**RAG = Retrieval Augmented Generation**

A technique where:

* LLM (e.g., GPT) + External Knowledge Base work together.
* The model retrieves relevant information from external sources like:

  * PDFs
  * Company databases
  * Personal documents
  * Websites
* Then generates accurate, updated, grounded answers.

### Why RAG is Needed?

ChatGPT cannot answer:

* ❌ current affairs
* ❌ personal emails
* ❌ private company docs

Because it wasn't trained on them.

**RAG fixes this by connecting your LLM with your own data.**

### Benefits of RAG

* ✔ Up-to-date information
* ✔ Privacy-safe (data stays with you)
* ✔ No size limit on documents
* ✔ Can handle large files using chunks

---

## RAG Components

To build a RAG app, you need 4 components:

1. **Document Loaders** → Load data
2. **Text Splitters** → Break into chunks
3. **Vector Databases** → Store embeddings
4. **Retrievers** → Fetch relevant info

**This video covers → Document Loaders**

---

## What are Document Loaders?

Tools used to load data from different sources into LangChain and convert it into a standard **Document object**.

Each Document has:

* **page_content** → The actual text
* **metadata** → Extra info like file name, page number, etc.

No matter what file you load (PDF, CSV, website, etc.), LangChain converts it into the same format.

---

## Important Document Loaders

### 1️⃣ TextLoader

Loads `.txt` files.

```python
from langchain_community.document_loaders import TextLoader
loader = TextLoader("cricket.txt", encoding="utf-8")
docs = loader.load()
```

**Output:** A list of Document objects.

**Use cases:** transcripts, log files, text snippets

---

### 2️⃣ PyPDFLoader

Loads PDF pages. Each page becomes one Document.

```python
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("deep_learning.pdf")
docs = loader.load()
```

If PDF has 23 pages → **23 Document objects**.

**Works best for text-based PDFs.** Not good for scanned images.

**Other PDF loaders:**

* PDFPlumberLoader → extract tables
* UnstructuredPDFLoader → scanned images
* PyMuPDFLoader → layout-heavy PDFs

---

### 3️⃣ DirectoryLoader

Loads multiple files from a folder.

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
loader = DirectoryLoader("books/", glob="*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()
```

If folder has 3 PDFs (326 + 392 + 468 pages) → **Total documents = 1186**

---

### Load vs Lazy Load

| Load()                   | Lazy Load()                  |
| ------------------------ | ---------------------------- |
| Loads everything at once | Loads one document at a time |
| Uses more memory         | Memory efficient             |
| Returns list             | Returns generator            |
| Good for small data      | Good for large datasets      |

---

### 4️⃣ WebBaseLoader

Loads content from web pages.

```python
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://flipkart.com/macbook")
docs = loader.load()
```

Works best with **static HTML pages**.

---

### 5️⃣ CSVLoader

Loads CSV rows. Each row = one Document.

```python
from langchain_community.document_loaders import CSVLoader
loader = CSVLoader("ads.csv")
docs = loader.load()
```

**Use cases:** analytics, column-based queries

---

## Other Loaders

LangChain also has loaders for:

* **Cloud storage** → AWS S3, GDrive, Dropbox
* **Social media** → Reddit, Twitter, Slack
* **Common file types** → JSON, HTML, YouTube transcripts, etc.

---

## Custom Document Loader

If LangChain doesn't have a loader for your data source, you can build your own:

```python
class CustomLoader(BaseLoader):
    def load(self):
        # your logic
```

---

## Summary

* We started **RAG development** using LangChain.
* First component learned: **Document Loaders**
* Covered Text, PDF, Directory, Web, and CSV loaders.
* Next videos will cover:
  ✔ Text Splitters
  ✔ Vector Databases
  ✔ Retrievers
  ➜ Finally build a complete **RAG app**.

---

End of Notes 🚀

🚀 **Text Splitting in LangChain – Simple Notes**

**What is Text Splitting?**
Text Splitting means breaking a large document—like long PDFs, books, articles—into small parts (chunks) so that a Language Model (LLM) can process them effectively.

---

### ❗ Why do we need Text Splitting?

LLMs cannot handle very large text at once because:

**Context Length Limit**
Every LLM has a limit.
Example: If a model accepts 50,000 tokens, and your PDF has 1,00,000+ words, you must split it.

**Better Embeddings**
Embedding a huge paragraph reduces meaning accuracy. Small chunks capture meaning better.

**Semantic Search works better**
Searching among small chunks returns more accurate results.

**Better Summarization**
LLMs give poor summaries for giant text; splitting improves output.

**Computational Efficiency**
Small chunks = Less memory + faster processing + parallel execution.

---

### 🔥 Four Types of Text Splitting

---

### 1️⃣ Length-Based Text Splitting

Split text based on a fixed size (characters or tokens).

**Pros**
✔️ Very simple & fast

**Cons**
❌ Doesn’t care about words, meaning, or sentence boundaries
❌ May cut words in half

**Code: Character-based Text Splitter**

```python
from langchain.text_splitter import CharacterTextSplitter

text = "Your long text here..."

splitter = CharacterTextSplitter(
    chunk_size=100,      # size of each chunk
    chunk_overlap=0,     # no overlap
    separator=""         # split exactly at limit
)

chunks = splitter.split_text(text)
print(chunks)
```

**🔁 Chunk Overlap**
Chunk Overlap means some part of the previous chunk is added at the start of the next chunk.

**Why?**
Helps maintain context continuity.
Recommended overlap: 10–20% of chunk size in RAG.

```python
splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)
```

---

### 2️⃣ Text-Structure Based Splitting

*(📌 Best for normal written text)*
Also called **Recursive Character Text Splitter**.

This method respects:

* Paragraphs
* Sentences
* Words
* Characters

It tries to split at sentence boundaries first, then words, then characters if needed.

🏆 Most used splitter in RAG

**Code: Recursive Character Text Splitter**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_text(text)
print(len(chunks), chunks[:2])
```

---

### 3️⃣ Document-Structure Based Splitting

Used when the document is not plain text, e.g.:

* Python / JavaScript code
* Markdown
* HTML

Each has its own structure and keywords (`class`, `def`, `<h1>`).
LangChain provides language-aware splitters.

**Code: Splitting Python Code**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import Language

code = """class A:
    def hello(self):
        print("hi")
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=200,
    chunk_overlap=20
)

chunks = splitter.split_text(code)
print(chunks)
```

**Splitting Markdown**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import Language

md = """# Title
## Features
- Point 1
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=200,
    chunk_overlap=10
)

chunks = splitter.split_text(md)
print(chunks)
```

---

### 4️⃣ Semantic Meaning Based Splitting *(⚠️ Experimental)*

This method splits text based on topic changes detected using embeddings.

**Idea:**

1. Convert each sentence into an embedding
2. Compare similarities
3. When meaning changes sharply → split

Useful when paragraphs contain multiple topics.

**Code: Semantic Splitting**

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation"
)

chunks = splitter.split_text(text)
print(chunks)
```

❌ Results are currently inconsistent
🔮 Likely to become popular in future RAG applications

---

### 🏁 Summary

| Splitter Type       | When to Use     | Best For                       |
| ------------------- | --------------- | ------------------------------ |
| Length-Based        | Simple cases    | Quick splitting, small text    |
| Recursive Character | Default choice  | RAG, embeddings, summarization |
| Document-Based      | Structured docs | Code, Markdown, HTML           |
| Semantic-Based      | Topic detection | Experimental research          |

---

### 🎯 What should YOU use?

👉 Always start with **RecursiveCharacterTextSplitter** — it's accurate, context-aware, and ideal for RAG pipelines.

---

End of Notes 🚀

📌 VECTOR STORES – COMPLETE SIMPLE NOTES + FULL CODE

🚀 **Introduction**

We are building RAG-based applications using LangChain. To build a RAG system, we need:

1️⃣ Document Loaders – already learned
2️⃣ Text Splitters – already learned
3️⃣ Vector Stores – today's topic (very important!)

Vector stores allow us to store embeddings and perform semantic similarity search.

---

❓ **Why Do We Need Vector Stores?**

Imagine making an IMDb-like website that stores movie details such as name, director, actors, release date, and genre. This works fine until you want to add a **Movie Recommendation System**.

---

❌ **First Try: Keyword-Based Similarity**

You compare movies based on similar keywords:

* same actor?
* same director?
* same year?
* same genre?

But this fails because **keywords ≠ meaning**.

Example:

| User watches    | Suggested              | Reality                 |
| --------------- | ---------------------- | ----------------------- |
| My Name Is Khan | Kabhi Alvida Naa Kehna | Totally different story |

Another case:

| Movies                             | Similar Meaning | Keywords           |
| ---------------------------------- | --------------- | ------------------ |
| Taare Zameen Par, A Beautiful Mind | Same core idea  | Different keywords |

So keyword matching is **not intelligent**.

---

🧠 **Better Solution: Compare Story Meaning**

Instead of matching keywords, compare the meaning of plots. But computers do not understand text, so we convert text into numerical vectors called **embeddings**.

---

📌 **What are Embeddings?**

Embeddings convert text meaning into numbers.

```
Text → Neural Network → Vector
```

Example embedding vector:

```
[0.78, -0.11, 0.62, ...] (512 dimensions)
```

Now we can compute similarity using **cosine similarity** – smaller angle → more similar.

---

⚠️ **Challenges When Using Embeddings**

As the database grows to lakhs of items:

1️⃣ Generating embeddings for all items
2️⃣ Storing embeddings efficiently – SQL databases can't store vectors properly
3️⃣ Performing semantic search FAST – comparing query with 10 lakh vectors is slow

---

🎯 **The Solution → Vector Stores**

A **vector store** is a system designed to:
✔ Store vectors (embeddings)
✔ Retrieve them efficiently
✔ Perform similarity search quickly
✔ Store metadata along with vectors

---

🔑 **Key Features of Vector Stores**

| Feature           | Purpose                              |
| ----------------- | ------------------------------------ |
| Storage           | Store embeddings + metadata          |
| Similarity Search | Find vectors similar to query        |
| Indexing          | Organize vectors for faster search   |
| CRUD              | Create, Read, Update, Delete vectors |

---

🔍 **Indexing (Fast Search)**

Without indexing:

```
Query → Compare with 10 lakh vectors → Slow
```

With clustering + indexing:

```
Query → Compare with clusters → Filter → Compare fewer vectors
```

Results: 10 lakh comparisons reduce to 1 lakh → Super fast 🚀

---

📍 **Where Are Vector Stores Used?**

| Application            | Why                       |
| ---------------------- | ------------------------- |
| RAG systems            | Store document embeddings |
| Recommendation systems | Find similar items        |
| Semantic search        | Search by meaning         |
| Image search           | Compare image embeddings  |

Anywhere embeddings are used → **vector store is required**.

---

📌 **Vector Store vs Vector Database**

| Vector Store        | Vector Database    |
| ------------------- | ------------------ |
| Only stores vectors | Full DB system     |
| Lightweight         | Enterprise scaling |
| Local experiments   | Production-ready   |

Examples:

* Vector Store → **FAISS**
* Vector DB → **Pinecone, Milvus, Weaviate**
* Hybrid → **Chroma**

Formula:

```
Vector Database = Vector Store + Database Features
```

---

🛠 **Vector Stores in LangChain**

LangChain supports: **Chroma, FAISS, Pinecone, Weaviate, Qdrant**

Common methods:

```
from_documents()
add_documents()
similarity_search()
delete()
update()
```

You can switch vector stores without changing most code.

---

🔥 **Chroma Vector Store**

Chroma is:

* Lightweight
* Open Source
* Ideal for Local RAG development

Chroma hierarchy:

```
Tenant → Database → Collection → Documents
```

Each document has:

* Embedding vector
* Metadata (extra info)

---

🧾 **COMPLETE WORKING CODE** (Run in Colab)

### STEP 1: Install libraries

```
!pip install langchain langchain-openai chromadb tiktoken
```

### STEP 2: Import modules

```
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
```

### STEP 3: Create Documents

```
doc1 = Document(page_content="Virat Kohli is an Indian cricketer and a great batsman.", metadata={"team": "Royal Challengers Bangalore"})

doc2 = Document(page_content="Rohit Sharma is the captain of Mumbai Indians.", metadata={"team": "Mumbai Indians"})

doc3 = Document(page_content="MS Dhoni is a former captain of Chennai Super Kings.", metadata={"team": "Chennai Super Kings"})

doc4 = Document(page_content="Jasprit Bumrah is a bowler for Mumbai Indians.", metadata={"team": "Mumbai Indians"})

doc5 = Document(page_content="Ravindra Jadeja is an all-rounder in Chennai Super Kings.", metadata={"team": "Chennai Super Kings"})

docs = [doc1, doc2, doc3, doc4, doc5]
```

### STEP 4: Create Vector Store

```
embeddings = OpenAIEmbeddings()
vector_store = Chroma(embedding_function=embeddings, persist_directory="my_chromadb", collection_name="sample")
```

### STEP 5: Add Documents

```
vector_store.add_documents(docs)
```

### STEP 6: View Data

```
vector_store.get(include=["embeddings", "documents", "metadatas"])
```

### STEP 7: Similarity Search

```
query = "Who among these is a bowler?"
vector_store.similarity_search(query, k=2)
```

### STEP 8: Similarity Search with Score

```
vector_store.similarity_search_with_score(query, k=2)
```

### STEP 9: Filter by Metadata

```
vector_store.similarity_search(query="player", k=10, filter={"team": "Chennai Super Kings"})
```

### STEP 10: Update a Document

```
updated_doc = Document(page_content="Virat Kohli, former captain of RCB, is known for his aggressive leadership.", metadata={"team": "Royal Challengers Bangalore"})

vector_store.update_document(document_id="1", document=updated_doc)
```

### STEP 11: Delete a Document

```
vector_store.delete(ids=["1"])
```

---

🎉 **CONCLUSION**

Now you understand:
✔ Why vector stores exist
✔ Why embeddings matter
✔ Difference between vector store vs vector database
✔ How LangChain integrates vector stores
✔ How to use ChromaDB

You are now one step away from building your own **RAG system** 🚀

---

⚡ **HOMEWORK**
Try replacing Chroma with:

```
from langchain.vectorstores import FAISS
```

Same code will work — LangChain has a unified API.

