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

# 🚀 LangChain Retrievers 

These notes explain **Retrievers in LangChain** in very simple English, step by step, with **clear concepts + code snippets**. This is mainly used in **RAG (Retrieval Augmented Generation)** systems.

---

## 1️⃣ Where Retrievers Fit in RAG (Big Picture)

To build a **RAG-based application**, you must understand these 4 core components:

1. **Document Loaders** – Load data (PDFs, web pages, text files)
2. **Text Splitters** – Break large text into small chunks
3. **Vector Stores** – Store embeddings of text
4. **Retrievers** – Fetch the most relevant documents for a query ✅

➡️ After learning these four, you are ready to build RAG systems.

---

## 2️⃣ What is a Retriever?

**Simple definition:**

> A Retriever is a LangChain component that **fetches relevant documents from a data source** based on a user’s query.

### How it works (Simple Flow)

```
User Query → Retriever → Data Source → Relevant Documents
```

* User asks a question
* Retriever searches a data source (vector store, Wikipedia, API, etc.)
* Retriever returns **multiple LangChain `Document` objects**

📌 Think of a Retriever like a **smart search engine**.

---

## 3️⃣ Important Points About Retrievers

### 🔹 Point 1: Multiple Retrievers Exist

LangChain does NOT have only one retriever.

There are **many retrievers**, each designed for different use cases.

---

### 🔹 Point 2: Retrievers Are Runnables

All retrievers are **Runnables**, just like:

* Models
* Prompts
* Chains

This means:

* You can call `.invoke()` on them
* You can plug them inside **chains**

```python
results = retriever.invoke("your query here")
```

---

### 🔹 Point 3: Why Retrievers Matter

In advanced RAG systems:

* Different retrievers are used
* Search strategies matter a lot
* Retriever choice improves accuracy

---

## 4️⃣ Types of Retrievers (Two Ways to Classify)

### ✅ Type 1: Based on **Data Source**

Examples:

* Wikipedia Retriever → Wikipedia articles
* Vector Store Retriever → Chroma, FAISS, etc.
* Arxiv Retriever → Research papers

---

### ✅ Type 2: Based on **Search Strategy**

Examples:

* Similarity Search
* MMR (Maximum Marginal Relevance)
* Multi-Query Retriever
* Contextual Compression Retriever

---

## 5️⃣ Wikipedia Retriever

### What is it?

A retriever that searches **Wikipedia** using Wikipedia APIs.

* Uses **keyword-based search** (not semantic search)
* Returns relevant Wikipedia articles as `Document` objects

---

### Code Example: Wikipedia Retriever

```python
from langchain_community.retrievers import WikipediaRetriever

# Create retriever
retriever = WikipediaRetriever(
    top_k_results=2,
    lang="en"
)

query = "Geopolitical history of India and Pakistan"

# Invoke retriever
docs = retriever.invoke(query)

# Print results
for doc in docs:
    print(doc.page_content[:500])
```

📌 This is a **retriever**, not a document loader, because it performs **search + relevance filtering**.

---

## 6️⃣ Vector Store Retriever (Most Common)

### What is it?

A retriever that fetches documents from a **vector store** using **semantic similarity**.

Used with:

* Chroma
* FAISS
* Weaviate

---

### How It Works

1. Documents → Embeddings
2. Store embeddings in vector DB
3. User query → Query embedding
4. Semantic similarity search
5. Return top-k documents

---

### Code Example: Vector Store Retriever (Chroma)

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

# Sample documents
docs = [
    Document(page_content="LangChain is a framework for LLM apps"),
    Document(page_content="Chroma is a vector database"),
]

embeddings = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

query = "What is Chroma used for?"
results = retriever.invoke(query)

for r in results:
    print(r.page_content)
```

---

### ❓ Why Use Retriever Instead of `similarity_search()`?

You can do this:

```python
vectorstore.similarity_search(query, k=2)
```

But retrievers are better because:

* They support **advanced strategies (MMR, Multi-query)**
* They are **runnables**
* They integrate easily into **RAG chains**

---

## 7️⃣ MMR – Maximum Marginal Relevance

### Problem with Normal Similarity Search

* Returns **very similar documents**
* Causes **redundant results**

Example:

* Doc 1: Glaciers melting
* Doc 2: Glaciers melting again
* Doc 3: Deforestation

➡️ Poor diversity

---

### What MMR Solves

MMR selects documents that are:

✔ Relevant to the query
✔ Different from each other

> Relevant + Diverse

---

### MMR Formula Idea (Conceptual)

* Pick most relevant document first
* Then pick documents that are **less similar** to already selected ones

---

### Code Example: MMR Retriever

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "lambda_mult": 0.5
    }
)

query = "What is LangChain?"
results = retriever.invoke(query)

for r in results:
    print(r.page_content)
```

📌 `lambda_mult`:

* `1.0` → behaves like similarity search
* `0.0` → maximum diversity

---

## 8️⃣ Multi-Query Retriever

### Problem It Solves

User queries can be **ambiguous**.

Example:

> "How can I stay healthy?"

Meaning could be:

* Diet
* Exercise
* Mental health

---

### Solution: Multi-Query Retriever

* Uses an **LLM** to generate multiple sub-queries
* Searches each sub-query
* Merges results
* Removes duplicates

---

### Flow

```
User Query
   ↓
LLM generates multiple queries
   ↓
Retriever searches for each
   ↓
Results merged & deduplicated
```

---

### Code Example: Multi-Query Retriever

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)

query = "How to improve energy levels and maintain balance"
results = multi_retriever.invoke(query)

for r in results:
    print(r.page_content)
```

📌 Best for **ambiguous or broad queries**.

---

## 9️⃣ Contextual Compression Retriever

### Problem It Solves

Documents may contain **mixed topics**.

Example document:

* Line 1: Grand Canyon
* Line 2: Photosynthesis
* Line 3: Tourism

Query:

> "What is photosynthesis?"

❌ Normal retriever returns full document

---

### Solution

* Retrieve documents first
* Then **compress documents using an LLM**
* Keep only query-relevant parts

---

### How It Works

1. Base retriever fetches documents
2. LLM removes irrelevant text
3. Short, clean output returned

---

### Code Example: Contextual Compression Retriever

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()

base_retriever = vectorstore.as_retriever()
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    document_compressor=compressor
)

query = "What is photosynthesis?"
results = compression_retriever.invoke(query)

for r in results:
    print(r.page_content)
```

📌 Best for:

* Long documents
* Mixed content
* Reducing LLM context length

---

## 🔟 Other Important Retrievers (Explore Later)

* Parent Document Retriever
* Time-Weighted Vector Retriever
* Self-Query Retriever
* Ensemble Retriever
* Multi-Retriever

📌 Official Docs:
[https://python.langchain.com/docs/modules/data_connection/retrievers/](https://python.langchain.com/docs/modules/data_connection/retrievers/)

---

## ✅ Why So Many Retrievers Exist?

Because:

* Simple RAG often gives poor results
* Different problems need different strategies
* Retriever tuning improves RAG accuracy

➡️ Advanced RAG = Better Retrievers

---

## 🧠 Final Summary

* Retrievers fetch relevant documents
* They are **runnable components**
* Multiple retrievers exist for different needs
* Advanced retrievers = better RAG systems

📌 In real projects, you will **experiment with retrievers** to improve performance.

---


# 📘 RAG (Retrieval Augmented Generation) 

---

## 1. What is this video about?

This video introduces **RAG (Retrieval Augmented Generation)**.

* RAG is one of the most useful and common applications of Generative AI.
* It is widely used to build **question-answering systems on private or recent data**.

**Instructor’s plan:**

* **This video** → Conceptual and theoretical understanding of RAG
* **Next video** → Build a complete RAG system using **LangChain**

---

## 2. Quick Recap: RAG Components (Already Covered)

Before learning RAG, the following core components were studied:

### 1️⃣ Document Loaders

* Load data from different sources:

  * PDFs
  * Websites
  * YouTube
  * Google Drive
  * AWS S3

### 2️⃣ Text Splitters

* Break large text into smaller, meaningful chunks

### 3️⃣ Vector Stores

* Store text in the form of **embeddings (vectors)**

### 4️⃣ Retrievers

* Perform **semantic search** to fetch relevant chunks from vector stores

Once these components are clear, understanding RAG becomes easy.

---

## 3. How we will understand RAG

We follow the **WHY → WHAT → HOW** approach:

* **Why** do we need RAG?
* **What** exactly is RAG?
* **How** does a RAG-based system work?

---

## 4. Understanding LLMs (Background)

### What are LLMs?

* LLMs (Large Language Models) are **transformer-based neural networks**.
* They have **billions of parameters** (weights and biases).

Examples:

* 7B, 13B, 70B, 175B parameter models

### Parametric Knowledge

* All knowledge of an LLM is stored inside its **parameters**.
* This is called **parametric knowledge**.
* More parameters → more knowledge → more powerful model

### How users interact with LLMs

1. User sends a **prompt**
2. LLM:

   * Understands the prompt
   * Searches its parametric knowledge
   * Generates an answer word by word

---

## 5. Problems with Normal Prompting (WHY RAG is needed)

### ❌ Problem 1: Private Data

* LLMs are **not trained on your private data**

Examples:

* Company documents
* Internal videos

Asking ChatGPT about them → ❌ No answer

---

### ❌ Problem 2: Recent / Current Data

* LLMs have a **knowledge cutoff date**
* They do not know:

  * Today’s news
  * Latest updates

Open-source models fail more in this case.

---

### ❌ Problem 3: Hallucination

* LLMs sometimes give **confident but incorrect answers**

Example:

> “Einstein played football for Germany” ❌

This happens because LLMs are **probabilistic models**.

---

## 6. First Solution Attempt: Fine-Tuning

### What is Fine-Tuning?

* Take a **pretrained LLM**
* Train it again on a **smaller domain-specific dataset**

**Example:**

* Train LLM on medical data → Better medical answers

---

### Types of Fine-Tuning

#### 1️⃣ Supervised Fine-Tuning

* Uses **labeled data**
* Format:

  * Prompt → Desired Output
* Requires thousands to lakhs of examples

#### 2️⃣ Continued Pre-Training

* Unsupervised method
* Feed raw text (e.g., lecture transcripts)
* Similar to original pretraining but on smaller domain data

#### 3️⃣ RLHF

* Reinforcement Learning with Human Feedback
* Helps align model behavior with human expectations

---

### Fine-Tuning Process (Supervised)

1. Collect labeled domain data
2. Choose method (Full FT / LoRA / QLoRA)
3. Train for a few epochs
4. Evaluate (accuracy, hallucination, safety)

---

## 7. Problems with Fine-Tuning ❌

* **Very expensive** (large models need heavy compute)
* **Requires expert engineers**
* **Frequent updates are costly**

  * New data → retrain again
  * Remove data → retrain again

👉 Fine-tuning is **not practical** when data changes frequently.

---

## 8. Second Solution: In-Context Learning

### What is In-Context Learning?

* LLM learns a task by seeing **examples inside the prompt**
* **No weight updates** happen
* Core ability of large models like GPT-3+

### Example: Sentiment Analysis

```
I love this phone → Positive
This app crashes → Negative
Camera is amazing → Positive
I hate the battery life → ?
```

**LLM Answer:** Negative

This is called **Few-shot Prompting**.

---

## 9. Emergent Property of LLMs

* In-context learning was **not explicitly programmed**
* It appeared when models became very large

Examples:

* GPT-1 / GPT-2 → ❌
* GPT-3 (175B) → ✅

📄 Famous paper:
**“Language Models are Few-Shot Learners”**

---

## 10. Improving In-Context Learning → RAG

### Key Idea

* Instead of giving **examples**, give **relevant context**

---

## 11. What is RAG?

**RAG = Retrieval Augmented Generation**

> RAG makes an LLM smarter by giving it **extra information at question time**.

---

## 12. RAG Example (Intuition)

* Website has a **2-hour lecture video**
* Student asks a question about **Gradient Descent**

Instead of:

* Sending entire transcript ❌

We:

* Retrieve only relevant parts (e.g. 5–25 minutes)
* Send those parts as **context**

---

## 13. RAG Prompt Example

```
You are a helpful assistant.
Answer only from the provided context.
If context is insufficient, say "I don't know".
```

---

## 14. RAG = 4-Step Pipeline

### Step 1: Indexing

Create an **external knowledge base**

Includes:

* Document Ingestion
* Text Chunking
* Embedding Generation
* Vector Storage

---

### Step 2: Retrieval

* Convert user query into embedding
* Search vector store
* Fetch most relevant chunks

---

### Step 3: Augmentation

* Combine:

  * User query
  * Retrieved context
* Create final prompt

---

### Step 4: Generation

* LLM generates answer using:

  * Parametric knowledge
  * Retrieved context

---

## 15. Indexing in Detail

### 1️⃣ Document Ingestion

* Load documents into memory

Tools:

* PDF Loader
* YouTube Loader
* Web Loader
* Google Drive / S3

---

### 2️⃣ Text Chunking

**Why split text?**

* LLM context length limit
* Better semantic search

**Rules:**

* Chunks must be meaningful
* Avoid abrupt splits

**Tools:**

* RecursiveCharacterTextSplitter
* Semantic Chunker
* HTML / Markdown splitters

---

### 3️⃣ Embeddings

* Convert each chunk → dense vector
* Captures semantic meaning
* Required for semantic search

Models:

* OpenAI Embeddings
* Sentence Transformers

---

### 4️⃣ Vector Store

Store:

* Text chunk
* Embedding
* Metadata

Examples:

* **Local:** FAISS, Chroma
* **Cloud:** Pinecone, Weaviate, Milvus, Qdrant

👉 This becomes your **external knowledge base**

---

## 16. Retrieval in Detail

Steps:

1. Convert query to embedding
2. Find closest vectors
3. Rank results
4. Select top chunks as context

Techniques:

* Cosine similarity
* MMR
* Contextual compression
* Re-ranking

---

## 17. Why RAG Solves the 3 Problems

### ✅ Private Data

* Context comes from your own data
* LLM answers from your documents

### ✅ Recent Data

* Add new documents to vector store
* No retraining required

### ✅ Hallucination

* LLM answers only from provided context
* Can say “I don’t know”
* Responses become grounded

---

## 18. RAG vs Fine-Tuning

| Feature               | RAG          | Fine-Tuning |
| --------------------- | ------------ | ----------- |
| Cost                  | Low          | Very High   |
| Retraining            | ❌ Not needed | ✅ Required  |
| Updates               | Easy         | Expensive   |
| Complexity            | Simple       | Complex     |
| Hallucination Control | Better       | Limited     |

---

## 19. Final Takeaway

* RAG is a **cheaper, simpler, and scalable** alternative to fine-tuning.
* Best choice when:

  * Data changes frequently
  * Private or recent data is needed

🚀 **Next Step:** Build a complete **RAG system using LangChain**

# 📘 Building a RAG System with LangChain Part 2 video second

## YouTube Chat using RAG (Simple English Notes)

---

## 1. What is this video about?

This video explains **how to build a practical RAG (Retrieval Augmented Generation) system using LangChain**.

### In the previous video:

* Learned **RAG theory**
* Why RAG is needed
* RAG vs Fine-tuning

### In this video:

* Build a **complete RAG system step-by-step**
* Use **real code examples**
* Solve a **real-world problem**

---

## 2. Problem Statement – YouTube Chat

### ❓ Problem

YouTube videos (podcasts, lectures) are very long (2–3 hours).
If you want to:

* Ask a question
* Get a summary
* Clear a doubt

👉 You must watch the full video.

### ✅ Solution

Build a **RAG-based YouTube Chat System** where:

* You give a YouTube video
* Ask any question
* System answers **using only that video’s content**

### Examples:

* “Is AI discussed in this podcast?”
* “Summarize this video in 5 points”
* “Explain this part of the lecture”

---

## 3. Possible UI Options (Not main focus)

* Chrome Extension (HTML, CSS, JavaScript)
* Streamlit Website
* Google Colab Notebook (used in this video)

📌 **Focus is on RAG logic, not UI**

---

## 4. RAG Architecture Used

We follow the **standard RAG flow**:

### 🔹 Step 1: Indexing

* Load document (YouTube transcript)
* Split text into chunks
* Create embeddings
* Store embeddings in vector store

### 🔹 Step 2: Retrieval

* Convert query to embedding
* Perform semantic search
* Fetch relevant chunks

### 🔹 Step 3: Augmentation

* Combine:

  * Retrieved context
  * User query
* Create a prompt

### 🔹 Step 4: Generation

* Send prompt to LLM
* Generate final answer

---

## 5. Requirements

### 🔑 OpenAI API Key

```python
import os
os.environ["OPENAI_API_KEY"] = "your_api_key_here"
```

---

## 6. Install Required Libraries

```bash
pip install langchain langchain-community langchain-openai faiss-cpu youtube-transcript-api
```

---

## 7. Step 1 – Load YouTube Transcript

We use **YouTubeTranscriptApi** (more reliable than LangChain YouTubeLoader).

### 🔹 Get Transcript Code

```python
from youtube_transcript_api import YouTubeTranscriptApi

video_id = "VIDEO_ID_HERE"  # only ID, not full URL
language = "en"  # use "hi" for Hindi videos

transcript = YouTubeTranscriptApi.get_transcript(
    video_id,
    languages=[language]
)
```

### 🔹 Transcript Format

```python
[
  {'text': 'sentence', 'start': 12.3, 'duration': 4.5},
  ...
]
```

### 🔹 Convert Transcript into Single Text

```python
full_text = " ".join([item["text"] for item in transcript])
```

✔️ Now we have the **complete video transcript as one string**

---

## 8. Step 2 – Text Splitting

Long text must be split into chunks.

### 🔹 Recursive Character Text Splitter

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_text(full_text)

len(chunks)  # example: 168 chunks
```

---

## 9. Step 3 – Create Embeddings & Vector Store

### 🔹 Embedding Model

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
```

### 🔹 FAISS Vector Store

```python
from langchain_community.vectorstores import FAISS

vector_store = FAISS.from_texts(chunks, embeddings)
```

✔️ **Indexing complete**

---

## 10. Step 4 – Create Retriever

Retriever fetches relevant chunks.

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
```

### 🔹 Test Retriever

```python
docs = retriever.invoke("What is DeepMind?")
len(docs)
```

✔️ Output → List of relevant documents

---

## 11. Step 5 – Prompt Template (Augmentation)

### 🔹 Prompt Template

```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, say \"I don't know\".

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)
```

---

## 12. Convert Retrieved Docs to Context String

```python
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
```

---

## 13. Step 6 – LLM Setup

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
```

---

## 14. Manual RAG (Before Chain)

```python
question = "Is the topic of aliens discussed in this video?"

docs = retriever.invoke(question)
context = format_docs(docs)

final_prompt = prompt.invoke({
    "context": context,
    "question": question
})

response = llm.invoke(final_prompt)
response.content
```

✔️ **RAG pipeline works**

---

## 15. Problem: Too Many Manual Steps ❌

We manually called:

* Retriever
* Prompt
* LLM

❌ Not scalable
❌ Not clean

---

## 16. Solution – Build LangChain Chain ✅

---

## 17. Parallel Chain (Context + Question)

### 🔹 Required Imports

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
```

### 🔹 Parallel Chain

```python
parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})
```

### 🔹 Test Parallel Chain

```python
parallel_chain.invoke("Who is Demis?")
```

Output:

```python
{
  "context": "...large text...",
  "question": "Who is Demis?"
}
```

---

## 18. Final RAG Chain

```python
from langchain_core.output_parsers import StrOutputParser

rag_chain = (
    parallel_chain
    | prompt
    | llm
    | StrOutputParser()
)
```

### 🔹 Single Call Execution

```python
rag_chain.invoke("Can you summarize the video?")
```

✔️ **One function → Full RAG pipeline**

---

## 19. Final Result

✅ Indexing
✅ Retrieval
✅ Augmentation
✅ Generation
✅ Clean & scalable code

---

## 20. Possible Improvements (Industry-Level)

### 🔹 UI Improvements

* Streamlit Website
* Chrome Extension

### 🔹 Evaluation

* RAGAS
* LangSmith

**Metrics:**

* Faithfulness
* Answer relevance
* Context recall
* Context precision

### 🔹 Indexing Improvements

* Fix transcript errors
* Translate non-English transcripts
* Semantic chunking
* Pinecone / cloud vector databases

### 🔹 Retrieval Improvements

* Query rewriting
* Multi-query retrieval
* Hybrid search
* MMR
* Re-ranking
* Contextual compression

### 🔹 Augmentation Improvements

* Better prompt templates
* Answer grounding
* Context window optimization

### 🔹 Generation Improvements

* Citations
* Guardrails
* Safety filters

### 🔹 Advanced RAG

* Multimodal RAG (text + image + video)
* Agentic RAG
* Memory-based RAG

---
# 📘 LangChain Tools & Agents – Simple English Notes

## 1. Playlist Recap

This LangChain playlist is divided into **3 parts**:

### Part 1 – LangChain Fundamentals

* Models
* Prompts
* Chains
* Other core components

### Part 2 – RAG (Retrieval Augmented Generation)

* Document Loaders
* Text Splitters
* Vector Stores
* Retrievers
* Built complete RAG-based systems

### Part 3 – Agents (Current Section)

* Tools (this video)
* Tool Calling (next video)
* Agents using LangChain & LangGraph

👉 To build **agents**, understanding **tools** is mandatory.

---

## 2. What is a Tool?

### Understanding LLM Limitations

LLMs have **only two core abilities**:

1. **Think (Reasoning)** – understand and break down questions
2. **Speak (Text Generation)** – generate answers in natural language

❌ LLMs **cannot**:

* Book tickets
* Call APIs
* Fetch live data
* Reliably solve complex math
* Run code
* Access databases

👉 LLMs are like a **human brain without hands and legs**.

---

## 3. Why Tools are Needed

### Definition

> **A Tool is a function that gives LLMs hands and legs**.

Tools allow LLMs to:

* Perform actions
* Call APIs
* Run code
* Interact with systems

### Simple Definition

> **A tool is just a Python function packaged so that an LLM can understand and call it when needed.**

---

## 4. Tools + LLM = Agents

### Agent Definition

> An AI Agent is an LLM-powered system that can **think, decide, and take actions using external tools and APIs**.

### Breakdown

* **Reasoning & Decision Making** → LLM
* **Taking Action** → Tools

👉 LLM + Tools = **Agent**

---

## 5. Types of Tools in LangChain

### 1️⃣ Built-in Tools

Pre-built, production-ready tools provided by LangChain.

Examples:

* DuckDuckGo Search
* Wikipedia Query
* Python REPL
* Shell Tool
* HTTP Requests
* Gmail, Slack, SQL tools

You just **import and use** them.

---

### 2️⃣ Custom Tools

You create them when:

* Built-in tools don’t match your use case
* You want to call your own APIs
* You want LLMs to interact with your database or app logic

---

## 6. Using Built-in Tools

### Example: DuckDuckGo Search Tool

```python
from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

result = search_tool.invoke("IPL news")
print(result)
```

✅ Useful for:

* Live news
* Current events
* Web search for agents

---

### Example: Shell Tool

```python
from langchain_community.tools import ShellTool

shell = ShellTool()

result = shell.invoke("whoami")
print(result)
```

⚠️ **Warning**: Shell tool is powerful but risky in production.

---

## 7. Creating Custom Tools (3 Ways)

### Method 1️⃣: Using `@tool` Decorator (Most Common)

#### Step-by-step

1. Write a Python function
2. Add docstring (important)
3. Add type hints
4. Add `@tool` decorator

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b
```

#### Using the Tool

```python
result = multiply.invoke({"a": 3, "b": 5})
print(result)  # 15
```

---

### Tool Attributes

```python
print(multiply.name)
print(multiply.description)
print(multiply.args)
```

Every tool has:

* `name`
* `description`
* `arguments`

---

### Tool Schema (What LLM Sees)

LLM does NOT see your function code.
It sees a **JSON schema** like this:

```json
{
  "name": "multiply",
  "description": "Multiply two numbers",
  "parameters": {
    "type": "object",
    "properties": {
      "a": {"type": "integer"},
      "b": {"type": "integer"}
    },
    "required": ["a", "b"]
  }
}
```

---

## 8. Method 2️⃣: StructuredTool + Pydantic (Strict Validation)

Used for **production-grade agents**.

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(description="First number")
    b: int = Field(description="Second number")

def multiply(a: int, b: int) -> int:
    return a * b

multiply_tool = StructuredTool.from_function(
    func=multiply,
    name="multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput
)
```

✔️ Enforces strict input schema

---

## 9. Method 3️⃣: BaseTool Class (Advanced)

Best for:

* Async tools
* Deep customization

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel

class MultiplyInput(BaseModel):
    a: int
    b: int

class MultiplyTool(BaseTool):
    name = "multiply"
    description = "Multiply two numbers"
    args_schema = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b
```

```python
tool = MultiplyTool()
print(tool.invoke({"a": 4, "b": 6}))  # 24
```

---

## 10. Toolkits

### What is a Toolkit?

> A **Toolkit** is a collection of related tools bundled together.

### Benefits

* Reusability
* Clean organization
* Easy integration

---

### Example: Math Toolkit

```python
class MathToolkit:
    def get_tools(self):
        return [add, multiply]

math_toolkit = MathToolkit()
tools = math_toolkit.get_tools()
```

---

## 11. Summary

✔️ Tools give LLMs **action power**
✔️ Built-in tools are ready-to-use
✔️ Custom tools let LLMs interact with your systems
✔️ Tools + LLM = Agents
✔️ Toolkits group related tools

---

## 12. What’s Next?

➡️ **Next Video**: Tool Calling (connecting tools with LLMs)
➡️ After that: Full **Agents** using LangChain & LangGraph

---

🎯 **Key Takeaway**:

> If you want to build real-world AI agents, mastering **tools** is non-negotiable.

# 📘 LangChain Tool Calling – Simple English Notes

These notes explain **Tools, Tool Binding, Tool Calling, and Tool Execution** in LangChain, step‑by‑step, in **simple English**, with **important code snippets**.

---

## 1️⃣ Quick Revision: What LLMs Can and Cannot Do

### What LLMs are good at

* **Reasoning** → understanding a question by breaking it down
* **Text generation** → producing a human‑like answer

Think of an LLM as:

> 🧠 Good at thinking + 🗣️ good at speaking

### Biggest limitation of LLMs

LLMs **cannot perform actions** on their own:

* ❌ Cannot call APIs
* ❌ Cannot update databases
* ❌ Cannot run commands
* ❌ Cannot fetch real‑time data

They **only generate text**.

👉 To give LLMs *hands and legs*, we use **Tools**.

---

## 2️⃣ What Are Tools in LangChain?

* Tools are **Python functions**
* Each tool performs **one task**
* LLMs can *suggest* using a tool

### Every tool must have:

1. **Name**
2. **Description** (what the tool does)
3. **Input schema** (what inputs it expects)

---

## 3️⃣ Tool Creation (Example: Multiply Tool)

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """
    Given two numbers a and b, this tool returns their product.
    """
    return a * b
```

### Testing the tool

```python
multiply.invoke({"a": 3, "b": 4})
# Output: 12
```

---

## 4️⃣ Tool Binding (Connecting Tool with LLM)

### What is Tool Binding?

Tool Binding means **registering tools with an LLM** so that:

* LLM knows which tools exist
* LLM knows what each tool does
* LLM knows how to call them

### Create an LLM

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
```

### Bind tools

```python
llm_with_tools = llm.bind_tools([multiply])
```

Now:

* `llm_with_tools` = LLM + tools

---

## 5️⃣ Tool Calling (LLM Suggests a Tool)

### Important concept

⚠️ **LLM does NOT run tools**

LLM only:

* Decides *which tool* is needed
* Generates *tool name + input arguments*

Actual execution is done by **you (the developer)**.

---

### Example 1: Normal Question

```python
llm_with_tools.invoke("Hi, how are you?")
```

✅ LLM replies normally
❌ No tool is used

---

### Example 2: Tool‑Required Question

```python
ai_msg = llm_with_tools.invoke("Can you multiply 3 with 10?")
```

### Tool Call Output

```python
ai_msg.tool_calls
```

Example output:

```json
[
  {
    "name": "multiply",
    "args": {"a": 3, "b": 10},
    "id": "tool_call_id",
    "type": "tool_call"
  }
]
```

👉 This is called **Tool Calling**.

---

## 6️⃣ Tool Execution (Running the Tool)

Now we **execute the tool manually**.

### Extract arguments and run tool

```python
tool_call = ai_msg.tool_calls[0]
result = multiply.invoke(tool_call)
```

### Tool Message

* Result is wrapped inside a **ToolMessage**
* This message can be sent back to the LLM

---

## 7️⃣ Maintaining Conversation History

LangChain works best with **message history**.

### Message types

* `HumanMessage`
* `AIMessage`
* `ToolMessage`

### Example flow

```python
from langchain_core.messages import HumanMessage

messages = []
messages.append(HumanMessage(content="Multiply 3 and 10"))

ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

# Execute tool
result = multiply.invoke(ai_msg.tool_calls[0])
messages.append(result)

# Send full context back to LLM
final_answer = llm.invoke(messages)
print(final_answer.content)
```

Output:

```
The product of 3 and 10 is 30.
```

---

## 8️⃣ Real‑World Example: Currency Converter

### Problem

LLMs **do not have real‑time currency rates**.

So we create tools:

1. Fetch conversion rate from API
2. Multiply amount with rate

---

## 9️⃣ Tool 1: Get Conversion Factor

```python
import requests
from langchain_core.tools import tool

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """
    Fetches real‑time currency conversion rate between two currencies.
    """
    url = f"https://api.exchangerate-api.com/v4/latest/{base_currency}"
    response = requests.get(url)
    data = response.json()
    return data["rates"][target_currency]
```

---

## 🔟 Tool 2: Convert Currency

```python
from langchain_core.tools import tool

@tool
def convert(amount: int, conversion_rate: float) -> float:
    """
    Converts base currency amount into target currency.
    """
    return amount * conversion_rate
```

---

## 1️⃣1️⃣ Injected Tool Arguments (Very Important)

### Problem

LLM may **hallucinate conversion rate** before tool execution.

### Solution

Use **InjectedToolArgument**

```python
from typing_extensions import Annotated
from langchain_core.tools import InjectedToolArgument

@tool
def convert(
    amount: int,
    conversion_rate: Annotated[float, InjectedToolArgument()]
) -> float:
    return amount * conversion_rate
```

Now:

* LLM will NOT fill `conversion_rate`
* Developer injects it manually after first tool execution

---

## 1️⃣2️⃣ Executing Multiple Tools in Order

```python
messages = [HumanMessage(content="Convert 10 USD to INR")]
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

conversion_rate = None

for tool_call in ai_msg.tool_calls:
    if tool_call["name"] == "get_conversion_factor":
        tool_msg = get_conversion_factor.invoke(tool_call)
        messages.append(tool_msg)
        conversion_rate = tool_msg.content

    if tool_call["name"] == "convert":
        tool_call["args"]["conversion_rate"] = conversion_rate
        tool_msg = convert.invoke(tool_call)
        messages.append(tool_msg)

final_answer = llm.invoke(messages)
print(final_answer.content)
```

---

## 1️⃣3️⃣ Is This an AI Agent?

❌ **No**

### Why?

Because:

* Developer controls execution
* Steps are manually coded
* LLM is not autonomous

---

## 1️⃣4️⃣ What Makes an AI Agent?

An AI Agent:

* Breaks problem into steps
* Chooses tools automatically
* Executes tools autonomously
* Needs minimal human control

👉 **This will be covered in the next video**.

---

## ✅ Final Summary

You learned:

* ✔️ Tool Creation
* ✔️ Tool Binding
* ✔️ Tool Calling
* ✔️ Tool Execution
* ✔️ Injected Tool Arguments
* ✔️ Real‑time API usage

🎯 This is the **foundation of AI Agents in LangChain**.

---

🚀 *Next step: Building a fully autonomous AI Agent*

# 🤖 AI Agents with LangChain – Simple English Notes

---

## 1. What this video is about

* This is the **last video** of the LangChain playlist
* Topic: **How to build AI Agents using LangChain**
* The video has **two parts**:

  * Conceptual understanding of AI Agents
  * Practical implementation of a basic AI Agent using LangChain

---

## 2. What problem do AI Agents solve? (Goa Trip Example)

### ❌ Traditional way (Manual & hectic)

Planning a **Delhi → Goa** trip manually means:

* Booking train or flight tickets
* Booking hotels
* Planning daily itinerary
* Booking local transport
* Comparing prices on many websites
* Making many decisions

Problems:

* Time-consuming
* Confusing
* Hard for elderly or non-tech users

### ✅ AI Agent way (Smart & automatic)

You just say:

> "Create a budget trip plan from Delhi to Goa from 1st–7th May"

AI Agent will:

* Understand your goal
* Break it into steps
* Search trains & flights
* Compare prices
* Suggest cheapest option
* Book tickets (with permission)
* Book hotels
* Plan daily itinerary
* Book local transport
* Track total cost
* Show final summary
* Adjust plan if you say **NO**

👉 User only says **YES / NO**
👉 Everything else is automated

---

## 3. What is an AI Agent? (Simple Definition)

### Simple words

An **AI Agent** is an intelligent system that:

* Takes a high-level goal
* Plans steps by itself
* Uses tools & APIs
* Executes tasks automatically
* Remembers context
* Adapts to new information
* Optimizes for best result

### Technical definition (simplified)

An AI Agent:

* Uses an LLM for reasoning
* Uses tools/APIs for actions
* Works in multiple steps
* Maintains memory & context
* Can re-plan if something changes

---

## 4. AI Agent vs LLM (Very Important)

### LLM (ChatGPT-like)

* Can reason
* Can generate text
* ❌ Cannot take real actions
* ❌ Cannot call APIs
* ❌ Cannot book tickets

### AI Agent

* Uses LLM for thinking
* Uses tools for actions
* Can call APIs
* Can search the web
* Can update databases
* Can automate workflows

👉 **AI Agent = LLM + Tools**

---

## 5. Core Characteristics of AI Agents

* **Goal-driven** – You tell *what*, not *how*
* **Planning ability** – Breaks big tasks into steps
* **Tool awareness** – Knows which tool to use
* **Context & memory** – Remembers preferences & progress
* **Adaptive** – Changes plan if something fails

---

## 6. How AI Agents work internally (High Level)

An AI Agent has:

* **LLM** → for reasoning
* **Tools** → for actions (search, API, DB, etc.)

### Flow

```
User Goal
   ↓
Agent thinks
   ↓
Agent uses tool
   ↓
Gets result
   ↓
Thinks again
   ↓
Final Answer
```

---

## 7. What is ReAct?

**ReAct = Reasoning + Acting**

It is a **design pattern** for AI Agents.

ReAct allows:

* Thinking using LLM
* Acting using tools
* Doing both in a loop

---

## 8. ReAct Loop (MOST IMPORTANT)

ReAct works in a **3-step loop**:

1. **Thought** – What should I do next?
2. **Action** – Which tool should I use?
3. **Observation** – What result did the tool give?

This loop repeats until the final answer is ready.

### Example: Capital & Population of France

**User Query**:

> What is the capital of France and its population?

**Iteration 1**

* Thought: I need the capital first
* Action: Search "capital of France"
* Observation: Paris

**Iteration 2**

* Thought: Now find population of Paris
* Action: Search "population of Paris"
* Observation: 2.1 million

**Iteration 3**

* Thought: I know the final answer

✅ **Final Answer**:
Paris has ~2.1M population

---

## 9. Why ReAct is powerful

* Handles multi-step problems
* Uses tools when required
* Transparent reasoning (debuggable)
* Better accuracy
* Ideal for real-world automation

---

## 10. LangChain AI Agent Architecture

Main components:

* Tool
* LLM
* Agent
* Agent Executor

---

## 11. Agent vs Agent Executor (Very Clear)

### Agent

* Thinks
* Plans
* Decides next action

### Agent Executor

* Executes actions
* Calls tools
* Manages the loop
* Orchestrates Thought → Action → Observation

👉 **Agent = Brain**
👉 **Agent Executor = Hands**

---

## 12. Basic LangChain Agent Code (Minimal & Important)

### Install libraries

```bash
pip install langchain langchain-openai langchain-community duckduckgo-search
```

### Create a search tool

```python
from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()
```

### Create LLM

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
```

### Load ReAct prompt

```python
from langchain import hub

prompt = hub.pull("hwchase17/react")
```

### Create ReAct Agent

```python
from langchain.agents import create_react_agent

agent = create_react_agent(
    llm=llm,
    tools=[search_tool],
    prompt=prompt
)
```

### Create Agent Executor

```python
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True
)
```

### Run the agent

```python
agent_executor.invoke({
    "input": "What are the three ways to reach Goa from Delhi?"
})
```

You will see:

* Agent thinking
* Tool usage
* Observations
* Final answer

---

## 13. Adding a Custom Tool (Weather Example)

### Custom weather tool

```python
from langchain.tools import tool
import requests

@tool
def get_weather(city: str) -> str:
    """Get current weather of a city"""
    url = f"http://api.weatherstack.com/current?access_key=API_KEY&query={city}"
    response = requests.get(url).json()
    return f"{city} temperature is {response['current']['temperature']}°C"
```

### Example query

```python
agent_executor.invoke({
    "input": "Find the capital of Madhya Pradesh and its current weather"
})
```

Agent will:

* Find capital → **Bhopal**
* Call weather API
* Return final answer

---

## 14. Full Agent Creation Flow (Memory Trick)

1. Create Tools
2. Create LLM
3. Choose Agent Design Pattern (ReAct)
4. Create Agent
5. Create Agent Executor
6. Invoke Agent

---

## 15. Important Reality Check (Very Important)

⚠️ **LangChain Agents are now considered old-style**

LangChain recommends:

* ❌ Not ideal for scalable production agents
* ✅ Use **LangGraph** instead

### LangGraph provides:

* Better control
* Stateful agents
* Production-grade workflows
* Scalable architectures

---

## 16. Why LangChain was still taught?

* To understand core concepts
* To learn:

  * Agent thinking
  * Tool calling
  * ReAct logic

👉 Concepts remain the same in **LangGraph**

---

## 17. Final Takeaway

* **AI Agent = LLM + Tools + Planning**
* **ReAct = Thought → Action → Observation loop**
* Agent thinks, Executor executes
* LangChain helps understand fundamentals
* **LangGraph is the future for real systems**

---

✅ End of Notes




