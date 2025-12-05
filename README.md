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



