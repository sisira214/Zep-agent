
# Zep-Agent 🧠💬

An AI agent built in Python that integrates **Zep** as a long-term memory layer and uses **Streamlit** for an interactive chat interface. This project demonstrates how to build a memory-aware conversational agent that can recall past interactions and maintain context across conversations.

---

## 🔍 Overview

**Zep-Agent** showcases how to use **Zep** as persistent memory for an AI agent. Instead of treating each user interaction as isolated, the agent stores and retrieves relevant conversational context from Zep, enabling more coherent and personalized responses.

The application is entirely **Streamlit-based**, making it easy to run and interact with locally.

---

## 🧠 Why Zep?

**Zep** provides long-term memory for AI agents by:
- Persisting conversation history
- Retrieving relevant past context
- Maintaining temporal memory across sessions
- Reducing hallucinations and improving response grounding

This makes it ideal for conversational agents that need continuity and personalization.

---

## 🚀 Features

✔ Streamlit-based chat UI  
✔ Long-term conversational memory using Zep  
✔ Context-aware responses across sessions  
✔ Simple, extensible Python agent architecture  
✔ Designed for experimentation with agent memory

---

## 🧰 Tech Stack

| Component | Technology |
|---------|------------|
| Language | Python |
| UI | Streamlit |
| Memory Layer | Zep |
| Agent Logic | Custom Python |
| Environment | `.env`-based config |

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sisira214/Zep-agent.git
   cd Zep-agent


2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Configuration

Create a `.env` file and add your required credentials:

```env
OPENAI_API_KEY="your_openai_key"
ZEP_API_KEY="your_zep_api_key"
ZEP_BASE_URL="https://your-zep-instance-url"
```

*(Variable names may vary slightly — check `agent.py` for exact usage.)*

---

## ▶️ Run the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

This will open a browser window where you can chat with the agent.
The agent will retrieve relevant memory from Zep and store new interactions automatically.

---

## 🧠 How It Works

1. User enters a message in the **Streamlit UI**
2. The agent queries **Zep** for relevant past context
3. Retrieved memory is injected into the prompt
4. The model generates a response using both current input and memory
5. New conversation data is stored back in Zep

This loop allows the agent to improve continuity over time.

---

## 📁 Project Structure

```
Zep-agent/
├── .gitignore
├── LICENSE
├── README.md
├── agent.py          # Core agent logic + Zep memory integration
├── app.py            # Streamlit chat interface
└── requirements.txt  # Dependencies
```

---

## 🤝 Contributing

Contributions are welcome, including:

* Improved memory retrieval strategies
* UI enhancements
* Multi-user support
* Additional agent workflows

Open an issue or submit a pull request to contribute.

---

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.


```
