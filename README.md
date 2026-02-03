# Health_Data_Agentic_RAG_chatbot

An intelligent, context-aware clinical data assistant built with **LangChain**, **Groq**, and **Streamlit**. This agent allows health professionals to query complex patient datasets using natural language, automatically generating SQLite queries and providing visual insights (charts/tables) or natural language summaries.

## 🚀 Key Features

* **Multi-Model Fallback**: Uses `Llama-3.3-70b` as the primary engine with an automatic fallback to `Qwen-2.5-32b` for high reliability.
* **Contextual Memory**: Remembers the last 5 turns of conversation to handle follow-up questions (e.g., "Show me the smokers," followed by "How many of *them* have high blood pressure?").
* **Intelligent Routing**: Automatically decides if the user needs a **Bar Chart**, a **Data Table**, or a **Plain Text** summary.
* **Production Guardrails**:
* **DML Protection**: Blocks `INSERT`, `UPDATE`, `DELETE`, and `DROP` commands.
* **PII Masking**: Specifically instructed to avoid mentioning database internals or private identifiers in natural language responses.
* **Safety Filtering**: Validates SQL syntax and query safety before execution.


* **Real-time Observability**: Sidebar tracking of token usage (Input/Output/Total) and the raw SQL generated for every request.

## 🛠️ Tech Stack

* **Frontend**: [Streamlit](https://streamlit.io/)
* **Orchestration**: [LangChain](https://www.langchain.com/)
* **LLM API**: [Groq Cloud](https://groq.com/)
* **Database**: SQLite (SQLAlchemy)
* **Data Handling**: Pandas & NumPy

## 📊 Database Schema

The agent is optimized for a star schema focused on clinical demographics and activity logs:

1. **`health_dataset_1`**: Clinical metrics (BP, Hemoglobin, BMI, Stress levels, etc.)
2. **`health_dataset_2_agg`**: Aggregated physical activity summaries.
3. **`health_dataset_2`**: Granular daily activity logs.

## 📥 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/health-data-agent.git
cd health-data-agent

```


2. **Install dependencies**:
```bash
pip install streamlit pandas sqlalchemy langchain-groq langchain-community groq

```


3. **Setup Database**:
Ensure your `health_data.db` SQLite file is in the root directory.
4. **Run the App**:
```bash
streamlit run app.py

```



## 📋 Usage Instructions

1. Open the application in your browser.
https://health-data-agentic-rag-chatbot.streamlit.app
2. Enter your **Groq API Key** in the sidebar.
3. Type a question in the chat input:
* *"Show a graph of blood pressure abnormality by age group."*
* *"List patients with high stress levels and BMI over 30."*
* *"What is the average hemoglobin level for smokers?"*


4. View the **Performance Stats** and **Generated SQL** in the sidebar to verify accuracy.

## 🛡️ Security & Logging

The system maintains a detailed operation log in `agent_ops.log`. It tracks:

* Model routing decisions.
* Token consumption per stage (Router, SQL Gen, Synthesis).
* User feedback and sentiment.
* Database errors and blocked unsafe queries.

---
