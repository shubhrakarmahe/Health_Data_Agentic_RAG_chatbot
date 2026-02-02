import streamlit as st
import pandas as pd
import logging
import time
import json
import ast
from datetime import datetime
from sqlalchemy import create_engine, text

# LangChain & Groq
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# ------------------------------------------------------------------
# 1. LOGGING & CONFIGURATION
# ------------------------------------------------------------------
st.set_page_config(page_title="Health Data Agent", page_icon="🛡️", layout="wide")

logging.basicConfig(
    filename="agent_ops.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)
logger = logging.getLogger("SQL_Agent")

SCHEMA_CONTEXT = """
Database Schema (SQLite):
1. Table: health_dataset_1 (Clinical & Demographics)
   - Primary Key: patient_number (TEXT)
   - Columns: blood_pressure_abnormality (TEXT) - values 'yes'/'no', 
              level_of_hemoglobin_g_per_dl (FLOAT), 
              genetic_pedigree_coefficient (FLOAT) - values between 0 and 1, 
              age (INT),
              bmi (INT), 
              sex (TEXT) - values male/female/men/women, 
              pregnancy (TEXT) - values 'yes'/'no', 
              smoking (TEXT) - values 'yes'/'no',
              salt_content_in_the_diet_mg_per_day (INT), 
              alcohol_consumption_ml_per_day (FLOAT),
              level_of_stress (TEXT) - values 'low', 'medium', 'high',
              chronic_kidney_disease (TEXT) - values 'yes'/'no', 
              adrenal_and_thyroid_disorders (TEXT) - values 'yes'/'no', 
              bmi_category (TEXT) - values 'Underweight', 'Normal', 'Overweight', 'Obese_class_I', 'Obese_class_II+',
              hemoglobin_category (TEXT) - values ['low', 'normal', 'high'],
              age_group (TEXT) - values ['<18', '18-29', '30-39', '40-49', '50-59', '60-69', '70+'],
              gpc_available (TEXT) - values 'yes'/'no'.

2. Table: health_dataset_2_agg (Activity Summary)
   - Primary Key: patient_number (TEXT)
   - Columns: total_physical_activity (INT), max_physical_activity (INT), 
     min_physical_activity (INT), mean_physical_activity (FLOAT), 
     no_of_days_missed_physical_activity (INT).

3. Table: health_dataset_2 (Daily Activity Logs)
   - Primary Key: patient_number (TEXT)
   - Columns: day_number (INT), physical_activity (INT).

Relationships:
- All tables act as a star schema joined by 'patient_number'.
"""

# ------------------------------------------------------------------
# 2. ROBUST AGENT LOGIC (With History Management)
# ------------------------------------------------------------------
class TokenTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.details = []

    def update(self, response, stage):
        try:
            usage = response.response_metadata.get('token_usage', {})
            i = usage.get('prompt_tokens', 0)
            o = usage.get('completion_tokens', 0)
            t = usage.get('total_tokens', 0)
            self.input_tokens += i
            self.output_tokens += o
            self.total_tokens += t
            self.details.append(f"Stage: {stage} | Input: {i} | Output: {o}")
        except:
            pass

class RobustSQLAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.db = SQLDatabase.from_uri("sqlite:///health_data.db")
        self.tracker = TokenTracker()
        self.primary_model = "llama-3.3-70b-versatile"
        self.fallback_model = "qwen-2.5-32b-instruct"

    def _get_llm(self, model_name):
        return ChatGroq(model=model_name, api_key=self.api_key, temperature=0)

    def _safe_invoke(self, chain, inputs, stage_name):
        try:
            llm = self._get_llm(self.primary_model)
            chain_with_llm = chain | llm
            res = chain_with_llm.invoke(inputs)
            self.tracker.update(res, f"{stage_name} (Llama)")
            return res, self.primary_model
        except Exception as e:
            logger.error(f"Primary model failed at {stage_name}: {e}")
            llm = self._get_llm(self.fallback_model)
            chain_with_llm = chain | llm
            res = chain_with_llm.invoke(inputs)
            self.tracker.update(res, f"{stage_name} (Qwen Fallback)")
            return res, self.fallback_model

    def run(self, user_query: str, chat_history: str = ""):
        self.tracker.reset()
        logger.info(f"START QUERY: {user_query}")
        
        # --- STEP 1: ROUTER ---
        router_prompt = ChatPromptTemplate.from_template(
            """
            Chat History: {history}
            Current Query: "{query}"
            Schema: {schema}
            
            Decide the output format:
            - "chart": visualization, graph, plot.
            - "table": "list", "show data", "dataframe".
            - "text": counts, summaries.
            
            Return JSON: {{ "action": "chart|table|text", "x_col": "...", "y_col": "..." }}
            """
        )
        router_res, _ = self._safe_invoke(router_prompt, {"query": user_query, "schema": SCHEMA_CONTEXT, "history": chat_history}, "Router")
        
        try:
            router_data = JsonOutputParser().parse(router_res.content)
        except:
            router_data = {"action": "text"}
            
        intent = router_data.get("action", "text")

        # --- STEP 2: SQL GENERATION ---
        sql_prompt = ChatPromptTemplate.from_template(
            """
            Role: You are a clinical expert in health data analysis and your role is to create SQL queries based on the user's request.
            Schema: {schema}
            Chat History: {history}
            Question: {query}
            
            Rules:
            1. Output ONLY valid SQLite SQL. No markdown.
            2. Use 'patient_number' for JOINS.
            3. Case insensitive matching for text columns.
            4. For charts/tables, ensure to SELECT x_col and y_col as specified.
            5. Limit results to 5 rows unless specified.
            6. Do NOT run DML statements (INSERT, UPDATE, DELETE, DROP).
            7. If unsure about column names, refer to the schema context.
            8. If the query is invalid or unsafe, respond with "INVALID".
            9. Always ensure SQL syntax correctness.
            10. Ensure to provide aliases for selected columns in SQL for clarity.
            11. If the user requests aggregated data (like counts, averages), use appropriate SQL aggregation functions.
            12. Provide proper labels when tabulating data for charts or tables.
            """
        )
        sql_res, _ = self._safe_invoke(sql_prompt, {"query": user_query, "schema": SCHEMA_CONTEXT, "history": chat_history}, "SQL Gen")
        
        generated_sql = sql_res.content.replace("```sql", "").replace("```", "").strip()
        
        # --- STEP 3: EXECUTION ---
        try:
            if any(k in generated_sql.upper() for k in ["DROP", "DELETE", "UPDATE"]):
                return {"error": "Unsafe Query Blocked", "sql": generated_sql}
            raw_data = self.db.run(generated_sql)
            try:
                data_obj = ast.literal_eval(raw_data) if isinstance(raw_data, str) else raw_data
            except:
                data_obj = raw_data
        except Exception as e:
            return {"error": f"Database Error: {str(e)}", "sql": generated_sql}

        # --- STEP 4: OUTPUT SYNTHESIS ---
        if intent == "text":
            syn_prompt = ChatPromptTemplate.from_template(
                "Chat History: {history}\nUser Question: {q}\nSQL Result: {r}\nTask: Answer concisely. No SQL. Don't mention 'database' or 'query'." \
                " Don't include any PII data in the answer."
            )
            syn_res, _ = self._safe_invoke(syn_prompt, {"q": user_query, "r": raw_data, "history": chat_history}, "Synthesis")
            final_answer = syn_res.content
        else:
            final_answer = "Here is the requested data."

        return {
            "intent": intent, "sql": generated_sql, "data": data_obj, 
            "answer": final_answer, "router_config": router_data,
            "token_stats": {"total": self.tracker.total_tokens, "input": self.tracker.input_tokens, "output": self.tracker.output_tokens}
        }

# ------------------------------------------------------------------
# 3. STREAMLIT UI
# ------------------------------------------------------------------
def main():
    
    
    if "last_stats" not in st.session_state:
        st.session_state.last_stats = None

    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("Groq API Key", type="password")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        if "last_sql" in st.session_state:
            with st.expander("View Generated SQL", expanded=False):
                st.code(st.session_state.last_sql, language="sql")

        if st.session_state.last_stats:
            with st.expander("📊 Performance Stats", expanded=False):
                s = st.session_state.last_stats
                st.metric("Total Tokens", s['total'])
                st.write(f"📥 Input: {s['input']}")
                st.write(f"📤 Output: {s['output']}")

        # Feedback Loop
        if "last_sql" in st.session_state:
            st.divider()
            st.subheader("📢 Feedback")
            with st.form("feedback_form"):
                sentiment = st.radio("Rate this result:", ["👍 Helpful", "👎 Incorrect"], horizontal=True)
                notes = st.text_area("How can we improve?")
                if st.form_submit_button("Submit Feedback"):
                    logger.info(f"FEEDBACK: {sentiment} - {notes} - SQL: {st.session_state.last_sql}")
                    st.success("Thank you! Feedback logged.")

    st.title("🩺 Health Data Agent")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("type") == "chart":
                df = pd.DataFrame(msg["data"])
                if not df.empty: st.bar_chart(df.set_index(df.columns[0]))
            elif msg.get("type") == "table":
                st.dataframe(pd.DataFrame(msg["data"]))

    # --- INPUT & EXECUTION ---
    if prompt := st.chat_input("Ex: How many patients are smokers?"):
        if not api_key:
            st.warning("Please enter API Key.")
            st.stop()
            
        # 1. Format Chat History for the LLM
        history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages[-5:]]) # Last 5 turns

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing context..."):
                agent = RobustSQLAgent(api_key)
                # 2. Pass history to the agent
                result = agent.run(prompt, chat_history=history_str)
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.session_state.last_sql = result["sql"]
                    st.write(result["answer"])
                    st.session_state.last_stats = result["token_stats"]
                    
                    if result["intent"] == "chart":
                        df = pd.DataFrame(result["data"])
                        st.bar_chart(df.set_index(df.columns[0]))
                    elif result["intent"] == "table":
                        st.dataframe(pd.DataFrame(result["data"]))
                    
                    #st.caption(f"⚡ Tokens: {result['token_stats']['total']}")
                    # 3. Add Token Stats (Small footer)
                    t_stats = result["token_stats"]
                    st.session_state.last_stats = t_stats

                    
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": result["answer"], 
                        "type": result["intent"], 
                        "data": result["data"]
                    })
                    st.rerun()

if __name__ == "__main__":
    main()
