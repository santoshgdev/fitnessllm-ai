#%%
import json
import os
import re

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from sqlalchemy import create_engine, text

os.environ["OLLAMA_HOST"] = "http://ollama:11434"
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
import ollama
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import BaseOutputParser
from langchain_community.agent_toolkits.sql.base import create_sql_agent

#%%
model = "DeepSeek-r1:8b"
ollama.pull(model)
model = OllamaLLM(model=model)

#%%
from sqlalchemy.dialects import registry
registry.register('bigquery', 'sqlalchemy_bigquery', 'BigQueryDialect')

#%%
dataset = "dev_strava"
bq_conn = f"bigquery://{os.environ['PROJECT_ID']}/{dataset}"
bq_conn_w_credentials = f"{bq_conn}?credentials={os.environ['PROJECT_ID']}"
db = SQLDatabase.from_uri(database_uri=bq_conn_w_credentials)


#%%
toolkit = SQLDatabaseToolkit(db=db, llm=model)
agent_executor = create_sql_agent(llm=model, toolkit=toolkit, verbose=True, handle_parsing_errors=True)


#%%
db = SQLDatabase.from_uri(bq_conn_w_credentials)
schema_info = db.get_table_info()

#%%
PROMPT_TEMPLATE = """You are a BigQuery SQL expert. Generate SQL queries using this schema:
{schema}

Form the SQL using only columns belonging to tables in the above schema. In bigquery dataset is organized as follows:

The athlete_id refers to each easier. For now there is only one athlete (me). The activity_id corresponds to each workout. 
You need to be able to join the time table with other tables using both the activity_id and index to be able to see the value at each time point for all tables except activity and athlete_summary.
For example, if you joined time with power, the data columns for both tables would you give the time point within a particular activity and its associated power. 

On the other hands, the activity table give the activity summary for each activity.

Current request: {question}
Respond with JSON containing "query" and "reasoning" keys like:
{{
  "query": "SELECT ...",
  "reasoning": "Step-by-step explanation"
}}"""


prompt = PromptTemplate(template=PROMPT_TEMPLATE,
                        input_variables=["question"],
                        partial_variables={"schema_info": schema_info})

#%%
class DeepSeekSQLOutputParser(BaseOutputParser):
    def parse(self, text: str) -> dict:
        try:
            # Handle common formatting issues
            text = re.search(r'\{.*\}', text, re.DOTALL).group()
            text = text.replace("'", "\"")  # Fix quotes
            text = text.replace("<thinking>", "").replace("</thinking>", "")
            return super().parse(text)
        except Exception as e:
            raise ValueError(f"Failed to parse model output: {text}") from e

    def get_format_instructions(self) -> str:
        return """Respond ONLY with JSON containing:
        {
            "query": "generated SQL query",
            "reasoning": "step-by-step explanation"
        }"""


def explicit_invocation(question: str):
    try:
        # Step 1: Prepare Inputs
        raw_input = {"question": question}
        print(f"[DEBUG] Raw Input: {raw_input}")

        # Step 2: Schema Injection
        schema = db.get_table_info()
        print(f"[DEBUG] Schema Length: {len(schema)} chars")

        # Step 3: Prompt Generation
        prompt = PROMPT_TEMPLATE.format(
            question=question,
            schema=schema
        )
        print(f"[DEBUG] Generated Prompt:\n{prompt[:500]}...")  # Show first 500 chars

        # Step 4: Model Invocation
        raw_output = ""
        try:
            raw_output = model.invoke(prompt)
            print(f"[DEBUG] Raw Model Output:\n{raw_output}")
        except Exception as e:
            print(f"[ERROR] Model Failed: {str(e)}")
            return None

        # Step 5: Output Parsing
        parser = DeepSeekSQLOutputParser()
        try:
            parsed = parser.parse(raw_output)
            print(f"[DEBUG] Parsed Output: {parsed}")
        except Exception as e:
            print(f"[ERROR] Parsing Failed. Raw Output:\n{raw_output}")
            raise

        # Step 6: Query Execution
        try:
            result = db.run(parsed["query"])
            print(f"[DEBUG] Query Result: {result[:200]}...")  # Truncate for readability
            return result
        except Exception as e:
            print(f"[ERROR] SQL Execution Failed. Query:\n{parsed['query']}")
            raise

    except Exception as e:
        print(f"[FATAL] Pipeline Failed: {str(e)}")
        return None

result = explicit_invocation("What's the average workout duration?")
print(result)