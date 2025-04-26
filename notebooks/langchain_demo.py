#%%
import json
import os
import re
from typing import Annotated, List, Dict

from google.api_core.exceptions import GoogleAPICallError
from google.cloud import bigquery
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from google.oauth2 import service_account
from google.cloud import bigquery
import json
from pydantic import BaseModel, Field
from google.oauth2 import service_account

from notebooks.langchain_demo_helper import get_bigquery_table_details

os.environ["OLLAMA_HOST"] = "http://ollama:11434"
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
import ollama
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import BaseOutputParser, PydanticOutputParser
from langchain_community.agent_toolkits.sql.base import create_sql_agent

#%%
model = "DeepSeek-r1:8b"
ollama.pull(model)
model = OllamaLLM(model=model)

#%%
from sqlalchemy.dialects import registry
registry.register('bigquery', 'sqlalchemy_bigquery', 'BigQueryDialect')

#%%
dataset = "dev_silver_strava"
bq_conn = f"bigquery://{os.environ['PROJECT_ID']}/{dataset}"
bq_conn_w_credentials = f"{bq_conn}?credentials={os.environ['PROJECT_ID']}"
db = SQLDatabase.from_uri(database_uri=bq_conn_w_credentials)


#%%
def get_dataset_details(dataset: str, project_id: str, tables: list[str]):
    bq_conn = f"bigquery://{os.environ['PROJECT_ID']}/{dataset}"
    bq_conn_w_credentials = f"{bq_conn}?credentials={os.environ['PROJECT_ID']}"

    return get_bigquery_table_details(project_id, dataset_id=dataset, table_subset=tables)



#%%
toolkit = SQLDatabaseToolkit(db=db, llm=model)
agent_executor = create_sql_agent(llm=model, toolkit=toolkit, verbose=True, handle_parsing_errors=True)


#%%
db = SQLDatabase.from_uri(bq_conn_w_credentials)
schema_info = db.get_table_info()

#%%
PROMPT_TEMPLATE = """You are a BigQuery SQL expert. Generate SQL queries using this schema:
**Dataset Context:**

**Available Tables Schema:**
Bronze: {bronze_schema_info}
Silver: {silver_schema_info}

**Constraints:**
1. Only use the tables under the Available Tables Schema: aggregate_streams, activity, and athlete_summary
2. Always make sure to use the table under the correct schema and prefix them with the schema name

    For example, dev_silver_strava has aggregate_stream and dev_bronze_strava has activity and athlete_summary
3. Never reference non-existent columns, tables, or schemas
4. WHERE statement for athlete_id can be omitted in current context
    
**User Query:** {question}
Respond with JSON containing "query" and "reasoning" keys like:
{{
  "query": "SELECT ...",
  "reasoning": "Step-by-step explanation"
}}"""


prompt = PromptTemplate(template=PROMPT_TEMPLATE,
                        input_variables=["question"],
                        partial_variables={"schema_info": schema_info})

#%%
class SQLExtraction(BaseModel):
    query: Annotated[str, Field(description="The SQL query to extract")]
    reasoning: Annotated[str, Field(description="Step-by-step explanation accompanying the query")]

output_parser = PydanticOutputParser(pydantic_object=SQLExtraction)

class DeepSeekSQLOutputParser(BaseOutputParser):
    def parse(self, text: str) -> dict:
        try:
            # Handle common formatting issues
            text = re.search(r'\{.*\}', text, re.DOTALL).group()
            text = text.replace("'", "\"")  # Fix quotes
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
            text = text.replace("<think>", "").replace("</think>", "")
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

        # Generate schema components
        silver_info = get_dataset_details(dataset="dev_silver_strava", project_id=os.environ['PROJECT_ID'], tables=None)
        bronze_info = get_dataset_details(dataset="dev_bronze_strava", project_id=os.environ['PROJECT_ID'],
                                          tables=['activity', 'athlete_summary'])

        # Step 3: Prompt Generation
        prompt = PROMPT_TEMPLATE.format(
            question=question,
            bronze_schema_info=bronze_info,
            silver_schema_info=silver_info
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
            parsed = output_parser.parse(raw_output)
            print(f"[DEBUG] Parsed Output: {parsed}")
        except Exception as e:
            print(f"[ERROR] Parsing Failed. Raw Output:\n{raw_output}")
            raise

        # Step 6: Query Execution
        try:
            result = db.run(parsed.query)
            print(f"[DEBUG] Query Result: {result[:200]}...")  # Truncate for readability
            return result
        except Exception as e:
            print(f"[ERROR] SQL Execution Failed. Query:\n{parsed['query']}")
            raise

    except Exception as e:
        print(f"[FATAL] Pipeline Failed: {str(e)}")
        return None

result = explicit_invocation("when's the last time I did yoga?")
print(result)