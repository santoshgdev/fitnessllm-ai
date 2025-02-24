#%%
import json
import os
import re
from typing import Annotated, List, Dict

from google.api_core.exceptions import GoogleAPICallError
from google.cloud import bigquery
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from pydantic import BaseModel, Field
from sqlalchemy import inspect

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
toolkit = SQLDatabaseToolkit(db=db, llm=model)
agent_executor = create_sql_agent(llm=model, toolkit=toolkit, verbose=True, handle_parsing_errors=True)


#%%
db = SQLDatabase.from_uri(bq_conn_w_credentials)
schema_info = db.get_table_info()

#%%
PROMPT_TEMPLATE = """You are a BigQuery SQL expert. Generate SQL queries using this schema:
**Dataset Context:**
- Project: {project_id}
- Dataset: dev_silver_strava
- Current Tables Available:
{table_details}

**Available Tables Schema:**
{table_schemas}

**Constraints:**
1. Use ONLY tables from: {table_names}
2. Always prefix tables with `dev_silver_strava`
3. Never reference non-existent columns
4. athlete_id can be omitted in current context

**Example Valid Queries:**
1. Recent activities:
    SELECT name, start_date
    FROM dev_silver_strava.activity_summary
    ORDER BY start_date DESC
    LIMIT 5
    
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


def get_bigquery_table_details(project_id: str, dataset_id: str) -> Dict:
    """Retrieves complete table metadata including schema, constraints, and samples.

    Args:
        project_id: GCP project ID
        dataset_id: BigQuery dataset name

    Returns:
        Dictionary with complete table metadata including:
        - columns with data types
        - primary/foreign keys
        - sample data
        - table relationships
    """
    client = bigquery.Client(project=project_id,
                             location="US")
    dataset_ref = client.dataset(dataset_id)

    metadata = {"tables": {}}

    try:
        # Get all tables in dataset
        tables = client.list_tables(dataset_ref)
        table_names = [t.table_id for t in tables]

        # Query INFORMATION_SCHEMA directly
        for table_name in table_names:
            table_data = {
                "columns": [],
                "primary_keys": [],
                "foreign_keys": [],
                "sample_data": [],
                "relationships": []
            }

            # 1. Get column metadata
            col_query = f"""
                SELECT 
                    column_name, 
                    table_name,
                    data_type,
                    is_nullable
                FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
                WHERE table_name = '{table_name}'
            """
            cols = client.query(col_query).result()
            table_data["columns"] = [dict(row) for row in cols]

            # 2. Get primary keys
            pk_query = f"""
                SELECT 
                    ccu.column_name
                FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS` tc
                JOIN `{project_id}.{dataset_id}.INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE` ccu
                    ON tc.constraint_name = ccu.constraint_name
                WHERE 
                    tc.table_name = '{table_name}'
                    AND tc.constraint_type = 'PRIMARY KEY'
            """
            pks = client.query(pk_query).result()
            table_data["primary_keys"] = [row.column_name for row in pks]

            # 3. Get foreign keys and relationships
            fk_query = f"""
                SELECT
                    rc.constraint_name,
                    ccu.table_name AS foreign_table,
                    ccu.column_name AS foreign_column,
                    kcu.column_name AS local_column
                FROM `{project_id}`.{dataset_id}.INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc
                JOIN `{project_id}`.{dataset_id}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
                    ON rc.constraint_name = kcu.constraint_name
                JOIN `{project_id}`.{dataset_id}.INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE ccu
                    ON rc.unique_constraint_name = ccu.constraint_name
                WHERE kcu.table_name = '{table_name}'
            """
            fks = client.query(fk_query).result()
            table_data["foreign_keys"] = [dict(row) for row in fks]

            # 4. Get sample data (first 5 rows)
            sample_query = client.query(
                f"SELECT * FROM `{project_id}.{dataset_id}.{table_name}` LIMIT 5"
            )
            table_data["sample_data"] = [
                dict(row) for row in sample_query.result()
            ]

            # 5. Find relationships based on foreign keys
            relationships = set()
            for fk in table_data["foreign_keys"]:
                relationships.add(f"{fk['local_column']} → {fk['foreign_table']}.{fk['foreign_column']}")
            table_data["relationships"] = list(relationships)

            metadata["tables"][table_name] = table_data

        # Add dataset-wide relationships
        metadata["dataset_relationships"] = _get_dataset_relationships(project_id, dataset_id)

    except GoogleAPICallError as e:
        print(f"BigQuery API error: {str(e)}")
    except Exception as e:
        print(f"General error: {str(e)}")

    return metadata


def _get_dataset_relationships(project_id: str, dataset_id: str) -> List[Dict]:
    """Identifies all dataset-level table relationships."""
    client = bigquery.Client(project=project_id)
    query = f"""
        SELECT
            rc.constraint_name,
            kcu.table_name AS from_table,
            kcu.column_name AS from_column,
            ccu.table_name AS to_table,
            ccu.column_name AS to_column
        FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS` rc
        JOIN `{project_id}.{dataset_id}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE` kcu
            ON rc.constraint_name = kcu.constraint_name
        JOIN `{project_id}.{dataset_id}.INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE` ccu
            ON rc.unique_constraint_name = ccu.constraint_name
    """
    results = client.query(query).result()
    return [dict(row) for row in results]

def explicit_invocation(question: str):
    try:
        # Step 1: Prepare Inputs
        raw_input = {"question": question}
        print(f"[DEBUG] Raw Input: {raw_input}")

        # Step 2: Schema Injection
        schema = db.get_table_info()
        print(f"[DEBUG] Schema Length: {len(schema)} chars")

        # Generate schema components
        out = get_bigquery_table_details(project_id=os.environ['PROJECT_ID'], dataset_id=dataset)

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