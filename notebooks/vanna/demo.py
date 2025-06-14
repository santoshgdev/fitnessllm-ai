#%%
import os
from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore
from fitnessllm_shared.cloud_utils import get_secret
from dotenv import load_dotenv
load_dotenv()

#%%
class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

secret = get_secret("LLM-OPENAI-DEV")

vn = MyVanna(config={'api_key': secret, 'model': 'o4-mini-2025-04-16', 'temperature': 1.0})

vn.connect_to_bigquery(project_id=os.environ['PROJECT_ID'])

#%%
sql = "SELECT * FROM dev_bronze_strava.INFORMATION_SCHEMA.COLUMNS"
df_information_schema = vn.run_sql(sql)
plan = vn.get_training_plan_generic(df_information_schema)
vn.train(plan=plan)

#%%
sql = "SELECT * FROM dev_silver_strava.INFORMATION_SCHEMA.COLUMNS"
df_information_schema = vn.run_sql(sql)
plan = vn.get_training_plan_generic(df_information_schema)
vn.train(plan=plan)
