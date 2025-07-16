import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from google.cloud import bigquery
from google.api_core.exceptions import BadRequest
import os


def validate_sql_from_sheet(sheet_url: str, creds_path: str):
    """
    Reads question-SQL pairs from a Google Sheet and performs a BigQuery dry run.

    Args:
        sheet_url (str): The URL of the Google Sheet.
        creds_path (str): The file path to the service account credentials JSON.
    """
    # 1. Authenticate and access the Google Sheet
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_file(creds_path, scopes=scope)
        client = gspread.authorize(creds)

        spreadsheet = client.open_by_url(sheet_url)
        worksheet = spreadsheet.get_worksheet(0)  # Get the first sheet

        # Get all records and convert to a Pandas DataFrame
        records = worksheet.get_all_records()
        df = pd.DataFrame.from_dict(records)
        print(f"Successfully loaded {len(df)} records from Google Sheet.\n")

    except Exception as e:
        print(f"Error accessing Google Sheet: {str(e)}")
        return

    # 2. Initialize BigQuery client
    bq_client = bigquery.Client(credentials=creds, project=creds.project_id)

    # 3. Loop through queries and perform a dry run
    print("--- Starting BigQuery Dry Runs ---")
    successful_validations = 0
    failed_validations = 0

    for index, row in df.iterrows():
        question = row['question']
        sql_query = row['sql']

        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)

        try:
            # Start the dry run query
            query_job = bq_client.query(sql_query, job_config=job_config)

            # A dry run query completes immediately
            bytes_processed = query_job.total_bytes_processed
            print(f"[SUCCESS] Question: '{question}'")
            print(f"  └── Query is valid. Will process {bytes_processed / 1e6:.2f} MB of data.\n")
            successful_validations += 1

        except BadRequest as e:
            # Handle invalid SQL queries
            print(f"[FAILURE] Question: '{question}'")
            print(f"  └── Query is invalid. Error: {e.errors[0]['message']}\n")
            failed_validations += 1

        except Exception as e:
            print(f"[ERROR] An unexpected error occurred for question: '{question}'")
            print(f"  └── Error: {e}\n")
            failed_validations += 1

    print("--- Validation Complete ---")
    print(f"Total Valid Queries: {successful_validations}")
    print(f"Total Invalid Queries: {failed_validations}")


if __name__ == "__main__":
    # --- CONFIGURE YOUR DETAILS HERE ---
    # 1. Paste the URL of your Google Sheet
    GOOGLE_SHEET_URL = os.environ["GOOGLE_SHEET"]

    validate_sql_from_sheet(sheet_url=GOOGLE_SHEET_URL, creds_path=os.environ["GOOGLE_APPLICATION_CREDENTIALS"])

