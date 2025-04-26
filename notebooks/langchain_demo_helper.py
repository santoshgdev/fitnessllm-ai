import json
import os
from typing import List, Dict

from google.oauth2 import service_account


def get_bigquery_table_details(project_id: str, dataset_id: str, table_subset: list[str] | None = None) -> Dict:
    """Retrieves complete table metadata including schema, constraints, and samples.

    Args:
        project_id: GCP project ID
        dataset_id: BigQuery dataset name

    Returns:
        Dictionary with complete table metadata including:
        - columns with data types
        - primary and foreign key constraints
        - sample data
        - table relationships
    """
    from google.cloud import bigquery
    json_data = json.load(open(os.environ['GOOGLE_APPLICATION_CREDENTIALS']))
    your_credentials = service_account.Credentials.from_service_account_info(json_data)
    client = bigquery.Client(project=project_id, location="US", credentials=your_credentials)
    dataset_ref = client.dataset(dataset_id)

    metadata = {"tables": {}}

    try:
        # Get all tables in the dataset.
        tables = client.list_tables(dataset_ref)
        if table_subset:
            table_names = [t.table_id for t in tables if t.table_id in table_subset]
        else:
            table_names = [t.table_id for t in tables]

        # Query INFORMATION_SCHEMA for each table.
        for table_name in table_names:
            table_data = {
                "columns": [],
                "primary_keys": [],
                "foreign_keys": [],
                "sample_data": [],
                "relationships": []
            }

            # 1. Get column metadata.
            col_query = f"""
                SELECT 
                    column_name, 
                    data_type,
                    is_nullable
                FROM `{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
                WHERE table_name = '{table_name}'
            """
            cols = client.query(col_query).result()
            table_data["columns"] = [dict(row) for row in cols]

            # 2. Get primary keys.
            pk_query = f"""
                SELECT 
                    ccu.column_name
                FROM `{dataset_id}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS` tc
                JOIN `{dataset_id}.INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE` ccu
                    ON tc.constraint_name = ccu.constraint_name
                WHERE 
                    tc.table_name = '{table_name}'
                    AND tc.constraint_type = 'PRIMARY KEY'
            """
            pks = client.query(pk_query).result()
            table_data["primary_keys"] = [row.column_name for row in pks]

            # 3. Get foreign keys.
            fk_query = f"""
                SELECT 
                  tc.constraint_name,
                  ccu.table_name AS foreign_table,
                  ccu.column_name AS foreign_column,
                  kcu.column_name AS local_column
                FROM `{dataset_id}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS` AS tc
                JOIN `{dataset_id}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE` AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                JOIN `{dataset_id}.INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE` AS ccu
                  ON tc.constraint_name = ccu.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND kcu.table_name = '{table_name}'
            """
            fks = client.query(fk_query).result()
            table_data["foreign_keys"] = [dict(row) for row in fks]

            # 4. Get sample data (first 5 rows).
            sample_query = client.query(
                f"SELECT * FROM `{dataset_id}.{table_name}` LIMIT 5"
            )
            table_data["sample_data"] = [dict(row) for row in sample_query.result()]

            # 5. Build relationships from foreign key definitions.
            relationships = set()
            for fk in table_data["foreign_keys"]:
                relationships.add(f"{fk['local_column']} → {fk['foreign_table']}.{fk['foreign_column']}")
            table_data["relationships"] = list(relationships)

            metadata["tables"][table_name] = table_data

        # Add dataset-wide relationships.
        metadata["dataset_relationships"] = _get_dataset_relationships(project_id, dataset_id)

    except Exception as e:
        print(f"General error: {str(e)}")

    return metadata


def _get_dataset_relationships(project_id: str, dataset_id: str) -> List[Dict]:
    """Identifies all dataset-level table relationships."""
    from google.cloud import bigquery
    client = bigquery.Client(project=project_id, location="US")
    query = f"""
        SELECT
            tc.constraint_name,
            kcu.table_name AS from_table,
            kcu.column_name AS from_column,
            ccu.table_name AS to_table,
            ccu.column_name AS to_column
        FROM `{dataset_id}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS` AS tc
        JOIN `{dataset_id}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE` AS kcu
            ON tc.constraint_name = kcu.constraint_name
        JOIN `{dataset_id}.INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE` AS ccu
            ON tc.constraint_name = ccu.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY'
    """
    results = client.query(query).result()
    return [dict(row) for row in results]
