def generate_grouped_query(val_mapping):
    # Define the CTE structure with column placeholders
    cte_template = """
    WITH cte_table AS (
      SELECT
        A,
        B,
        C,
        D,
        E
      FROM
        tab_temp
    ),
    
    pivot_table AS (
      SELECT
        A,
        B,
        {pivot_columns}
      FROM
        cte_table
      GROUP BY
        A, B
    )

    SELECT
      A,
      B,
      {final_columns}
    FROM
      pivot_table
    GROUP BY
      A, B;
    """
    
    # Create the pivot column definitions for the SQL query based on val_mapping
    pivot_columns = []
    final_columns = []

    for original_val, new_val in val_mapping.items():
        col_d = f"MAX(CASE WHEN LOWER(TRIM(C)) = '{original_val.lower()}' THEN D ELSE NULL END) AS {new_val}_ColD"
        col_e = f"MAX(CASE WHEN LOWER(TRIM(C)) = '{original_val.lower()}' THEN E ELSE NULL END) AS {new_val}_ColE"
        pivot_columns.append(col_d)
        pivot_columns.append(col_e)
        
        # In the final select, use the already aggregated columns from pivot_table
        final_columns.append(f"{new_val}_ColD")
        final_columns.append(f"{new_val}_ColE")

    # Join the generated columns with commas
    pivot_columns_str = ",\n        ".join(pivot_columns)
    final_columns_str = ",\n      ".join(final_columns)

    # Insert the generated pivot and final columns into the template
    final_query = cte_template.format(pivot_columns=pivot_columns_str, final_columns=final_columns_str)
    
    return final_query


val_mapping = {"sce_code": "Sce_name", "sce_code2": "Sce_name2"}
print(generate_grouped_query(val_mapping))


      