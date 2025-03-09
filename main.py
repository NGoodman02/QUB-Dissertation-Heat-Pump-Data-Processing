import os
import sqlite3
from datetime import datetime, timezone
import pytz
import pandas as pd
import re
import openpyxl

# Define file paths
BASE_PATH = "/Users/admin/Documents/QUB Final Year/Dissertation/JCC Stadium Data/Consolidated AH Data Logs/2024"
EXCEL_FILE = "/Users/admin/Documents/QUB Final Year/Dissertation/Data Spreadsheet Files/ModbusTablesForV2-9.xlsx"
OUTPUT_FILE = "/Users/admin/Documents/QUB Final Year/Dissertation/Data Spreadsheet Files/Filtered_ModbusTables_Data.xlsx"
DB_FILE = "/Users/admin/Documents/QUB Final Year/Dissertation/Output_DB.db"

# List of request names to process
REQUEST_NAMES = ["HP1_HR1", "HP1_HR2a", "HP1_HR2b", "HM1_HR1", "HM1_HR2"]

# List of desired variables to include
DESIRED_VARIABLES = [
    "Formatted Time",  # Ensure the time column is explicitly included
    "HP1 (Air Source) - Primary exchanger setpoint currently used (Scaled - /10 °C)",
    "HP1 (Air Source) - Evaporator inlet temperature (Scaled - /10 °C)",
    "HP1 (Air Source) - Evaporator outlet temperature (Scaled - /10 °C)",
    "HP1 (Air Source) - Condenser inlet temperature (Scaled - /10 °C)",
    "HP1 (Air Source) - Circuit 1 high pressure (Scaled - /10 bar)",
    "HP1 (Air Source) - Circuit 1 low pressure (Scaled - /10 bar)",
    "HP1 (Air Source) - External temperature (Scaled - /10 °C)",
    "HP1 (Air Source) - Power absorbed by the unit (Scaled - /10 W)",
    "HP1 (Air Source) - Total energy absorbed by the unit (Scaled - kWh)",
    "HM1 (First stage) - Energy_Total (Scaled - /10 MWh)",
    "HM1 (First stage) - Power_Inst (Scaled - kW)"
]

# Function to connect to the database
def connect_to_database(db_path):
    try:
        return sqlite3.connect(db_path)
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return None

# Function to query the database for a specific request name
def query_database_for_request(cursor, request_name):
    try:
        query = f"SELECT response_data, time FROM modbus_data WHERE request_name = '{request_name}'"
        cursor.execute(query)
        return cursor.fetchall()
    except sqlite3.Error as e:
        print(f"Error executing query for {request_name}: {e}")
        return []

# Function to format epoch timestamps to local time with timezone awareness
def format_epoch_to_local(epoch_timestamp, timezone_str='Europe/London'):
    try:
        # Create a timezone-aware datetime object in UTC
        utc_datetime = datetime.fromtimestamp(epoch_timestamp, tz=timezone.utc)
        # Convert it to the desired local timezone
        local_datetime = utc_datetime.astimezone(pytz.timezone(timezone_str))
        return local_datetime.strftime('%m-%d-%Y, %I:%M %p')
    except Exception as e:
        print(f"Error formatting timestamp {epoch_timestamp}: {e}")
        return None

# Function to create a dictionary of parameters from the Excel file
def create_parameter_dict(hr_ref, excel_file):
    parameter_dict = {}
    try:
        # Load all sheets into a dictionary of DataFrames
        sheets = pd.read_excel(excel_file, sheet_name=None)

        # Iterate through each sheet to find all rows matching the HR ref
        for sheet_name, df in sheets.items():
            if "HR ref" in df.columns:
                rows = df[df["HR ref"].str.strip().str.lower() == hr_ref.lower()]
                for _, row in rows.iterrows():
                    description = row.get("Description", f"Unknown Parameter for {hr_ref}")
                    uom = row.get("UoM", "")
                    scaling_factor = 1  # Default scaling factor
                    if isinstance(uom, str):
                        match = re.search(r'\d+', uom)  # Extract integer from UoM
                        if match:
                            scaling_factor = int(match.group())
                    parameter_dict[description] = {"scaling_factor": scaling_factor, "uom": uom}

    except Exception as e:
        print(f"Error creating parameter dictionary for {hr_ref}: {e}")

    return parameter_dict

# Function to insert data into SQLite database
def insert_data_into_db(connection, data):
    try:
        cursor = connection.cursor()

        # Create table if it doesn't exist
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS modbus_data (
            "Formatted Time" TEXT,
            "Parameter Name" TEXT,
            "Scaled Value" REAL
        )
        '''
        cursor.execute(create_table_query)

        # Insert rows into the table
        for row in data:
            formatted_time = row["Formatted Time"]
            for param_name, value in row.items():
                if param_name != "Formatted Time":
                    cursor.execute("INSERT INTO modbus_data (\"Formatted Time\", \"Parameter Name\", \"Scaled Value\") VALUES (?, ?, ?)",
                                   (formatted_time, param_name, value))

        connection.commit()
        print("Data successfully inserted into database.")
    except sqlite3.Error as e:
        print(f"Error inserting data into database: {e}")

# Main function to process data
def main():
    # Initialize a list to store all data
    all_data = []

    # Recursively traverse the folder structure
    for root, _, files in os.walk(BASE_PATH):
        for file in files:
            if file.endswith(".db"):
                db_path = os.path.join(root, file)
                print(f"Processing file: {db_path}")

                # Step 1: Connect to SQLite Database
                connection = connect_to_database(db_path)
                if not connection:
                    continue

                cursor = connection.cursor()

                # Step 2: Process each request name
                for request_name in REQUEST_NAMES:
                    parameter_dict = create_parameter_dict(request_name, EXCEL_FILE)
                    results = query_database_for_request(cursor, request_name)

                    if not results:
                        print(f"No matching data found for request_name = '{request_name}'. Skipping.")
                        continue

                    # Process Results
                    processed_data = []
                    for row in results:
                        response_data = row[0].decode("utf-8") if isinstance(row[0], bytes) else row[0]
                        epoch_timestamp = row[1]
                        formatted_time = format_epoch_to_local(epoch_timestamp)

                        # Split response_data into individual numbers
                        try:
                            numbers = response_data.strip("b'[]").split(",")  # Remove b'[] and split by comma
                            numbers = [float(num.strip()) for num in numbers]  # Convert to floats
                        except ValueError as e:
                            print(f"Error processing response_data: {response_data}. Error: {e}")
                            continue

                        # Add time and scaled numbers with parameter names to the processed data
                        processed_row = {"Formatted Time": formatted_time}
                        for i, number in enumerate(numbers):
                            # Use parameter name and scaling factor from the dictionary
                            if i < len(parameter_dict):
                                param_name = list(parameter_dict.keys())[i]
                                scaling_factor = parameter_dict[param_name]["scaling_factor"]
                                uom = parameter_dict[param_name]["uom"]
                                processed_row[f"{param_name} (Scaled - {uom})"] = number / scaling_factor
                            else:
                                processed_row[f"{request_name}_Value {i + 1}"] = number  # Default fallback

                        processed_data.append(processed_row)

                    # Add processed data to all_data list
                    all_data.extend(processed_data)

                # Close the Database Connection
                connection.close()

    # Step 3: Convert all data into a DataFrame
    final_df = pd.DataFrame(all_data)

    # Step 4: Filter the DataFrame to include only the desired variables
    final_df = final_df[["Formatted Time"] + [col for col in final_df.columns if col in DESIRED_VARIABLES]]

    # Step 5: Remove duplicate column names (if any)
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    # Step 6: Convert 'Formatted Time' to datetime
    final_df['Formatted Time'] = pd.to_datetime(final_df['Formatted Time'], errors='coerce', format='%m-%d-%Y, %I:%M %p')

    # Step 7: Group by 'Formatted Time' to ensure no duplicate timestamps
    final_df = final_df.groupby('Formatted Time', as_index=False).last()

    # Step 8: Sort by 'Formatted Time'
    final_df = final_df.sort_values(by="Formatted Time")

    # Step 9: Insert data into SQLite database
    db_connection = connect_to_database(DB_FILE)
    if db_connection:
        insert_data_into_db(db_connection, all_data)
        db_connection.close()

    # Step 10: Write the data to a single sheet in the output Excel file

    with pd.ExcelWriter(OUTPUT_FILE) as writer:
        final_df.to_excel(writer, sheet_name="All Data", index=False)

    print(f"Filtered data has been written to {OUTPUT_FILE} and SQLite database.")

# Execute the script
if __name__ == "__main__":
    main()


