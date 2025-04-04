# Import required libraries
import mysql.connector
from mysql.connector import Error
import pandas as pd

# Step 1: Create a MySQL database connection
def create_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',  # Replace with your MySQL host
            user='root',      # Replace with your MySQL username
            password='Saikumar@2105',  # Replace with your MySQL password
            database='fraud_detection_db'  # Replace with your database name
        )
        print("MySQL connection established")
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

# Step 2: Create a table to store predictions
def create_table(connection):
    try:
        cursor = connection.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                Transaction_ID INT,
                Time INT,
                Amount FLOAT,
                Prediction INT,
                Actual_Class INT
            )
        ''')
        print("Table 'transactions' created successfully")
    except Error as e:
        print(f"Error creating table: {e}")

# Step 3: Insert predictions from CSV into the table
def insert_predictions_from_csv(connection, csv_file):
    try:
        # Read the CSV file
        predictions_df = pd.read_csv(r"C:\Users\Sai\OneDrive\Desktop\sravani\sravani\Fraud_detection\Fraud.csv")

        # Strip any extra spaces in column names
        predictions_df.columns = predictions_df.columns.str.strip()

        # Insert data into MySQL
        cursor = connection.cursor()
        for _, row in predictions_df.iterrows():
           cursor.execute('''
    INSERT INTO transactions (Transaction_ID, Time, Amount, Prediction, Actual_Class)
    VALUES (%s, %s, %s, %s, %s)
''', (int(row['Transaction_ID']), int(row['Time']), float(row['Amount']), int(row['Prediction']), int(row['Actual_Class'])))

        connection.commit()
        print("Predictions inserted into MySQL successfully")
    except Error as e:
        print(f"Error inserting predictions: {e}")

# Step 4: Retrieve predictions for dashboard use
def retrieve_predictions(connection):
    try:
        cursor = connection.cursor()
        cursor.execute('SELECT * FROM transactions WHERE Prediction = 1')  # Fetch only flagged transactions
        results = cursor.fetchall()
        print("Retrieved predictions from MySQL:")
        for row in results:
            print(row)
        return results
    except Error as e:
        print(f"Error retrieving predictions: {e}")
        return None

# Main function to execute all steps
def main():
    # Path to your existing CSV file
    csv_file = 'fraud_predictions.csv'

    # Step 1: Create connection
    connection = create_connection()
    if connection is None:
        return

    # Step 2: Create table
    create_table(connection)

    # Step 3: Insert predictions from CSV
    insert_predictions_from_csv(connection, csv_file)

    # Step 4: Retrieve predictions
    retrieve_predictions(connection)

    # Close connection
    connection.close()
    print("MySQL connection closed")

# Run the main function
if __name__ == "__main__":
    main()