import mysql.connector
from mysql.connector import Error
import pandas as pd

# ----------------------------------------
# 1. Connect to the database
# ----------------------------------------
def create_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='Sravs#0801',
            database='sravani'
        )
        print("MySQL connection established")
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

# ----------------------------------------
# 2. Create required tables if they don't exist
# ----------------------------------------
def create_tables(connection):
    try:
        cursor = connection.cursor()

        # Create transactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                Transaction_ID INT UNIQUE,
                Time INT,
                Amount FLOAT,
                Prediction INT,
                Actual_Class INT
            )
        ''')
        print("Table 'transactions' verified/created")

        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                email VARCHAR(255) NOT NULL UNIQUE,
                password VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        print("Table 'users' verified/created")

    except Error as e:
        print(f"Error creating tables: {e}")

# ----------------------------------------
# 3. Optional: Clear transactions table
# ----------------------------------------
def clear_transactions_table(connection):
    try:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM transactions")
        connection.commit()
        print("'transactions' table cleared (for dev/testing)")
    except Error as e:
        print(f"Error clearing transactions: {e}")

# ----------------------------------------
# 4. Insert predictions
# ----------------------------------------
def insert_predictions_from_csv(connection, csv_file):
    try:
        predictions_df = pd.read_csv(csv_file)
        predictions_df.columns = predictions_df.columns.str.strip()

        print(f"Loaded {len(predictions_df)} rows from: {csv_file}")

        cursor = connection.cursor()

        cursor.execute("CREATE TEMPORARY TABLE temp_transactions LIKE transactions")

        for _, row in predictions_df.iterrows():
            cursor.execute('''
                INSERT INTO temp_transactions 
                (Transaction_ID, Time, Amount, Prediction, Actual_Class)
                VALUES (%s, %s, %s, %s, %s)
            ''', (int(row['Transaction_ID']), int(row['Time']),
                  float(row['Amount']), int(row['Prediction']), int(row['Actual_Class'])))

        cursor.execute('''
            INSERT INTO transactions
            SELECT t.* FROM temp_transactions t
            LEFT JOIN transactions m ON t.Transaction_ID = m.Transaction_ID
            WHERE m.Transaction_ID IS NULL
        ''')

        cursor.execute('SELECT ROW_COUNT()')
        new_rows = cursor.fetchone()[0]

        cursor.execute("DROP TEMPORARY TABLE IF EXISTS temp_transactions")
        connection.commit()

        print(f"Inserted {new_rows} new records.")
        print(f"Skipped {len(predictions_df) - new_rows} duplicates.")
    except Error as e:
        print(f"Error inserting predictions: {e}")
        connection.rollback()

# ----------------------------------------
# 5. Retrieve & print predictions (flagged)
# ----------------------------------------
def retrieve_predictions(connection):
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT DISTINCT * FROM transactions")
        results = cursor.fetchall()
        print("Flagged Transactions:")
        for row in results:
            print(row)
        return results
    except Error as e:
        print(f"Error retrieving predictions: {e}")
        return None

# ----------------------------------------
# 6. Main Execution
# ----------------------------------------
def main():
    csv_file = 'Fraud.csv'  # Make sure this is your latest dataset
    #Connect mysql and create tables
    connection = create_connection()
    if connection is None:
        return

    create_tables(connection)
    
    # Uncomment the next line ONLY if you want to reset data each time (useful for dev only)
    # clear_transactions_table(connection)
    
    #insert the csv into mysql tables
    insert_predictions_from_csv(connection, csv_file)
    #Rretrieve the data from sql
    retrieve_predictions(connection)

    connection.close()
    print("🔚 MySQL connection closed")

if __name__ == "__main__":
    main()
