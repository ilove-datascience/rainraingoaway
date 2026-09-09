import os
import mysql.connector
from mysql.connector import Error
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)
DEFAULT_DB_CONFIG = {
	"host": "localhost",
	"user": "root",
	"password": "",
	"database": "WeatherBot"
}

def _load_db_config():
	"""Load database credentials from environment variables."""
	return {
		"host": os.getenv("MYSQLHOST", DEFAULT_DB_CONFIG["host"]),
		"user": os.getenv("MYSQLUSER", DEFAULT_DB_CONFIG["user"]),
		"password": os.getenv("MYSQL_ROOT_PASSWORD", DEFAULT_DB_CONFIG["password"]),
		"database": os.getenv("MYSQL_DATABASE", DEFAULT_DB_CONFIG["database"]),
	}



def _get_db_connection():
	"""Get a MySQL connection using cached credentials."""
	config = _load_db_config()
	return mysql.connector.connect(
		host=config.get("host"),
		user=config.get("user"),
		password=config.get("password"),
		database=config.get("database")
	)
 
from mysql.connector import Error


def add_user(userid) -> bool:
    connection = None
    cursor = None

    try:
        connection = _get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO users (userid)
            VALUES (%s)
            """,
            (userid,)
        )

        connection.commit()

        return True

    except Error as e:
        print(f"Failed to add user {userid}: {e}")

        if connection and connection.is_connected():
            connection.rollback()

        return False

    finally:
        if cursor:
            cursor.close()

        if connection and connection.is_connected():
            connection.close()

def add_location(userid, lat, long)-> bool:
    connection = None
    cursor = None

    try:
        connection = _get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
                INSERT INTO user_location (
                    userid,
                    latitude,
                    longitude
                )
                VALUES (%s, %s, %s)

                ON DUPLICATE KEY UPDATE
                    latitude = VALUES(latitude),
                    longitude = VALUES(longitude)
            """,
            (
                userid,
                lat,
                long
            )
        )
        cursor.execute(
            """
                UPDATE users
                SET location_flag = 1
                WHERE userid = %s
            """,
            (userid,)
        )


        

        connection.commit()

        return True

    except Error as e:
        print(f"Failed to add/update location for {userid}: {e}")

        if connection and connection.is_connected():
            connection.rollback()

        return False

    finally:
        if cursor:
            cursor.close()

        if connection and connection.is_connected():
            connection.close()



def save_mode_choice(userid, mode)->bool:
    connection = None
    cursor = None
    connection = _get_db_connection()
    cursor = connection.cursor(dictionary=True)
    userid=userid
    try:
        cursor.execute(
                    """
                    UPDATE users 
                    SET mode = %s
                    WHERE userid = %s
                    """,
                    (mode,userid)
                )
        
        connection.commit()
        return True 
        
    except Error as e:
        print(f"Failed to add user {userid}: {e}")

        if connection and connection.is_connected():
            connection.rollback()

        return False
    
    finally:
            if cursor:
                cursor.close()
    
            if connection and connection.is_connected():
                connection.close()

def get_autoupdate_users()-> list:
    connection = None
    cursor = None

    try:
        connection = _get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT userid
            FROM users
            WHERE mode = %s
            """,
            ("automatic",)
        )

        rows = cursor.fetchall()

        return [row[0] for row in rows]

    except Error as e:
        print(f"Failed to get automatic users: {e}")
        return []

    finally:
        if cursor:
            cursor.close()

        if connection and connection.is_connected():
            connection.close()

def get_location(userid):
    connection=None
    cursor = None 
    try:
        connection = _get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT latitude, longitude
            FROM user_location
            WHERE userid = %s
            """,
            (userid,)
        )

        row = cursor.fetchone()
        return row
    except Error as e:
        print(f"Failed to get automatic users: {e}")
        return []

    finally:
        if cursor:
            cursor.close()

        if connection and connection.is_connected():
            connection.close()
        
def main():
    connection = None
    cursor = None

    # Dummy test data
    test_chat_id = 999999915
    test_longitude = 103.8198
    test_latitude = 1.3521
    #test_datetime = datetime.now()

    try:
        # ----------------------
        # Test connection
        # ----------------------
        print("Connecting to MySQL...")

        connection = _get_db_connection()

        if connection.is_connected():
            print("Database connection successful!")

        cursor = connection.cursor(dictionary=True)

        # ----------------------
        # Test WRITE
        # ----------------------
        print("\nTesting write...")

        insert_query_loc = """
            INSERT INTO user_location
                (userid, longitude, latitude)
            VALUES (%s, %s, %s)
        """
        
        insert_query_user= """INSERT INTO users 
                    (userid)
                    values (%s)"""
        cursor.execute(
            insert_query_user,
            (
                test_chat_id,
        
                #test_datetime
            )
        )
        
        cursor.execute(
            insert_query_loc,
            (
                test_chat_id,
                test_longitude,
                test_latitude,
                #test_datetime
            )
        )

        connection.commit()

        print("Write successful!")

        # ----------------------
        # Test READ
        # ----------------------
        print("\nTesting read...")

        select_query = """
            SELECT
                userid,
                longitude,
                latitude,
                last_updated 
            FROM user_location
            WHERE userid = %s
        """

        cursor.execute(select_query, (test_chat_id,))

        result = cursor.fetchone()

        if result:
            print("Read successful!")
            print(result)
        else:
            print("Could not find inserted row.")

        # ----------------------
        # Clean up test row
        # ----------------------
        cursor.execute(
            "DELETE FROM user_location WHERE userid = %s",
            (test_chat_id,)
        )

        connection.commit()

        print("\nTest row deleted.")

    except Error as e:
        print("\nMySQL error:")
        print(e)

        if connection and connection.is_connected():
            connection.rollback()

    finally:
        if cursor:
            cursor.close()

        if connection and connection.is_connected():
            connection.close()
            print("Database connection closed.")


if __name__ == "__main__":
    main()