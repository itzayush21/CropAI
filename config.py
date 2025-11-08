import os
from urllib.parse import quote_plus
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

class Config:
    # Replace the placeholders below with your actual MySQL credentials
    MYSQL_USER = os.environ.get('MYSQL_USER', 'root')
    MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', 'yourpassword')
   
    MYSQL_HOST = os.environ.get('MYSQL_HOST', 'localhost')
    MYSQL_PORT = os.environ.get('MYSQL_PORT', 3306)
    MYSQL_DB = os.environ.get('MYSQL_DB', 'cropai')

    SQLALCHEMY_DATABASE_URI = (
        f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
    )
    print(MYSQL_PASSWORD)
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'cropai-secret-key')
