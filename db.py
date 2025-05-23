import certifi
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import logging

# Configure logger
logger = logging.getLogger(__name__)

# MongoDB connection string
MONGO_CONNECTION_STRING = "mongodb+srv://ShreyasSatre:Test%4012345678@aihealth.8xngetl.mongodb.net/?retryWrites=true&w=majority&appName=AIHealth"  # Replace with your MongoDB connection string

# Database and collection names
DB_NAME = "resume_parser_db"
USERS_COLLECTION = "users"
RESUME_COLLECTION = "uploaders_resumes"  # Assuming this is where resume data is stored

# Initialize global client and database objects
client = None
db = None
users_collection = None
resume_collection = None

def initialize_db():
    """Initialize the database connection."""
    global client, db, users_collection, resume_collection
    
    try:
        ca = certifi.where()
        # Create MongoDB client
        client = MongoClient(MONGO_CONNECTION_STRING, tlsCAFile=ca, serverSelectionTimeoutMS=10000)
        
        # Ping the server to verify connection
        client.admin.command('ping')
        
        # Get database and collections
        db = client[DB_NAME]
        users_collection = db[USERS_COLLECTION]
        resume_collection = db[RESUME_COLLECTION]
        
        # Create indexes for better performance
        users_collection.create_index("email", unique=True)
        
        logger.info("Successfully connected to MongoDB")
        return True
    
    except ConnectionFailure as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        return False
    
    except Exception as e:
        logger.error(f"An error occurred while connecting to MongoDB: {e}")
        return False

def close_db_connection():
    """Close the database connection."""
    global client
    if client:
        client.close()
        logger.info("MongoDB connection closed")

def get_users_collection():
    """Get the users collection."""
    global users_collection
    return users_collection

def get_resume_collection():
    """Get the resume collection."""
    global resume_collection
    return resume_collection