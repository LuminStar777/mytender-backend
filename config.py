import logging
import os
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from langchain_community.chat_models import ChatPerplexity
from langchain_core.callbacks import CallbackManager
from langchain_core.tracers import LangChainTracer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langsmith import Client
from markitdown import MarkItDown
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from openai import OpenAI
from services.helper import LoggingRateLimiter

log = logging.getLogger(__name__)

# Add these lines after your initial logging setup, near the top of the file
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

load_dotenv()
# set the following env variables:
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://eu.api.smith.langchain.com/"
# check if the LANGCHAIN_PROJECT environment variable is set, if not, set it to "mytender"
if "LANGCHAIN_PROJECT" not in os.environ:
    os.environ["LANGCHAIN_PROJECT"] = "mytender_local"
assert os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY is not set"


RETRIEVE_SUBTOPIC_CHUNKS = 20
RELEVANCE_THRESHOLD = 6  # Threshold for including chunks in results (0-10 scale)
# Initialize LangSmith client
project_name = "mytender"
ls_client = Client()
tracer = LangChainTracer(project_name=project_name, client=ls_client)
callback_manager = CallbackManager([tracer])

# MODEL4 = 'gpt-4'
GPT_DEFAULT_MODEL = "gpt-4o"
GPT_FAST_MODEL = "gpt-4o-mini"
CHUNK_SIZE = (200, 2000)

MAX_TITLE_REQUESTS = 10000

# Add rate limiters before model initialization
bedrock_rate_limiter = LoggingRateLimiter(
    name="Bedrock",
    requests_per_second=3.0,  # Reduced from 50 to 3 RPS (180 RPM)
    max_bucket_size=5,  # Reduced burst capacity
    check_every_n_seconds=0.01,  # Keep fast checks
)

gemini_rate_limiter = LoggingRateLimiter(
    name="Gemini",
    requests_per_second=33,  # 2000 RPM for paid tier Gemini Pro
    max_bucket_size=50,  # Allow good burst capacity
    check_every_n_seconds=0.01,  # Fast checks for high throughput
)

openai_rate_limiter = LoggingRateLimiter(
    name="OpenAI",
    requests_per_second=33,  # 2000 RPM like Gemini
    max_bucket_size=50,  # Allow good burst capacity
    check_every_n_seconds=0.01,  # Fast checks for high throughput
)

# Update OpenAI instances to use rate limiter
openai_instance = ChatOpenAI(
    model_name=GPT_DEFAULT_MODEL,
    verbose=False,
    temperature=0,
    callback_manager=callback_manager,
    rate_limiter=openai_rate_limiter,
)

openai_instance_mini = ChatOpenAI(
    model_name=GPT_FAST_MODEL,
    verbose=False,
    temperature=0,
    callback_manager=callback_manager,
    rate_limiter=openai_rate_limiter,
)

openai_fast_instance = ChatOpenAI(
    model_name=GPT_FAST_MODEL,
    verbose=False,
    temperature=0,
    callback_manager=callback_manager,
    rate_limiter=openai_rate_limiter,
)

claude_instance = ChatAnthropic(
    model="claude-3-5-sonnet-20240620", anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY")
)

gemini_15pro_instance = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    verbose=False,
    temperature=0,
    callback_manager=callback_manager,
    rate_limiter=gemini_rate_limiter,
)

openai_embedding = OpenAIEmbeddings()
perplexity = ChatPerplexity(
    model='sonar-pro',
    temperature=0.0,
)

groq_mixtral = chat = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
groq_llama = chat = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")

bedrock_llama = ChatBedrock(
    model_id="us.meta.llama3-2-3b-instruct-v1:0",
    region_name="us-east-1",
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    rate_limiter=bedrock_rate_limiter,
)

bedrock_claude35 = ChatBedrock(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    region_name="us-east-1",
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    rate_limiter=bedrock_rate_limiter,
)

parsing_llm_client = OpenAI()
markdown_client = MarkItDown(llm_client=parsing_llm_client, llm_model="gpt-4o")

try:
    ollama_instance = OllamaLLM(base_url="http://localhost:11434", model="llama2")
    ollama_embedding = OllamaEmbeddings()
except:
    pass

CHROMA_FOLDER = os.environ.get("CHROMA_FOLDER", 'chroma_db')
DOC_ENTRIES_COLLECTION = os.environ.get("DOC_ENTRIES_COLLECTION", 'doc_entries_local')
USER_ADMIN_COLLECTION = os.environ.get("USER_ADMIN_COLLECTION", 'user_admin_local')
BIDS_COLLECTION = os.environ.get("BIDS_COLLECTION", 'bids_local')
FILE_COLLECTION = os.environ.get("FILE_COLLECTION", 'files_local')

# LLM assignments with explicit naming
llm_outline_fallback = openai_instance  # Fallback model for outline generation
llm_compliance = openai_instance  # Model for compliance processing
llm_section_compliance = openai_instance  # Model for section-level compliance extraction
llm_chain_default = openai_instance  # Default model for general chain processing
llm_outline = gemini_15pro_instance  # Model for generating outlines
llm_tender_insights = openai_instance  # Model for tender insights
llm_tender_library_chat = openai_instance  # Model for tender library chat
llm_fallback = bedrock_claude35  # Fallback model when primary fails
llm_post_process = openai_instance  # Model for post-processing
llm_post_process_fallback = bedrock_claude35  # Fallback model for post-processing
llm_writing_plan = openai_fast_instance  # Model for writing plans
llm_chunk_title = openai_instance_mini  # Model for generating chunk titles
llm_bid_intel_summary = openai_instance  # Model for generating bid intel summary
llm_opportunity_information = openai_instance  # Model for generating opportunity information
llm_cover_letter = openai_instance  # Model for generating cover letter
llm_exec_summary = openai_instance  # Model for generating executive summary
llm_diagram = openai_instance  # Model for generating diagrams
llm_keywords_for_subtopic = openai_instance_mini  # Model for generating keywords for subtopic
llm_evidence_rewriting = openai_instance  # Model for generating keywords for subtopic


embedder = OpenAIEmbeddings()

STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY_LIVE")
STRIPE_WEBHOOK_KEY = os.environ.get("STRIPE_WEBHOOK_SECRET_LIVE")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
master_password = os.environ.get("MASTER_PASSWORD")
# AWS SES region
AWS_SES_REGION_NAME = os.environ.get("AWS_SES_REGION_NAME")
AWS_SES_REGION_ENDPOINT = os.environ.get("AWS_SES_REGION_ENDPOINT")

SECRET_KEY = "secre11222t"
ALGORITHM = "HS256"

HOSTNAME = "44.208.84.199"
MONGO_USERNAME = os.environ.get("MONGO_USERNAME", "spark")
MONGO_PASSWORD = os.environ["MONGO_PASSWORD"]
MONGO_DB = "spark"  # The database you want to connect to
MONGO_AUTH_DB = "admin"  # Database where the user is authenticated
connection_string = (
    f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{HOSTNAME}:27017/{MONGO_DB}"
    f"?authSource={MONGO_AUTH_DB}"
)

ADMIN_USERNAME = "adminuser"

# Initialize mongo_client with the current event loop
mongo_client = AsyncIOMotorClient(connection_string)
mongo_db = mongo_client[MONGO_DB]


# Instead of initializing fs directly, create a function to get it
async def get_gridfs():
    """Get GridFS instance using the current event loop"""
    # pylint: disable=protected-access
    if not hasattr(get_gridfs, '_fs'):
        get_gridfs._fs = AsyncIOMotorGridFSBucket(mongo_db, bucket_name=FILE_COLLECTION)
    return get_gridfs._fs


file_collection = mongo_db[FILE_COLLECTION]

admin_collection = mongo_db[USER_ADMIN_COLLECTION]
doc_entry_collection = mongo_db[DOC_ENTRIES_COLLECTION]
queries_collection = mongo_db["queries"]
feedback_collection = mongo_db["feedback"]
template_collection = mongo_db["templates"]
bids_collection = mongo_db[BIDS_COLLECTION]
account_creation_tokens = mongo_db["account_creation_tokens"]


async def load_user_config(login):
    """Load user configuration"""
    user_config = await admin_collection.find_one({"login": login})
    return user_config


async def load_admin_prompts():
    """Load admin prompts"""
    prompts = await admin_collection.find_one({"login": "adminuser"})
    return prompts

'''OKTA configurations'''
OKTA_ISSUER = os.getenv('OKTA_ISSUER')
OKTA_AUDIENCE = os.getenv('OKTA_AUDIENCE')  # "api://default"
JWKS_URL = os.getenv('JWKS_URL')  # f"{OKTA_ISSUER}/v1/keys"

class OktaGroupingConfig(BaseModel):
    GROUP_MAPPING: Dict[str, List[str]] = {
        "Admin": ["admin:all", "read:all"],
        "Viewer": ["read:only"],
        "Editor": ["read:all", "write:all"]
    }
    DEFAULT_PERMISSIONS: List[str] = ["read:self"]