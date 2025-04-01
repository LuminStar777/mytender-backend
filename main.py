import configparser
import logging

import uvicorn
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from api import app
from config import admin_collection, load_admin_prompts
from services.helper import init_logger

# Create a ConfigParser object
config = configparser.ConfigParser()

# Read the configuration file
config.read('config.ini')

# Get the port number
port = int(config['DEFAULT']['Port'])
ssl_certfile = config['DEFAULT']['ssl_certfile']
ssl_keyfile = config['DEFAULT']['ssl_keyfile']

load_dotenv()

init_logger(screenlevel=logging.INFO, filename="default")

log = logging.getLogger(__name__)


def fastapi_interface():
    origins = ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            ssl_certfile=ssl_certfile,
            ssl_keyfile=ssl_keyfile,
            log_config=None,
            access_log=False,
            workers=1,  # Use single worker to avoid event loop issues
        )
    except FileNotFoundError:
        log.warning("SSL certificates not found")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_config=None,
            access_log=False,
            workers=1,  # Use single worker to avoid event loop issues
        )


async def update_prompt_in_collections():
    adminuser = await load_admin_prompts()
    admin_collection.update_many(
        {},
        {
            "$set": {
                "question_extractor": adminuser["question_extractor"],
            }
        },
    )


if __name__ == "__main__":
    # asyncio.run(update_prompt_in_collections())
    fastapi_interface()
