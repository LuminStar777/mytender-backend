import re

import requests

SENDER_USER = "user"
SENDER_AGENT = "agent"
MESSAGE_STREAM_START = "stream_start"
MESSAGE_STREAM_TOKEN = "stream_token"
MESSAGE_STREAM_END = "stream_end"
MESSAGE_ERROR = "error"
MESSAGE_NORMAL = "message"
MESSAGE_PING = "ping"

url = "https://raw.githubusercontent.com/hyperreality/American-British-English-Translator/master/data/american_spellings.json"
american_to_british_dict = requests.get(url, timeout=10).json()


def britishize(string):
    for american_spelling, british_spelling in american_to_british_dict.items():
        # Using a regex pattern to match whole words only
        pattern = r'\b' + re.escape(american_spelling) + r'\b'
        string = re.sub(pattern, british_spelling, string)
    return string
