import logging
from datetime import datetime, timedelta

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

TOKEN = "xoxb-6170747087975-7119039926821-bqpeaRxvVGqZXHB4UxDTz5GD"
WEBHOOK = "https://hooks.slack.com/services/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"
CHANNEL = "mytender-io-customer-support"
CHANNEL_ID = "C06U3P7P05S"

log = logging.getLogger(__name__)


# pylint: disable=inconsistent-return-statements,


def get_messages(user):
    resp = []
    client = WebClient(token=TOKEN)
    try:
        # Fetch the conversation history
        response = client.conversations_history(channel=CHANNEL_ID)

        # Check if the response is OK
        if response["ok"]:
            messages = response["messages"]
            now = datetime.now()
            for message in messages:
                timestamp = float(message.get("ts"))
                message_time = datetime.fromtimestamp(timestamp)
                if now - timedelta(hours=72) <= message_time <= now:
                    if "@" + user in message.get("text") or user + ":" in message.get("text"):
                        filtered_text = message.get("text")
                        # Remove username from the message
                        if user + ":" in filtered_text:
                            filtered_text = filtered_text.replace(user + ":", "")
                            filtered_text = "!" + filtered_text
                        if "@" + user in filtered_text:
                            filtered_text = filtered_text.replace("@" + user, "")

                        # Create a message dictionary with id and text
                        msg_dict = {
                            "id": message.get("ts"),  # Use the timestamp as a unique id
                            "text": filtered_text.strip(),  # Strip any leading/trailing whitespace
                        }
                        resp.append(msg_dict)
            return resp
        else:
            log.warning(f"Failed to fetch history: {response['error']}")

    except SlackApiError as e:
        log.warning(f"Error fetching conversation history: {e.response['error']}")


def send_message(user, message):
    client = WebClient(token=TOKEN)
    try:
        response = client.chat_postMessage(channel=CHANNEL, text=user + ": " + message)
        log.info(f"Message sent: {response['ts']}")
    except SlackApiError as e:
        log.warning(f"Error posting message: {e}")
