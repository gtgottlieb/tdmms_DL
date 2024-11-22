"""Module to send notifications using Discord."""

import os
import sys
import requests

ROOT_DIR = os.path.abspath(os.path.join(__file__, '../../../'))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))

from api_config import DISCORD_WEBHOOK_URL

def notify(message: str):
    """
    Function that posts a request to Discord. This request contains
    a message that will be shown on the configured phone.
    """
    data = {"content": message}

    try:
        requests.post(DISCORD_WEBHOOK_URL, json=data)
    except Exception as e:
        print("Tried sending notification with message '{}'".format(message))
        print("But this error occured:\n{}".format(e))

    return None

if __name__ == '__main__':
    notify('Test notification')