"""Module to send notifications using Pushover."""

import os
import sys

import requests

ROOT_DIR = os.path.abspath(os.path.join(__file__, '../../../'))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))

try:
  from api_config import PUSHOVER_APP_KEY, PUSHOVER_USER_KEY
except:
  PUSHOVER_APP_KEY = None
  PUSHOVER_USER_KEY = None

def notify(message: str):
  assert (PUSHOVER_APP_KEY and PUSHOVER_USER_KEY), 'Pushover API keys not found.'

  try:
    requests.post("https://api.pushover.net/1/messages.json", data = {
      "token": PUSHOVER_APP_KEY,
      "user": PUSHOVER_USER_KEY,
      "message": message
    })
  except Exception as e:
    print("Tried sending notification with message '{}'".format(message))
    print("But this error occured:\n{}".format(e))

  return None