import logging
from sanic_jwt.decorators import protected
import json
from sanic import Sanic
from sanic import Blueprint, response
from sanic.request import Request
from typing import Text, Optional, List, Dict, Any
import jwt
from rasa.core.channels.channel import UserMessage, OutputChannel
from rasa.core.channels.channel import InputChannel
from rasa.core.channels.channel import CollectingOutputChannel
import datetime as dt

logger = logging.getLogger(__name__)


class MyIO(InputChannel):
    """A custom http input channel for Alexa.
    You can find more information on custom connectors in the 
    Rasa docs: https://rasa.com/docs/rasa/user-guide/connectors/custom-connectors/
    """

    @classmethod
    def name(cls):
        return "alexa_assistant"

    # Sanic blueprint for handling input. The on_new_message
    # function pass the received message to Rasa Core
    # after you have parsed it
    def blueprint(self, on_new_message):

        botbase = Blueprint("botbase", __name__)

        # required route: use to check if connector is live
        @botbase.route("/", methods=["GET"])
        async def health(request):
            return response.json({"status": "ok"})

        # @botbase.get('/webhook/login/')
        # @protected(botbase)
        @botbase.route("/webhook/login/", methods=["POST"])
        # @protected(botbase)
        async def login(request):
            user_id = request.json.get("id")
            payload = {"user": {"username": user_id, "role": "admin"}}
            signed = jwt.encode(payload, 'secretkey', algorithm='HS256')
            return response.json({"token": signed})

        # required route: defines
        @botbase.route("/webhook", methods=["POST"])
        async def receive(request):
            # get the json request sent by Alexa

            # payload = request.json.get("type")
            # check to see if the user is trying to launch the skill
            try:
                token = request.headers.get("Authorization")
                print(token)
                decoded_token = jwt.decode(
                    token, "secretkey", algorithms=["HS256"])
                if decoded_token:
                    intenttype = request.json.get("type")
                    sender_id = request.json.get(
                        "sender")  # method to get sender_id

                    input_channel = self.name()  # method to fetch input channel
                    metadata = self.get_metadata(
                        request)  # method to get metadat

                    # if the user is starting the skill, let them know it worked & what to do next
                    if intenttype == "LaunchRequest":

                        message = "Hello! Welcome to this Rasa-powered chatbot. You can start by saying 'hi' to Rodie."
                        session = "false"
                        sender_id = request.json.get(
                            "sender")  # method to get sender_id

                        input_channel = self.name()  # method to fetch input channel
                        metadata = self.get_metadata(
                            request)  # method to get metadata

                        r = {
                            'message': [
                                {
                                    'recipient_id': 'default',
                                    'custom': {
                                        'description': message,
                                        'timestamp': '{}:{}'.format(dt.datetime.now().hour, dt.datetime.now().minute),
                                        'request_type': "text.request",
                                        'data_type': 'text',

                                    }
                                }
                            ]
                        }
                    else:
                        # get the Alexa-detected intent
                        intent = request.json.get("text")

                        # makes sure the user isn't trying to end the skill
                        if intent == "bye":

                            session = "true"
                            message = "Talk to you later"
                            sender_id = request.json.get(
                                "sender")  # method to get sender_id

                            input_channel = self.name()  # method to fetch input channel
                            metadata = self.get_metadata(
                                request)  # method to get metadata
                        else:
                            # get the user-provided text from the slot named "text"
                            text = text = request.json.get("text")
                            sender_id = request.json.get(
                                "sender")  # method to get sender_id

                            input_channel = self.name()  # method to fetch input channel
                            metadata = self.get_metadata(
                                request)  # method to get metadata

                            # initialize output channel
                            out = CollectingOutputChannel()
                            print("Metadata:", metadata)
                            print("Sender:", sender_id)
                            print("Inputchannel:", input_channel)
                            # send the user message to Rasa & wait for the
                            # response to be sent back
                            await on_new_message(UserMessage(text, out))
                            # extract the text from Rasa's response
                            # responses = [m["text"] for m in out.messages]
                            # message = responses[0]
                            # if len(responses)> 0:
                            #     message = responses[0]

                            # else :

                            session = "false"
                            message = out.messages
                            print(message)

                            if('custom' in message[0].keys()):
                                print(message)

                                r = {
                                    'message': message
                                }

                            else:
                                r = {
                                    'message': [
                                        {
                                            'recipient_id': 'default',
                                            'custom': {
                                                'description': message[0]['text'],
                                                'timestamp': '{}:{}'.format(dt.datetime.now().hour, dt.datetime.now().minute),
                                                'request_type': "text.request",
                                                'data_type': 'text',

                                            }
                                        }
                                    ]
                                }

            except (jwt.exceptions.InvalidSignatureError, jwt.exceptions.DecodeError) as error:
                print(error)
                r = {
                    "message": [
                        {
                            "recipient_id": "default",
                            "custom": {
                                "description": "You are not authorised to communicate with Rodie. Please reach us on our email: tinashemashaya21@outlook.com/",
                                "timestamp": '{}:{}'.format(dt.datetime.now().hour, dt.datetime.now().minute),
                                "request_type": "text.request",
                                "data_type": "text",

                            }
                        }
                    ]
                }

            return response.json(r)

        return botbase
