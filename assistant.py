import os
import asyncio
import json
import logging
from typing import Dict
import websockets
import requests
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PORT = int(os.getenv('PORT', 5050))
MAKE_WEBHOOK_URL = os.getenv('MAKE_WEBHOOK_URL', '<your Make.com URL>')

# Check if API key is present
if not OPENAI_API_KEY:
    logger.error('Missing OpenAI API key. Please set it in the .env file.')
    exit(1)

# Initialize FastAPI app
app = FastAPI()

# System message template
SYSTEM_MESSAGE = """
### Role
You are an AI assistant named Sophie, working at Bart's Automotive. Your role is to answer customer questions about automotive services and repairs, and assist with booking tow services.
### Persona
- You have been a receptionist at Bart's Automotive for over 5 years.
- You are knowledgeable about both the company and cars in general.
- Your tone is friendly, professional, and efficient.
- You keep conversations focused and concise, bringing them back on topic if necessary.
- You ask only one question at a time and respond promptly to avoid wasting the customer's time.
### Conversation Guidelines
- Always be polite and maintain a medium-paced speaking style.
- When the conversation veers off-topic, gently bring it back with a polite reminder.
### First Message
The first message you receive from the customer is their name and a summary of their last call, repeat this exact message to the customer as the greeting.
### Handling FAQs
Use the function `question_and_answer` to respond to common customer queries.
### Booking a Tow
When a customer needs a tow:
1. Ask for their current address.
2. Once you have the address, use the `book_tow` function to arrange the tow service.
"""

# Constants
VOICE = 'alloy'  # The voice for AI responses
# PORT is already defined from environment variables

# Session management: Store session data for ongoing calls
sessions: Dict[str, Dict] = {}

# Event types to log to the console for debugging purposes
LOG_EVENT_TYPES = [
    'response.content.done',
    'rate_limits.updated',
    'response.done',
    'input_audio_buffer.committed',
    'input_audio_buffer.speech_stopped',
    'input_audio_buffer.speech_started',
    'session.created',
    'response.text.done',
    'conversation.item.input_audio_transcription.completed'
]

@app.get("/")
async def root():
    """
    Root route to check if the server is running.
    """
    return {"message": "Twilio Media Stream Server is running!"}

@app.post("/incoming-call")
async def handle_incoming_call(request: Request):
    """
    Handle incoming calls from Twilio.
    """
    try:
        twilio_params = await request.json()
    except Exception as e:
        logger.error(f"Error parsing request body: {e}")
        return JSONResponse(content={"error": "Invalid request"}, status_code=400)
    
    caller_number = twilio_params.get("From", "Unknown")
    session_id = twilio_params.get("CallSid")
    logger.info(f"Incoming call from {caller_number}, session ID: {session_id}")

    # Default first message
    first_message = "Hello, welcome to Bart's Automotive. How can I assist you today?"

    # Fetch personalized message from Make.com webhook
    try:
        response = requests.post(
            MAKE_WEBHOOK_URL,
            headers={'Content-Type': 'application/json'},
            json={"route": "1", "data1": caller_number, "data2": "empty"}
        )
        if response.ok:
            response_data = response.json()
            first_message = response_data.get('firstMessage', first_message)
            logger.info(f"Parsed firstMessage from Make.com: {first_message}")
        else:
            logger.error(f"Failed to send data to Make.com webhook: {response.status_code} {response.reason}")
    except Exception as e:
        logger.error(f"Error sending data to Make.com webhook: {e}")

    # Set up session data
    session = {
        "transcript": "",
        "streamSid": None,
        "callerNumber": caller_number,
        "callDetails": twilio_params,
        "firstMessage": first_message
    }
    sessions[session_id] = session

    # Return TwiML response to Twilio
    twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://{request.client.host}:{PORT}/media-stream">
            <Parameter name="firstMessage" value="{first_message}" />
            <Parameter name="callerNumber" value="{caller_number}" />
        </Stream>
    </Connect>
</Response>"""

    return JSONResponse(content=twiml_response, media_type="text/xml")

@app.websocket("/media-stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket route to handle real-time media streams.
    """
    await websocket.accept()
    logger.info("Client connected to media-stream")

    # Extract headers for session identification
    try:
        headers = websocket.headers
        session_id = headers.get('x-twilio-call-sid', f"session_{int(asyncio.get_event_loop().time())}")
        session = sessions.get(session_id, {"transcript": ""})
        sessions[session_id] = session

        caller_number = session.get("callerNumber", "Unknown")
        first_message = session.get("firstMessage", "Hello, how can I assist you?")

        logger.info(f"Handling WebSocket for session: {session_id}, Caller: {caller_number}")
    except Exception as e:
        logger.error(f"Error extracting session information: {e}")
        await websocket.close()
        return

    # Open a WebSocket connection to OpenAI Realtime API
    openai_ws_url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
    headers_openai = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1"
    }

    try:
        async with websockets.connect(openai_ws_url, extra_headers=headers_openai) as openai_ws:
            logger.info("Connected to the OpenAI Realtime API")

            # Send session update to OpenAI
            session_update = {
                "type": "session.update",
                "session": {
                    "turn_detection": {"type": "server_vad"},
                    "input_audio_format": "g711_ulaw",
                    "output_audio_format": "g711_ulaw",
                    "voice": VOICE,
                    "instructions": SYSTEM_MESSAGE,
                    "modalities": ["text", "audio"],
                    "temperature": 0.8,
                    "input_audio_transcription": {
                        "model": "whisper-1"
                    },
                    "tools": [
                        {
                            "type": "function",
                            "name": "question_and_answer",
                            "description": "Get answers to customer questions about automotive services and repairs",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "question": {"type": "string"}
                                },
                                "required": ["question"]
                            }
                        },
                        {
                            "type": "function",
                            "name": "book_tow",
                            "description": "Book a tow service for a customer",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "address": {"type": "string"}
                                },
                                "required": ["address"]
                            }
                        }
                    ],
                    "tool_choice": "auto"
                }
            }
            await openai_ws.send(json.dumps(session_update))
            logger.info("Sent session update to OpenAI")

            # Prepare and send the first message
            first_message_payload = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": first_message}]
                }
            }
            await openai_ws.send(json.dumps(first_message_payload))
            logger.info("Sent first message to OpenAI")

            # Trigger AI to generate a response
            await openai_ws.send(json.dumps({"type": "response.create"}))
            logger.info("Triggered AI to generate a response")

            async def receive_from_twilio():
                """
                Receive messages from Twilio and forward to OpenAI.
                """
                while True:
                    try:
                        message = await websocket.receive_text()
                        data = json.loads(message)

                        if data.get("event") == "start":
                            stream_sid = data["start"].get("streamSid")
                            call_sid = data["start"].get("callSid")
                            custom_parameters = data["start"].get("customParameters", {})
                            first_message = custom_parameters.get("firstMessage", "Hello, how can I assist you?")
                            caller_number = custom_parameters.get("callerNumber", "Unknown")

                            # Update session data
                            session["callerNumber"] = caller_number
                            session["firstMessage"] = first_message
                            sessions[session_id] = session

                            logger.info(f"CallSid: {call_sid}, StreamSid: {stream_sid}")
                            logger.info(f"First Message: {first_message}, Caller Number: {caller_number}")

                            # Prepare the first message
                            queued_first_message = {
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "message",
                                    "role": "user",
                                    "content": [{"type": "input_text", "text": first_message}]
                                }
                            }

                            await openai_ws.send(json.dumps(queued_first_message))
                            logger.info("Sent queued first message to OpenAI")

                            # Trigger AI to generate a response
                            await openai_ws.send(json.dumps({"type": "response.create"}))
                            logger.info("Triggered AI to generate a response after first message")

                        elif data.get("event") == "media":
                            audio_payload = data["media"].get("payload")
                            if openai_ws.open:
                                audio_append = {
                                    "type": "input_audio_buffer.append",
                                    "audio": audio_payload
                                }
                                await openai_ws.send(json.dumps(audio_append))
                                logger.info("Appended audio buffer to OpenAI")
                    except Exception as e:
                        logger.error(f"Error receiving from Twilio: {e}")
                        break

            async def receive_from_openai():
                """
                Receive messages from OpenAI and forward to Twilio.
                """
                while True:
                    try:
                        response = await openai_ws.recv()
                        response_data = json.loads(response)

                        # Handle audio response
                        if response_data.get("type") == "response.audio.delta" and "delta" in response_data:
                            audio_response = {
                                "event": "media",
                                "streamSid": session.get("streamSid"),
                                "media": {"payload": response_data["delta"]}
                            }
                            await websocket.send_text(json.dumps(audio_response))
                            logger.info("Sent audio delta to Twilio")

                        # Handle function calls
                        if response_data.get("type") == "response.function_call_arguments.done":
                            function_name = response_data.get("name")
                            arguments = response_data.get("arguments")
                            try:
                                args = json.loads(arguments)
                            except json.JSONDecodeError:
                                logger.error("Invalid JSON in function call arguments")
                                continue

                            if function_name == "question_and_answer":
                                question = args.get("question")
                                if question:
                                    try:
                                        webhook_response = await send_to_webhook({
                                            "route": "3",
                                            "data1": question,
                                            "data2": ""
                                        })
                                        answer_message = webhook_response.get("message", "I'm sorry, I couldn't find an answer to that question.")
                                        thread_id = webhook_response.get("thread", "")

                                        # Update session with thread ID if provided
                                        if thread_id:
                                            session["threadId"] = thread_id

                                        function_output_event = {
                                            "type": "conversation.item.create",
                                            "item": {
                                                "type": "function_call_output",
                                                "role": "system",
                                                "output": answer_message
                                            }
                                        }
                                        await openai_ws.send(json.dumps(function_output_event))
                                        logger.info("Sent function call output to OpenAI")

                                        # Trigger AI to generate a response based on the answer
                                        instructions = f"Respond to the user's question \"{question}\" based on this information: {answer_message}. Be concise and friendly."
                                        response_create = {
                                            "type": "response.create",
                                            "response": {
                                                "modalities": ["text", "audio"],
                                                "instructions": instructions
                                            }
                                        }
                                        await openai_ws.send(json.dumps(response_create))
                                        logger.info("Triggered AI to generate a response based on Q&A")
                                    except Exception as e:
                                        logger.error(f"Error processing question_and_answer: {e}")
                                        send_error_response(openai_ws)

                            elif function_name == "book_tow":
                                address = args.get("address")
                                if address:
                                    try:
                                        webhook_response = await send_to_webhook({
                                            "route": "4",
                                            "data1": session.get("callerNumber"),
                                            "data2": address
                                        })
                                        booking_message = webhook_response.get("message", "I'm sorry, I couldn't book the tow service at this time.")

                                        function_output_event = {
                                            "type": "conversation.item.create",
                                            "item": {
                                                "type": "function_call_output",
                                                "role": "system",
                                                "output": booking_message
                                            }
                                        }
                                        await openai_ws.send(json.dumps(function_output_event))
                                        logger.info("Sent book_tow output to OpenAI")

                                        # Trigger AI to generate a response based on the booking
                                        instructions = f"Inform the user about the tow booking status: {booking_message}. Be concise and friendly."
                                        response_create = {
                                            "type": "response.create",
                                            "response": {
                                                "modalities": ["text", "audio"],
                                                "instructions": instructions
                                            }
                                        }
                                        await openai_ws.send(json.dumps(response_create))
                                        logger.info("Triggered AI to generate a response based on booking")
                                    except Exception as e:
                                        logger.error(f"Error processing book_tow: {e}")
                                        send_error_response(openai_ws)

                        # Log agent response
                        if response_data.get("type") == "response.done":
                            agent_content = response_data.get("response", {}).get("output", [])
                            transcript = next((item.get("content", {}).get("transcript") for item in agent_content if "transcript" in item.get("content", {})), "Agent message not found")
                            session["transcript"] += f"Agent: {transcript}\n"
                            logger.info(f"Agent ({session_id}): {transcript}")

                        # Log user transcription
                        if response_data.get("type") == "conversation.item.input_audio_transcription.completed":
                            user_transcript = response_data.get("transcript", "").strip()
                            session["transcript"] += f"User: {user_transcript}\n"
                            logger.info(f"User ({session_id}): {user_transcript}")

                        # Log other relevant events
                        if response_data.get("type") in LOG_EVENT_TYPES:
                            logger.info(f"Received event: {response_data.get('type')}, Data: {response_data}")

                    except Exception as e:
                        logger.error(f"Error processing OpenAI message: {e}")
                        break

            # Run both receiving functions concurrently
            await asyncio.gather(
                receive_from_twilio(),
                receive_from_openai()
            )

        async def send_to_webhook(payload: Dict) -> Dict:
            """
            Send data to the Make.com webhook and return the JSON response.
            """
            logger.info(f"Sending data to webhook: {payload}")
            try:
                response = requests.post(
                    MAKE_WEBHOOK_URL,
                    headers={'Content-Type': 'application/json'},
                    json=payload
                )
                logger.info(f"Webhook response status: {response.status_code}")
                if response.ok:
                    response_data = response.json()
                    logger.info(f"Webhook response data: {response_data}")
                    return response_data
                else:
                    logger.error(f"Failed to send data to webhook: {response.status_code} {response.reason}")
                    return {}
            except Exception as e:
                logger.error(f"Error sending data to webhook: {e}")
                return {}
            
    except Exception as e:
        logger.error(f"Can not connect to OpenAI ws socket server: {e}")
        await openai_ws.close()
        return
    
def send_error_response(openai_ws):
    """
    Helper function to send an error response to OpenAI.
    """
    error_response = {
        "type": "response.create",
        "response": {
            "modalities": ["text", "audio"],
            "instructions": "I apologize, but I'm having trouble processing your request right now. Is there anything else I can help you with?",
        }
    }
    asyncio.create_task(openai_ws.send(json.dumps(error_response)))
    logger.info("Sent error response to OpenAI")

if __name__ == "___MAIN__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
