
import os
import asyncio
import websockets
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PORT = os.getenv('PORT', 5050)
MAKE_WEBHOOK_URL = "<your Make.com URL>"

# Check if API key is present
if not OPENAI_API_KEY:
    raise EnvironmentError('Missing OpenAI API key. Please set it in the .env file.')

# Initialize FastAPI app
app = FastAPI()

# System message template
SYSTEM_MESSAGE = """
### Role
You are an AI assistant named Sophie, working at Bart's Automotive...
"""

# Sessions for managing ongoing calls
sessions = {}

# Root route - For checking server status
@app.get("/")
async def root():
    return {"message": "Twilio Media Stream Server is running!"}

# Handle incoming calls from Twilio
@app.post("/incoming-call")
async def handle_incoming_call(request: Request):
    twilio_params = await request.json()
    caller_number = twilio_params.get("From", "Unknown")
    session_id = twilio_params.get("CallSid")

    print(f"Incoming call from {caller_number}, session ID: {session_id}")

    # Default first message
    first_message = "Hello, welcome to Bart's Automotive. How can I assist you today?"

    # Fetch personalized message from Make.com
    try:
        response = requests.post(
            MAKE_WEBHOOK_URL,
            headers={'Content-Type': 'application/json'},
            json={"route": "1", "data1": caller_number, "data2": "empty"}
        )
        if response.ok:
            response_data = response.json()
            first_message = response_data.get('firstMessage', first_message)
        else:
            print(f"Webhook error: {response.status_code}")
    except Exception as e:
        print(f"Error fetching from webhook: {e}")

    # Set up session data
    session = {
        "transcript": "",
        "streamSid": None,
        "callerNumber": caller_number,
        "firstMessage": first_message
    }
    sessions[session_id] = session

    # Return TwiML for Twilio to connect the call to the media stream
    twiml_response = f"""
    <?xml version=\"1.0\" encoding=\"UTF-8\"?>
    <Response>
        <Connect>
            <Stream url="wss://{request.client.host}/media-stream">
                <Parameter name="firstMessage" value="{first_message}" />
                <Parameter name="callerNumber" value="{caller_number}" />
            </Stream>
        </Connect>
    </Response>
    """
    return JSONResponse(content=twiml_response, media_type="text/xml")

# WebSocket route for handling real-time media streams
@app.websocket("/media-stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to media-stream")

    # Simulate OpenAI WebSocket connection
    async with websockets.connect(
        "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01",
        extra_headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}
    ) as openai_ws:
        session_id = f"session_{websocket.client.host}_{websocket.client.port}"
        session = sessions.get(session_id, {"transcript": ""})
        first_message = session.get("firstMessage", "Hello, how can I assist you?")
        
        # Send session configuration to OpenAI WebSocket
        session_update = {
            "type": "session.update",
            "session": {
                "turn_detection": {"type": "server_vad"},
                "input_audio_format": "g711_ulaw",
                "output_audio_format": "g711_ulaw",
                "voice": "alloy",
                "instructions": SYSTEM_MESSAGE,
                "modalities": ["text", "audio"],
                "temperature": 0.8
            }
        }
        await openai_ws.send(session_update)

        # Send first message
        await openai_ws.send({
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": first_message}]}
        })

        # Handle incoming media stream from Twilio
        while True:
            try:
                data = await websocket.receive_text()
                # Simulate sending the data to OpenAI WebSocket
                await openai_ws.send({
                    "type": "input_audio_buffer.append",
                    "audio": data
                })
                # Simulate receiving a response from OpenAI WebSocket
                response = await openai_ws.recv()
                await websocket.send_text(response)
            except Exception as e:
                print(f"Error handling WebSocket: {e}")
                break

    await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(PORT))
