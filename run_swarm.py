from dotenv import load_dotenv
from openai import OpenAI
from swarm import Swarm
from agents import router_agent
import json

ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",        
    api_key="ollama"            
)

def pretty_print_messages(messages) -> None:
    for message in messages:
        if message["role"] != "assistant":
            continue

        # print agent name in blue
        print(f"\033[94m{message['sender']}\033[0m:", end=" ")

        # print response, if any
        if message["content"]:
            print(message["content"])

        # print tool calls in purple, if any
        tool_calls = message.get("tool_calls") or []
        if len(tool_calls) > 1:
            print()
        for tool_call in tool_calls:
            f = tool_call["function"]
            name, args = f["name"], f["arguments"]
            arg_str = json.dumps(json.loads(args)).replace(":", "=")
            print(f"\033[95m{name}\033[0m({arg_str[1:-1]})")

def run_demo_loop(
    starting_agent, context_variables=None, stream=False, debug=True
) -> None:
    client = Swarm(client=ollama_client)
    print("Starting Ollama Swarm CLI")

    messages = []
    agent = starting_agent

    while True:
        user_input = input("\033[90mUser\033[0m: ")
        messages.append({"role": "user", "content": user_input})

        response = client.run(
            agent=agent,
            messages=messages,
            context_variables=context_variables or {},
            stream=stream,
            debug=debug,
        )

        pretty_print_messages(response.messages)

        messages.extend(response.messages)
        agent = response.agent

if __name__ == "__main__":
    run_demo_loop(router_agent)