import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import anthropic  # or cohere, depending on the choice
import os  
from dotenv import load_dotenv

load_dotenv()

# Configure database connection
engine = create_engine('sqlite:///rss-feed-database.db')
# Set up Anthropic or Cohere API Client
anthropic_client = anthropic.Anthropic(api_key=os.environ['anthropic_token']) # for Anthropic
# cohere_client = cohere.Client("your_cohere_api_key")        # for Cohere

# Function to query the database
def query_database(sql_query):
    with engine.connect() as connection:
        data = pd.read_sql(sql_query, connection)
    return data

model_name = "claude-3-5-sonnet-20240620"

def interpret_prompt(prompt):
    response = anthropic_client.messages.create(
        model=model_name,
        max_tokens=100,
        messages=[
            {"role": "user", "content": f"Please analyze the following prompt and determine the data to retrieve, insights needed, and type of visualization:\nPrompt: {prompt}"}
        ]
    )
    return response.content[0].text

def generate_insight_from_data(data):
    data_summary = data.describe().to_string()
    response = anthropic_client.messages.create(
        model=model_name,
        max_tokens=150,
        messages=[
            {"role": "user", "content": f"Analyze the following data summary and provide insights:\n{data_summary}"}
        ]
    )
    return response.content[0].text

# Visualization function
def create_visualization(data):
    plt.figure(figsize=(10, 6))
    data.plot(kind='bar', x='category', y='value')  # Customize based on data
    plt.title("Generated Visualization")
    plt.xlabel("Category")
    plt.ylabel("Value")
    plt.savefig("chart.png")
    return "chart.png"

# Streamlit App
st.title("Agent-Driven AI BI Dashboard")

# User prompt input
user_prompt = st.text_input("Enter your analysis request (e.g., 'Show me last quarter sales trends')")
print(f"User Prompt: {user_prompt}")
if user_prompt:
    # Step 1: Interpret the command
    interpreted_action = interpret_prompt(user_prompt)
    print(f"Interpreted Action: {interpreted_action}")
    # Step 2: Query Data
    data = query_database(interpreted_action)
    print(f"Data: {data}")
    # Step 3: Generate Insights
    insight = generate_insight_from_data(data)

    # Step 4: Create Visualization
    chart_path = create_visualization(data)

    # Display Results
    st.subheader("AI-Generated Insight")
    st.write(insight)
    st.subheader("Data Visualization")
    st.image(chart_path)
