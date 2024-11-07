from swarm import Agent
import os
import sqlite3

model = os.getenv('LLM_MODEL', 'qwen2.5-coder:7b')
conn = sqlite3.connect('rss-feed-database.db')
cursor = conn.cursor()

with open("ai-news-complete-tables.sql", "r") as table_schema_file:
    table_schemas = table_schema_file.read()
    
def get_sql_agent_instructions():
    return f"""You are a SQL expert who takes in a request from a user for information
    they want to retrieve from the DB, creates a SELECT statement to retrieve the
    necessary information, and then invoke the function to run the query and
    get the results back to then report to the user the information they wanted to know.
    
    Here are the table schemas for the DB you can query:
    
    {table_schemas}

    Write all of your SQL SELECT statements to work 100% with these schemas and nothing else.
    You are always willing to create and execute the SQL statements to answer the user's question.
    """

def run_sql_select_statement(sql):
    """Executes a SQL SELECT statement and returns the results of running the SELECT. Make sure you have a full SQL SELECT query created before calling this function."""
    print(f"Executing SQL statement: {sql}")
    cursor.execute(sql)
    records = cursor.fetchall()

    if not records:
        return "No results found."
    
    # Get column names
    column_names = [description[0] for description in cursor.description]
    
    # Calculate column widths
    col_widths = [len(name) for name in column_names]
    for row in records:
        for i, value in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(value)))
    
    # Format the results
    result_str = ""
    
    # Add header
    header = " | ".join(name.ljust(width) for name, width in zip(column_names, col_widths))
    result_str += header + "\n"
    result_str += "-" * len(header) + "\n"
    
    # Add rows
    for row in records:
        row_str = " | ".join(str(value).ljust(width) for value, width in zip(row, col_widths))
        result_str += row_str + "\n"
    
    return result_str    

#todo query url from vector DB and order it.
def submit_order(description):
    """Submit a order for the user."""
    return {"response": f"order created for {description}"}

def send_email(email_address,message):
    """Send an email to the user."""
    response = f"Email sent to: {email_address} with message: {message}"
    return {"response": response}


router_agent = Agent(
    name="Router Agent",
    instructions="Determine which agent is best suited to handle the user's request, and transfer the conversation to that agent. we have two agents which are DB Agent and Order Agent",
    model="qwen2.5:3b"
)
db_agent = Agent(
    name="DB Agent",
    instructions=get_sql_agent_instructions() + "\n\nHelp the user execute sql statement to get query result.",
    functions=[run_sql_select_statement],
    model=model
)
order_agent = Agent(
    name="Order Agent",
    instructions="You are an order agent who deals with questions about company's products, such as iphone, ipad, adjustable desk, software service etc.",
    functions=[submit_order, send_email],
    model = model
)

def transfer_back_to_router():
    """Call this function if a user is asking about a topic that is not handled by the current agent."""
    return router_agent


def transfer_to_db_agent():
    return db_agent


def transfer_to_order_agent():
    return order_agent


router_agent.functions = [transfer_to_db_agent, transfer_to_order_agent]
db_agent.functions.append(transfer_back_to_router)
order_agent.functions.append(transfer_back_to_router)
