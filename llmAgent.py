from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.utilities import SQLDatabase
import os
import ast
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from typing_extensions import TypedDict
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain import hub
from langchain import PromptTemplate
from langgraph.graph import END, StateGraph, START
from typing import Annotated, Literal
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display
import mysql.connector
import pandas as pd
import pymysql
load_dotenv()
def connect_db(database):
    db = SQLDatabase.from_uri(f"mysql+pymysql://root:88888888@localhost/{database}")
    return db

def connect_llm():
    load_dotenv()
    return ChatOpenAI(model=os.environ.get('MODEL_NAME'), base_url=os.environ.get("ARK_API_URL"), api_key=os.environ.get("ARK_API_KEY"))

class State(TypedDict):
    database: str
    question: str
    query: str
    result: str
    answer: str
    res: str
    imgpth: str
    plot_request: str

def get_databases_list():
    connection = mysql.connector.connect(
        host="localhost",       # MySQL 主机地址
        user="root",            # 用户名
        password="88888888"     # 密码
    )
    cursor = connection.cursor()
    cursor.execute("SHOW DATABASES")
    databases = cursor.fetchall()
    res = []
    for db in databases:
        res.append(db[0])
    return res

query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")
# query_prompt_template.messages[0].pretty_print()
global db
global llm
db = connect_db("netflows")
llm = connect_llm()

def switch_database(state: State):
    prompt = """
    Given an input question, switch the database to the one that is most relevant to the question.
    Pay attention to use only the database names that you can see in the database description. Be careful to only output the database name and only output one.
    Only use the following databases:
    {database_list}
    Question: {input}
    """
    prompt = prompt.format(database_list=get_databases_list(), input=state["question"])
    res = llm.invoke(prompt).content
    global db
    db = connect_db(res)
    print('switch to database:', res)
    return {"database": res}

def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    # structured_llm = llm.with_structured_output(QueryOutput)
    # result = structured_llm.invoke(prompt)
    result = llm.invoke(prompt).content
    print("============================SQL Query============================")
    print(result)
    print("=================================================================")
    return {"query": result}

def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDataBaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}

def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    print(prompt)
    response = llm.invoke(prompt)
    return {"answer": response.content}

def plot_result(state: State):
    """Plot the SQL result."""
    plot_prompt = """
    You are a data scientist. Given the following SQL result and User requriement, please generate the Python function and using matplotlib to meet the user requirement and return it to me.
    SQL Result: {result}
    User Requirement: {question}
    Pay attention to only return the executable python code, and always save the img as 'plot.png', the function name should be plot(data).
    examples are given below:
    def plot(data):
        # plot the data
        plt.savefig("plot.png")
        return "plot.png"
    imgpth = plot(data)
    """
    prompt = plot_prompt.format(result=state["result"], question=state["plot_request"])
    resp = llm.invoke(prompt).content
    resp = resp.strip().split('\n')[1:-1]
    resp = '\n'.join(resp).strip()
    print(resp)
    data = ast.literal_eval(state["result"])
    imgpth = "plot.png"
    try:
        exec(resp)
        exec('imgpth = plot(data)')
        # plot(data)
        return {"imgpth": imgpth}
    except:
        return {"imgpth": "error"}

def SQLAgent():
    def should_continue(state: State) -> Literal[END, "write_query", "execute_query"]:
        res = state.get("res", "no")
        if res == "yes":
            return "execute_query"
        elif res == "no":
            return "write_query"
        else:
            return END
    workflow = StateGraph(State)
    workflow.add_node("switch_database", switch_database)
    workflow.add_node("write_query", write_query)
    workflow.add_node("execute_query", execute_query)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("plot_result", plot_result)
    workflow.add_edge(START, "switch_database")
    workflow.add_edge("switch_database", "write_query")
    workflow.add_conditional_edges("write_query", should_continue)
    workflow.add_edge("execute_query", "generate_answer")
    workflow.add_edge("generate_answer", "plot_result")
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory, interrupt_after=["write_query"], interrupt_before=["plot_result"])
    config = {"configurable": {"thread_id": "1"}}
    display(Image(app.get_graph().draw_mermaid_png()))
    question = input(">")
    for step in app.stream(
        {"question": question},
        config,
        stream_mode="updates",
    ):
        print(step)
    user_approval = "yes"
    while user_approval != "exit":
        try:
            user_approval = input("Do you want to go to execute query? (yes/no/exit): ")
        except Exception:
            user_approval = "no"

        if user_approval.lower() == "yes":
            app.update_state(config, {"res": "yes"})
        elif user_approval.lower() == "no":
            app.update_state(config, {"res": "no"})
        else:
            app.update_state(config, {"res": "exit"})
            print("Agent Exit by User.")
            exit()
        for step in app.stream(None, config, stream_mode="updates"):
            print(step)
        if user_approval.lower() == "yes":
            break
    try:
        plot_request = input("What do you want to plot? (e.g. histogram, scatter plot): ")
    except Exception:
        plot_request = "no"
    if plot_request == 'no':
        exit()
    else:
        app.update_state(config, {"plot_request": plot_request})
    for step in app.stream(None, config, stream_mode="updates"):
        print(step)

SQLAgent()