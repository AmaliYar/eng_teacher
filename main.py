from asyncio import run
from Model import LangModel
from Agent import LessonState, Agent, Graph
from Dbase import Postgres
from langchain.agents import Tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage


llm = LangModel()
base = Postgres
state = LessonState(is_word_exist=False)
agent = Agent(llm, base())
llm.model = llm.model.bind_tools([agent.request_info])


# state["user_message"] = 'Hi!, I want to learn a new word! My word: satisfaction'
state["user_message"] = 'Hi!, I want to check my words!'
# msg = "I need some expert guidance for building an AI agent. Could you request assistance for me?"

graph = Graph()
graph.build_default_graph(agent)
# graph.human_in_the_loop_test(agent)
pipeline = graph.gph.compile(checkpointer=InMemorySaver())


config = {"configurable": {"thread_id": "1"}}

events = pipeline.stream(
    state,
    config,
    stream_mode="debug",
)

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
