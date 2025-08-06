from asyncio import run
from Model import LangModel
from Agent import LessonState, Agent, Graph
from Dbase import Postgres
llm = LangModel()
base = Postgres
state = LessonState(is_word_exist=False)
agent = Agent(llm, base())

state["user_message"] = 'Hi!, I want to learn a new word! My word: satisfaction'
graph = Graph()
graph.build_default_graph(agent)
pipeline = graph.gph.compile()
pipeline.invoke(state)
