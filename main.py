from asyncio import run
from Model import LangModel
from Agent import LessonState, Agent
llm = LangModel()
state = LessonState()
agent = Agent(llm)
state["user_message"] = "Hi ! I want to refresh my knowledge in some words"
res = run(agent.get_working_mode(state))
print(res)
