from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END, START
from typing_extensions import TypedDict, Annotated
from typing import Literal, Optional, Final
from langchain_core.tools import tool
from Dbase import Postgres
from langgraph.prebuilt import create_react_agent, tools_condition, ToolNode
from langchain_mistralai.chat_models import ChatMistralAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import Field, BaseModel
from langgraph.types import interrupt

LessonsTypes = Literal['learn', 'train', 'test']
DIALOG_STEPS = 3
STEPS_TO_CHECK_RESULT = 1
LEARNING_STOP_WORD = '*Conclusion*'
LEARNING_RATE = 0.25


class LessonState(TypedDict):
    user_message: str
    current_word: str
    working_mode: str
    words: list
    learning_progress: float
    words_amount: int
    successful_learning: bool
    current_word_translations: dict
    is_word_exist: bool
    messages: list
    message: str


class Agent:

    def __init__(self, model, db: Postgres):
        self.model = model
        self.db = db

    def get_working_mode(self, state: LessonState) -> LessonState:
        class WorkingModes(BaseModel):
            """Probable working modes, based on user question. Choose one."""
            mode: Literal['new', 'training', 'test'] = Field(
                description="Choose, which working mode had supposed by user request.")
            word: Optional[str] = Field(
                default='',
                description="Word, which user wants to study")
        prompt = PromptTemplate.from_template("""You are language assistant. You have 2 main task: 
                        1) detect working mode, which has supposed by user question
                        2) if user pointed a word, which he want to study, find it and remember
                        {msg}
                        """)
        structured_instance = self.model.model.with_structured_output(WorkingModes)
        response = structured_instance.invoke(prompt.invoke({'msg': state['user_message']}))
        state['working_mode'] = response.mode
        state['current_word'] = response.word
        return state

    def add_words_to_vocab(self, state: LessonState) -> LessonState:
        QUERY_ADD_NEW_WORD = 'INSERT INTO words (word, as_noun, learning_progress, as_verb, as_adjective) ' \
                             'VALUES (%s, %s, %s, %s, %s)'
        self.db.send_query(QUERY_ADD_NEW_WORD,
                           (state['current_word'],
                            state['current_word_translations']['noun'],
                            0.0,
                            state['current_word_translations']['verb'],
                            state['current_word_translations']['adjective'],
                            ))
        return state

    def get_words_from_vocab(self, state: LessonState) -> LessonState:
        QUERY_FIND_WORDS = "SELECT * FROM words WHERE word=%s"
        query_result = self.db.send_query(QUERY_FIND_WORDS, (state['current_word'],))
        if query_result:
            state['learning_progress'] = query_result[0][2]
            state['is_word_exist'] = True
        return state

    def update_progress(self, state:LessonState) -> LessonState:
        state['learning_progress'] += LEARNING_RATE
        QUERY_UPDATE_PROGRESS = "UPDATE words SET learning_progress =%s WHERE word =%s"
        self.db.send_query(QUERY_UPDATE_PROGRESS, (state['learning_progress'], state['current_word']))
        return state

    def start_training(self, state: LessonState) -> LessonState:
        #todo: add conditional handling of input data: can be one word, or more than one

        agent = self.get_dialog_agent(state['current_word'])
        print('welcome the agent. He knows your word yet')
        while not self.check_finalizer_in_the_last_message(LEARNING_STOP_WORD,
                                                           agent,
                                                           {"configurable": {"thread_id": "1"}},
                                                           state):
            print(agent.invoke({'messages': {'role': 'user', 'content': input('Your answer:')}},
                               {"configurable": {"thread_id": "1"}}).get('messages')[-1].content)

        return state

    def get_dialog_agent(self, current_word: str):
        return create_react_agent(model=ChatMistralAI(
            api_key=self.model.api_key, model='mistral-medium-latest', temperature=0.8),
                                  tools=[],
                                  # response_format=LearningState,
                                  checkpointer=MemorySaver(), prompt=
                                  f"""
                <instruction>
                You are system, which helps peoples to learn new words. 
                Before sending messages for user, think about your interaction plan with user, 
                plan your action careful - it is very important ! 
                Next you can find scenario of interaction with user:
                step 0 - a student sends you a word, which he wants to learn
                step 1 - you show 3 phrases-examples with user's word and 3 sentences with user's word
                step 2 - You generate question for user. Question must to be related with user's word
                step 3 - user sends you an answer for question
                step 4 - User generates a question for you. Question must to be related with user's word.
                step 5 - You answer on user question
                step 6 - Based on above steps results, you do a decision: has student learned the word or not.
                Follow it step by step. 
                </instruction/>
                Next you can find restrictions, which provide you, how to avoid mistakes in your work:
                <restrictions>
                Be careful in steps 2 and 4, which has pointed in instruction: wait user answer before crossing to 
                the next step! Do not show him task from more than one step! It is very important! 
                Do not force user to send answer from two steps in the one message!
                </restrictions/>

                Next you can find output format features:
                
                <output format>
                After successful completion 6th step, you must to generate a conclusion, 
                which contains only one of two words:
                "success" - if student gave you a right translation
                "fail" - if student did a mistake
                You must also use specific keyword Conclusion
                </output format>
                Here you can find your final message examples:
                <output format examples>
                *Conclusion*: success
                *Conclusion*: fail
                </output format examples>
                User's word today: {current_word}
                """)

    def start_learning_new_words(self, state: LessonState) -> LessonState:
        #todo: add validation word in database
        state = self.get_words_from_vocab(state)
        if not state['is_word_exist']:
            print(f'It is a new word, forming metadata in database')
            structured_input = self.model.model.with_structured_output(method="json_mode")
            model_response = structured_input.invoke([
                        SystemMessage(
                            content="""
                            <instruction>
                            You are system for preparing data to loading into database. Your working algorithm:
                            1) detect in user message word, which he wants to learn. Remember this word like <current_word>
                            2) detect noun form of <current_word> in English, save it like <noun>. 
                            Do not save same by sense words ! If from does not exist, remember NULL for <noun>. 
                            3) detect verb form of <current_word> in English, save it like <verb>. 
                            Do not save same by sense words ! If from does not exist, remember NULL for <verb>
                            4) detect adjective form of <current_word> in English, save it like <adjective>. 
                            Do not save same by sense words ! If from does not exist, remember NULL for <adjective>
                            </instruction>
                            next you can find specific for output format:
                            <output format>
                            Return user next JSON format:
                            {
                            "current_word": <current_word>,
                            "current_word_translations": {
                                                                "noun": <noun>,
                                                                "verb": <verb>,
                                                                "adjective": <adjective>
                            
                                                        }
                            }
                            Your answer mustn't contain another information rather JSON. It is very important !
                            </output format>
                            """

                        ),
                HumanMessage(content=state["current_word"])
                    ])
            state.update(model_response)
            self.add_words_to_vocab(state)
        self.start_training(state)
        return state

    def start_test(self, state: LessonState):
        class UserWords(BaseModel):
            """words amount, which user wants to learn"""
            words: Optional[int] = Field(default=None, description="words amount, which user wants to learn")

        response = self.model.model.with_structured_output(UserWords).invoke(
            [SystemMessage(
                content="""
                You are language learning assistant.
                Your main task - calculate, how many words user wants today to train.
                Sometimes, user can forget to specify words amount. Then, you must request this information.
                Request words until user provides you this information, because it is very important!
                """),
             HumanMessage(content=state['user_message'])])
        state['words_amount'] = response.words
        return state

    def decide_to_finish_lesson(self, state: LessonState) -> LessonState:
        pass

    def return_working_mode(self, state: LessonState) -> str:
        return state['working_mode']

    def check_lesson_completeness(self, state: LessonState) -> str:
        if state['successful_learning']:
            state = self.get_words_from_vocab(state)
            self.update_progress(state)
            try:
                self.db.connection.commit()
            except Exception as e:
                self.db.connection.close()
                print("Transaction rolled back due to error:", e)
            finally:
                self.db.connection.close()
            return 'finish'
        return state['working_mode']

    def check_info_fullfillment(self, state: LessonState):
        if state['words_amount']:
            return 'next'
        return 'get_info'

    def check_finalizer_in_the_last_message(self, content, messages_holder, params: dict, state):
        if messages_holder.checkpointer.get(params):
            if content in messages_holder.checkpointer.get(params)['channel_values']['messages'][-1].content:
                state['successful_learning'] = True
                return True
        return False

    def chatbot(self, state):
        message = self.model.model.invoke(state["messages"][-1].content)
        assert len(message.tool_calls) <= 1
        return {"messages": [message]}

    def request_info(self, state: LessonState):
        """Request information from user about words amount which user wants to learn"""
        # response = interrupt('Give me amount of words!!!')
        state['words_amount'] = int(input('How many words...'))
        return state


class Graph:
    def __init__(self, states=LessonState):
        self.gph = StateGraph(states)

    def build_default_graph(self, agent: Agent):
        # tools = []
        # tools = [request_info]
        # tool_node = ToolNode(tools=tools)
        self.gph.add_node("welcome_page", agent.get_working_mode)
        # self.gph.add_node("tools", tool_node)
        self.gph.add_node("training", agent.start_training)
        self.gph.add_node("learning", agent.start_learning_new_words)
        self.gph.add_node("testing", agent.start_test)
        self.gph.add_node("ask_info", agent.request_info)
        self.gph.add_node("check_progress", agent.decide_to_finish_lesson)
        self.gph.add_conditional_edges("welcome_page", agent.return_working_mode,
                                       {'new': "learning",
                                        'training': "training",
                                        'test': "testing"})
        self.gph.add_conditional_edges("testing", agent.check_info_fullfillment,
                                       {'get_info': 'ask_info',
                                        'next': 'testing'})
        self.gph.add_edge("training", "check_progress")
        self.gph.add_edge("learning", "check_progress")
        self.gph.add_edge("testing", "check_progress")
        # self.gph.add_edge("tools", "testing")
        self.gph.add_conditional_edges("check_progress", agent.check_lesson_completeness,
                                       {'new': "learning",
                                        'training': "training",
                                        'test': "testing",
                                        'finish': END})
        self.gph.set_entry_point("welcome_page")



# while True:
#     model_response = get_dialog_agent().invoke(
#         {"messages": [{"role": "user", "content": input("Your message:")}]},
#         {"configurable": {"thread_id": "1"}})
#     print(model_response['messages'][-1].content)
