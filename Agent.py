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
from langgraph.types import interrupt, Command
import time

LessonsTypes = Literal['learn', 'train', 'test']
DIALOG_STEPS = 3
STEPS_TO_CHECK_RESULT = 1
LEARNING_STOP_WORD = 'Conclusion'
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

    async def get_working_mode(self, state: LessonState) -> LessonState:
        class WorkingModes(BaseModel):
            """Probable working modes, based on user question. Choose one."""
            mode: Literal['new', 'training', 'test'] = Field(
                description="""Choose, which working mode had supposed by user request.
                new - describes user intention to learn a new word;
                training - describes user intention to train words, which he learned earlier;
                test - describes user intention to check his knowledge - aka examination
                """)
            word: Optional[str] = Field(
                default='',
                description="Word, which user wants to study")

        prompt = PromptTemplate.from_template("""You are language assistant. You have 2 main task: 
                        1) detect working mode, which has supposed by user question
                        2) if user pointed a word, which he want to study, find it and remember
                        {msg}
                        """)
        # structured_instance = self.model.model.with_structured_output(WorkingModes)
        # response = structured_instance.invoke(prompt.invoke({'msg': state['user_message']}))
        response = await send_request_to_model(self.model.model.with_structured_output(WorkingModes),
                                               prompt.invoke({'msg': state['user_message']}))
        state['working_mode'] = response.mode
        state['current_word'] = response.word
        return state

    def add_words_to_vocab(self, state: LessonState) -> LessonState:
        QUERY_ADD_NEW_WORD = 'INSERT INTO words (word, as_noun, learning_progress, as_verb, as_adjective, noun_rus,' \
                             ' adj_rus, verb_rus) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)'
        self.db.send_query(QUERY_ADD_NEW_WORD,
                           (state['current_word'],
                            state['current_word_translations']['noun'],
                            0.0,
                            state['current_word_translations']['verb'],
                            state['current_word_translations']['adjective'],
                            state['current_word_translations']['noun_rus'],
                            state['current_word_translations']['adj_rus'],
                            state['current_word_translations']['verb_rus'],
                            ))
        return state

    def get_word_from_vocab(self, state: LessonState) -> LessonState:
        QUERY_FIND_WORD = "SELECT * FROM words WHERE word=%s"
        query_result = self.db.send_query(QUERY_FIND_WORD, (state['current_word'],))
        if query_result:
            state['learning_progress'] = query_result[0][2]
            state['is_word_exist'] = True
        return state

    def get_words_from_vocab(self, state: LessonState) -> list:
        QUERY_FIND_WORDS = "SELECT * FROM words ORDER BY learning_progress LIMIT %s"
        query_result = self.db.send_query(QUERY_FIND_WORDS, (state['words_amount'],))
        words = [parse_query(word) for word in query_result]
        return words

    def update_progress(self, state: LessonState, increasing:bool = True) -> LessonState:
        if increasing:
            state['learning_progress'] += LEARNING_RATE
        else:
            if state['learning_progress'] > 0:
                state['learning_progress'] -= LEARNING_RATE
        QUERY_UPDATE_PROGRESS = "UPDATE words SET learning_progress =%s WHERE word =%s"
        self.db.send_query(QUERY_UPDATE_PROGRESS, (state['learning_progress'], state['current_word']))
        return state

    async def start_training(self, state: LessonState):
        if not state['current_word']:
            state['current_word'] = input('You have forgot to point a word. Which word do you want to train?')
            state['working_mode'] = 'learn'
            return Command(goto='learning', update=state)

        agent = self.get_dialog_agent(state['current_word'])
        print('welcome the agent. He knows your word yet')
        while not self.check_finalizer_in_the_last_message(LEARNING_STOP_WORD,
                                                           agent,
                                                           {"configurable": {"thread_id": "1"}},
                                                           state):
            model_answer = await send_request_to_model(agent, {'messages': {'role': 'user', 'content': input('Your answer:')}},
                                        {"configurable": {"thread_id": "1"}})
            print(model_answer.get('messages')[-1].content)

        return Command(goto='check_progress', update=state)

    def get_dialog_agent(self, current_word: str):
        return create_react_agent(model=ChatMistralAI(
            api_key=self.model.api_key, model='mistral-medium-latest', temperature=0.8),
            tools=[],
            # response_format=LearningState,
            checkpointer=MemorySaver(), prompt=
            f"""
                <instruction>
                You are english language teacher, which helps peoples to learn new words. 
                Before sending messages for user, plan your interaction with user, 
                your plan must to follow a next [scenario] - it is very important ! 
                
                Next you can find [scenario] of interaction with user:
                step 0 - a student sends you a word, which he wants to learn
                step 1 - you show 3 phrases-examples with user's word and 3 sentences with user's word. 
                Also point word's russian translation.
                step 2 - You generate question for user. Question must to be related with user's word
                step 3 - user sends you an answer for question
                step 4 - User generates a question for you. Question must to be related with user's word.
                step 5 - You answer on user question
                step 6 - Based on above steps results, you do a decision: has student learned the word or not.
                Follow it step by step. 
                </instruction>
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
                You must also use specific keyword [Conclusion]
                </output format>
                
                Here you can find several final message examples:
                
                <output format examples>
                1) Conclusion: success
                2) Conclusion: fail
                </output format examples>
                
                User's word today: {current_word}
                """)

    async def start_learning_new_words(self, state: LessonState) -> LessonState:
        # todo: add validation word in database
        state = self.get_word_from_vocab(state)
        if not state['is_word_exist']:
            print(f'It is a new word, forming metadata in database')
            structured_input = self.model.model.with_structured_output(method="json_mode")
            #todo: handle more determined exit from dialog
            model_response = await send_request_to_model(
                structured_input,
                [
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
                               5) translate adjective form of <noun> to Russian, save it like <noun_rus>. 
                               Do not save same by sense words ! If from does not exist, remember NULL for <noun_rus>
                               6) translate verb form of <noun> to Russian, save it like <verb_rus>. 
                               Do not save same by sense words ! If from does not exist, remember NULL for <verb_rus>
                               7) translate adjective form of <noun> to Russian, save it like <adj_rus>. 
                               Do not save same by sense words ! If from does not exist, remember NULL for <adj_rus>
                               </instruction>
                               next you can find specific for output format:
                               <output format>
                               Return user next JSON format:
                               {
                               "current_word": <current_word>,
                               "current_word_translations": {
                                                                   "noun": <noun>,
                                                                   "verb": <verb>,
                                                                   "adjective": <adjective>,
                                                                   "noun_rus": <noun_rus>,
                                                                   "verb_rus": <verb_rus>,
                                                                   "adj_rus": <adj_rus>
                                                                   

                                                           }
                               }
                               Your answer mustn't contain another information rather JSON. It is very important !
                               </output format>
                               """
                    ),
                    HumanMessage(content=state["current_word"])
                ]
            )
            state.update(model_response)
            self.add_words_to_vocab(state)
        await self.start_training(state)
        return state

    async def start_test(self, state: LessonState):
        class UserWords(BaseModel):
            """words amount, which user wants to learn"""
            words: Optional[int] = Field(default=None, description="words amount, which user wants to learn")

        response = await send_request_to_model(self.model.model.with_structured_output(UserWords),
                                         [SystemMessage(content="""You are language learning assistant.
                Your main task - detect, how many words user wants today to train.
                Sometimes, user can forget to specify words amount. Then, you must to point it, like "0 words"
                """), HumanMessage(content=state['user_message'])])
        if not response.words:
            response.words = int(input('Give me amount of words!!!'))
            print(f'requesting words amount from user')
            # todo: fix interrupt bug - doesn't work
            # response.words = interrupt()
        state['words_amount'] = response.words
        state['words'] = self.get_words_from_vocab(state)
        print(f'detected {response.words} words')
        for word in state['words']:
            word['learned'] = check_user_answer(word, input(f'get your translation for the word {word.get("word")}:'))
        state['successful_learning'] = False not in [word['learned'] for word in state['words']]
        return Command(goto="check_progress", update=state)

    def decide_to_finish_lesson(self, state: LessonState) -> LessonState:
        pass

    def return_working_mode(self, state: LessonState) -> str:
        return state['working_mode']

    # пробелма - есть вариант когда исследуется одно слово - это один набор атрибутов, а есть пакетный режим обработки
    # нескольких слов - здесь другой набор .... Надо какое-то решение плюс-минус адекватное продумать ска
    def check_lesson_completeness(self, state: LessonState) -> LessonState:
        increasing_flag = False
        if state['successful_learning']:
            if 'words' not in state.keys() or not state['words']:
                state = self.get_word_from_vocab(state)
                self.update_progress(state)
                self.db.commit_transaction()
                return state
            increasing_flag = True
        for word in state['words']:
            state['current_word'] = word['word']
            state['learning_progress'] = word['learning_progress']
            self.update_progress(state, increasing=increasing_flag)
        self.db.commit_transaction()
        return state

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

    def check_user_intention(self, state: LessonState):
        working_mode_key = 'working_mode'
        user_answer = input(f'Today we had a {state.get(working_mode_key)}. Would you like to study something else ?')
        if 'no' in user_answer.lower():
            self.db.end_session()
            return Command(goto=END)
        state['user_message'] = input('Inform me, what do you want to do next ?')
        # это какой-то конченный костыль
        state['words'] = []
        state['is_word_exist'] = False
        return Command(goto='welcome_page', update=state)


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
        self.gph.add_node("check_progress", agent.check_lesson_completeness)
        self.gph.add_node("check_intention", agent.check_user_intention)
        self.gph.add_conditional_edges("welcome_page", agent.return_working_mode,
                                       {'new': "learning",
                                        'training': "training",
                                        'test': "testing"})
        # self.gph.add_edge("training", "check_progress")
        self.gph.add_edge("learning", "check_progress")
        # self.gph.add_edge("testing", "check_progress")
        self.gph.add_edge("check_progress", "check_intention")
        self.gph.set_entry_point("welcome_page")



def check_user_answer(word_metadata: dict, user_answer: str):
    return user_answer.lower() in word_metadata.values()


def parse_query(resulted_row: list) -> dict:
    return {
        'word': resulted_row[0],
        'as_noun': resulted_row[1],
        'learning_progress': resulted_row[2],
        'as_verb': resulted_row[3],
        'as_adjective': resulted_row[4],
        'learned': False
    }


async def send_request_to_model(receiver, request, config=None):
    response = None
    while not response:
        try:
            print('trying to send request for model')
            response = await receiver.ainvoke(request, config)
            print(f'200')
            return response
        except Exception as e:
            print(f'unstable connection... trying again. Cause: {e}')
            time.sleep(10)
