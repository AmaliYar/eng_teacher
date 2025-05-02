from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from typing import Literal

LessonsTypes = Literal['new_words', 'training', 'test']


class LessonState(TypedDict):
    current_word: str
    working_mode: LessonsTypes
    words: list
    learning_progress: float
    words_amount: int
    successful_learning: bool


class Agent:
    def __init__(self, model):
        self.model = model

    def get_working_mode(self, state:LessonState) -> LessonState:
        pass

    def add_words_to_vocab(self, state: LessonState) -> LessonState:
        pass

    def get_words_from_vocab(self, state: LessonState) -> LessonState:
        pass

    def start_training(self, state: LessonState) -> LessonState:
        pass

    def start_learning_new_words(self, state: LessonState) -> LessonState:
        pass

    def start_test(self, state: LessonState) -> LessonState:
        pass

    def decide_to_finish_lesson(self, state: LessonState) -> LessonState:
        pass

    def return_working_mode(self, state: LessonState) -> str:
        pass

    def check_lesson_completeness(self, state: LessonState) -> str:
        pass


class Graph:
    def __init__(self, states=LessonState):
        self.gph = StateGraph(states)

    def build_default_graph(self, agent: Agent):
        self.gph.add_node("welcome_page", agent.get_working_mode)
        self.gph.add_node("training", agent.start_training)
        self.gph.add_node("learning", agent.start_learning_new_words)
        self.gph.add_node("testing", agent.start_test)
        self.gph.add_node("check_progress", agent.decide_to_finish_lesson)
        self.gph.add_conditional_edges("welcome_page", agent.return_working_mode,
                                       {'new_words': "learning",
                                        'training': "training",
                                        'test': "testing"})
        self.gph.add_edge("training", "check_progress")
        self.gph.add_edge("learning", "check_progress")
        self.gph.add_edge("testing", "check_progress")
        self.gph.add_conditional_edges("check_progress", agent.check_lesson_completeness,
                                       {'new_words': "learning",
                                        'training': "training",
                                        'test': "testing",
                                        'finish': END})


