from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate
from db import Postgres, Qdrant
GLOBAL_PROMPT = """
Ты русскоговряший ассистент технической поддержки.
Твоя задача, помогать пользователям в решении вопросов, связанных с приготовлением хлеба
Будь вежлив, игнорируй агрессию.
"""
INTRO_PROMPT = """Ты русскоговряший ассистент технической поддержки. Твоя основная задача: оценивать,
                    насколько вопросы пользователей связаны с приготовлением хлеба.
                    Ты должен оценивать степень связности по шкале от 0 до 100, где 0 - вопрос пользователя
                    совершенно не связан с темой приготовления хлеба, а 100 - вопрос пользователя максимально связан
                    с выпеканием хлеба.
                    Формируя ответ, пошагово аргументируй свой ответ, оценивай,
                    насколько слова в предложении могут характеризовать процесс приготовления хлеба. Это очень важно !
                    В ответе пользователю ты должен записать только число!!!! 
                    Примеры оценок:
                    Пример номер 1:
                    вопрос пользователя: "Кто такие имбирные человечки ? "
                    твоя оценка: 0
                    Пример номер 2:
                    вопрос пользователя: "Сколько дрожжей необходимо на кило ржаного ?"
                    твоя оценка: 100
                    \n{format_instructions}\n{question}
                """
BORDER_VAL = 80
response_schemas = [
    ResponseSchema(name="similarity", description="степень связности вопроса с темой приготовления хлеба", type="int"),
    ResponseSchema(name="description", description="пояснения полученной оценки связности вопроса с темой приготовления хлеба"),
]


class CurrentState(TypedDict):
    genuine_question: str
    answer_context: list
    related_rows_id: list[set]
    supposed_edited_genuine_question: str
    context_similarity: int
    model_answer: str


class StatesHandler:
    rel_db: Postgres
    vect_db: Qdrant

    def __init__(self, model):
        self.model = model

    def add_rel_database(self, db: Postgres):
        self.rel_db = db
        pass

    def add_vect_database(self, db: Qdrant):
        self.vect_db = db
        pass

    async def detect_thematic_correspondence(self, cs: CurrentState):
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        presetting_prompt = PromptTemplate(
            template=INTRO_PROMPT,
            input_variables=["question"],
            partial_variables={"format_instructions": format_instructions}
        )
        actions_chain = presetting_prompt | self.model | output_parser
        cs['context_similarity'] = (await actions_chain.ainvoke({"question": cs["genuine_question"]}))['similarity']
        return cs

    async def get_answer(self, cs: CurrentState):
        prompt_for_valid_data = ChatPromptTemplate(
            [("system", GLOBAL_PROMPT), ("human", "{question}")]
        )
        if cs['context_similarity'] > BORDER_VAL:
            answer_chain = prompt_for_valid_data | self.model | StrOutputParser()
            cs['model_answer'] = await answer_chain.ainvoke({"question": cs["genuine_question"]})
        else:
            cs['model_answer'] = "Ваш вопрос не относится к хлебам, динах"
        return cs

    async def find_similar_context(self, cs: CurrentState):
        if self.vect_db:
            cs['answer_context'] = await self.vect_db.retriever.ainvoke(cs['genuine_question'])
        else:
            cs['answer_context'] = [None]
            print('Firstly, you need to initialise vector store instance')
        return cs







