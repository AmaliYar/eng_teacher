from abc import ABC
from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
import getpass
import time


class LangModel:
    model: ChatMistralAI
    model_instance: str
    embedder: MistralAIEmbeddings
    id: int

    def __init__(self):
        self.api_key = getpass.getpass("Api key:")
        self.model = ChatMistralAI(model="ministral-3b-2410", api_key=self.api_key, temperature=0.5, timeout=250,
                                   max_concurrent_requests=2)
        self.model_instance = 'mistral-embed'
        # self.embedder = Mistral(api_key=self.api_key)
        self.embedder = MistralAIEmbeddings(api_key=self.api_key)
        self.connection = False
        self.get_connection_with_model()

    def embedding_data(self, data: list):
        return self.embedder.embeddings.create(model=self.model_instance, inputs=data)

    async def get_connection_with_model(self):
        while not self.connection:
            try:
                print('trying to set connection with mistral')
                model_answer = await self.model.ainvoke('Hi mistral')
                self.connection = True
                print(f'connection with mistral has established. Model\'s answer: {model_answer.content}')
            except Exception as e:
                print(f'unstable connection... trying again. Cause: {e}')
                time.sleep(10)




