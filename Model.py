from abc import ABC
from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
import getpass


class Agent(ABC):
    pass


class LangModel:
    model: ChatMistralAI
    model_instance: str
    embedder: MistralAIEmbeddings
    id: int

    def __init__(self):
        self.api_key = getpass.getpass("Api key:")
        self.model = ChatMistralAI(model="mistral-large-2407", api_key=self.api_key)
        self.model_instance = 'mistral-embed'
        # self.embedder = Mistral(api_key=self.api_key)
        self.embedder = MistralAIEmbeddings(api_key=self.api_key)

    def embedding_data(self, data: list):
        return self.embedder.embeddings.create(model=self.model_instance, inputs=data)
