import os
import sys

import constants

from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import JSONLoader
# from langchain_openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = constants.apikey

query = sys.argv[1]

loader = JSONLoader(file_path='intents.json',jq_schema='.content')
index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query(query,llm=ChatOpenAI()))