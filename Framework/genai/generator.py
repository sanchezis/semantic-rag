import nest_asyncio
nest_asyncio.apply()

import os
import datetime
import openai
import init
import pickle

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from openai import OpenAI

from pydantic import BaseModel, Field
from typing import Dict, Optional, Any
from typing import List, Dict, Tuple

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv();
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES" # if you are running in macos Apple chip
os.environ["TOKENIZERS_PARALLELISM"] = "false"


INDEX_NAME = 'semantic-search-rag'
ENGINE = 'text-embedding-3-small'
NAMESPACE = 'default'

FINAL_ANSWER_TOKEN = "Assistant Response:"
STOP = '[END]'
PROMPT_TEMPLATE = """

Today is {today} and you can retrieve information from a database. Respond to the user's input as best as you can.

Here is an example of the conversation format:

[START]
User Input: the input question you must answer
Context: retrieved context from the database
Context Document: context document used
Context Score : a score from 0 - 1 of how strong the information is a match
Assistant Thought: This context has sufficient information to answer the question.
Assistant Response: your final answer to the original input question which could be I don't have sufficient information to answer the question.
[END]
[START]
User Input: another input question you must answer
Context: more retrieved context from the database
Context Document: context document used
Context Score : another score from 0 - 1 of how strong the information is a match
Assistant Thought: This context does not have sufficient information to answer the question.
Assistant Response: your final answer to the second input question which could be I don't have sufficient information to answer the question.
[END]
[START]
User Input: another input question you must answer
Context: more retrieved context from the database
Context Document: context document used
Context Score : another score from 0 - 1 of how strong the information is a match
Assistant Thought: A previous piece of context has the answer to this question
Assistant Response: your final answer to the second input question which could be I don't have sufficient information to answer the question.
[END]
[START]
User Input: another input question you must answer
Context: NO CONTEXT FOUND
Context Document: NONE
Context Score : 0
Assistant Thought: We either could not find something or we don't need to look something up
Assistant Response: I'm sorry I don't know.
[END]

Begin:

{running_convolution}
"""


class RagBot(BaseModel):
    llm: Any
    prompt_template: str = PROMPT_TEMPLATE
    stop_pattern: List[str] = [STOP]
    user_inputs: List[str] = []
    ai_responses: List[str] = []
    contexts: List[Tuple[str, float]] = []
    verbose: bool = False
    threshold: float = 0.6
    top_k:int = 4

    def query_from_pinecone(self, query, top_k=4, include_metadata=True):
        from Framework.utils.pinecone_helper import query_from_pinecone
        pinecone_key = os.environ.get('PINECONE_API_KEY')
        pc = Pinecone(
            api_key=pinecone_key
        )
        index = pc.Index(name=INDEX_NAME)
        return query_from_pinecone(query, index, top_k, include_metadata)

    # @property
    # def running_convolution(self):
    #     convolution = ''
    #     for index in range(len(self.user_inputs)):
    #         convolution += f'[START]\nUser Input: {self.user_inputs[index]}\n'
    #         convolution += f'Context: {self.contexts[index][0]}\nContext URL: {self.contexts[index][1]}\nContext Score: {self.contexts[index][2]}\n'
    #         if len(self.ai_responses) > index:
    #             convolution += self.ai_responses[index]
    #             convolution += '\n[END]\n'
    #     return convolution.strip()

    @property
    def running_convolution(self):
        convolution = ''
        for index in range(len(self.user_inputs)):
            user_input = self.user_inputs[index]
            context_text, doc_id, score = self.contexts[index]

            if context_text == 'NO CONTEXT FOUND':
                context_block = "[NO CONTEXT FOUND]"
            else:
                context_block = f"Document: {doc_id}\nScore: {score:.2f}\nText: \"{context_text}\""

            ai_response = self.ai_responses[index] if len(self.ai_responses) > index else ''
            convolution += f"[START]\nUser Input: {user_input}\n{context_block}\n{ai_response}\n[END]\n"

        return convolution.strip()


    def run(self, question: str):
        self.user_inputs.append(question)
        top_responses = self.query_from_pinecone(question, self.top_k)

        for top_response in top_responses: 
            if self.verbose:
                print(top_response['score'])
            if top_response['score'] >= self.threshold:
                self.contexts.append(
                        (
                        top_response['metadata']['text'], 
                        top_response['metadata']['document'], 
                        top_response['score']
                        )
                )
            else:
                self.contexts.append(('NO CONTEXT FOUND', 'NONE', 0))

        prompt = self.prompt_template.format(  
                today = datetime.date.today(),
                running_convolution=self.running_convolution
        )
        if self.verbose:
            print('--------')
            print('PROMPT')
            print('--------')
            print(prompt)
            print('--------')
            print('END PROMPT')
            print('--------')
        generated = self.llm.generate(prompt, stop=self.stop_pattern)
        if self.verbose:
            print('--------')
            print('GENERATED')
            print('--------')
            print(generated)
            print('--------')
            print('END GENERATED')
            print('--------')
        self.ai_responses.append(generated)
        if FINAL_ANSWER_TOKEN in generated:
            generated = generated.split(FINAL_ANSWER_TOKEN)[-1]
        return generated
    
    
    
# Define a class for the Chat Language Model
class OpenAIChatLLM(BaseModel):
    model: str = 'gpt-4o'  # Default model to use
    temperature: float = 0.0  # Default temperature for generating responses

    # Method to generate a response from the model based on the provided prompt
    def generate(self, prompt: str, stop: List[str] = None):
        # Create a completion request to the OpenAI API with the given parameters
        # Initialize the OpenAI client with the API key from user data
        client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            stop=stop
        )

        # Return the generated response content
        return response.choices[0].message.content