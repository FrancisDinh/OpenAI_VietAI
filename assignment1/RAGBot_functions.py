from langchain.docstore.document import Document
from utils import search_doc_from_knowledge_base, get_doc_pages
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import openai
import random


def create_qa_chain(**params):
    llm = OpenAI(**params)
    chain = load_qa_chain(llm, chain_type='stuff')
    return chain


def answer_idk():
    answers = [
        "I'm sorry, but I don't have that information.",
        "Unfortunately, I don't have the answer you're looking for.",
        "I'm not familiar with that topic, so I can't provide any insight.",
        "I'm afraid I don't know the answer to your question."
    ]
    return random.choice(answers)


def answer_idk_ext_context():
    answers = [
        "I'm sorry, but I don't have that information in your PDF. Do you want me to use my knowledge?",
        "Unfortunately, your PDF doesn't have the answer you're looking for. Do you want to search for it externally?",
        "I'm not familiar with that topic, so I can't provide any insight. Do you want me to use my knowledge?",
        "I'm afraid I don't know the answer to your question. Do you want me to use my knowledge?"
    ]
    return random.choice(answers)


def answer_default(question):
    # Define parameters for the OpenAI ChatCompletion API
    params = dict(model="gpt-3.5-turbo",
                  temperature=0,
                  messages=[{"role": "user", "content": question}])

    # Generate a response from the AI model
    response = openai.ChatCompletion.create(**params)
    content = response.choices[0].message.content

    return content


def answer_from_doc(chain, doc, question: str):
    # Run the question-answering chain to find the answer to the question
    response = chain.run(
        input_documents=doc, question=question)

    return response


def what_is_the_question(question):
    question_template = f"""
    Here is the question user is asking {question}
    """
    params = dict(model="gpt-3.5-turbo",
                  messages=[{"role": "user", "content": question}])

    # Generate a response from the AI model
    response = openai.ChatCompletion.create(**params)
    content = response.choices[0].message.content

    return content
