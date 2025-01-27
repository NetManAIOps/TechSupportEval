from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
from .utils import langchain_create_vectorstore, get_test_question

ID = 1
NAME = 'gpt_langchain'
TITLE = 'GPT 4o Mini'
MODEL = 'gpt-4o-mini-2024-07-18'
TOP_K = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

PROMPT = PromptTemplate(
    template=(
        "You are a technical support expert specializing in resolving user issues. "
        "Use the following pieces of retrieved context to provide a solution to the user's technical support question. "
        "If the context does not provide enough information to solve the problem, say that you don't know. "
        "Ensure your answer is clear, actionable, and concise.\n\n"
        "Question: {question}\n\n"
        "Context: {context}\n\n"
        "Answer:"
    ),
    input_variables=["question", "context"]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Step 3: Prompt 模板
def get_prompt(instruction):
    template = (
        f"You are a helpful assistant following strict instructions. "
        f"Instruction: {instruction} "
        f"Context: {{context}} "
        f"Question: {{question}} "
        f"Provide a detailed and structured answer:"
    )
    return PromptTemplate(template=template, input_variables=["context", "question"])


def process_question(inputs: dict):
    question_id = inputs['question_id']
    question = inputs['question']
    reference_doc = inputs['reference_doc']
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    llm = ChatOpenAI(model=MODEL, temperature=0)
    vectorstore = langchain_create_vectorstore(NAME, question_id, reference_doc, splitter=splitter)

    qa_chain = (
        {
            "context": vectorstore.as_retriever(search_kwargs={'k': TOP_K}) | format_docs,
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return qa_chain.invoke(question)


if __name__ == "__main__":
    inputs = get_test_question()
    question_id = inputs['question_id']
    question = inputs['question']
    print(f'Question {question_id}:')
    print(question)
    print('\n\n===========\n\n')
    print('Answer:')
    print(process_question(inputs))
