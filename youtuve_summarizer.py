from langchain.document_loaders import YoutubeLoader
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain import PromptTemplate
from langchain.chains import LLMChain
from main import llm
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.tools import YouTubeSearchTool

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

embeddings = OpenAIEmbeddings(openai_api_key="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")

def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs

def create_db_from_youtube_url(url):
    loader = YoutubeLoader.from_youtube_url(url)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    query = "Get the introduction of the video in 1000 words"
    response = get_response_from_query(db, query, 4)
    return response

def get_exact_youtube_url(video_description):
    tool = YouTubeSearchTool()
    output = tool.run(video_description)[2:-2].split("', '")[0]
    return "https://www.youtube.com" + output



tools = [
    Tool.from_function(
        func=create_db_from_youtube_url,
        name="YoutubeTranscript",
        description="useful for when you need to get youtube transcript from video url"
    ),
    Tool.from_function(
        func=get_exact_youtube_url,
        name="YoutubeExactUrlFinder",
        description="useful for when you need to find exact youtube url from provided input"
    ),
]

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent.run("What is Joe Rogan asking to RDJ about Ironman in his podcast?")
