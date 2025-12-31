from pydantic import BaseModel,Field
from typing import List
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

class Source(BaseModel):
    """Schema for the source used by the agent"""
    url :str = Field(description="Provides the url of the source from which news is derived")

class AgentResponse(BaseModel):
    """Schema of the response from the agent"""
    ans : str  = Field(description="Provides latest news requested by the user")
    sources : List[Source] = Field(default_factory=list,description="Provides news sources")


llm = ChatOpenAI(
    model="gpt-3.5-turbo",  # or "gpt-4", etc.
    temperature=0,
    max_retries=2 # controls creativity vs. determinism
)

tools = [TavilySearch()]

search_agent = create_agent(llm, tools,response_format=AgentResponse)


def main():
    #response = search_agent.invoke(llm_input["messages"])
    response = search_agent.invoke({"messages": [{"role": "user", "content": "Give me world news in 5 concise points"}]})
    return response.get("structured_response", None)

if __name__ == "__main__":
    print(main())
