#!/usr/bin/env python3
import os
import json
import requests
from nanda_adapter import NANDA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
load_dotenv()

def search_web(query: str, num_results: int = 5) -> list:
    """
    Search the web using Serper API and return relevant results
    
    Args:
        query (str): Search query
        num_results (int): Number of results to return
        
    Returns:
        list: List of search results with title, snippet, and link
    """
    serper_api_key = os.getenv("SERPER_API_KEY")
    if not serper_api_key:
        print("Warning: SERPER_API_KEY not found. Web search will be disabled.")
        return []
    
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": serper_api_key,
        "Content-Type": "application/json"
    }
    
    payload = {
        "q": query,
        "num": num_results
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        # Extract organic results
        for item in data.get("organic", []):
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link": item.get("link", "")
            })
        
        return results
    except Exception as e:
        print(f"Error in web search: {e}")
        return []

def create_megabrain_agent():
    """Create a LangChain-powered megabrain agent with web search capabilities"""

    # Initialize the LLM
    llm = ChatAnthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-3-sonnet-20240229"  # Using Sonnet for better reasoning
    )

    def megabrain_improvement(message_text: str) -> str:
        """
        Megabrain agent that uses web search to provide comprehensive answers
        
        Args:
            message_text (str): User's question or message
            
        Returns:
            str: Comprehensive answer with web search results
        """
        try:
            # First, determine if this is a question that would benefit from web search
            question_check_prompt = PromptTemplate(
                input_variables=["message"],
                template="""Analyze if the following message is a question that would benefit from current, factual information from the web. 
                Respond with only 'YES' or 'NO'.
                
                Examples that need web search:
                - Questions about current events, news, or recent developments
                - Questions about specific facts, statistics, or data
                - Questions about technology, companies, or products
                - Questions about recent research or studies
                - Questions about current prices, availability, or status
                
                Examples that don't need web search:
                - General philosophical questions
                - Personal advice or opinions
                - Creative writing prompts
                - Simple greetings or casual conversation
                
                Message: {message}
                
                Response:"""
            )
            
            question_check_chain = question_check_prompt | llm | StrOutputParser()
            needs_search = question_check_chain.invoke({"message": message_text}).strip().upper()
            
            search_results = []
            if needs_search == "YES":
                print(f"🔍 Searching the web for: {message_text}")
                search_results = search_web(message_text, num_results=5)
                print(f"📊 Found {len(search_results)} search results")
            
            # Create the main response prompt
            if search_results:
                # Format search results for the prompt
                search_context = "\n\n".join([
                    f"Title: {result['title']}\nSnippet: {result['snippet']}\nLink: {result['link']}"
                    for result in search_results
                ])
                
                response_prompt = PromptTemplate(
                    input_variables=["message", "search_results"],
                    template="""You are a Megabrain AI assistant with access to current web information. 
                    Answer the user's question comprehensively using the provided search results and your knowledge.
                    
                    Guidelines:
                    - Provide a thorough, accurate, and helpful answer
                    - Use the search results to support your response with current information
                    - Cite specific sources when referencing search results
                    - If search results don't fully answer the question, use your knowledge to fill gaps
                    - Be clear about what information comes from web search vs. your training data
                    - Structure your response logically and clearly
                    
                    User's Question: {message}
                    
                    Web Search Results:
                    {search_results}
                    
                    Comprehensive Answer:"""
                )
                
                response_chain = response_prompt | llm | StrOutputParser()
                result = response_chain.invoke({
                    "message": message_text,
                    "search_results": search_context
                })
            else:
                # No web search needed, provide a direct response
                response_prompt = PromptTemplate(
                    input_variables=["message"],
                    template="""You are a Megabrain AI assistant. Answer the user's question or respond to their message 
                    in a helpful, intelligent, and comprehensive manner.
                    
                    Guidelines:
                    - Provide thoughtful, accurate, and helpful responses
                    - Be conversational but informative
                    - If it's a question, give a complete answer
                    - If it's a statement, respond appropriately
                    - Use your knowledge to provide valuable insights
                    
                    User's Message: {message}
                    
                    Response:"""
                )
                
                response_chain = response_prompt | llm | StrOutputParser()
                result = response_chain.invoke({"message": message_text})
            
            return result.strip()
            
        except Exception as e:
            print(f"Error in megabrain improvement: {e}")
            # Fallback response
            return f"I apologize, but I encountered an error while processing your message: '{message_text}'. Please try rephrasing your question or let me know how I can help you in a different way."

    return megabrain_improvement

def main():
    """Main function to start the megabrain agent"""

    # Check for API keys
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set your ANTHROPIC_API_KEY environment variable")
        return
    
    if not os.getenv("SERPER_API_KEY"):
        print("Warning: SERPER_API_KEY not found. Web search will be disabled.")
        print("To enable web search, get a free API key from https://serper.dev/")

    # Create megabrain agent
    megabrain_logic = create_megabrain_agent()

    # Initialize NANDA with megabrain logic
    nanda = NANDA(megabrain_logic)

    # Start the server
    print("Starting Megabrain Agent with LangChain and Web Search...")
    print("I can answer questions using current web information!")

    domain = os.getenv("DOMAIN_NAME", "localhost")

    if domain != "localhost":
        # Production with SSL
        nanda.start_server_api(os.getenv("ANTHROPIC_API_KEY"), domain)
    else:
        # Development server
        nanda.start_server()

if __name__ == "__main__":
    main()