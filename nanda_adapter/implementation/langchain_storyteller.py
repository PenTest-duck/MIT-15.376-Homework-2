#!/usr/bin/env python3
import os
from nanda_adapter import NANDA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
load_dotenv()

def create_storyteller_improvement():
    """Create a LangChain-powered storyteller improvement function"""

    # Initialize the LLM
    llm = ChatAnthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-3-haiku-20240307"
    )

    # Create a prompt template for story transformation
    prompt = PromptTemplate(
        input_variables=["message"],
        template="""Transform the following message into a short, engaging story (exactly 1 paragraph). 
        The story should capture the essence and meaning of the original message while presenting it as a narrative.
        Make it creative, engaging, and well-written. Use vivid descriptions and storytelling elements.
        Keep it to one paragraph only.
        
        Original message: {message}
        
        Short story:"""
    )

    # Create the chain
    chain = prompt | llm | StrOutputParser()

    def storyteller_improvement(message_text: str) -> str:
        """Transform message into a short story"""
        try:
            result = chain.invoke({"message": message_text})
            return result.strip()
        except Exception as e:
            print(f"Error in storyteller improvement: {e}")
            return f"Once upon a time, there was a message that said: {message_text}. And that was the beginning of an interesting tale."  # Fallback story transformation

    return storyteller_improvement

def main():
    """Main function to start the storyteller agent"""

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set your ANTHROPIC_API_KEY environment variable")
        return

    # Create storyteller improvement function
    storyteller_logic = create_storyteller_improvement()

    # Initialize NANDA with storyteller logic
    nanda = NANDA(storyteller_logic)

    # Start the server
    print("Starting Storyteller Agent with LangChain...")
    print("All messages will be transformed into short stories!")

    domain = os.getenv("DOMAIN_NAME", "localhost")

    if domain != "localhost":
        # Production with SSL
        nanda.start_server_api(os.getenv("ANTHROPIC_API_KEY"), domain)
    else:
        # Development server
        nanda.start_server()

if __name__ == "__main__":
    main()