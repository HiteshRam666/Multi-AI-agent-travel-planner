import gradio as gr
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
import os

class PlannerState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "The messages in the conversation"]
    city: str
    interests: List[str]
    itinerary: str
    additional_details: str  # To store additional details from the user

# Define the LLM
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]
llm = ChatOpenAI()

# Define the itinerary prompt (now including additional details)
itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant. Create a detailed trip itinerary for {city} based on the user's interests: {interests}. Consider the following details: {additional_details}. Provide a brief description and a bulleted itinerary."),
    ("human", "Create an itinerary for my trip.")
])

def input_city(city: str, state: PlannerState) -> PlannerState:
    return {
        **state,
        "city": city,
        "messages": state['messages'] + [HumanMessage(content=city)],
    }

def input_interests(interests: str, state: PlannerState) -> PlannerState:
    return {
        **state,
        "interests": [interest.strip() for interest in interests.split(',')],
        "messages": state['messages'] + [HumanMessage(content=interests)],
    }

def input_additional_details(additional_details: str, state: PlannerState) -> PlannerState:
    return {
        **state,
        "additional_details": additional_details,
        "messages": state['messages'] + [HumanMessage(content=additional_details)],
    }

def create_itinerary(state: PlannerState) -> str:
    response = llm.invoke(itinerary_prompt.format_messages(city=state['city'], interests=", ".join(state['interests']), additional_details=state['additional_details']))
    state["itinerary"] = response.content
    state["messages"] += [AIMessage(content=response.content)]
    return response.content

# Define the Gradio application
def travel_planner(city: str, interests: str, additional_details: str):
    # Initialize state with default values
    state = {
        "messages": [],
        "city": city,
        "interests": [interest.strip() for interest in interests.split(',')],
        "itinerary": "",
        "additional_details": additional_details,
    }

    # Process the city, interests, and additional details inputs
    state = input_city(city, state)
    state = input_interests(interests, state)
    state = input_additional_details(additional_details, state)

    # Generate the itinerary
    itinerary = create_itinerary(state)

    return itinerary

# Build the Gradio interface
interface = gr.Interface(
    fn=travel_planner,
    theme='Yntec/HaleyCH_Theme_Orange_Green',
    inputs=[
        gr.Textbox(label="Enter the city for your day trip"),
        gr.Textbox(label="Enter your interests (comma-separated)"),
        gr.Textbox(label="Enter additional details (e.g., number of days, budget, specific interests)"),
    ],
    outputs=gr.Textbox(label="Generated Itinerary"),
    title="Travel Itinerary Planner",
    description="Enter a city, your interests, and additional details to generate a personalized day trip itinerary."
)

# Launch the Gradio application
interface.launch(share=True)
