import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
import json
import os
import re
from schema_definitions import QuestionScenario
from pydantic import ValidationError

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("API Key not found. Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

chat_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, google_api_key=API_KEY)

st.title("AI-Powered Question & Scenario Generator")
st.write("Enter the organization metadata and scenario context to generate relevant questions and responses.")

organization_size = st.text_input("Organization Size (e.g., Large, Medium, Small)")
organization_country = st.text_input("Organization Country (e.g., USA, India, Germany)")
organization_industry = st.text_input("Industry (e.g., AI, Healthcare, Finance)")
context = st.text_area("Context (Scenario Description)")

def format_list(text):
    if isinstance(text, str):  
        lines = text.strip().split("\n")
        formatted_list = [line.lstrip("-").strip() for line in lines if line.strip()]
        return formatted_list
    return text

def clean_scenario(text):
    if isinstance(text, str) and "\n-" in text:
        return text.replace("\n-", "; ")
    return text

if st.button("Generate Questions & Scenarios"):
    if not (organization_size and organization_country and organization_industry and context):
        st.warning("Please provide all required inputs.")
    else:
        organization_metadata = f"{organization_size}, {organization_country}, {organization_industry}"

        prompt_template = PromptTemplate(
            input_variables=["organization_metadata", "context"],
            template="""
            Generate a structured set of insights based on the following inputs:

            - Organization Metadata: {organization_metadata}
            - Context: {context}

            ### Expected JSON Output Format
            ```json
            [
                {{
                    "question": "Key question based on input",
                    "scenario": "Detailed scenario description",
                    "insights": ["Insight 1", "Insight 2"],
                    "thought_process": "Logical reasoning behind the insights",
                    "actions": ["Action 1", "Action 2"]
                }}
            ]
            ```
            Ensure that the response is structured and includes actionable recommendations in proper JSON format.
            """
        )

        formatted_prompt = prompt_template.format(
            organization_metadata=organization_metadata, 
            context=context
        )

        st.info("Generating insights... Please wait...")
        
        response = chat_model.invoke([HumanMessage(content=formatted_prompt)])
        
        try:
            response_text = response.content.strip()
            response_text = re.sub(r"```(json)?", "", response_text, flags=re.MULTILINE).strip()
            parsed_json = json.loads(response_text)

            for item in parsed_json:
                item["insights"] = format_list(item["insights"])
                item["actions"] = format_list(item["actions"])
                item["scenario"] = clean_scenario(item["scenario"])  

            validated_response = [QuestionScenario(**item) for item in parsed_json]

            st.success("Generated Insights:")
            for idx, item in enumerate(validated_response):
                with st.expander(f"Question {idx+1}: {item.question}"):
                    st.write(f"**Scenario:** {item.scenario}")
                    st.write(f"**Insights:**")
                    st.write(f"- " + "\n- ".join(item.insights))
                    st.write(f"**Thought Process:** {item.thought_process}")
                    st.write(f"**Actions:**")
                    st.write(f"- " + "\n- ".join(item.actions))

        except json.JSONDecodeError:
            st.error("LLM output is not valid JSON. Here’s the raw response:")
            st.text(response.content)
        
        except ValidationError as e:
            st.error(f"Validation failed: {str(e)}")
            st.text(response_text)
