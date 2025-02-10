import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
import os
from output_schema import output_schema  
import json

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("API Key not found.")

print("Expected JSON Schema:")
print(json.dumps(output_schema, indent=2))

chat_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

prompt_template = PromptTemplate(
    input_variables=["organization_metadata", "context"],
    template="""
    Generate a set of questions and scenarios based on the following inputs:

    - **Organization Metadata:** {organization_metadata}
    - **Context:** {context}

    ### **Expected JSON Output Format**
    ```json
    [
        {{
            "question": "Generated question based on input",
            "scenario": "Detailed scenario",
            "impact": "Possible impact of this scenario on the organization",
            "response": "Suggested response to address the situation"
        }},
        {{
            "question": "Another generated question",
            "scenario": "Another detailed scenario",
            "impact": "Impact of this scenario",
            "response": "Suggested response"
        }}
    ]
    ```
    Ensure that the response is **detailed** and considers both the **employee perspective** and **organization impact**.
    """
)

organization_metadata = """
Company Size: 500,
Country: USA,
Industry: IT,
Role: Senior Developer,
Policy: No clear work-life balance policy.
"""

context = """
Scenario: A senior developer leaves due to dissatisfaction with project scope, feeling unchallenged. 
This developer has been contributing to social media on topics unrelated to their job.
"""

formatted_prompt = prompt_template.format(
    organization_metadata=organization_metadata, context=context
)

print("Generating questions and scenarios...")

response = chat_model.invoke([HumanMessage(content=formatted_prompt)])

try:
    response_text = response.content.strip("```json").strip("```")  
    parsed_response = json.loads(response_text) 
    print("Generated Questions and Scenarios:")
    print(json.dumps(parsed_response, indent=2))
except json.JSONDecodeError:
    print("Warning: LLM output is not valid JSON. Here’s the raw response:")
    print(response.content)

