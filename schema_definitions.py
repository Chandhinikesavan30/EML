from pydantic import BaseModel, Field
from typing import List

class QuestionScenario(BaseModel):
    question: str = Field(..., title="Generated Question")
    scenario: str = Field(..., title="Scenario Description")
    insights: List[str] = Field(..., title="Key Insights from the Scenario")
    thought_process: str = Field(..., title="Thought Process Behind Insights")
    actions: List[str] = Field(..., title="Recommended Actions")

OutputSchema = List[QuestionScenario]
