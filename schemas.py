from typing import List
from pydantic import BaseModel, Field


class Route(BaseModel):
    category: str = Field(
        "One of these words only: programming, math, general")


class Answer(BaseModel):
    answer: str = Field("The answer of the question")


class Summary(BaseModel):
    summary: str = Field("A one line summary of the question")


class ChatResponse(BaseModel):
    answer: str = Field("The final answer for the user")
    summary: str = Field("A one line summary of the answer")
    keywords: List[str] = Field("3 to 5 important keywords")
    difficulty: str = Field("Easy, Medium or Hard")
    confidence: float = Field(
        ge=0,
        le=1,
        description="How confident the answer is, between 0 and 1"
    )
