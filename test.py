from openai import OpenAI
import os
from pydantic import BaseModel
LLM_API_KEY = os.getenv("OPENAI_API_KEY")
print("LLM_API_KEY", LLM_API_KEY)
client = OpenAI(api_key=LLM_API_KEY)



class StoryResponse(BaseModel):
    story: str

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

response = client.responses.parse(
    model="gpt-5-mini",
    input=[
        {"role": "system", "content": "Extract the event information."},
        {
            "role": "user",
            "content": "Alice and Bob are going to a science fair on Friday.",
        },
    ],
    text_format=CalendarEvent,
)

event = response.output_parsed


print(event)