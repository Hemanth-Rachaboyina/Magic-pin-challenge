import os
import json
from pydantic import BaseModel
from typing import Type, TypeVar
from dotenv import load_dotenv
import openai

load_dotenv()

openai_model = "gpt-4o"

T = TypeVar('T', bound=BaseModel)

def generate_with_gemini(system_prompt: str, user_prompt: str, schema: Type[T], model_name: str = "gemini-2.5-flash") -> T:
    from google import genai
    from google.genai import types
    
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.2,
        ),
    )
    return schema.model_validate_json(response.text)


def generate_with_openai(system_prompt: str, user_prompt: str, schema: Type[T], model_name: str = openai_model) -> T:
    
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format=schema,
        temperature=0.2
    )
    return response.choices[0].message.parsed



# Removed commented out legacy functions
