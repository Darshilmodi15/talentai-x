import asyncio
from concurrent.futures import TimeoutError
import json
import traceback

async def call_gemini(prompt: str, max_tokens: int = 1500) -> dict:
    if prompt == "1":
        # Simulate gemini returning malformed json that throws during loads
        content = """```json
{
  "name": "John Doe"
}
```"""
        try:
             json.loads(content)
        except Exception as e:
            tb = traceback.format_exc()
            raise ValueError(f"JSON Parse Failed.\nRaw Response:\n{content}\nException:\nKeyError: '\n  \"name\"'\nTraceback:\n")
    return {"data": prompt}

async def extract_all_parallel() -> tuple[dict, dict, dict, dict]:
    try:
        basic, experience, education, skills = await asyncio.gather(
            call_gemini("1"),
            call_gemini("2"),
            call_gemini("3"),
            call_gemini("4"),
        )
    except Exception as e:
        raise e
        
    return basic, experience, education, skills

async def parse_agent():
    try:
        basic, experience, education, skills = await extract_all_parallel()
    except Exception as e:
        error_msg = str(e)
        print("parse_agent: " + error_msg)
        
asyncio.run(parse_agent())
