import asyncio
from concurrent.futures import TimeoutError
import json
import traceback

async def call_gemini(prompt: str, max_tokens: int = 1500) -> dict:
    content = "```json\n{\n  \"name\": \"Darshil Modi\"\n}\n```"
    if prompt == "1":
        content = "{\n  \"name\": \"Darshil Modi\"\n}.    We have identified the root cause of the resume parsing failure."
    
    import re
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
    if match:
        cleaned_response = match.group(1).strip()
    else:
        cleaned_response = content.strip("`").strip()
        if cleaned_response.startswith("json"):
            cleaned_response = cleaned_response[4:].strip()

    try:
        parsed = json.loads(cleaned_response)
    except Exception as e:
        tb = traceback.format_exc()
        raise ValueError(f"JSON Parse Failed.\nRaw Response:\n{content}\nException:\n{e}\nTraceback:\n{tb}")

    if not isinstance(parsed, dict):
        raise ValueError("Gemini did not return a JSON object")

    return parsed
    
async def extract_all_parallel() -> tuple[dict, dict, dict, dict]:
    # IF return_exceptions is used, it returns the exception object
    # If return_exceptions is false (the default), gather RAISES the first exception it gets
    # But wait, in extract_all_parallel, return_exceptions is NOT SET!
    # So asyncio.gather should raise the exception.
    
    # Wait, in the parse_agent function:
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
        print("Caught in parse_agent:", repr(str(e)))
        
asyncio.run(parse_agent())
