import asyncio
from concurrent.futures import TimeoutError

async def test_key_error():
    # If the gather doesn't use return_exceptions=True, the first exception is raised directly
    pass

def merge_extractions(basic, experience, education, skills):
    return {
        "name": basic.get("name"),
    }

async def call_gemini(prompt: str, max_tokens: int = 1500) -> dict:
    if prompt == "2":
        return ValueError("Error in experience") # Accidentally returning an exception object instead of raising
    return {"name": "test"}
    
async def extract_all_parallel() -> tuple[dict, dict, dict, dict]:
    basic, experience, education, skills = await asyncio.gather(
        call_gemini("1"),
        call_gemini("2"), # Returns Exception instead of raising it
        call_gemini("3"),
        call_gemini("4"),
    )
    return basic, experience, education, skills

async def parse_agent():
    try:
        basic, experience, education, skills = await extract_all_parallel()
        # what if experience is an Exception?
        merge_extractions(basic, experience, education, skills)
    except Exception as e:
        print(repr(str(e)))
        
asyncio.run(parse_agent())
