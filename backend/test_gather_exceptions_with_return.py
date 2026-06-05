import asyncio
import json

async def call_gemini(prompt: str) -> dict:
    if prompt == "1":
        content = "```json\n{\n  \"name\": \"Darshil Modi\"\n}\n```"
        try:
             json.loads(content)
        except Exception as e:
            raise ValueError(f"JSON Parse Failed.\nRaw Response:\n{content}\nException:\nKeyError: '\n  \"name\"'\nTraceback:\n")
        return {"data": prompt}

async def extract_all_parallel() -> tuple[dict, dict, dict, dict]:
    results = await asyncio.gather(
        call_gemini("1"),
        call_gemini("2"),
        call_gemini("3"),
        call_gemini("4"),
        return_exceptions=True
    )
    
    for r in results:
        if isinstance(r, Exception):
            raise r
            
    return tuple(results)

async def parse_agent():
    try:
        basic, experience, education, skills = await extract_all_parallel()
    except Exception as e:
        error_msg = str(e)
        print("parse_agent: " + error_msg)
        
asyncio.run(parse_agent())
