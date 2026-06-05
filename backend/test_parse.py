import asyncio
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def call_gemini(prompt: str, max_tokens: int = 1500) -> dict:
    content = "```json\n{\n  \"name\": \"John Doe\"\n}\n```"
    
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
        logger.exception("JSON parse failed")
        import traceback
        tb = traceback.format_exc()
        raise ValueError(f"JSON Parse Failed.\nRaw Response:\n{content}\nException:\n{e}\nTraceback:\n{tb}")

    if not isinstance(parsed, dict):
        raise ValueError("Gemini did not return a JSON object")

    return parsed

async def extract_all_parallel() -> tuple[dict, dict, dict, dict]:
    basic, experience, education, skills = await asyncio.gather(
        call_gemini("1", max_tokens=800),
        call_gemini("2", max_tokens=1500),
        call_gemini("3", max_tokens=600),
        call_gemini("4", max_tokens=1000),
    )
    return basic, experience, education, skills

async def main():
    try:
        await extract_all_parallel()
        print("Success")
    except Exception as e:
        print(repr(str(e)))
        
asyncio.run(main())
