import asyncio
from typing import Optional, Any
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def call_gemini(prompt: str, max_tokens: int = 1500) -> dict:
    if prompt == "1":
        # Simulate gemini failure
        raise ValueError("JSON Parse Failed.\nRaw Response:\n{\n  \"name\": \"John Doe\"\n}\nException:\nKeyError: '\n  \"name\"'\nTraceback:\n")
    return {"data": prompt}

async def extract_all_parallel(raw_text: str) -> tuple[dict, dict, dict, dict]:
    text_chunk = raw_text[:6000]

    basic, experience, education, skills = await asyncio.gather(
        call_gemini("1", max_tokens=800),
        call_gemini("2", max_tokens=1500),
        call_gemini("3", max_tokens=600),
        call_gemini("4", max_tokens=1000),
    )
    return basic, experience, education, skills

async def main():
    try:
        await extract_all_parallel('test')
    except Exception as e:
        print(repr(str(e)))
        
asyncio.run(main())
