import asyncio
import json

async def fetch_basic():
    return json.loads("{\n  \"name\"")  # JSONDecodeError

async def main():
    try:
        await asyncio.gather(fetch_basic())
    except Exception as e:
        print(repr(str(e)))
        print(type(e).__name__)

asyncio.run(main())
