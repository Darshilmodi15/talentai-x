import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

content = "```json\n{\n  \"name\": \"John Doe\"\n}\n```"

import re
match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
if match:
    cleaned_response = match.group(1).strip()
else:
    cleaned_response = content.strip("`").strip()
    if cleaned_response.startswith("json"):
        cleaned_response = cleaned_response[4:].strip()

print("Cleaned:")
print(repr(cleaned_response))
try:
    parsed = json.loads(cleaned_response)
    print("Parsed dict:", parsed)
except Exception as e:
    logger.exception("JSON parse failed")
    import traceback
    tb = traceback.format_exc()
    raise ValueError(f"JSON Parse Failed.\nRaw Response:\n{content}\nException:\n{e}\nTraceback:\n{tb}")
