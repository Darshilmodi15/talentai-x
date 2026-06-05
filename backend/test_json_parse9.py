import json

content = """{
  "name": "Darshil Modi",
  "email": "example@email.com"
}.    We have identified the root cause of the resume parsing failure."""

import re
match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
if match:
    cleaned = match.group(1).strip()
else:
    cleaned = content.strip("`").strip()
    if cleaned.startswith("json"):
        cleaned = cleaned[4:].strip()

try:
    json.loads(cleaned)
except Exception as e:
    print(repr(str(e)))
