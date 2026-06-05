import json

# Can we simulate the actual string that caused the KeyError?
content = """```json
{
  "name": "Darshil Modi",
  "
  \"name\"": "duplicate?"
}
```"""

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
    print(type(e).__name__)
    print(repr(str(e)))
