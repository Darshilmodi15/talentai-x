import json

content = """
Here is the JSON:

```json
{
  "name": "Darshil Modi"
}
```
"""

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
    print(parsed)
except Exception as e:
    print(repr(str(e)))
