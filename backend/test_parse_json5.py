import json

text = """```json
{
  "name": "Darshil Modi",
  "email": "example@email.com",
  "experience": [
    {
      "company": "Tech Corp",
      "duration_months": 24,
      "bullets": ["Did stuff"]
    }
  ]
}
```"""

import re
match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
if match:
    cleaned = match.group(1).strip()
    
# Now simulate what happens during merge_extractions
parsed = json.loads(cleaned)

def merge_extractions(basic, experience, education, skills):
    return {
        "name": basic.get("name"),
    }
    
try:
    merge_extractions(text, {}, {}, {}) # If someone accidentally passed the raw text instead of parsed JSON to merge
except Exception as e:
    print(repr(str(e)))
