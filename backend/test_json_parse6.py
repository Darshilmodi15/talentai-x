import json

content = """
```json
{
  "name": "Darshil"
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

parsed = json.loads(cleaned_response)

def merge_extractions(basic, experience, education, skills):
    print("merge")
    return {
        "name": basic.get("name"),
    }

basic = parsed
experience = "\n  \"name\"" # What if experience is malformed and some other string?
education = {}
skills = {}

try:
    merge_extractions(basic, experience, education, skills)
except Exception as e:
    print(repr(str(e)))
