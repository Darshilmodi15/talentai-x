import json

content = "```\n{\n  \"name\": \"Darshil Modi\"\n}\n```"

cleaned_response = content.strip("`").strip()
if cleaned_response.startswith("json"):
    cleaned_response = cleaned_response[4:].strip()

try:
    parsed = json.loads(cleaned_response)
    print(parsed)
except Exception as e:
    print(repr(str(e)))
