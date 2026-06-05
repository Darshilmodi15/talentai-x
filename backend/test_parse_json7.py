import json

content = """
{
  "name": "Darshil"
}
"""

try:
    parsed = json.loads(content)
except Exception as e:
    parsed = e # Let's say parsed was somehow assigned the exception

def merge_extractions(basic, experience, education, skills):
    return {
        "name": basic.get("name"),
    }

try:
    merge_extractions(parsed, {}, {}, {})
except Exception as e:
    print(repr(str(e)))
