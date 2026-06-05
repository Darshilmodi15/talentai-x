import json

text = """{
  "basic": {
    "name": "Darshil"
  }
}"""
try:
    parsed = json.loads(text)
    print(parsed["basic"]["\n  \"name\""])
except Exception as e:
    print(type(e).__name__, repr(str(e)))
