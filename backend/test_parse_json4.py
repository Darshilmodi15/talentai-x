import json

# What if Gemini returns malformed JSON where there is a newline in the key, like:
content = """{
  "
  \"name\"": "John"
}"""
try:
    parsed = json.loads(content)
    print(parsed)
except Exception as e:
    print(repr(str(e)))
