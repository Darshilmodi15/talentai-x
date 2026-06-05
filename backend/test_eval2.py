import ast

def merge_extractions(basic: dict, experience: dict, education: dict, skills: dict) -> dict:
    return {
        "name": basic.get("name"),
    }

basic = {'\n  "name"': 'John'}

# this works
print(basic.get('\n  "name"'))

# what if the dictionary is actually a string that looks like a dict?
# wait, if basic is a string, basic.get() will raise AttributeError

