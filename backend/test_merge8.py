def merge_extractions(basic: dict, experience: dict, education: dict, skills: dict) -> dict:
    return {
        "name": basic.get("name"),
    }

basic = type('obj', (object,), {'get': lambda self, key: 'val'})()
try:
    merge_extractions(basic, {}, {}, {})
except Exception as e:
    print(repr(str(e)))
