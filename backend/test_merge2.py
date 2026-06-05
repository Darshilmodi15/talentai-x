def merge_extractions(basic: dict, experience: dict, education: dict, skills: dict) -> dict:
    return {
        "name": basic.get("name"),
    }

basic = "string"
try:
    merge_extractions(basic, {}, {}, {})
except Exception as e:
    print(repr(str(e)))
