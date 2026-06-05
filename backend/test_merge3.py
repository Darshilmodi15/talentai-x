def merge_extractions(basic: dict, experience: dict, education: dict, skills: dict) -> dict:
    return {
        "name": basic["name"],
    }

basic = "string"
try:
    merge_extractions(basic, {}, {}, {})
except Exception as e:
    print(type(e).__name__, repr(str(e)))
