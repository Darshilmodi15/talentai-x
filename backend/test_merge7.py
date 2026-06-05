def merge_extractions(basic: dict, experience: dict, education: dict, skills: dict) -> dict:
    return {
        "experience": experience.get("experience", []),
        "experience_months_total": experience.get("experience_months_total", 0),
    }

basic = {}
experience = "\n  \"name\"" 
education = {}
skills = {}

try:
    merge_extractions(basic, experience, education, skills)
except Exception as e:
    print(repr(str(e)))
