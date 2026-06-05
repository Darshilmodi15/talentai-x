def merge_extractions(basic: dict, experience: dict, education: dict, skills: dict) -> dict:
    return {
        "name": basic.get("name"),
        "email": basic.get("email"),
        "phone": basic.get("phone"),
        "location": basic.get("location"),
        "summary": basic.get("summary"),
        "linkedin_url": basic.get("linkedin_url"),
        "github_url": basic.get("github_url"),
        "portfolio_url": basic.get("portfolio_url"),
        "other_urls": basic.get("other_urls", []),
        "experience": experience.get("experience", []),
        "experience_months_total": experience.get("experience_months_total", 0),
        "education": education.get("education", []),
        "skills": skills.get("skills", []),
        "certifications": skills.get("certifications", []),
        "projects": skills.get("projects", []),
        "publications": skills.get("publications", []),
    }

basic = None
try:
    merge_extractions(basic, {}, {}, {})
except Exception as e:
    print(repr(str(e)))
