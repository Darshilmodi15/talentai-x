import os
os.environ["ALLOWED_ORIGINS"] = 'https://talentai-x.vercel.app'
try:
    from app.core.config import Settings
    s = Settings()
    print("Parsed:", s.ALLOWED_ORIGINS)
    print("Type:", type(s.ALLOWED_ORIGINS))
    print("Is origin in list?", "https://talentai-x.vercel.app" in s.ALLOWED_ORIGINS)
except Exception as e:
    print("Error:", repr(e))

