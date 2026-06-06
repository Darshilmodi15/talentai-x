import os
os.environ["ALLOWED_ORIGINS"] = '["http://localhost:3000,https://talentai-x.vercel.app"]'
from app.core.config import Settings
s = Settings()
print("Parsed:", s.ALLOWED_ORIGINS)
print("Is allowed?", "https://talentai-x.vercel.app" in s.ALLOWED_ORIGINS)
