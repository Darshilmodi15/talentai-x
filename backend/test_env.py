import os
from app.core.config import Settings

# Test with a JSON array string
os.environ["ALLOWED_ORIGINS"] = '["http://localhost:3000", "https://talentai-x.vercel.app"]'
s1 = Settings()
print("JSON array:", s1.ALLOWED_ORIGINS)

# Test with comma separated string
os.environ["ALLOWED_ORIGINS"] = 'http://localhost:3000,https://talentai-x.vercel.app'
s2 = Settings()
print("Comma separated:", s2.ALLOWED_ORIGINS)

