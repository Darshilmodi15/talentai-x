import os
from app.core.config import Settings

cases = [
    '["http://localhost:3000","https://talentai-x.vercel.app"]',
    '["http://localhost:3000,https://talentai-x.vercel.app"]',
    'http://localhost:3000,https://talentai-x.vercel.app',
    'https://talentai-x.vercel.app/',
    '["https://talentai-x.vercel.app/"]'
]

for c in cases:
    os.environ["ALLOWED_ORIGINS"] = c
    try:
        s = Settings()
        print(f"Input: {c}")
        print(f"Output: {s.ALLOWED_ORIGINS}")
    except Exception as e:
        print(f"Error for {c}: {e}")

