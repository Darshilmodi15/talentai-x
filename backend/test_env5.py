import os
from app.core.config import Settings

os.environ["ALLOWED_ORIGINS"] = '["http://localhost:3000,https://talentai-x.vercel.app/"]'
s = Settings()
print(s.ALLOWED_ORIGINS)

