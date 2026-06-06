from fastapi import Request
from starlette.middleware.cors import CORSMiddleware
from app.core.config import settings

print("ALLOWED_ORIGINS =", settings.ALLOWED_ORIGINS)

# Instantiate the middleware directly to test
app_dummy = lambda scope, receive, send: None
cors = CORSMiddleware(
    app=app_dummy,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

headers = [
    (b"origin", b"https://talentai-x.vercel.app"),
    (b"access-control-request-method", b"POST"),
    (b"access-control-request-headers", b"x-api-key"),
]

from starlette.datastructures import Headers
req_headers = Headers(raw=headers)

resp = cors.preflight_response(req_headers)
print("Status:", resp.status_code)
print("Headers:", resp.headers)
print("Body:", resp.body)
