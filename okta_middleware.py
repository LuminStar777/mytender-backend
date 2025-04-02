import os
from fastapi import Request
from fastapi.responses import JSONResponse
from okta_jwt_verifier import JWTVerifier
from starlette.middleware.base import BaseHTTPMiddleware


class OktaJWTMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        OKTA_DOMAIN =os.getenv("OKTA_ISSUER")
        OKTA_ISSUER = f"{OKTA_DOMAIN}/oauth2/default"
        OKTA_CLIENT_ID = os.getenv("OKTA_CLIENT_ID")

        accessToken = request.headers.get("authorization", None)

        if accessToken is not None:
            try:
                accessToken = accessToken.split(" ")[1]
                jwt_verifier = JWTVerifier(OKTA_ISSUER, OKTA_CLIENT_ID, "api://default")
                await jwt_verifier.verify_access_token(accessToken)
                return await call_next(request)
            except Exception as e:
                return JSONResponse({"error": f"Unauthorized token or {e}"}, status_code=401)
        else:
            return JSONResponse({"error": "Unauthorized Access"}, status_code=401)