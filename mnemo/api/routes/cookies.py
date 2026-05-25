from fastapi import APIRouter, Request
from mnemo.models import UpdateCookiesRequest, UpdateCookiesResponse

router = APIRouter()

@router.post("/cookies", response_model=UpdateCookiesResponse)
async def update_cookies(req: UpdateCookiesRequest, request: Request) -> UpdateCookiesResponse:
    settings = request.app.state.settings
    settings.cookies_path.parent.mkdir(parents=True, exist_ok=True)
    settings.cookies_path.write_text(req.cookies)
    settings.cookies_path.chmod(0o600)
    return UpdateCookiesResponse(status="success", message="Cookies updated successfully")
