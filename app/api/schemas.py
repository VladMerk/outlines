from pydantic import BaseModel


class StartResponse(BaseModel):
    thread_id: str
    ai_message: str


class UserReply(BaseModel):
    thread_id: str
    user_input: str
