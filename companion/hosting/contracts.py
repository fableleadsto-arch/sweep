"""Text only: callers cannot provide tools, models, file paths or system prompts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=6000)

    @model_validator(mode="after")
    def assistant_limit(self):
        if self.role == "assistant" and len(self.content) > 2000:
            raise ValueError("assistant history exceeds 2000 characters")
        return self


class TextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    messages: list[Message] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def bounded_conversation(self):
        if sum(len(message.content) for message in self.messages) > 10000:
            raise ValueError("conversation exceeds 10000 characters")
        if self.messages[-1].role != "user" or not self.messages[-1].content.strip():
            raise ValueError("last message must contain a user request")
        return self


class PairRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    invitation: str = Field(min_length=49, max_length=49, repr=False)
    device_name: str = Field(min_length=1, max_length=80)
