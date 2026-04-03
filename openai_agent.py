import os
from typing import Optional
from urllib.error import HTTPError, URLError

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed

from agent_utils import (
    DEFAULT_SYSTEM_PROMPT,
    MAX_RESPONSE_PARSE_ATTEMPTS,
    build_user_prompt,
    make_observation,
    parse_move,
)
from base_agent import BaseAgent, Board
from engine import POSSIBLE_MOVES, check_move_valid

try:
    from openai import APIConnectionError, APIStatusError, OpenAI
except ImportError:
    APIConnectionError = None
    APIStatusError = None
    OpenAI = None

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
RETRYABLE_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def is_retryable_request_error(exc: BaseException) -> bool:
    if isinstance(exc, (URLError,)):
        return True
    if isinstance(exc, HTTPError):
        return exc.code in RETRYABLE_HTTP_STATUS_CODES
    if APIConnectionError is not None and isinstance(exc, APIConnectionError):
        return True
    if APIStatusError is not None and isinstance(exc, APIStatusError):
        return exc.status_code in RETRYABLE_HTTP_STATUS_CODES
    return False


class OpenAIAgent(BaseAgent):
    def __init__(
        self,
        name: str = "OpenAIAgent",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        temperature: float = 0.1,
        timeout_seconds: float = 300.0,
        history_size: Optional[int] = 3,
    ):
        resolved_base_url = (
            api_base_url
            or os.getenv("OPENAI_API_BASE_URL")
            or os.getenv("LLM_API_BASE_URL")
            or DEFAULT_OPENAI_BASE_URL
        )
        super().__init__(name)
        self.model = model or os.getenv("LLM_MODEL")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base_url = resolved_base_url
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.history_size = history_size if history_size is not None else int(os.getenv("LLM_HISTORY_SIZE", "3"))
        self.last_response = ""
        self.observation_history: list[dict[str, object]] = []
        self._openai_client: Optional[OpenAI] = None

    def reset(self) -> None:
        self.last_response = ""
        self.observation_history.clear()

    def _legal_moves(self, board: Board) -> list[str]:
        return [move for move in POSSIBLE_MOVES if check_move_valid(board, move)]

    def _get_openai_client(self) -> OpenAI:
        if OpenAI is None:
            raise RuntimeError("The OpenAI Python SDK is not installed. Add the `openai` package to use this path.")

        if self._openai_client is None:
            client_kwargs: dict[str, object] = {"timeout": self.timeout_seconds}
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            if self.api_base_url:
                client_kwargs["base_url"] = self.api_base_url
            self._openai_client = OpenAI(**client_kwargs)

        return self._openai_client

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_fixed(1),
        retry=retry_if_exception(is_retryable_request_error),
    )
    def _request_completion(self, system_prompt: str, user_prompt: str) -> str:
        if not self.model:
            raise ValueError("OpenAIAgent requires a model. Set LLM_MODEL or pass model=.")

        client = self._get_openai_client()
        response = client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=user_prompt,
            reasoning={"effort": "medium"},
        )

        if getattr(response, "output_text", None):
            return response.output_text

        if hasattr(response, "model_dump"):
            response_dict = response.model_dump()
            for item in response_dict.get("output", []):
                if item.get("type") != "message":
                    continue
                for content_item in item.get("content", []):
                    if content_item.get("type") == "output_text" and content_item.get("text"):
                        return content_item["text"]

        raise RuntimeError(f"Unsupported OpenAI Responses output format: {response}")

    def get_move(self, board: Board) -> tuple[str, int]:
        legal_moves = self._legal_moves(board)
        if not legal_moves:
            return "", 0

        current_observation = make_observation(board)
        invalid_response = None

        for _ in range(MAX_RESPONSE_PARSE_ATTEMPTS):
            user_prompt = build_user_prompt(
                observation_history=self.observation_history,
                current_observation=current_observation,
                history_size=self.history_size,
                invalid_response=invalid_response,
            )
            raw_response = self._request_completion(DEFAULT_SYSTEM_PROMPT, user_prompt)

            self.last_response = raw_response
            move = parse_move(raw_response)
            if move is not None:
                self.observation_history.append(current_observation)
                return move, 0

            invalid_response = raw_response

        return "NONE", 0
