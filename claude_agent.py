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
    from anthropic import APIConnectionError, APIStatusError, Anthropic
except ImportError:
    APIConnectionError = None
    APIStatusError = None
    Anthropic = None

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


class ClaudeAgent(BaseAgent):
    def __init__(
        self,
        name: str = "ClaudeAgent",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout_seconds: float = 300.0,
        history_size: Optional[int] = 3,
    ):
        super().__init__(name)
        self.model = model or os.getenv("ANTHROPIC_MODEL") or os.getenv("LLM_MODEL") or "claude-sonnet-4-20250514"
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_response_attempts = MAX_RESPONSE_PARSE_ATTEMPTS
        self.timeout_seconds = timeout_seconds
        self.history_size = history_size if history_size is not None else int(os.getenv("LLM_HISTORY_SIZE", "3"))
        self.last_response = ""
        self.observation_history: list[dict[str, object]] = []
        self._client: Optional[Anthropic] = None

    def reset(self) -> None:
        self.last_response = ""
        self.observation_history.clear()

    def _legal_moves(self, board: Board) -> list[str]:
        return [move for move in POSSIBLE_MOVES if check_move_valid(board, move)]

    def _get_client(self) -> Anthropic:
        if Anthropic is None:
            raise RuntimeError(
                "The Anthropic Python SDK is not installed. Add the `anthropic` package to use this agent."
            )

        if self._client is None:
            client_kwargs: dict[str, object] = {"timeout": self.timeout_seconds}
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            self._client = Anthropic(**client_kwargs)

        return self._client

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_fixed(1),
        retry=retry_if_exception(is_retryable_request_error),
    )
    def _request_completion(self, system_prompt: str, user_prompt: str) -> str:
        client = self._get_client()
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )

        for block in message.content:
            if block.type == "text":
                return block.text

        raise RuntimeError(f"Unexpected Anthropic response format: {message}")

    def get_move(self, board: Board) -> tuple[str, int]:
        legal_moves = self._legal_moves(board)
        if not legal_moves:
            return "", 0

        current_observation = make_observation(board)
        invalid_response = None

        for _ in range(self.max_response_attempts):
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
