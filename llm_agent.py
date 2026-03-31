import json
import os
import re
from typing import Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed

from base_agent import BaseAgent, Board
from engine import POSSIBLE_MOVES, check_move_valid

ACTION_TAG_PATTERN = re.compile(r"<action>\s*(LEFT|RIGHT|UP|DOWN|NONE)\s*</action>", re.IGNORECASE)
DEFAULT_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
RETRYABLE_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}
MAX_RESPONSE_PARSE_ATTEMPTS = 5

DEFAULT_SYSTEM_PROMPT = """
You are an agent in the game 2048.

At every step, you receive a 4x4 board where 0 means empty.
You must output exactly one action: "UP", "DOWN", "LEFT", or "RIGHT".

Game rules:
- Tiles slide in the chosen direction.
- Equal tiles merge into their sum.
- A tile merges at most once per move.
- After every valid move, a new tile spawns in an empty cell.
- Moves that do not change the board are invalid and should be avoided.
- The episode ends when no valid moves remain.

Goal:
Maximize cumulative reward over the episode by choosing strong, valid moves.

Respond using XML in exactly this structure:
<response>
  <reasoning>brief explanation here</reasoning>
  <action>move</action>
</response>
""".strip()


def format_board(board: Board) -> str:
    return json.dumps(board)


def normalize_chat_completions_url(url: Optional[str]) -> str:
    if not url:
        return DEFAULT_CHAT_COMPLETIONS_URL

    normalized = url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/completions"):
        return normalized[: -len("/completions")] + "/chat/completions"
    if normalized.endswith("/v1"):
        return normalized + "/chat/completions"
    if normalized.endswith("/v1/"):
        return normalized.rstrip("/") + "/chat/completions"
    return normalized + "/v1/chat/completions"


def is_retryable_request_error(exc: BaseException) -> bool:
    if isinstance(exc, URLError):
        return True
    if isinstance(exc, HTTPError):
        return exc.code in RETRYABLE_HTTP_STATUS_CODES
    return False


class LLMAgent(BaseAgent):
    def __init__(
        self,
        name: str = "LLMAgent",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        temperature: float = 0.1,
        timeout_seconds: float = 60.0,
        history_size: Optional[int] = 3,
    ):
        super().__init__(name)
        self.model = model or os.getenv("LLM_MODEL")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base_url = normalize_chat_completions_url(
            api_base_url or os.getenv("LLM_API_BASE_URL")
        )
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.history_size = history_size if history_size is not None else int(os.getenv("LLM_HISTORY_SIZE", "3"))
        self.last_response = ""
        self.observation_history: list[dict[str, object]] = []

    def reset(self) -> None:
        self.last_response = ""
        self.observation_history = []

    def _legal_moves(self, board: Board) -> list[str]:
        return [move for move in POSSIBLE_MOVES if check_move_valid(board, move)]

    def _make_observation(self, board: Board, legal_moves: list[str]) -> dict[str, object]:
        return {
            "board": board,
            "valid_actions": legal_moves,
            "done": False,
        }

    def _build_user_prompt(
        self,
        current_observation: dict[str, object],
        invalid_response: Optional[str] = None,
    ) -> str:
        previous_observations = self.observation_history[-self.history_size :] if self.history_size > 0 else []
        payload = {
            "previous_observations": previous_observations,
            "current_observation": current_observation,
        }
        prompt = f"Observation:\n{json.dumps(payload, indent=2)}"
        if invalid_response is not None:
            prompt += (
                "\n\nYour previous response was invalid for this state. "
                "Reply again using exactly one <response> block with an <action> tag "
                f"containing one of: {', '.join(current_observation['valid_actions'])}, NONE."
            )
        return prompt

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_fixed(1),
        retry=retry_if_exception(is_retryable_request_error),
    )
    def _send_request(self, request: Request) -> str:
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return response.read().decode("utf-8")

    def _request_completion(self, system_prompt: str, user_prompt: str) -> str:
        if not self.model:
            raise ValueError("LLMAgent requires a model. Set LLM_MODEL or pass model=.")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        request = Request(
            self.api_base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            body = self._send_request(request)
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed with status {exc.code}: {error_body}") from exc
        except URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

        data = json.loads(body)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM response shape: {data}") from exc

        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
            ]
            if text_parts:
                return "".join(text_parts)

        raise RuntimeError(f"Unsupported LLM content format: {content}")

    def _parse_move(self, raw_response: str, legal_moves: list[str]) -> Optional[str]:
        action_tag_match = ACTION_TAG_PATTERN.search(raw_response)
        if not action_tag_match:
            return None

        normalized = action_tag_match.group(1).upper()
        if normalized == "NONE":
            return "NONE"
        if normalized in POSSIBLE_MOVES:
            return normalized
        return None

    def get_move(self, board: Board) -> Tuple[str, int]:
        legal_moves = self._legal_moves(board)
        if not legal_moves:
            return "", 0

        current_observation = self._make_observation(board, legal_moves)
        invalid_response = None

        for _ in range(MAX_RESPONSE_PARSE_ATTEMPTS):
            user_prompt = self._build_user_prompt(current_observation, invalid_response=invalid_response)
            raw_response = self._request_completion(DEFAULT_SYSTEM_PROMPT, user_prompt)
            self.last_response = raw_response
            move = self._parse_move(raw_response, legal_moves)
            if move is not None:
                self.observation_history.append(current_observation)
                return move, 0
            invalid_response = raw_response

        return "NONE", 0
