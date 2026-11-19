"""Native Antigravity (ACP) LLM/VLM client for FSR-VLN.

Drives FSR-VLN's LLM instruction parsing and VLM slow-reasoning verification
directly through the official local Antigravity CLI ACP server (`agy acp --stdio`)
with native multimodal image content blocks—no reverse proxy and no OSS upload required.

Configurable model / reasoning effort:
  - Hydra / YAML config:
      models:
        llm:
          backend: antigravity        # antigravity | openai | azure | offline
          model: gemini-4-neon-high   # or gemini-4-neon-xhigh (overrides effort)
          effort: high                # high | extra (xhigh) | medium | low
  - Environment variables (override defaults when config does not specify):
      FSR_VLN_LLM_BACKEND=antigravity
      ANTIGRAVITY_MODEL=gemini-4-neon-high   # or gemini-4-neon-xhigh
      ANTIGRAVITY_EFFORT=high                # high | extra | xhigh
      AGY_BIN=agy                            # optional path to agy binary
"""

from __future__ import annotations

import atexit
import base64
from dataclasses import dataclass
import json
import mimetypes
import os
import shutil
import subprocess
import threading
import urllib.request
from typing import Any, Dict, List, Optional, Tuple, Union


NEON_EFFORT_TO_MODEL = {
    "low": "gemini-4-neon-low",
    "medium": "gemini-4-neon-medium",
    "high": "gemini-4-neon-high",
    "extra": "gemini-4-neon-xhigh",
    "xhigh": "gemini-4-neon-xhigh",
    "extra_high": "gemini-4-neon-xhigh",
    "extra-high": "gemini-4-neon-xhigh",
    "neon-low": "gemini-4-neon-low",
    "neon-medium": "gemini-4-neon-medium",
    "neon-high": "gemini-4-neon-high",
    "neon-extra": "gemini-4-neon-xhigh",
    "neon-xhigh": "gemini-4-neon-xhigh",
}

DEFAULT_ANTIGRAVITY_MODEL = "gemini-4-neon-high"


def _cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    """Safely read nested attribute/key path from OmegaConf DictConfig or dict."""
    cur = cfg
    if cur is None:
        return default
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, None)
        else:
            cur = getattr(cur, part, None)
    return default if cur is None else cur


def resolve_antigravity_model(
    model: Optional[str] = None,
    effort: Optional[str] = None,
    cfg: Any = None,
) -> str:
    """Resolve an Antigravity Gemini 4 Neon model ID (High / XHigh configurable).

    Priority:
      1. Explicit model (`model` arg > `ANTIGRAVITY_MODEL` env > `cfg.models.llm.model`)
      2. Reasoning effort (`effort` arg > `ANTIGRAVITY_EFFORT` env > `cfg.models.llm.effort`)
         where `high` -> `gemini-4-neon-high` and `extra`/`xhigh` -> `gemini-4-neon-xhigh`
      3. Default: `gemini-4-neon-high`
    """
    env_model = os.getenv("ANTIGRAVITY_MODEL", "").strip()
    env_effort = os.getenv("ANTIGRAVITY_EFFORT", "").strip()

    raw_cfg_model = _cfg_get(cfg, "models.llm.model", None)
    cfg_model = "" if raw_cfg_model is None else str(raw_cfg_model).strip()
    raw_cfg_effort = _cfg_get(cfg, "models.llm.effort", None)
    cfg_effort = "" if raw_cfg_effort is None else str(raw_cfg_effort).strip()

    chosen_model = (model or env_model or cfg_model).strip()
    if chosen_model:
        return NEON_EFFORT_TO_MODEL.get(chosen_model.lower(), chosen_model)

    chosen_effort = (effort or env_effort or cfg_effort).strip().lower()
    if chosen_effort in NEON_EFFORT_TO_MODEL:
        return NEON_EFFORT_TO_MODEL[chosen_effort]

    return DEFAULT_ANTIGRAVITY_MODEL


def resolve_llm_backend(cfg: Any = None) -> str:
    """Determine active LLM backend: antigravity | openai | azure | offline."""
    env_backend = os.getenv("FSR_VLN_LLM_BACKEND", "").strip().lower()
    if env_backend:
        return env_backend

    cfg_backend = str(_cfg_get(cfg, "models.llm.backend", "") or "").strip().lower()
    if cfg_backend:
        return cfg_backend

    agy_bin = os.getenv("AGY_BIN", "agy")
    if shutil.which(agy_bin):
        return "antigravity"
    if os.getenv("AZURE_OPENAI_API_KEY", "").strip():
        return "azure"
    if os.getenv("OPENAI_API_KEY", "").strip():
        return "openai"
    return "offline"


def _load_image_as_base64(image_ref: str, max_side: int = 640) -> Tuple[str, str]:
    """Convert a local path, file:// URI, data URI, or HTTP URL to (mime_type, base64_str).

    Automatically encodes local images as compact JPEG (bounded by `max_side`) so
    multi-image VLM calls over ACP stay fast and crisp.
    """
    ref = image_ref.strip()

    if ref.startswith("data:"):
        header, b64_data = ref.split(",", 1)
        mime = header[5:].split(";")[0] or "image/png"
        return mime, b64_data

    if ref.startswith("file://"):
        ref = ref[len("file://"):]

    if ref.startswith("http://") or ref.startswith("https://"):
        with urllib.request.urlopen(ref, timeout=30) as resp:
            raw = resp.read()
            mime = resp.headers.get_content_type() or "image/png"
        return mime, base64.b64encode(raw).decode("ascii")

    if not os.path.exists(ref):
        raise FileNotFoundError(f"Image file not found for Antigravity VLM call: {ref}")

    try:
        from io import BytesIO
        from PIL import Image as PILImage

        with PILImage.open(ref) as im:
            im = im.convert("RGB")
            w, h = im.size
            if max(w, h) > max_side:
                scale = max_side / float(max(w, h))
                im = im.resize((int(w * scale), int(h * scale)), PILImage.Resampling.LANCZOS)
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=85)
            return "image/jpeg", base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        mime, _ = mimetypes.guess_type(ref)
        if not mime or not mime.startswith("image/"):
            mime = "image/png"
        with open(ref, "rb") as f:
            raw = f.read()
        return mime, base64.b64encode(raw).decode("ascii")


def _messages_to_acp_prompt(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert chat messages (system/user/assistant + multimodal images) to native ACP blocks."""
    system_parts: List[str] = []
    conversation_blocks: List[Dict[str, Any]] = []

    non_system = [m for m in messages if m.get("role") != "system"]
    has_multi_turn = len(non_system) > 1

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            if isinstance(content, str):
                system_parts.append(content.strip())
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        system_parts.append(str(item.get("text", "")).strip())
            continue

        if has_multi_turn:
            label = "User:" if role == "user" else "Assistant:"
            conversation_blocks.append({"type": "text", "text": label})

        if isinstance(content, str):
            conversation_blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    conversation_blocks.append({"type": "text", "text": item})
                elif isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "text":
                        conversation_blocks.append(
                            {"type": "text", "text": str(item.get("text", ""))}
                        )
                    elif item_type == "image_url":
                        url_obj = item.get("image_url", {})
                        url = url_obj.get("url", "") if isinstance(url_obj, dict) else str(url_obj)
                        mime, b64_data = _load_image_as_base64(url)
                        conversation_blocks.append(
                            {"type": "image", "mimeType": mime, "data": b64_data}
                        )
                    elif item_type == "image":
                        if "data" in item:
                            conversation_blocks.append(
                                {
                                    "type": "image",
                                    "mimeType": item.get("mimeType", "image/png"),
                                    "data": item["data"],
                                }
                            )
                        elif "path" in item:
                            mime, b64_data = _load_image_as_base64(item["path"])
                            conversation_blocks.append(
                                {"type": "image", "mimeType": mime, "data": b64_data}
                            )

    prefix_lines = [
        "Direct-response mode: Answer strictly based on the text and images provided below. "
        "Do NOT call any tools, do NOT run commands, and do NOT read external files. "
        "Follow the exact output format requested."
    ]
    if system_parts:
        prefix_lines.append("System Instructions:\n" + "\n".join(system_parts))

    prompt_blocks: List[Dict[str, Any]] = [
        {"type": "text", "text": "\n\n".join(prefix_lines)}
    ]
    prompt_blocks.extend(conversation_blocks)
    if has_multi_turn:
        prompt_blocks.append({"type": "text", "text": "Assistant:"})
    return prompt_blocks


@dataclass
class _Message:
    content: str
    role: str = "assistant"


@dataclass
class _Choice:
    message: _Message
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class _ChatCompletionResponse:
    choices: List[_Choice]
    model: str
    usage: Optional[Dict[str, Any]] = None


class _CompletionsNamespace:
    def __init__(self, client: "AntigravityACPClient") -> None:
        self._client = client

    def create(
        self,
        *,
        model: Optional[str] = None,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> _ChatCompletionResponse:
        del temperature, seed, kwargs
        resolved_model = resolve_antigravity_model(model=model) if model else self._client.default_model
        text, usage = self._client.generate_from_messages(messages, model=resolved_model)
        return _ChatCompletionResponse(
            choices=[_Choice(message=_Message(content=text))],
            model=resolved_model,
            usage=usage,
        )


class _ChatNamespace:
    def __init__(self, client: "AntigravityACPClient") -> None:
        self.completions = _CompletionsNamespace(client)


class AntigravityACPClient:
    """Persistent native Antigravity ACP (`agy acp --stdio`) multimodal client."""

    def __init__(
        self,
        default_model: Optional[str] = None,
        agy_bin: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> None:
        self.default_model = resolve_antigravity_model(model=default_model)
        self.agy_bin = agy_bin or os.getenv("AGY_BIN", "agy")
        self.cwd = cwd or os.getcwd()
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._req_id = 0
        self.chat = _ChatNamespace(self)
        atexit.register(self.close)

    def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return

        if not shutil.which(self.agy_bin):
            raise RuntimeError(
                f"Antigravity CLI '{self.agy_bin}' not found in PATH. "
                "Install/login to Antigravity CLI (`agy`) or set AGY_BIN."
            )

        self._proc = subprocess.Popen(
            [self.agy_bin, "acp", "--stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._req_id = 0
        init_res, _ = self._rpc(
            "initialize",
            {"protocolVersion": 1, "clientCapabilities": {}},
        )
        if "error" in init_res:
            raise RuntimeError(f"Antigravity ACP initialize failed: {init_res['error']}")

    def _send_raw(self, payload: Dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

    def _rpc(self, method: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        assert self._proc is not None and self._proc.stdout is not None
        self._req_id += 1
        rid = self._req_id
        self._send_raw(
            {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
        )

        chunks: List[str] = []
        while True:
            line = self._proc.stdout.readline()
            if not line:
                err_out = ""
                if self._proc.stderr is not None:
                    try:
                        err_out = self._proc.stderr.read()
                    except Exception:
                        pass
                self.close()
                raise RuntimeError(f"Antigravity ACP process exited unexpectedly. stderr={err_out}")

            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Handle server-to-client permission requests defensively by denying tool execution
            if "id" in msg and "method" in msg:
                req_method = msg.get("method", "")
                if req_method == "session/request_permission":
                    options = msg.get("params", {}).get("options", [])
                    reject_id = None
                    for opt in options:
                        if "reject" in str(opt.get("kind", "")).lower():
                            reject_id = opt.get("optionId")
                            break
                    outcome = (
                        {"outcome": "selected", "optionId": reject_id}
                        if reject_id is not None
                        else {"outcome": "cancelled"}
                    )
                    self._send_raw(
                        {"jsonrpc": "2.0", "id": msg["id"], "result": {"outcome": outcome}}
                    )
                else:
                    self._send_raw(
                        {
                            "jsonrpc": "2.0",
                            "id": msg["id"],
                            "error": {"code": -32601, "message": f"Unsupported client method: {req_method}"},
                        }
                    )
                continue

            if msg.get("id") == rid:
                return msg, "".join(chunks)

            if "method" in msg:
                upd = msg.get("params", {}).get("update", {})
                if upd.get("sessionUpdate") == "agent_message_chunk":
                    content = upd.get("content", {})
                    if content.get("type") == "text":
                        chunks.append(content.get("text", ""))

    def generate_from_messages(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Run a stateless multimodal prompt over a fresh ACP session."""
        target_model = resolve_antigravity_model(model=model) if model else self.default_model
        prompt_blocks = _messages_to_acp_prompt(messages)

        with self._lock:
            self._ensure_started()
            sess_res, _ = self._rpc(
                "session/new",
                {"cwd": self.cwd, "mcpServers": []},
            )
            if "error" in sess_res:
                raise RuntimeError(f"Antigravity ACP session/new failed: {sess_res['error']}")
            session_id = sess_res["result"]["sessionId"]

            try:
                set_res, _ = self._rpc(
                    "session/set_model",
                    {"sessionId": session_id, "modelId": target_model},
                )
                if "error" in set_res:
                    raise RuntimeError(
                        f"Antigravity ACP session/set_model({target_model}) failed: {set_res['error']}"
                    )

                prompt_res, text = self._rpc(
                    "session/prompt",
                    {"sessionId": session_id, "prompt": prompt_blocks},
                )
                if "error" in prompt_res:
                    raise RuntimeError(
                        f"Antigravity ACP session/prompt failed: {prompt_res['error']}"
                    )
                usage = prompt_res.get("result", {}).get("usage")
                # Strip transient ACP retry banner lines (e.g. "⚠ The stream was interrupted...")
                clean_lines = [
                    ln for ln in text.splitlines()
                    if not ln.strip().startswith("⚠")
                ]
                return "\n".join(clean_lines).strip(), usage
            finally:
                try:
                    self._rpc("session/close", {"sessionId": session_id})
                except Exception:
                    pass

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


_SINGLETON_LOCK = threading.Lock()
_SINGLETON_CLIENT: Optional[AntigravityACPClient] = None


def get_antigravity_client(default_model: Optional[str] = None) -> AntigravityACPClient:
    global _SINGLETON_CLIENT
    resolved = resolve_antigravity_model(model=default_model)
    with _SINGLETON_LOCK:
        if _SINGLETON_CLIENT is None:
            _SINGLETON_CLIENT = AntigravityACPClient(default_model=resolved)
        else:
            _SINGLETON_CLIENT.default_model = resolved
        return _SINGLETON_CLIENT


def create_llm_client(cfg: Any = None) -> Tuple[Any, str]:
    """Create the configured LLM/VLM client.

    Defaults to native Antigravity ACP (`gemini-4-neon-high`, configurable to
    `gemini-4-neon-xhigh` via `models.llm.effort=extra` / `ANTIGRAVITY_EFFORT=extra`
    or `models.llm.model=gemini-4-neon-xhigh` / `ANTIGRAVITY_MODEL=gemini-4-neon-xhigh`).
    """
    backend = resolve_llm_backend(cfg)

    if backend == "antigravity":
        model_id = resolve_antigravity_model(cfg=cfg)
        client = get_antigravity_client(default_model=model_id)
        print(f"[LLM] Using native Antigravity ACP backend with model: {model_id}")
        return client, model_id

    if backend == "azure":
        from openai import AzureOpenAI

        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
        if not (azure_api_key and azure_endpoint and azure_deployment):
            raise RuntimeError(
                "Azure backend requires AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT."
            )
        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        )
        return client, azure_deployment

    if backend == "openai":
        from openai import OpenAI

        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_api_key:
            raise RuntimeError("OpenAI backend requires OPENAI_API_KEY.")
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
        model = os.getenv("OPENAI_MODEL", "gpt-4o").strip() or "gpt-4o"
        client = OpenAI(api_key=openai_api_key, base_url=base_url)
        return client, model

    raise RuntimeError(
        f"Unsupported or unavailable LLM backend '{backend}'. "
        "Use `agy` (Antigravity native) or set FSR_VLN_LLM_BACKEND=openai|azure."
    )
