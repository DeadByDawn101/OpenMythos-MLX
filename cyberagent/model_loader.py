"""
Universal Model Loader — Load any RavenX agent model from any runtime.

Backends (auto-detected):
  1. MLX native     — Apple Silicon, safetensors, fastest for Mac
  2. GGUF/Ollama    — ollama serve, any hardware, broadest compatibility
  3. GGUF/llama.cpp — llama-server or llama-cli, raw GGUF on disk
  4. OpenAI API     — Any /v1/chat/completions endpoint (mlx_lm, vLLM, TGI, OpenRouter)
  5. Agent mode     — Direct tool-calling with MCP broker wired in

This is the piece that makes CyberAgent harness-agnostic.
Same harness code, different model.load() call.

RavenX AI Labs LLC — September 2026
"""

import json
import os
import subprocess
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger("cyberagent.loader")


# ─────────────────────────────────────────────────────────
# Abstract Backend
# ─────────────────────────────────────────────────────────

class ModelBackend(ABC):
    """Abstract model backend — all runtimes implement this."""

    @abstractmethod
    def generate(self, system: str, prompt: str,
                 max_tokens: int = 2048,
                 temperature: float = 0.1) -> str:
        """Generate a completion from system + user prompt."""
        pass

    @abstractmethod
    def generate_with_tools(self, system: str, prompt: str,
                            tools: List[Dict],
                            max_tokens: int = 2048) -> Dict:
        """Generate with tool-calling support (agent mode)."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


# ─────────────────────────────────────────────────────────
# Backend: MLX Native
# ─────────────────────────────────────────────────────────

class MLXBackend(ModelBackend):
    """Apple Silicon native — loads safetensors directly via mlx-lm."""

    def __init__(self, model_id: str):
        from mlx_lm import load, generate
        self._generate = generate
        self.model_id = model_id
        logger.info(f"[MLX] Loading {model_id}...")
        self.model, self.tokenizer = load(model_id)
        logger.info(f"[MLX] Ready")

    @property
    def name(self) -> str:
        return f"mlx:{self.model_id}"

    def generate(self, system: str, prompt: str,
                 max_tokens: int = 2048, temperature: float = 0.1) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        full_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self._generate(
            self.model, self.tokenizer,
            prompt=full_prompt, max_tokens=max_tokens,
            temp=temperature
        )

    def generate_with_tools(self, system: str, prompt: str,
                            tools: List[Dict], max_tokens: int = 2048) -> Dict:
        # MLX doesn't natively support tool calling — simulate via prompt injection
        tool_desc = json.dumps(tools, indent=2)
        augmented_system = f"{system}\n\nAvailable tools:\n{tool_desc}\n\nTo call a tool, output: {{\"tool\": \"name\", \"args\": {{...}}}}"
        output = self.generate(augmented_system, prompt, max_tokens)
        # Try to parse tool call from output
        try:
            tool_call = json.loads(output)
            return {"type": "tool_call", "content": tool_call}
        except json.JSONDecodeError:
            return {"type": "text", "content": output}


# ─────────────────────────────────────────────────────────
# Backend: Ollama
# ─────────────────────────────────────────────────────────

class OllamaBackend(ModelBackend):
    """Ollama — works with GGUF models via ollama serve."""

    def __init__(self, model_name: str = "ravenx-cyberagent",
                 host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host
        # Verify ollama is running
        try:
            import urllib.request
            urllib.request.urlopen(f"{host}/api/tags", timeout=5)
            logger.info(f"[Ollama] Connected to {host}")
        except Exception:
            logger.warning(f"[Ollama] Server not reachable at {host}")
            logger.warning(f"         Start with: ollama serve")

    @property
    def name(self) -> str:
        return f"ollama:{self.model_name}"

    def generate(self, system: str, prompt: str,
                 max_tokens: int = 2048, temperature: float = 0.1) -> str:
        import urllib.request
        payload = json.dumps({
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }).encode()

        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(req, timeout=300)
        data = json.loads(resp.read())
        return data.get("message", {}).get("content", "")

    def generate_with_tools(self, system: str, prompt: str,
                            tools: List[Dict], max_tokens: int = 2048) -> Dict:
        import urllib.request
        payload = json.dumps({
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "tools": tools,
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(req, timeout=300)
        data = json.loads(resp.read())
        msg = data.get("message", {})
        if msg.get("tool_calls"):
            return {"type": "tool_call", "content": msg["tool_calls"]}
        return {"type": "text", "content": msg.get("content", "")}


# ─────────────────────────────────────────────────────────
# Backend: OpenAI-compatible API
# ─────────────────────────────────────────────────────────

class OpenAIBackend(ModelBackend):
    """Any OpenAI-compatible /v1/chat/completions endpoint.

    Works with: mlx_lm.server, vLLM, TGI, OpenRouter, LM Studio,
    llama.cpp --server, and actual OpenAI.
    """

    def __init__(self, endpoint: str = "http://localhost:8080",
                 model: str = "default",
                 api_key: Optional[str] = None):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        logger.info(f"[API] Endpoint: {self.endpoint}")

    @property
    def name(self) -> str:
        return f"api:{self.endpoint}"

    def _call(self, payload: dict) -> dict:
        import urllib.request
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            f"{self.endpoint}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers
        )
        resp = urllib.request.urlopen(req, timeout=300)
        return json.loads(resp.read())

    def generate(self, system: str, prompt: str,
                 max_tokens: int = 2048, temperature: float = 0.1) -> str:
        data = self._call({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        return data["choices"][0]["message"]["content"]

    def generate_with_tools(self, system: str, prompt: str,
                            tools: List[Dict], max_tokens: int = 2048) -> Dict:
        data = self._call({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "tools": tools,
            "max_tokens": max_tokens,
        })
        msg = data["choices"][0]["message"]
        if msg.get("tool_calls"):
            return {"type": "tool_call", "content": msg["tool_calls"]}
        return {"type": "text", "content": msg.get("content", "")}


# ─────────────────────────────────────────────────────────
# Backend: OpenMythos Native RDT
# ─────────────────────────────────────────────────────────

class OpenMythosBackend(ModelBackend):
    """Native OpenMythos with RDT depth looping."""

    def __init__(self, config=None, weights: Optional[str] = None):
        import mlx.core as mx
        from open_mythos_mlx import OpenMythos, mythos_1b
        self.config = config or mythos_1b()
        self.model = OpenMythos(self.config)
        if weights:
            self.model.load_weights(weights)
            mx.eval(self.model.parameters())
        total = sum(p.size for p in self.model.parameters().values())
        logger.info(f"[OpenMythos] {total:,} params loaded")

    @property
    def name(self) -> str:
        return "openmythos:native"

    def generate(self, system: str, prompt: str,
                 max_tokens: int = 2048, temperature: float = 0.1) -> str:
        # OpenMythos native requires tokenizer — return prompt for external processing
        return f"[OpenMythos depth analysis requested]\nSystem: {system}\nPrompt: {prompt}"

    def generate_with_tools(self, system: str, prompt: str,
                            tools: List[Dict], max_tokens: int = 2048) -> Dict:
        return {"type": "text", "content": self.generate(system, prompt, max_tokens)}

    def depth_forward(self, tokens, n_loops: int = 8):
        """Raw depth-extrapolated forward pass — for research/training."""
        import mlx.core as mx
        token_array = mx.array(tokens).reshape(1, -1)
        return self.model(token_array, n_loops=n_loops)


# ─────────────────────────────────────────────────────────
# RATH Protocol System Prompt
# ─────────────────────────────────────────────────────────

RATH_SYSTEM = """You are RavenX-CyberAgent, an autonomous security assessment agent.

Execute the RATH protocol for EVERY finding:
1. ATTACK SURFACE — Map all inputs, outputs, trust boundaries
2. EXPLOIT — Construct proof-of-concept with exact payload
3. IMPACT — Calculate CVSS score, assign CWE ID, map MITRE ATT&CK TTP
4. REMEDIATION — Provide specific code-level fix (not generic advice)
5. DOCUMENT — Output in structured format: [SEVERITY] CVSS X.X — Name
6. PREVENT — Recommend detection rules (YARA/Sigma/Snort)

Rules:
- ALWAYS include CVSS, CWE, and MITRE ATT&CK references
- Be clinical, precise, and paranoid
- Trust is binary: if input is untrusted, treat it as hostile
- Never execute live exploits without explicit HITL gate approval
- Use tool_call for command execution when available
"""

# ─────────────────────────────────────────────────────────
# RATH Tool Definitions (for agent-mode tool calling)
# ─────────────────────────────────────────────────────────

RATH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_semgrep",
            "description": "Run Semgrep static analysis on a file or directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "File or directory path"},
                    "config": {"type": "string", "description": "Semgrep config (default: auto)", "default": "auto"},
                },
                "required": ["target"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_nmap",
            "description": "Run Nmap port scan on a target host",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "IP address or hostname"},
                    "ports": {"type": "string", "description": "Port range (default: top 1000)", "default": ""},
                },
                "required": ["target"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_nuclei",
            "description": "Run Nuclei vulnerability scanner with templates",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "URL to scan"},
                    "severity": {"type": "string", "description": "Filter by severity", "default": "critical,high"},
                },
                "required": ["target"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file for analysis",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_patch",
            "description": "Write a remediation patch to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Output file path"},
                    "content": {"type": "string", "description": "Patch content"},
                },
                "required": ["path", "content"]
            }
        }
    },
]


# ─────────────────────────────────────────────────────────
# Universal Loader
# ─────────────────────────────────────────────────────────

RAVENX_MODELS = {
    "cyberagent-gguf": "deadbydawn101/RavenX-CyberAgent-Qwen3.6-35B-A3B-Opus-4.7-OpenMythos-Pentester-BugHunter-RATH-GGUF",
    "cyberagent-mlx": "deadbydawn101/RavenX-CyberAgent-Qwen3.6-35B-A3B-Opus-4.7-OpenMythos-Pentester-BugHunter-RATH-mlx",
    "chaos-agent": "deadbydawn101/RavenXAiLabs-Chaos-Agent-Qwen3.8-27B-Frontier-Intelligence-Injected-OBLITERATED-MLX",
}


def load_model(backend: str = "auto",
               model: Optional[str] = None,
               endpoint: Optional[str] = None,
               api_key: Optional[str] = None,
               weights: Optional[str] = None) -> ModelBackend:
    """Universal model loader — auto-detects the best backend.

    Args:
        backend: "auto", "mlx", "ollama", "api", "openmythos"
        model: Model ID (HF repo, ollama name, or RAVENX_MODELS key)
        endpoint: API endpoint URL
        api_key: API key for authenticated endpoints

    Returns:
        ModelBackend ready for generate() / generate_with_tools()

    Examples:
        # Auto-detect: tries MLX first, then Ollama, then API
        backend = load_model(model="cyberagent-mlx")

        # Explicit Ollama
        backend = load_model(backend="ollama", model="ravenx-cyberagent")

        # Any API endpoint
        backend = load_model(backend="api", endpoint="http://localhost:8080")

        # OpenAI
        backend = load_model(backend="api", endpoint="https://api.openai.com",
                             model="gpt-4o", api_key="sk-...")

        # Native OpenMythos
        backend = load_model(backend="openmythos", weights="model.safetensors")
    """
    # Resolve RavenX model aliases
    if model in RAVENX_MODELS:
        model = RAVENX_MODELS[model]

    if backend == "auto":
        return _auto_detect(model, endpoint, api_key)

    if backend == "mlx":
        return MLXBackend(model or "cyberagent-mlx")
    elif backend == "ollama":
        return OllamaBackend(model or "ravenx-cyberagent")
    elif backend == "api":
        return OpenAIBackend(
            endpoint=endpoint or "http://localhost:8080",
            model=model or "default",
            api_key=api_key
        )
    elif backend == "openmythos":
        return OpenMythosBackend(weights=weights)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _auto_detect(model: Optional[str], endpoint: Optional[str],
                 api_key: Optional[str]) -> ModelBackend:
    """Try backends in order: MLX → Ollama → API."""

    # 1. Try MLX (best for Apple Silicon)
    if model and ("mlx" in model.lower() or "safetensors" in str(model)):
        try:
            return MLXBackend(model)
        except Exception as e:
            logger.info(f"[Auto] MLX failed: {e}")

    # 2. Try Ollama
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return OllamaBackend(model or "ravenx-cyberagent")
    except Exception:
        logger.info("[Auto] Ollama not running")

    # 3. Try API
    ep = endpoint or "http://localhost:8080"
    try:
        import urllib.request
        urllib.request.urlopen(f"{ep}/v1/models", timeout=2)
        return OpenAIBackend(endpoint=ep, model=model or "default", api_key=api_key)
    except Exception:
        logger.info(f"[Auto] API not available at {ep}")

    raise RuntimeError(
        "No model backend available.\n"
        "  MLX:    pip install mlx-lm && provide --model <hf-repo>\n"
        "  Ollama: ollama serve && ollama pull ravenx-cyberagent\n"
        "  API:    python -m mlx_lm.server --model <hf-repo>"
    )
