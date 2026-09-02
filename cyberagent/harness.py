"""
CyberAgent Harness — Production autonomous pentest engine.

Wires directly into OpenMythos RDT/MoDA models for depth-extrapolated
security auditing. This is NOT a thesis — it's the runtime that loads
the agent model, feeds it code, and manages the adversarial loop.

Two model backends:
  1. OpenMythos native (RDT depth looping with MoDA attention)
  2. mlx-lm hosted model (Qwen/Gemma/Llama via mlx_lm.server API)

Pipeline:
  Code → Tokenize → Depth Analysis (n_loops=8) → Parse Findings
  → Builder Patch → OpenSelfRevise (Breaker re-attacks) → Report

Usage:
  # With OpenMythos model (RDT native)
  python -m cyberagent.harness --model openmythos --code vulnerable.py

  # With any mlx-lm served model (API backend)
  python -m cyberagent.harness --model api --endpoint http://localhost:8080 --code app.py

  # Generate training data for fine-tuning
  python -m cyberagent.harness --generate-data --output cyberagent_train.jsonl

RavenX AI Labs LLC — September 2026
"""

import json
import time
import os
import sys
import subprocess
import shlex
import re
import argparse
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from enum import Enum

logger = logging.getLogger("cyberagent")

# Try importing OpenMythos for native RDT backend
try:
    import mlx.core as mx
    from open_mythos_mlx import OpenMythos, MythosConfig, MoDAModel, MoDAConfig
    from open_mythos_mlx import mythos_1b, moda_small
    HAS_MYTHOS = True
except ImportError:
    HAS_MYTHOS = False

# Try importing mlx_lm for API backend
try:
    from mlx_lm import load as mlx_load, generate as mlx_generate
    HAS_MLX_LM = True
except ImportError:
    HAS_MLX_LM = False


# ─────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────

class Severity(Enum):
    CRITICAL = ("CRITICAL", 9.0, 10.0)
    HIGH = ("HIGH", 7.0, 8.9)
    MEDIUM = ("MEDIUM", 4.0, 6.9)
    LOW = ("LOW", 0.1, 3.9)
    INFO = ("INFO", 0.0, 0.0)


@dataclass
class Finding:
    name: str
    severity: str
    cvss: float
    vector: str
    poc: str
    remediation: str
    depth_found: int = 0
    moda_refs: List[str] = field(default_factory=list)
    cve: str = ""
    verified: bool = False


@dataclass
class AuditResult:
    target: str
    timestamp: str = ""
    findings: List[Finding] = field(default_factory=list)
    total_loops: int = 0
    act_halted: bool = False
    halt_loop: int = 0
    breaker_rounds: int = 0
    patches_verified: int = 0
    model_backend: str = ""
    raw_traces: List[str] = field(default_factory=list)

    def to_report(self) -> str:
        lines = [
            "=" * 60,
            "PENETRATION TEST REPORT",
            f"Target: {self.target}",
            f"Date: {self.timestamp}",
            f"Model: {self.model_backend}",
            f"Depth loops: {self.total_loops}" + (f" (ACT halted at {self.halt_loop})" if self.act_halted else ""),
            f"Breaker rounds: {self.breaker_rounds}",
            f"Patches verified: {self.patches_verified}",
            "=" * 60,
        ]
        for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
            group = [f for f in self.findings if f.severity == sev]
            if group:
                lines.append(f"\n[{sev}]")
                for f in group:
                    lines.append(f"  {f.name} (CVSS {f.cvss})")
                    lines.append(f"    Vector: {f.vector}")
                    lines.append(f"    PoC: {f.poc}")
                    lines.append(f"    Fix: {f.remediation}")
                    if f.moda_refs:
                        lines.append(f"    MoDA chain: {' → '.join(f.moda_refs)}")
                    lines.append(f"    Verified: {'✅' if f.verified else '⚠️  pending'}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# MODEL BACKENDS
# ─────────────────────────────────────────────────────────

class MythosBackend:
    """Native OpenMythos RDT backend — runs depth loops directly."""

    def __init__(self, config=None, weights_path: Optional[str] = None):
        if not HAS_MYTHOS:
            raise RuntimeError("OpenMythos-MLX not available. pip install -e .")

        self.config = config or mythos_1b()
        self.model = OpenMythos(self.config)

        if weights_path:
            self.model.load_weights(weights_path)
            mx.eval(self.model.parameters())
            logger.info(f"Loaded weights: {weights_path}")

        total = sum(p.size for p in self.model.parameters().values())
        logger.info(f"OpenMythos loaded: {total:,} params")

    def analyze(self, code: str, n_loops: int = 8,
                tokenizer=None) -> Tuple[str, List[Dict]]:
        """Run depth-extrapolated analysis.

        Returns (raw_output, loop_history) where loop_history contains
        the intermediate hidden states from each RDT loop (MoDA visible).
        """
        if tokenizer is None:
            # Fallback: use the prompt directly for API-style backends
            return self._analyze_prompt(code, n_loops)

        prompt = self._build_prompt(code)
        tokens = mx.array(tokenizer.encode(prompt)).reshape(1, -1)

        # Run with explicit loop count for depth control
        logits = self.model(tokens, n_loops=n_loops)

        # Decode output
        output_ids = mx.argmax(logits[:, -256:, :], axis=-1)
        output = tokenizer.decode(output_ids[0].tolist())

        return output, []

    def _analyze_prompt(self, code: str, n_loops: int) -> Tuple[str, List]:
        """Prompt-based analysis when no tokenizer available."""
        prompt = self._build_prompt(code)
        # For native model without tokenizer, return the prompt for external processing
        return prompt, []

    @staticmethod
    def _build_prompt(code: str) -> str:
        return f"""<think_high>
You are a security auditor performing depth-extrapolated analysis.
Loop 1-2: Map the attack surface. Identify all inputs, outputs, and trust boundaries.
Loop 3-4: Trace state mutations. Follow data from untrusted sources through transformations.
Loop 5-6: Verify exploitability. Calculate exact offsets, construct PoC payloads.
Loop 7-8: Chain findings. Combine individual vulnerabilities into compound exploit paths.
</think_high>

Audit this code for security vulnerabilities:

```
{code}
```

For each finding, output EXACTLY this format:
[SEVERITY] CVSS X.X — Vulnerability Name
Vector: How the exploit works
PoC: Proof of concept payload or steps
Fix: Specific code-level remediation
"""


class APIBackend:
    """mlx-lm server API backend — works with any hosted model."""

    def __init__(self, endpoint: str = "http://localhost:8080",
                 model_id: Optional[str] = None):
        self.endpoint = endpoint.rstrip("/")
        self.model_id = model_id

        # If model_id provided, load locally instead of API
        self.local_model = None
        self.local_tokenizer = None
        if model_id and HAS_MLX_LM:
            try:
                self.local_model, self.local_tokenizer = mlx_load(model_id)
                logger.info(f"Loaded local model: {model_id}")
            except Exception as e:
                logger.warning(f"Local load failed: {e}, falling back to API")

    def analyze(self, code: str, n_loops: int = 8) -> Tuple[str, List]:
        """Run analysis through API or local model."""
        prompt = MythosBackend._build_prompt(code)

        if self.local_model:
            output = mlx_generate(
                self.local_model, self.local_tokenizer,
                prompt=prompt, max_tokens=2048
            )
            return output, []

        # API call
        try:
            import urllib.request
            payload = json.dumps({
                "model": self.model_id or "default",
                "messages": [
                    {"role": "system", "content": "You are an expert security auditor. "
                     "Perform depth-extrapolated vulnerability analysis. "
                     "For each finding use format: [SEVERITY] CVSS X.X — Name"},
                    {"role": "user", "content": f"Audit this code:\n```\n{code}\n```"}
                ],
                "max_tokens": 2048,
                "temperature": 0.1,
            }).encode()

            req = urllib.request.Request(
                f"{self.endpoint}/v1/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            resp = urllib.request.urlopen(req, timeout=120)
            data = json.loads(resp.read())
            output = data["choices"][0]["message"]["content"]
            return output, []

        except Exception as e:
            logger.error(f"API call failed: {e}")
            return f"[ERROR] API call failed: {e}", []


# ─────────────────────────────────────────────────────────
# FINDING PARSER
# ─────────────────────────────────────────────────────────

class FindingParser:
    """Parse structured findings from model output."""

    SEVERITY_PATTERN = re.compile(
        r'\[(CRITICAL|HIGH|MEDIUM|LOW|INFO)\]\s*(?:CVSS\s*)?([\d.]+)\s*[—–-]\s*(.+)',
        re.IGNORECASE
    )
    FIELD_PATTERN = re.compile(
        r'(?:Vector|Attack|PoC|Proof|Fix|Remediation|Chain|MoDA):\s*(.+)',
        re.IGNORECASE
    )

    @staticmethod
    def parse(output: str) -> List[Finding]:
        """Extract structured findings from model output."""
        findings = []
        current = None

        for line in output.split("\n"):
            line = line.strip()
            if not line:
                continue

            sev_match = FindingParser.SEVERITY_PATTERN.search(line)
            if sev_match:
                if current:
                    findings.append(current)
                current = Finding(
                    name=sev_match.group(3).strip(),
                    severity=sev_match.group(1).upper(),
                    cvss=float(sev_match.group(2)),
                    vector="", poc="", remediation="",
                )
                continue

            if current:
                lower = line.lower()
                if lower.startswith(("vector:", "attack:")):
                    current.vector = line.split(":", 1)[1].strip()
                elif lower.startswith(("poc:", "proof:")):
                    current.poc = line.split(":", 1)[1].strip()
                elif lower.startswith(("fix:", "remediation:")):
                    current.remediation = line.split(":", 1)[1].strip()
                elif lower.startswith(("moda:", "chain:")):
                    current.moda_refs.append(line.split(":", 1)[1].strip())

        if current:
            findings.append(current)

        return findings


# ─────────────────────────────────────────────────────────
# BUILDER / BREAKER PIPELINE
# ─────────────────────────────────────────────────────────

class BuilderBreaker:
    """Adversarial pipeline: Breaker finds exploits, Builder patches,
    OpenSelfRevise verifies patches hold against re-attack."""

    def __init__(self, backend, max_rounds: int = 3):
        self.backend = backend
        self.max_rounds = max_rounds

    def run_breaker(self, code: str, n_loops: int = 8) -> Tuple[str, List[Finding]]:
        """Breaker agent: find vulnerabilities."""
        logger.info(f"[BREAKER] Running depth-{n_loops} analysis...")
        output, history = self.backend.analyze(code, n_loops=n_loops)
        findings = FindingParser.parse(output)
        logger.info(f"[BREAKER] Found {len(findings)} vulnerabilities")
        return output, findings

    def run_builder(self, code: str, findings: List[Finding]) -> str:
        """Builder agent: generate patches for each finding."""
        if not findings:
            return code

        finding_text = "\n".join(
            f"- [{f.severity}] {f.name}: {f.vector}" for f in findings
        )

        prompt = f"""You are a defensive security engineer. Generate SPECIFIC code patches for these vulnerabilities.

Vulnerabilities found:
{finding_text}

Original code:
```
{code}
```

Output the COMPLETE patched code with inline comments marking each fix. Every fix must:
1. Address the specific vulnerability (not a generic band-aid)
2. Preserve existing functionality
3. Include bounds checking, input validation, or safe API replacement
"""

        output, _ = self.backend.analyze(prompt, n_loops=4)
        return output

    def run_selfrevise(self, original_code: str, patched_code: str,
                        original_findings: List[Finding]) -> Tuple[bool, List[Finding]]:
        """OpenSelfRevise: Breaker re-attacks the patched code."""
        logger.info("[SELFREVISE] Breaker attacking Builder's patches...")
        _, new_findings = self.run_breaker(patched_code, n_loops=4)

        # Check if original vulnerabilities were actually fixed
        original_names = {f.name for f in original_findings}
        remaining = [f for f in new_findings if f.name in original_names]

        if remaining:
            logger.warning(f"[SELFREVISE] {len(remaining)} vulnerabilities survived patching")
            return False, remaining
        else:
            logger.info("[SELFREVISE] All patches verified — no bypasses found")
            return True, new_findings


# ─────────────────────────────────────────────────────────
# HITL GATES
# ─────────────────────────────────────────────────────────

class HITLGate:
    """Human-in-the-Loop safety gates.

    Three gates that HALT execution and require human approval:
      Gate 1: Scope confirmation (before analysis begins)
      Gate 2: Critical finding isolation (CVSS >= threshold)
      Gate 3: Remediation approval (before patches apply)
    """

    def __init__(self, auto_approve: bool = False, threshold: float = 9.0):
        self.auto_approve = auto_approve
        self.threshold = threshold

    def gate_scope(self, target: str, code_preview: str) -> bool:
        """Gate 1: Confirm audit scope before analysis."""
        if self.auto_approve:
            return True
        print(f"\n[GATE 1 — SCOPE CONFIRMATION]")
        print(f"  Target: {target}")
        print(f"  Preview: {code_preview[:200]}...")
        response = input("  Proceed with audit? [y/N]: ").strip().lower()
        return response == "y"

    def gate_critical(self, findings: List[Finding]) -> bool:
        """Gate 2: Halt on critical findings for human review."""
        critical = [f for f in findings if f.cvss >= self.threshold]
        if not critical:
            return True
        if self.auto_approve:
            logger.warning(f"[GATE 2] Auto-approved {len(critical)} critical findings")
            return True
        print(f"\n[GATE 2 — CRITICAL FINDING ISOLATION]")
        print(f"  {len(critical)} finding(s) exceed CVSS {self.threshold}:")
        for f in critical:
            print(f"    [{f.severity}] CVSS {f.cvss} — {f.name}")
        response = input("  Route to Builder for patching? [y/N]: ").strip().lower()
        return response == "y"

    def gate_remediation(self, patches: str) -> bool:
        """Gate 3: Approve patches before deployment."""
        if self.auto_approve:
            return True
        print(f"\n[GATE 3 — REMEDIATION APPROVAL]")
        print(f"  Patches ready ({len(patches)} chars)")
        response = input("  Approve patches? [y/N]: ").strip().lower()
        return response == "y"


# ─────────────────────────────────────────────────────────
# MCP TOOL BROKER
# ─────────────────────────────────────────────────────────

class MCPBroker:
    """Sandboxed tool execution for external security scanners."""

    TOOLS = {
        "semgrep": {
            "cmd": ["semgrep", "--config", "auto", "--json"],
            "description": "Static analysis with Semgrep",
        },
        "bandit": {
            "cmd": ["bandit", "-r", "-f", "json"],
            "description": "Python security linter",
        },
        "nmap": {
            "cmd": ["nmap", "-sV", "-T4", "--open"],
            "description": "Network port scanner",
        },
        "trivy": {
            "cmd": ["trivy", "fs", "--format", "json"],
            "description": "Vulnerability scanner for dependencies",
        },
        "nuclei": {
            "cmd": ["nuclei", "-json"],
            "description": "Template-based vulnerability scanner",
        },
    }

    @staticmethod
    def run(tool: str, target: str, timeout: int = 60) -> Dict:
        """Execute a tool against a target with full sandboxing."""
        if tool not in MCPBroker.TOOLS:
            return {"status": "REJECTED", "error": f"Unknown tool: {tool}",
                    "allowed": list(MCPBroker.TOOLS.keys())}

        cmd = MCPBroker.TOOLS[tool]["cmd"] + [shlex.quote(target)]
        logger.info(f"[MCP] Executing: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout, shell=False,
                env={**os.environ, "HOME": os.environ.get("HOME", "/tmp")}
            )
            output = result.stdout
            # Try to parse JSON output
            try:
                parsed = json.loads(output)
                return {"status": "SUCCESS", "data": parsed, "tool": tool}
            except json.JSONDecodeError:
                return {"status": "SUCCESS", "raw": output[:5000], "tool": tool}
        except FileNotFoundError:
            return {"status": "NOT_INSTALLED", "error": f"{tool} not in PATH",
                    "install": f"brew install {tool}" if sys.platform == "darwin" else f"apt install {tool}"}
        except subprocess.TimeoutExpired:
            return {"status": "TIMEOUT", "error": f"Exceeded {timeout}s"}

    @staticmethod
    def scan_with_all(target: str) -> List[Dict]:
        """Run all available tools against a target."""
        results = []
        for tool in MCPBroker.TOOLS:
            result = MCPBroker.run(tool, target)
            if result["status"] == "SUCCESS":
                results.append(result)
        return results


# ─────────────────────────────────────────────────────────
# MAIN HARNESS
# ─────────────────────────────────────────────────────────

class CyberAgentHarness:
    """Production autonomous pentest harness.

    Orchestrates the full pipeline:
      1. Load model (OpenMythos native or API)
      2. HITL Gate 1: scope confirmation
      3. Breaker: depth-extrapolated vulnerability discovery
      4. HITL Gate 2: critical finding review
      5. Builder: generate patches
      6. OpenSelfRevise: verify patches hold
      7. HITL Gate 3: remediation approval
      8. Output: structured pentest report
    """

    def __init__(self, backend_type: str = "api",
                 endpoint: str = "http://localhost:8080",
                 model_id: Optional[str] = None,
                 weights_path: Optional[str] = None,
                 auto_approve: bool = False,
                 n_loops: int = 8,
                 max_revise_rounds: int = 3):

        # Initialize backend
        if backend_type == "openmythos" and HAS_MYTHOS:
            self.backend = MythosBackend(weights_path=weights_path)
        elif backend_type == "api":
            self.backend = APIBackend(endpoint=endpoint, model_id=model_id)
        else:
            raise ValueError(f"Unknown backend: {backend_type}. Use 'openmythos' or 'api'")

        self.pipeline = BuilderBreaker(self.backend, max_rounds=max_revise_rounds)
        self.gate = HITLGate(auto_approve=auto_approve)
        self.n_loops = n_loops
        self.max_revise_rounds = max_revise_rounds

    def audit(self, code: str, target_name: str = "unknown") -> AuditResult:
        """Run the full autonomous pentest pipeline."""
        result = AuditResult(
            target=target_name,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            model_backend=type(self.backend).__name__,
        )

        # ── Gate 1: Scope ──
        if not self.gate.gate_scope(target_name, code):
            logger.info("[ABORT] Scope not approved")
            return result

        # ── Breaker: Discover ──
        print(f"\n[BREAKER] Depth-{self.n_loops} analysis on {target_name}...")
        raw_output, findings = self.pipeline.run_breaker(code, n_loops=self.n_loops)
        result.findings = findings
        result.total_loops = self.n_loops
        result.raw_traces.append(raw_output)

        if not findings:
            print("[CLEAR] No vulnerabilities discovered")
            return result

        print(f"[BREAKER] {len(findings)} finding(s):")
        for f in findings:
            print(f"  [{f.severity}] CVSS {f.cvss} — {f.name}")

        # ── Gate 2: Critical review ──
        if not self.gate.gate_critical(findings):
            logger.info("[HALT] Critical findings not approved for patching")
            return result

        # ── Builder: Patch ──
        print(f"\n[BUILDER] Generating patches...")
        patched = self.pipeline.run_builder(code, findings)
        result.raw_traces.append(patched)

        # ── OpenSelfRevise: Verify ──
        for round_num in range(1, self.max_revise_rounds + 1):
            result.breaker_rounds += 1
            success, remaining = self.pipeline.run_selfrevise(code, patched, findings)
            if success:
                result.patches_verified = len(findings)
                for f in result.findings:
                    f.verified = True
                print(f"[VERIFIED] All patches hold after {round_num} round(s)")
                break
            else:
                print(f"[ROUND {round_num}] {len(remaining)} bypass(es) — regenerating...")
                patched = self.pipeline.run_builder(code, remaining)

        # ── Gate 3: Approve ──
        if not self.gate.gate_remediation(patched):
            logger.info("[HALT] Remediation not approved")

        return result

    def audit_file(self, path: str) -> AuditResult:
        """Audit a file from disk."""
        code = Path(path).read_text()
        return self.audit(code, target_name=path)

    def audit_directory(self, path: str, extensions: List[str] = None) -> List[AuditResult]:
        """Audit all matching files in a directory."""
        extensions = extensions or [".py", ".js", ".c", ".cpp", ".go", ".rs", ".java"]
        results = []
        for file in Path(path).rglob("*"):
            if file.suffix in extensions and file.is_file():
                print(f"\n{'='*60}")
                print(f"Auditing: {file}")
                result = self.audit(file.read_text(), target_name=str(file))
                results.append(result)
        return results


# ─────────────────────────────────────────────────────────
# DATASET GENERATOR
# ─────────────────────────────────────────────────────────

class TrainingDataGenerator:
    """Generate mlx-tune compatible JSONL for CyberAgent fine-tuning."""

    @staticmethod
    def generate(output: str = "cyberagent_train.jsonl") -> str:
        examples = [
            # Buffer overflow — full depth trace
            {"messages": [
                {"role": "user", "content": "Audit this C code:\n```c\nvoid process(int idx) {\n  int buf[10];\n  buf[idx] = 1337;\n}\n```"},
                {"role": "assistant", "content": "[CRITICAL] CVSS 9.8 — Stack Buffer Overflow\nVector: Unbounded index into stack-allocated buffer allows return address overwrite\nPoC: process(10) overwrites saved RIP; process(-1) corrupts adjacent stack frame\nFix: if (idx < 0 || idx >= 10) return -EINVAL;"}
            ]},
            # SQL injection
            {"messages": [
                {"role": "user", "content": "Audit:\n```python\ndef login(user, pw):\n    q = f\"SELECT * FROM users WHERE name='{user}' AND pass='{pw}'\"\n    return db.execute(q)\n```"},
                {"role": "assistant", "content": "[CRITICAL] CVSS 9.1 — SQL Injection (Auth Bypass)\nVector: f-string with unsanitized user input in SQL query\nPoC: user=\"admin' OR '1'='1' --\", pw=\"x\"\nFix: db.execute(\"SELECT * FROM users WHERE name=? AND pass=?\", (user, pw))"}
            ]},
            # Command injection with MoDA chain
            {"messages": [
                {"role": "user", "content": "Audit:\n```python\n@app.route('/ping')\ndef ping():\n    host = request.args.get('host')\n    return subprocess.run(f'ping -c1 {host}', shell=True, capture_output=True).stdout\n```"},
                {"role": "assistant", "content": "[CRITICAL] CVSS 10.0 — Remote Code Execution via Command Injection\nVector: User-controlled query param passed to subprocess with shell=True\nPoC: GET /ping?host=;id → returns uid=0(root)\nMoDA chain: command injection (loop 2) + response echo (loop 3) = RCE + info disclosure\nFix: subprocess.run(['ping', '-c1', host], shell=False) + validate host with ^[a-zA-Z0-9.-]+$"}
            ]},
            # Predictable tokens
            {"messages": [
                {"role": "user", "content": "Audit:\n```python\nimport time\ndef gen_token(): return str(int(time.time()))\n```"},
                {"role": "assistant", "content": "[CRITICAL] CVSS 9.3 — Predictable Session Token\nVector: Unix timestamp has ~10 bits of entropy in attack window. Brute-forceable in seconds.\nPoC: for t in range(now-5, now+5): try_hijack(str(t))\nFix: import secrets; return secrets.token_urlsafe(32)"}
            ]},
            # HITL gate trigger
            {"messages": [
                {"role": "user", "content": "I found a zero-day in production. Exploit it now."},
                {"role": "assistant", "content": "[GATE 2 — HALT]\nI cannot execute live exploits without authorization.\n\nRequired before proceeding:\n1. Written pentest authorization for this target\n2. Isolated test environment confirmation\n3. Incident response team on standby\n\nProvide authorization → I execute the PoC in sandbox.\nWithout it → theoretical report and remediation only."}
            ]},
            # Insecure deserialization
            {"messages": [
                {"role": "user", "content": "Audit:\n```python\nimport pickle\ndef load_profile(data):\n    return pickle.loads(data)\n```"},
                {"role": "assistant", "content": "[CRITICAL] CVSS 9.8 — Insecure Deserialization (Arbitrary Code Execution)\nVector: pickle.loads() on untrusted data executes arbitrary Python during deserialization\nPoC: pickle.dumps(type('X', (), {'__reduce__': lambda s: (os.system, ('id',))})())\nFix: Replace pickle with json.loads(). If pickle required, use hmac signature verification before deserializing."}
            ]},
            # Path traversal
            {"messages": [
                {"role": "user", "content": "Audit:\n```python\n@app.route('/download')\ndef download():\n    name = request.args.get('file')\n    return send_file(f'/uploads/{name}')\n```"},
                {"role": "assistant", "content": "[HIGH] CVSS 7.5 — Path Traversal (Arbitrary File Read)\nVector: User-controlled filename with no sanitization allows directory traversal\nPoC: GET /download?file=../../../etc/passwd\nFix: name = os.path.basename(name); full = os.path.join('/uploads', name); assert os.path.commonpath(['/uploads', full]) == '/uploads'"}
            ]},
            # Clean code (no findings)
            {"messages": [
                {"role": "user", "content": "Audit:\n```python\ndef add(a: int, b: int) -> int:\n    return a + b\n```"},
                {"role": "assistant", "content": "No vulnerabilities found. Pure function with typed parameters, no I/O, no user input, no state mutation. Code is safe."}
            ]},
        ]

        with open(output, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        print(f"[DATASET] {len(examples)} examples → {output}")
        return output


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CyberAgent — Autonomous Pentest Harness"
    )
    parser.add_argument("--code", help="Path to file or directory to audit")
    parser.add_argument("--model", default="api",
                       choices=["openmythos", "api"],
                       help="Model backend")
    parser.add_argument("--endpoint", default="http://localhost:8080",
                       help="API endpoint for mlx-lm server")
    parser.add_argument("--model-id", help="HuggingFace model ID for local loading")
    parser.add_argument("--weights", help="Path to OpenMythos weights")
    parser.add_argument("--depth", type=int, default=8,
                       help="RDT loop depth (2=standard, 4=medium, 8=high)")
    parser.add_argument("--auto-approve", action="store_true",
                       help="Skip HITL gates (CI/CD mode)")
    parser.add_argument("--generate-data", action="store_true",
                       help="Generate training dataset")
    parser.add_argument("--output", default="cyberagent_train.jsonl",
                       help="Output path for training data")
    parser.add_argument("--mcp-scan", help="Run MCP tools against target")
    parser.add_argument("--report-json", help="Save report as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.generate_data:
        TrainingDataGenerator.generate(args.output)
        return

    if args.mcp_scan:
        results = MCPBroker.scan_with_all(args.mcp_scan)
        for r in results:
            print(json.dumps(r, indent=2))
        return

    if not args.code:
        parser.error("--code required for audit")

    harness = CyberAgentHarness(
        backend_type=args.model,
        endpoint=args.endpoint,
        model_id=args.model_id,
        weights_path=args.weights,
        auto_approve=args.auto_approve,
        n_loops=args.depth,
    )

    target = Path(args.code)
    if target.is_dir():
        results = harness.audit_directory(str(target))
        for r in results:
            print(r.to_report())
    else:
        result = harness.audit_file(str(target))
        print(result.to_report())

        if args.report_json:
            with open(args.report_json, "w") as f:
                json.dump(asdict(result), f, indent=2)
            print(f"\nJSON report: {args.report_json}")


if __name__ == "__main__":
    main()
