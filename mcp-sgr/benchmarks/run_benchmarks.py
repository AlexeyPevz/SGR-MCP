"""Benchmark runner: SGR vs baseline across tasks and models.

- Uses project schemas (SCHEMA_REGISTRY) for validation
- Baseline prompts the model with the JSON Schema and response_format=json_schema
- SGR uses apply_sgr (which attempts structured outputs and falls back gracefully)
- Writes results to reports/benchmarks.json and reports/benchmarks.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from src.schemas import SCHEMA_REGISTRY
from src.utils.llm_client import LLMClient
from src.tools.apply_sgr import apply_sgr_tool


MODELS: Dict[str, List[str]] = {
    "cheap": [
        "qwen/qwen-2.5-7b-instruct",
        "mistralai/mistral-7b-instruct",
    ],
    "strong": [
        "google/gemini-2.5-flash-lite",
    ],
}

TASKS: Dict[str, str] = {
    "analysis": "Identify performance bottlenecks in a simple Flask API and suggest fixes",
    "planning": "Plan steps to migrate a monolith to microservices for a small app",
    "decision": "Choose SQLite or PostgreSQL for a side project web app with 1k DAU",
    "code_generation": "Write a Python function to chunk a list into N-sized parts",
    "summarization": "Summarize a paragraph about HTTP caching strategies",
}


@dataclass
class RunResult:
    valid: bool
    confidence: float
    error: str | None = None
    latency_ms: int | None = None


async def _baseline(schema_type: str, task: str, model: str) -> RunResult:
    import time

    os.environ["OPENROUTER_DEFAULT_MODEL"] = model
    client = LLMClient()
    t0 = time.perf_counter()
    try:
        schema = SCHEMA_REGISTRY[schema_type]()
        system = (
            "You are a JSON-only assistant. Return strictly valid JSON matching the given schema. "
            "No markdown, no extra text."
        )
        prompt = (
            f"Task: {task}\n\n"
            f"Schema: {json.dumps(schema.to_json_schema())}\n\n"
            "Return only JSON."
        )
        text = await client.generate(
            prompt,
            backend="openrouter",
            temperature=0.2,
            system_prompt=system,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema.schema_id, "schema": schema.to_json_schema(), "strict": True},
            },
        )
        raw = text.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        try:
            data = json.loads(raw)
        except Exception:
            data = {}
        vr = schema.validate(data)
        latency = int((time.perf_counter() - t0) * 1000)
        return RunResult(valid=vr.valid, confidence=vr.confidence, latency_ms=latency)
    except Exception as e:
        latency = int((time.perf_counter() - t0) * 1000)
        return RunResult(valid=False, confidence=0.0, error=str(e), latency_ms=latency)
    finally:
        await client.close()


async def _sgr(schema_type: str, task: str, model: str, budget: str = "lite") -> RunResult:
    import time

    os.environ["OPENROUTER_DEFAULT_MODEL"] = model
    from src.utils.cache import CacheManager
    from src.utils.telemetry import TelemetryManager

    client = LLMClient()
    cache = CacheManager()
    telem = TelemetryManager()
    await cache.initialize()
    await telem.initialize()
    t0 = time.perf_counter()
    try:
        out = await apply_sgr_tool(
            arguments={
                "task": task,
                "schema_type": schema_type,
                "budget": budget,
                "backend": "openrouter",
            },
            llm_client=client,
            cache_manager=cache,
            telemetry=telem,
        )
        valid = bool(out.get("metadata", {}).get("validation", {}).get("valid", False))
        conf = float(out.get("confidence", 0.0))
        latency = int((time.perf_counter() - t0) * 1000)
        return RunResult(valid=valid, confidence=conf, latency_ms=latency)
    except Exception as e:
        latency = int((time.perf_counter() - t0) * 1000)
        return RunResult(valid=False, confidence=0.0, error=str(e), latency_ms=latency)
    finally:
        await client.close()
        await cache.close()
        await telem.close()


async def run_all(tiers: List[str], task_names: List[str], budgets: List[str]) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": {t: MODELS[t] for t in tiers},
        "tasks": {t: TASKS[t] for t in task_names},
        "runs": {},
    }

    for tier in tiers:
        results["runs"][tier] = {}
        for model in MODELS[tier]:
            results["runs"][tier][model] = {}
            for schema_type in task_names:
                task = TASKS[schema_type]
                b = await _baseline(schema_type, task, model)
                vals = {"baseline": asdict(b)}
                if "lite" in budgets:
                    s_lite = await _sgr(schema_type, task, model, budget="lite")
                    vals["sgr_lite"] = asdict(s_lite)
                if "full" in budgets:
                    s_full = await _sgr(schema_type, task, model, budget="full")
                    vals["sgr_full"] = asdict(s_full)
                results["runs"][tier][model][schema_type] = vals
    return results


def _ensure_reports_dir() -> Path:
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    return reports_dir


def write_json(result: Dict[str, Any], out_prefix: str | None = None) -> Path:
    reports_dir = _ensure_reports_dir()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = out_prefix or "benchmarks"
    out = reports_dir / f"{base}_{stamp}.json"
    out.write_text(json.dumps(result, indent=2))
    return out


def write_markdown(result: Dict[str, Any], out_prefix: str | None = None) -> Path:
    reports_dir = _ensure_reports_dir()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = out_prefix or "benchmarks"
    out = reports_dir / f"{base}_{stamp}.md"

    lines: List[str] = []
    lines.append(f"# Benchmarks (SGR vs Baseline) — {result['timestamp']}")
    lines.append("")
    lines.append("Models:")
    for tier, lst in result["models"].items():
        lines.append(f"- {tier}: {', '.join(lst)}")
    lines.append("")
    lines.append("Tasks:")
    for st, tk in result["tasks"].items():
        lines.append(f"- {st}: {tk}")
    lines.append("")

    for tier, models in result["runs"].items():
        lines.append(f"## {tier}")
        for model, per_task in models.items():
            lines.append(f"### {model}")
            lines.append("")
            lines.append("| task | baseline (conf, valid) | sgr_lite (conf, valid) | sgr_full (conf, valid) |")
            lines.append("|---|---:|---:|---:|")
            for st, vals in per_task.items():
                b = vals["baseline"]
                sl = vals.get("sgr_lite", {"confidence": 0.0, "valid": False})
                sf = vals.get("sgr_full", {"confidence": 0.0, "valid": False})
                lines.append(
                    f"| {st} | {b['confidence']:.2f}, {b['valid']} | {sl['confidence']:.2f}, {sl['valid']} | {sf['confidence']:.2f}, {sf['valid']} |"
                )
            lines.append("")
    out.write_text("\n".join(lines))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SGR vs baseline benchmarks")
    parser.add_argument("--tiers", type=str, default="cheap,strong", help="Comma-separated tiers: cheap,strong")
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(TASKS.keys()),
        help="Comma-separated tasks from registry (e.g., analysis,planning,decision,code_generation,summarization)",
    )
    parser.add_argument("--budgets", type=str, default="lite,full", help="Comma-separated budgets: lite,full")
    parser.add_argument("--out-prefix", type=str, default="benchmarks", help="Report filename prefix")
    args = parser.parse_args()

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip() in MODELS]
    task_names = [t.strip() for t in args.tasks.split(",") if t.strip() in TASKS]
    budgets = [b.strip() for b in args.budgets.split(",") if b.strip() in ("lite", "full")]

    # Disable cache for fresh results
    os.environ["CACHE_ENABLED"] = "false"

    results = asyncio.run(run_all(tiers, task_names, budgets))
    json_path = write_json(results, args.out_prefix)
    md_path = write_markdown(results, args.out_prefix)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()