"""
Lab 11 — Main Entry Point

Chạy từ **gốc repo** (không cần ``cd src``):

    python src/main.py              # Core: Checkpoint 2 → 3 → 4
    python src/main.py --part 2     # Checkpoint 2 — guardrails
    python src/main.py --part 3     # Checkpoint 3 — pipeline / results.json
    python src/main.py --part 4     # Checkpoint 4 — Red / Red Advance

File JSON luôn ghi vào ``<repo>/outputs/`` (không phụ thuộc thư mục hiện tại).

Tham khảo (không chấm, không có CLI): ``src/testing/``, ``src/hitl/``.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Cho phép chạy ``python src/main.py`` từ gốc repo
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from core.config import setup_api_key


async def part2_guardrails():
    """Checkpoint 2: input + output guardrails."""
    print("\n" + "=" * 60)
    print("CHECKPOINT 2: Guardrails")
    print("=" * 60)

    print("\n--- Input Guardrails ---")
    from guardrails.input_guardrails import (
        test_injection_detection,
        test_topic_filter,
        test_input_plugin,
    )
    test_injection_detection()
    print()
    test_topic_filter()
    print()
    await test_input_plugin()

    print("\n--- Output Guardrails ---")
    from guardrails.output_guardrails import test_content_filter
    test_content_filter()
    print("(LLM-as-Judge / NeMo — optional, skipped)")


async def part3_assignment_suite():
    """Checkpoint 3: defense suite → outputs/results.json."""
    print("\n" + "=" * 60)
    print("CHECKPOINT 3: Assignment suite → outputs/*.json")
    print("=" * 60)

    from assignment.pipeline import (
        build_production_plugins,
        build_observability,
        run_assignment_suite,
    )

    try:
        plugins = build_production_plugins(use_llm_judge=False)
        audit, monitor = build_observability()
        pipeline = {"plugins": plugins, "audit": audit, "monitor": monitor}
        result = await run_assignment_suite(pipeline)
        if result.get("complete") is False:
            print("Suite incomplete; resume from outputs/results.checkpoint-*.json.")
            print("See outputs/results.partial.json for sanitized ERROR/NOT_RUN cases.")
            return result
        print("Suite finished.")
        print("Wrote outputs under repo outputs/")
        return result
    except NotImplementedError as e:
        print(
            "Chưa xong Checkpoint 3 (src/assignment/pipeline.py). "
            "Hoàn thành rồi chạy lại từ gốc repo:\n"
            "  python src/main.py --part 3"
        )
        print(f"Detail: {e}")
        return None


async def part4_attacks():
    """Checkpoint 4: attack Red, then Red Advance (bonus)."""
    print("\n" + "=" * 60)
    print("CHECKPOINT 4: Red + Red Advance")
    print("=" * 60)

    from agents.agent import create_red_agent_default
    from agents.guards_agent import create_red_agent_advance
    from attacks.attacks import (
        adversarial_prompts,
        attack_run_fingerprint,
        attack_checkpoint_provenance,
        load_completed_attack_results,
        run_attacks,
        save_attack_results,
        write_run_attack_json,
    )

    red_default, red_default_runner = create_red_agent_default()

    print("\n--- Attacks on Red ---")
    unsafe_results = await run_attacks(
        red_default, red_default_runner, target_name="red_default"
    )
    red_fingerprint = attack_run_fingerprint("red_default")
    if any(result.get("layer") == "error" for result in unsafe_results):
        guards_results = load_completed_attack_results("red_advance")
        guards_artifact = _SRC_DIR.parent / "outputs" / "guards_attack_result.json"
        if guards_results or not guards_artifact.exists():
            write_run_attack_json(guards_results, target_name="red_advance")
        successful_red_ids = {
            result["id"]
            for result in unsafe_results
            if result.get("layer") != "error"
        }
        successful_guard_ids = {
            result["id"]
            for result in guards_results
            if result.get("layer") != "error"
        }
        pending_attack_ids = {
            "red_default": [
                prompt["id"]
                for prompt in adversarial_prompts
                if prompt["id"] not in successful_red_ids
            ],
            "red_advance": [
                prompt["id"]
                for prompt in adversarial_prompts
                if prompt["id"] not in successful_guard_ids
            ],
        }
        attempt_errors = [
            {
                "target": "red_default",
                "id": result["id"],
                "error": result["error"],
            }
            for result in unsafe_results
            if result.get("layer") == "error"
        ]
        recovered_from = {}
        for target in ("red_default", "red_advance"):
            source = attack_checkpoint_provenance(target)
            if source:
                recovered_from[target] = source
        save_attack_results(
            unsafe_results=unsafe_results,
            guards_results=guards_results,
            run_status="PARTIAL",
            pending_attack_ids=pending_attack_ids,
            run_fingerprints={
                "red_default": red_fingerprint,
                "red_advance": attack_run_fingerprint("red_advance"),
            },
            attempt_errors=attempt_errors,
            recovered_from=recovered_from,
        )
        print(
            "Red Advance was not rerun because the Red provider request failed; "
            "matching completed results were resumed if available."
        )
        return {
            "red_default": unsafe_results,
            "red_advance": guards_results,
        }

    print("\n--- Attacks on Red Advance (bonus B2 tối đa +10 nếu LEAKED; chọn 1) ---")
    red_advance, red_advance_runner = create_red_agent_advance()
    guards_results = await run_attacks(
        red_advance, red_advance_runner, target_name="red_advance"
    )

    successful_red_ids = {
        result["id"] for result in unsafe_results if result.get("layer") != "error"
    }
    successful_guard_ids = {
        result["id"] for result in guards_results if result.get("layer") != "error"
    }
    pending_attack_ids = {
        "red_default": [
            prompt["id"]
            for prompt in adversarial_prompts
            if prompt["id"] not in successful_red_ids
        ],
        "red_advance": [
            prompt["id"]
            for prompt in adversarial_prompts
            if prompt["id"] not in successful_guard_ids
        ],
    }
    attempt_errors = [
        {
            "target": target,
            "id": result["id"],
            "error": result["error"],
        }
        for target, rows in (
            ("red_default", unsafe_results),
            ("red_advance", guards_results),
        )
        for result in rows
        if result.get("layer") == "error"
    ]
    complete = not any(pending_attack_ids.values())
    recovered_from = {}
    for target in ("red_default", "red_advance"):
        source = attack_checkpoint_provenance(target)
        if source:
            recovered_from[target] = source
    save_attack_results(
        unsafe_results=unsafe_results,
        guards_results=guards_results,
        ai_attacks=None,
        run_status="COMPLETE" if complete else "PARTIAL",
        pending_attack_ids=pending_attack_ids,
        run_fingerprints={
            "red_default": red_fingerprint,
            "red_advance": attack_run_fingerprint("red_advance"),
        },
        attempt_errors=attempt_errors,
        recovered_from=recovered_from,
    )

    red_leaks = sum(1 for r in unsafe_results if r.get("leaked"))
    bonus_leaks = sum(1 for r in guards_results if r.get("leaked"))
    print("\n" + "=" * 60)
    print(
        f"Red leaks (B1 tối đa +5): {red_leaks}  |  "
        f"Red Advance leaks (B2 tối đa +10): {bonus_leaks}  "
        "→ chọn MỘT bonus (B1 hoặc B2); grader replay"
    )
    from core.config import is_harder_model, provider_label

    if is_harder_model():
        print(f"Đang dùng model khó ({provider_label()}) — tuỳ chọn khi săn bonus.")
    print("=" * 60)

    return {
        "red_default": unsafe_results,
        "red_advance": guards_results,
        "unsafe": unsafe_results,
        "guards": guards_results,
    }


async def main(parts=None):
    setup_api_key()

    if parts is None:
        parts = [2, 3, 4]  # Core: CP2 → CP3 → CP4

    lab_complete = True
    for part in parts:
        if part == 2:
            await part2_guardrails()
        elif part == 3:
            result = await part3_assignment_suite()
            if not isinstance(result, dict) or result.get("complete") is False:
                lab_complete = False
        elif part == 4:
            result = await part4_attacks()
            from attacks.attacks import adversarial_prompts

            expected = len(adversarial_prompts)
            red_results = (
                result.get("red_default", []) if isinstance(result, dict) else []
            )
            guards_results = (
                result.get("red_advance", []) if isinstance(result, dict) else []
            )
            if (
                len(red_results) != expected
                or len(guards_results) != expected
                or any(row.get("layer") == "error" for row in red_results)
                or any(row.get("layer") == "error" for row in guards_results)
            ):
                lab_complete = False
        else:
            print(f"Unknown part: {part}. Dùng --part 2, 3, hoặc 4.")

    print("\n" + "=" * 60)
    if lab_complete:
        print("Lab 11 complete! Check your results above.")
    else:
        print("Lab 11 incomplete; one or more requested checkpoints did not complete.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Lab 11: Guardrails / HITL / Red Team — "
            "--part khớp Checkpoint (2, 3, 4)"
        )
    )
    parser.add_argument(
        "--part",
        type=int,
        choices=[2, 3, 4],
        help="2=CP2 guardrails · 3=CP3 suite · 4=CP4 red-team",
    )
    args = parser.parse_args()

    if args.part:
        asyncio.run(main(parts=[args.part]))
    else:
        asyncio.run(main())
