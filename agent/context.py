"""
Context assembler.

Calls each enabled tool and returns a structured dict + a formatted
string block ready to inject into an LLM prompt.

Pass `enabled_tools` as a subset to run ablation experiments — only
those tools will be called and included in the prompt context.
"""
from __future__ import annotations

import pandas as pd
from typing import Any

from tools.historical_kpi    import query as historical_kpi_query
from tools.traffic_prediction import query as traffic_forecast_query
from tools.alarm_fault        import query as alarm_fault_query
from tools.interference       import query as interference_query
from tools.energy_pricing     import query as energy_pricing_query

ALL_TOOLS = frozenset([
    "historical_kpi",
    "traffic_forecast",
    "alarm_fault",
    "interference",
    "energy_pricing",
])


def assemble(
    timestamp: pd.Timestamp | str,
    cell_ids: list[int],
    current_utilization: dict[int, float],
    enabled_tools: frozenset[str] = ALL_TOOLS,
) -> tuple[dict[str, Any], str]:
    """
    Returns (raw_context_dict, formatted_prompt_block).

    raw_context_dict  — structured data for logging / correlation analysis
    formatted_prompt_block — human-readable text injected into the LLM prompt
    """
    ctx: dict[str, Any] = {"tools_used": list(enabled_tools)}
    lines: list[str]    = ["### Tool Context\n"]

    if "historical_kpi" in enabled_tools:
        data = historical_kpi_query(cell_id=None)
        ctx["historical_kpi"] = data
        lines.append("**Historical KPI (last 24 h per cell):**")
        for cid in sorted(data):
            d = data[cid]
            lines.append(
                f"  Cell {cid}: mean={d['mean_throughput_mbps']} Mbps  "
                f"util={d['mean_utilization_pct']}%  "
                f"sleep_rate={d['sleep_rate']:.1%}  "
                f"p10={d['p10_throughput_mbps']} / p90={d['p90_throughput_mbps']} Mbps"
            )
        lines.append("")

    if "traffic_forecast" in enabled_tools:
        data = traffic_forecast_query(timestamp=timestamp, cell_id=None, horizon=3)
        ctx["traffic_forecast"] = data
        lines.append("**Traffic forecast (next 3 intervals):**")
        for cid in sorted(data):
            steps = ", ".join(
                f"t+{s['step']}={s['predicted_mbps']} Mbps" for s in data[cid]
            )
            lines.append(f"  Cell {cid}: {steps}")
        lines.append("")

    if "alarm_fault" in enabled_tools:
        data = alarm_fault_query(timestamp=timestamp)
        ctx["alarm_fault"] = data
        active = {cid: v for cid, v in data.items() if v["active"]}
        lines.append("**Active faults:**")
        if active:
            for cid, v in active.items():
                lines.append(f"  Cell {cid}: {v['fault_type']} (severity={v['severity']})")
        else:
            lines.append("  None")
        lines.append("")

    if "interference" in enabled_tools:
        # Pre-compute for all cells sleeping (worst case); planner uses this as a lookup
        ctx["interference"] = {}
        lines.append("**Inter-cell interference (if cell sleeps → neighbor load increase):**")
        for cid in cell_ids:
            result = interference_query(
                sleeping_cells=[cid],
                current_utilization=current_utilization,
            )
            ctx["interference"][cid] = result
            if result["neighbor_load_pct"]:
                impacts = ", ".join(
                    f"Cell {n}→{v}%" for n, v in result["neighbor_load_pct"].items()
                )
                safe_tag = "SAFE" if result["safe"] else "OVERLOAD RISK"
                lines.append(f"  Sleep Cell {cid}: {impacts}  [{safe_tag}]")
        lines.append("")

    if "energy_pricing" in enabled_tools:
        data = energy_pricing_query(timestamp=timestamp, horizon=3)
        ctx["energy_pricing"] = data
        forecast_str = ", ".join(
            f"t+{s['step']}=${s['price_per_kwh']}/kWh ({s['tier']})"
            for s in data["forecast"]
        )
        lines.append(
            f"**Energy pricing:** current=${data['current_price_per_kwh']}/kWh "
            f"(tier={data['current_tier']})"
        )
        lines.append(f"  Forecast: {forecast_str}")
        lines.append("")

    return ctx, "\n".join(lines)
