from __future__ import annotations

from typing import Any


class RulesEngine:
    def process_context(self, context: dict[str, Any]) -> dict[str, Any]:
        alerts: list[str] = []
        severity = "NONE"

        if context["people"]["unknown_count"] > 0:
            alerts.append("Unknown person visible")
            severity = "HIGH"

        if context["focus_target"]:
            alerts.append(f"Focus target: {context['focus_target']['name']}")
            severity = "HIGH" if severity == "NONE" else severity
        elif context.get("requested_target"):
            alerts.append(f"Searching for: {context['requested_target']}")
            severity = "MEDIUM" if severity == "NONE" else severity

        if context["objects"]["counts"].get("cell phone", 0) > 0:
            alerts.append("Phone detected in scene")
            if severity == "NONE":
                severity = "MEDIUM"

        if len(context["people"]["tracks"]) >= 4 and severity == "NONE":
            alerts.append("Crowd density elevated")
            severity = "LOW"

        return {"alerts": alerts, "severity": severity}
