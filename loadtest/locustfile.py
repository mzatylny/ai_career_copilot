from __future__ import annotations

import os

from locust import HttpUser, between, task


class CareerCopilotUser(HttpUser):
    wait_time = between(0.5, 2.0)

    def on_start(self) -> None:
        key = os.getenv("AI_COPILOT_API_KEY", "")
        self.headers = {"X-API-Key": key} if key else {}

    @task(5)
    def health(self) -> None:
        self.client.get("/api/health")

    @task(2)
    def analyze_gap(self) -> None:
        self.client.post(
            "/api/analyze-gap",
            headers=self.headers,
            json={
                "resume_text": "Senior Python engineer delivering APIs, data systems and AI products. "
                * 3,
                "job_description_text": "Senior AI Engineer with Python, RAG, APIs, observability and cloud. "
                * 3,
                "target_seniority": "senior",
            },
        )
