"""Grounding prompt tests; these tests never call the Gemini service."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import generation
import config
from retrieval import RetrievedChunk


def test_configured_model_name_is_passed_to_gemini_client(monkeypatch):
    configured_model = "gemini-3.1-flash-lite"
    captured = {}

    class FakeModels:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return type("Response", (), {"text": "Generated answer."})()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(config, "GEMINI_MODEL_NAME", configured_model)

    answer = generation._generate(
        FakeClient(), "System instruction", "User question"
    )

    assert answer == "Generated answer."
    assert captured["model"] == configured_model
    assert captured["contents"] == "User question"


def test_404_explains_how_to_select_a_supported_model(monkeypatch):
    class FakeModels:
        def generate_content(self, **kwargs):
            raise RuntimeError(
                "404 NOT_FOUND: model is not found or is not supported"
            )

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(config, "GEMINI_MODEL_NAME", "unavailable-model")

    with pytest.raises(RuntimeError, match="GEMINI_MODEL=<model-id>") as error:
        generation._generate(FakeClient(), "System instruction", "User question")

    assert "unavailable to this API key" in str(error.value)
    assert "https://ai.google.dev/gemini-api/docs/models" in str(error.value)


def test_rag_generation_includes_question_and_cited_retrieved_context(monkeypatch):
    chunk = RetrievedChunk(
        text="The report describes the retrieval pipeline.",
        source="report.pdf",
        chunk_index=3,
        page_num=4,
        distance=0.1,
        score=0.9,
    )
    captured = {}

    monkeypatch.setattr(generation, "_get_client", lambda: object())

    def fake_generate(client, system_prompt, user_message, temperature):
        captured.update(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temperature,
        )
        return "The retrieval pipeline is described in [1]."

    monkeypatch.setattr(generation, "_generate", fake_generate)

    answer = generation.generate_rag_answer(
        "How does retrieval work?", [chunk]
    )

    assert answer.endswith("[1].")
    assert "ONLY the context passages" in captured["system_prompt"]
    assert "[1] Source: report.pdf, page 4" in captured["user_message"]
    assert chunk.text in captured["user_message"]
    assert "Question: How does retrieval work?" in captured["user_message"]


def test_no_retrieved_context_does_not_call_gemini(monkeypatch):
    def unexpected_client():
        raise AssertionError("No-context response should not call Gemini.")

    monkeypatch.setattr(generation, "_get_client", unexpected_client)

    answer = generation.generate_rag_answer("Question?", [])

    assert "No relevant documents" in answer


def test_rag_generation_rejects_blank_question():
    with pytest.raises(ValueError, match="Question"):
        generation.generate_rag_answer("  ", [])
