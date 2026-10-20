"""
generation.py
-------------
Grounded answer generation using the Google Gemini API.

This module takes:
  1. A user question.
  2. A list of retrieved context chunks (from retrieval.py).

And returns:
  - A grounded answer that cites only the provided context.
  - (Optionally) A plain LLM answer with no context for comparison.

Design notes
------------
- The Google Gen AI SDK (google-genai) is used, NOT the older
  google-generativeai library. The correct import is ``google.genai``.
- Model name and API key are read from environment variables (config.py).
  They are never hardcoded here.
- The system prompt instructs the model to answer ONLY from context and
  to say "I don't know" if the context is insufficient. This is a key
  RAG principle that reduces hallucination.
- Both a RAG answer and a plain LLM answer can be generated for
  side-by-side comparison (useful for the assignment evaluation).

Usage:
    from generation import generate_rag_answer, generate_plain_answer
    answer = generate_rag_answer(question, retrieved_chunks)
"""

import logging
from typing import List, Optional

from google import genai
from google.genai import types as genai_types

import config
from retrieval import RetrievedChunk, format_context

logger = logging.getLogger(__name__)


# ── Prompt templates ──────────────────────────────────────────────────────────

# RAG system prompt: model is strictly grounded in the provided context.
RAG_SYSTEM_PROMPT = (
    "You are a helpful academic assistant. "
    "Answer the user's question using ONLY the context passages provided below. "
    "If the context does not contain enough information to answer, say: "
    "'I don't have enough information in the provided documents to answer this question.' "
    "Do not use prior knowledge beyond what is in the context. "
    "Always cite which source passage(s) you used, referencing them by their [number]."
)

# Plain LLM system prompt: no context constraint (for comparison).
PLAIN_SYSTEM_PROMPT = (
    "You are a helpful academic assistant. "
    "Answer the user's question as clearly and accurately as you can."
)


# ── Gemini client factory ─────────────────────────────────────────────────────

def _get_client() -> genai.Client:
    """
    Create and return a Google Gen AI client.

    Raises
    ------
    RuntimeError if GEMINI_API_KEY is not set.
    """
    config.validate_api_key()
    return genai.Client(api_key=config.GEMINI_API_KEY)


# ── Core generation function ──────────────────────────────────────────────────

def _generate(
    client: genai.Client,
    system_prompt: str,
    user_message: str,
    temperature: float = 0.2,
) -> str:
    """
    Call the Gemini API and return the generated text.

    Parameters
    ----------
    client        : Authenticated Gen AI client.
    system_prompt : Instructions prepended to the conversation.
    user_message  : The full user turn (question + context if any).
    temperature   : Sampling temperature (lower = more deterministic).

    Returns
    -------
    Generated text string.

    Raises
    ------
    RuntimeError if the API call fails or returns no text.
    """
    logger.info(
        "Calling Gemini model=%s (temp=%.2f, prompt_chars=%d)",
        config.GEMINI_MODEL_NAME,
        temperature,
        len(system_prompt) + len(user_message),
    )

    try:
        response = client.models.generate_content(
            model=config.GEMINI_MODEL_NAME,
            contents=user_message,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=1024,
            ),
        )
    except Exception as exc:
        error_code = getattr(exc, "code", None)
        error_message = str(exc).lower()
        if error_code == 404 or str(error_code) == "404" or (
            "404" in error_message
            and ("not found" in error_message or "not_found" in error_message)
        ):
            raise RuntimeError(
                f"Gemini model '{config.GEMINI_MODEL_NAME}' is unavailable to this "
                "API key or does not support generateContent. Select a supported "
                "text-generation model from https://ai.google.dev/gemini-api/docs/models "
                "by setting GEMINI_MODEL=<model-id> in the project-root .env file, "
                "then restart the app."
            ) from exc
        raise RuntimeError(
            f"Gemini API call failed: {exc}\n"
            "Check that your GEMINI_API_KEY is valid and the model name is correct."
        ) from exc

    # Extract text from the response
    try:
        answer = response.text
    except Exception:
        answer = None

    if not answer:
        raise RuntimeError(
            "Gemini returned an empty response. "
            "The content may have been blocked by safety filters, "
            "or the model did not produce any output."
        )

    logger.info("Generation complete. Response length: %d chars.", len(answer))
    return answer.strip()


# ── Public API ────────────────────────────────────────────────────────────────

def generate_rag_answer(
    question: str,
    chunks: List[RetrievedChunk],
    temperature: float = 0.2,
) -> str:
    """
    Generate a grounded answer using retrieved context chunks.

    Parameters
    ----------
    question    : The user's question.
    chunks      : Retrieved context chunks from retrieval.retrieve_chunks().
    temperature : Model sampling temperature.

    Returns
    -------
    Answer string grounded in the provided context.

    Raises
    ------
    ValueError  if question is empty.
    RuntimeError if chunks is empty or API call fails.
    """
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    if not chunks:
        logger.warning(
            "generate_rag_answer called with no retrieved chunks. "
            "The model has no context to work with."
        )
        return (
            "No relevant documents were found in the knowledge base for your question. "
            "Please make sure documents have been ingested and try rephrasing."
        )

    # Build the user message: context block + question
    context_block = format_context(chunks)
    user_message = (
        f"Context passages:\n\n"
        f"{context_block}\n\n"
        f"---\n"
        f"Question: {question}"
    )

    client = _get_client()
    return _generate(client, RAG_SYSTEM_PROMPT, user_message, temperature)


def generate_plain_answer(
    question: str,
    temperature: float = 0.7,
) -> str:
    """
    Generate a plain LLM answer with NO retrieved context.

    Used for side-by-side comparison with the RAG answer to demonstrate
    how grounding reduces hallucination.

    Parameters
    ----------
    question    : The user's question.
    temperature : Model sampling temperature (higher = more varied output).

    Returns
    -------
    Answer string from the model without any document context.

    Raises
    ------
    ValueError  if question is empty.
    RuntimeError if the API call fails.
    """
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    client = _get_client()
    return _generate(client, PLAIN_SYSTEM_PROMPT, question, temperature)


# ── CLI quick-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Minimal test — prints a plain answer (no documents required)
    import sys

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is RAG in AI?"
    print(f"Question: {question}\n")
    print("Plain LLM answer (no context):")
    try:
        answer = generate_plain_answer(question)
        print(answer)
    except RuntimeError as e:
        print(f"ERROR: {e}")
