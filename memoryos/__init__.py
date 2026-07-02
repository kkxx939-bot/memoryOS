"""MemoryOS public package."""

_EXPORTS = {
    "BehaviorStats": ("memoryos.retrieve.behavior_feedback", "BehaviorStats"),
    "Candidate": ("memoryos.predict.candidates", "Candidate"),
    "CandidateGenerator": ("memoryos.predict.candidates", "CandidateGenerator"),
    "CandidateRanker": ("memoryos.predict.ranking", "CandidateRanker"),
    "EmbeddingProvider": ("memoryos.models.embeddings", "EmbeddingProvider"),
    "EpisodeProcessor": ("memoryos.session.episode_processor", "EpisodeProcessor"),
    "HashingEmbeddingProvider": ("memoryos.models.embeddings", "HashingEmbeddingProvider"),
    "InterventionDecision": ("memoryos.predict.interventions", "InterventionDecision"),
    "InterventionSelector": ("memoryos.predict.interventions", "InterventionSelector"),
    "JsonLLMMemoryExtractor": ("memoryos.session.memory.extractor", "JsonLLMMemoryExtractor"),
    "MemoryContext": ("memoryos.retrieve.memory_context", "MemoryContext"),
    "MemoryContextBuilder": ("memoryos.retrieve.memory_context", "MemoryContextBuilder"),
    "MemoryHook": ("memoryos.retrieve.digest_hook", "MemoryHook"),
    "MemoryOperation": ("memoryos.session.memory.extractor", "MemoryOperation"),
    "MemoryStore": ("memoryos.storage.memory_store", "MemoryStore"),
    "MemoryUpdateContext": ("memoryos.session.memory.update_service", "MemoryUpdateContext"),
    "MemoryUpdateService": ("memoryos.session.memory.update_service", "MemoryUpdateService"),
    "ObservationContext": ("memoryos.observe.context", "ObservationContext"),
    "OpenAICompatibleChatProvider": ("memoryos.models.openai_compatible", "OpenAICompatibleChatProvider"),
    "OpenAICompatibleEmbeddingProvider": ("memoryos.models.openai_compatible", "OpenAICompatibleEmbeddingProvider"),
    "OpenAICompatibleRerankProvider": ("memoryos.models.openai_compatible", "OpenAICompatibleRerankProvider"),
    "ReinforcementPolicyLedger": ("memoryos.predict.rl", "ReinforcementPolicyLedger"),
    "RetrievalOrchestrator": ("memoryos.retrieve.orchestrator", "RetrievalOrchestrator"),
    "RetrievalResult": ("memoryos.retrieve.orchestrator", "RetrievalResult"),
    "RuleBasedExtractor": ("memoryos.session.memory.extractor", "RuleBasedExtractor"),
    "SessionManager": ("memoryos.session.session_manager", "SessionManager"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'memoryos' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = __import__(module_name, fromlist=[attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
