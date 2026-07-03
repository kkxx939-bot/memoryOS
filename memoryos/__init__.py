"""MemoryOS public package."""

_EXPORTS = {
    "BehaviorStats": ("memoryos.application.learning.behavior_feedback", "BehaviorStats"),
    "Candidate": ("memoryos.application.prediction.candidate_generator", "Candidate"),
    "CandidateGenerator": ("memoryos.application.prediction.candidate_generator", "CandidateGenerator"),
    "CandidateRanker": ("memoryos.application.prediction.candidate_ranker", "CandidateRanker"),
    "EmbeddingProvider": ("memoryos.infrastructure.providers.embedding_provider", "EmbeddingProvider"),
    "EpisodeProcessor": ("memoryos.application.episode.episode_service", "EpisodeProcessor"),
    "HashingEmbeddingProvider": ("memoryos.infrastructure.providers.embedding_provider", "HashingEmbeddingProvider"),
    "InterventionDecision": ("memoryos.application.intervention.intervention_selector", "InterventionDecision"),
    "InterventionSelector": ("memoryos.application.intervention.intervention_selector", "InterventionSelector"),
    "JsonLLMMemoryExtractor": ("memoryos.application.memory.extractor", "JsonLLMMemoryExtractor"),
    "MemoryContext": ("memoryos.application.retrieval.memory_context_builder", "MemoryContext"),
    "MemoryContextBuilder": ("memoryos.application.retrieval.memory_context_builder", "MemoryContextBuilder"),
    "MemoryHook": ("memoryos.interfaces.hooks.memory_digest_hook", "MemoryHook"),
    "MemoryOperation": ("memoryos.application.memory.extractor", "MemoryOperation"),
    "MemoryStore": ("memoryos.infrastructure.repositories.memory_repository", "MemoryStore"),
    "MemoryUpdateContext": ("memoryos.application.memory.update_service", "MemoryUpdateContext"),
    "MemoryUpdateService": ("memoryos.application.memory.update_service", "MemoryUpdateService"),
    "ObservationContext": ("memoryos.domain.scene.observation", "ObservationContext"),
    "OpenAICompatibleChatProvider": ("memoryos.infrastructure.providers.openai_compatible", "OpenAICompatibleChatProvider"),
    "OpenAICompatibleEmbeddingProvider": ("memoryos.infrastructure.providers.openai_compatible", "OpenAICompatibleEmbeddingProvider"),
    "OpenAICompatibleRerankProvider": ("memoryos.infrastructure.providers.openai_compatible", "OpenAICompatibleRerankProvider"),
    "ReinforcementPolicyLedger": ("memoryos.application.learning.rl_calibrator", "ReinforcementPolicyLedger"),
    "RetrievalOrchestrator": ("memoryos.application.retrieval.retrieval_service", "RetrievalOrchestrator"),
    "RetrievalResult": ("memoryos.application.retrieval.retrieval_service", "RetrievalResult"),
    "RuleBasedExtractor": ("memoryos.application.memory.extractor", "RuleBasedExtractor"),
    "SessionManager": ("memoryos.application.session.session_manager", "SessionManager"),
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
