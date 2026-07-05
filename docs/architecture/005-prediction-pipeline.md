# Prediction Pipeline

The main pipeline is:

Observation -> NormalizeObservation -> SimilarBehaviorRetrieval -> CandidateActionPolicyRanking -> ActionContextRetrieval -> PolicyGate -> Executor -> PredictionLedger -> Async Commit.

PredictionEngine does not write durable Memory. It records PredictionLedger and returns a PolicyDecision. Long-term Memory, Behavior, and ActionPolicy updates are performed by async commit workers.
