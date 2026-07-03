# Safety Policy

PolicyGate evaluates:

- prediction confidence
- candidate score
- action risk level
- auto_execute_allowed
- recent negative feedback
- cooldown status
- resource and skill availability
- user policy memory

Low-risk actions may execute only when authorized and high confidence. Medium-risk actions default to ask_user. High-risk and private actions are blocked from automatic execution.
