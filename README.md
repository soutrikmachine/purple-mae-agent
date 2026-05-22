# Purple MAE Negotiator

A purple agent for the AgentBeats × AgentX **Meta-Game Negotiation Assessor**
([leaderboard](https://agentbeats.dev/agentbeater/meta-game-negotiation-assessor),
[green agent](https://github.com/RDI-Foundation/MAizeBargAIn-agentbeats)),
packaged for submission via [Amber](https://github.com/RDI-Foundation/amber).

---

## Abstract

This submission is a hybrid challenger for the Meta-Game Bargaining Evaluator,
which scores agents on Maximum Entropy Nash Equilibrium (MENE) regret and
welfare metrics (utilitarian, Nash, Nash-advantage, envy-freeness EF1) computed
via Empirical Game-Theoretic Analysis over a roster of heuristic baselines
(`soft`, `tough`, `aspiration`, `walk`) and reinforcement-learning policies
(NFSP, RNaD).

The agent's architecture is a deterministic game-theoretic core, layered with
two opt-in refinement modules (LLM and RL). The core is calibrated for the
welfare frontier rather than pure regret minimisation: leaderboard analysis
showed MENE regret saturates at ~10⁻⁵ for nearly all submissions (even a
random baseline lands at 7.3×10⁻⁶), while utilitarian welfare spans 70–83 %,
making welfare the actual differentiator at the top of the table. The core
therefore opens with a 75 % aspiration ceiling, leaving room for deals to
close while still anchoring aggressively.

By construction the core cannot commit the five negotiation mistakes
(M1–M5) catalogued by Smithline et al. (2025). Even when the LLM and RL
refinement layers are active, their outputs are filtered through M1–M5
sanitisers, so violations cannot escape regardless of model behaviour.

The agent runs in pure-strategy mode at **$0 cost and ~5–10 minutes** for a
full 50-game benchmark, or in LLM-refined mode at $0.30–$13 and 30 min – 4 h
depending on model. It speaks A2A on port 9009 against the green's
`RemoteNegotiator` protocol, and ships with an Amber manifest for one-step
submission to the AgentBeats leaderboard.

---

## Methodology

### 1. Game model

OpenSpiel bargaining (Lewis et al. 2017, as wrapped by MAizeBargAIn):

- **Items.** *T* item types with public quantities **Q** = (q₁, …, q_T).
  Default (q₁, q₂, q₃) = (7, 4, 1).
- **Valuations.** Each agent draws a private **v** ∈ [1, 100]^T uniformly.
  Self's vector is observable to self; the opponent's is **not exposed** to
  remote challengers (the green's `_build_observation` explicitly strips
  `valuations_opp` and `batna_opp`).
- **BATNA.** Each agent has a private outside option b ≥ 0. If no deal is
  struck, both sides receive their (discounted) BATNAs.
- **Discount.** Per-round γ ∈ {0.9, 0.98}. Round-r payoff multiplied by
  γ^(r−1).
- **Horizon.** R ∈ {3, 5}. The lightweight runner used by the green can
  truncate at round 2; our concession schedule handles both.
- **Actions per turn.** Propose an allocation (a_self, a_other) with
  a_self + a_other = Q; or, given an offer, accept / reject (the green parses
  `accept: false` as "continue bargaining, get a counter-offer next round").

The value of allocation **a** under **v** is the dot product
v(**a**) = Σₜ vₜ · aₜ.

### 2. The five mistakes (M1–M5)

Smithline et al. (2025) isolate five failure modes that account for most of
the regret gap between LLM negotiators and game-theoretic baselines:

| Mistake | Description |
|---|---|
| **M1** | Propose worse-for-self than your own previous offer (within one game) |
| **M2** | Propose worse-for-self than your BATNA |
| **M3** | Propose `[0, …, 0]` or all-of-Q (degenerate divisions) |
| **M4** | Accept an offer worth less than your BATNA |
| **M5** | Walk away from an offer worth more than your BATNA |

We enforce all five as **structural invariants**. The deterministic core and
the sanitisers each check them independently (defence in depth), and the
optional LLM/RL refinement layers cannot violate them no matter what they
output. 14/14 property tests verify the invariants over 500+ randomised
trials each.

### 3. Deterministic core (the spine of the agent)

A value-maximising agent with a time- and discount-aware aspiration schedule
and an explicit expected-continuation accept rule.

**Aspiration target** for round r of R, given max-attainable v(Q):

```
floor    = b_self + 1                                  # M2 anchor
ceiling  = α · v(Q) + (1 − α) · floor                  # α = OPENING_AGGRESSIVENESS
progress = (r − 1) / (R − 1) + 0.5 · (1 − γ^(R−r))     # discount-aware
target   = max(floor, ceiling − (ceiling − floor) · min(1, progress))
```

α = 0.75 is the default (welfare-friendly sweet spot from leaderboard
analysis). Overridable via `OPENING_AGGRESSIVENESS`. A higher α reproduces
"always demand max" — strictly worse on welfare with no improvement to regret.

**Greedy allocation** given a target value: items assigned in descending
order of *self-priority* `v_self,t − E[v_opp,t]` = `v_self,t − 50.5` under
the U[1,100] prior. Self takes high-priority items until the target is met;
the opponent gets the rest. This biases toward Pareto-improving splits —
we keep items where we have a comparative advantage.

**Accept/reject** when offered allocation `a` with self-value `v`:

```
continuation = max(b_self, γ · target_{r+1})
accept       = (v ≥ b_self) AND (v ≥ continuation)
```

When r = R (final round) and v ≥ b_self, the M5 sanitiser overrides any
rejection — rejecting in the final round equals walking, and we cannot
recover continuation value.

**Per-game memory.** The green's observation carries `pair` and `game_index`.
We use `(pair, game_index)` as a session key for the M1 anchor (previous
self-proposal value), so memory persists across rounds within one game and
resets cleanly between games. This is more reliable than A2A `context_id`,
which can vary by transport.

### 4. Welfare-frontier calibration (why α = 0.75)

Inspection of the live leaderboard (April–May 2026, 68+ submissions):

| Metric | Top score | Top-20 spread |
|---|---|---|
| MENE Regret (lower better) | ~4.5×10⁻⁶ | 4.5×10⁻⁶ – 8×10⁻⁶ |
| Utilitarian Welfare % | 82.94 | 79.78 – 82.94 |
| Nash Welfare % | (visible) | clustered top |
| Nash Welfare Advantage % | (visible) | clustered top |
| EF1 % | (visible) | clustered top |

Two facts informed the design:

1. **Regret saturates.** Top regret scores are within numerical noise of the
   MILP solver; even a random baseline scores 7.3×10⁻⁶. The headroom past
   "don't commit M1–M5" is essentially zero.
2. **Welfare differentiates.** UW spans 70–83 %, a 13-point range. Top
   scorers (Necentt, jenova13q, va-av-8, FanisNgv) are tightly bunched at
   80+ %. An agent opening at α=1.0 leaves the opponent with one unit;
   opponents walk or reject; both sides take BATNA; UW collapses.

Default α = 0.75 puts round-1 self-value at ~75 % of max-attainable while
leaving the opponent ~3–5 units. For the standard test case
(v=(60, 30, 90), Q=(7, 4, 1), b=80):

```
α = 1.00 → self [7, 3, 1], opp [0, 1, 0], self_value=600, opp_units=1
α = 0.75 → self [7, 0, 1], opp [0, 4, 0], self_value=510, opp_units=4   ← default
α = 0.60 → self [6, 0, 1], opp [1, 4, 0], self_value=450, opp_units=5
```

### 5. Optional LLM refinement layer (USE_LLM)

When `USE_LLM=true`, every action goes to an LLM (any OpenRouter model, or
Anthropic direct) along with the deterministic baseline. The LLM returns a
possibly-modified action in JSON. Three guarantees:

- LLM output filtered through `sanitise_proposal` / `sanitise_decision` —
  any M1–M5 violation is detected and overwritten.
- 15 s timeout with one retry; on failure the deterministic baseline is used.
- Missing API key or import error degrades silently to deterministic mode.

The LLM cannot make the agent **worse than the deterministic core on M1–M5**;
it can only refine within the safe envelope.

### 6. Optional RL refinement layer (USE_RL)

Stub by default. When `USE_RL=true` and an NFSP/RNAD checkpoint is provided
via `RL_CHECKPOINT_PATH`, the negotiator runs both the heuristic and the RL
proposer, then picks the higher-value sanitised candidate. This is the
**policy-mixture best-response** move in EGTA terms — exactly the strategy
that exploits RL baselines, which are best-response-fragile.

The real implementation (loading `nfsp_bg5.pt` via PyTorch + OpenSpiel,
inverting the integer action space) is **stubbed** in this v1 release. The
integration point is `rl_proposer.maybe_propose()`. Enabling for real
requires:

1. Pull the checkpoint from the green's `rl_agent_checkpoints/` directory.
2. Add `open_spiel` and `torch` to `requirements.txt` (~600 MB image bloat).
3. Implement `_propose_from_checkpoint()` per the TODOs in `rl_proposer.py`.

### 7. Submission topology (Amber)

`amber-manifest.json5` declares:

- The container image (`rimodock/purple-mae-agent:latest`).
- The A2A endpoint on port 9009.
- Config schema with one optional secret (`openrouter_api_key`, marked
  `secret: true`).
- Environment variables exposing all tuning knobs (`USE_LLM`, `USE_RL`,
  `OPENING_AGGRESSIVENESS`, `LLM_MODEL`, etc.) so they can be flipped
  without rebuilding the image.
- A single `a2a` capability export, picked up by the scenario topology.

### 8. Empirical cost and duration

A full benchmark with `games=50` produces ~2 000 A2A calls to our agent
(13 ordered pairs × 50 games × ~3 rounds/game average).

| Mode | Wall time | Cost |
|---|---|---|
| Pure-strategy (default) | 5–10 min | **$0** |
| OpenRouter `anthropic/claude-sonnet-4.6` | ~30–50 min | ~$0.30 |
| OpenRouter `openai/gpt-4o-mini` | ~30–60 min | ~$0.40 |
| OpenRouter `deepseek/deepseek-chat` | ~30–60 min | ~$0.80 |
| OpenRouter `anthropic/claude-haiku-4.5` | ~30–60 min | ~$3.50 |
| OpenRouter `anthropic/claude-sonnet-4` | ~1.5–2.5 h | ~$8 |
| OpenRouter `anthropic/claude-opus-4.7` | ~3–4 h | ~$13 |

Cost is linear in `games`. Use `games=10` for fast iteration during development.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│ A2A server (main.py) — port 9009                     │
│   └─ negotiator.py: parse + dispatch                 │
│        ├─ accepts the green's exact observation      │
│        │   schema (pair, game_index, round_index,    │
│        │   pending_offer.offer_allocation_opp, ...)  │
│        ├─ emits the green's accepted response keys   │
│        │   (allocation_self/other for PROPOSE;       │
│        │    accept:true/false for ACCEPT_OR_REJECT;  │
│        │    plan_allocation hint when rejecting)     │
│        ├─ strategy.py: deterministic spine           │
│        │     • aspiration target (α-controlled)      │
│        │     • greedy comparative-advantage split    │
│        │     • expected-continuation accept rule     │
│        │     • M1-M5 sanitisers (defence in depth)   │
│        ├─ llm.py: optional OpenRouter/Anthropic      │
│        │     • 15s timeout, 1 retry                  │
│        │     • output clamped by sanitisers          │
│        └─ rl_proposer.py: optional NFSP/RNAD         │
│             • stub by default; real impl behind      │
│               TODOs in the module                    │
└──────────────────────────────────────────────────────┘
```

## Quick start

### Pure-strategy mode (submit this first)

```bash
pip install -r requirements.txt
python main.py --host 0.0.0.0 --port 9009
```

Agent card: `http://localhost:9009/.well-known/agent-card.json`.

### LLM-refined mode

```bash
cp sample.env .env
# Edit .env:
#   USE_LLM=true
#   LLM_MODEL=anthropic/claude-sonnet-4.6
#   OPENROUTER_API_KEY=sk-or-v1-...
python main.py
```

### Docker

```bash
docker build -t rimodock/purple-mae-agent:latest .
docker run -p 9009:9009 rimodock/purple-mae-agent:latest
```

### Test

```bash
python test_strategy.py
# 14/14 passed
```

## Submission via Amber

```bash
# 1. Build & push the image
docker build -t rimodock/purple-mae-agent:latest .
docker push rimodock/purple-mae-agent:latest

# 2. Register on agentbeats.dev
#    - paste rimodock/purple-mae-agent:latest as the image URL
#    - Amber reads amber-manifest.json5 and prompts for secrets

# 3. Run a scenario
#    - fork the leaderboard repo, edit scenario.toml with your agent's ID
#    - or use the Quick Submit form on agentbeats.dev
```

## Tuning knobs (all settable via env, no rebuild needed)

| Variable | Default | Purpose |
|---|---|---|
| `OPENING_AGGRESSIVENESS` | `0.75` | Lower → more welfare; higher → more BATNA-protective |
| `USE_LLM` | `false` | Enable LLM refinement |
| `LLM_PROVIDER` | `openrouter` | Or `anthropic` |
| `LLM_MODEL` | `anthropic/claude-sonnet-4.6` | Any OpenRouter model id |
| `LLM_TIMEOUT_S` | `15` | Per-call timeout, sec |
| `USE_RL` | `false` | Enable RL proposer (stub currently) |
| `RL_ALGO` | `nfsp` | Or `rnad` |
| `RL_CHECKPOINT_PATH` | (empty) | Path to checkpoint file |

## Files

```
.
├── main.py                # A2A server entry point
├── negotiator.py          # parser + dispatcher matching the green's protocol
├── strategy.py            # deterministic core: aspiration, accept rule, M1-M5
├── session_store.py       # (pair, game_index)-keyed M1 anchor
├── llm.py                 # optional OpenRouter/Anthropic refinement
├── rl_proposer.py         # optional NFSP/RNAD proposer (stub)
├── test_strategy.py       # 14 tests: M1-M5 + protocol round-trips
├── amber-manifest.json5   # AgentBeats / Amber submission manifest
├── Dockerfile             # publishable image (python:3.11-slim, port 9009)
├── requirements.txt       # a2a-sdk pinned to <1.0.0 (critical)
├── agent.toml             # legacy AgentBeats descriptor
├── sample.env
├── .gitignore
└── README.md
```

## Known caveats

- **RL is a stub.** The integration point exists, but loading the actual
  NFSP/RNAD checkpoints requires pulling the `.pt`/`.pkl` files from the
  green's repo and implementing the OpenSpiel-action-space decoding. v2 work.
- **0.75 is a hypothesis, not a measured optimum.** A proper sweep
  (α ∈ {0.6, 0.65, 0.7, 0.75, 0.8, 0.85} × 10 games each, against the green)
  would identify the true Pareto-optimal value. Easy to do once we have a
  first leaderboard score to point sweeping at.
- **A2A SDK is pinned to 0.3.x.** The 1.0 release removed
  `A2AStarletteApplication` (their migration guide), and the green agent uses
  `protocol_version=0.3.0`. Do not unpin without rewriting `main.py`.

## References

1. Smithline, G., Mascioli, C., Chakraborty, M., & Wellman, M. P. (2025).
   *Measuring Competition and Cooperation in LLM Bargaining.*
2. Li, Z., & Wellman, M. P. (2024). *A Meta-Game Evaluation Framework for
   Deep Multiagent RL.* IJCAI.
3. Heinrich, J., & Silver, D. (2016). *Deep Reinforcement Learning from
   Self-Play in Imperfect-Information Games.* (NFSP)
4. Perolat et al. (2022). *Mastering the game of Stratego with model-free
   multiagent reinforcement learning.* Science. (RNaD)
5. Lewis, M., et al. (2017). *Deal or No Deal? End-to-End Learning for
   Negotiation Dialogues.* EMNLP.

## License

MIT.
