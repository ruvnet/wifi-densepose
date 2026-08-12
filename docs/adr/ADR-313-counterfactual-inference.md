# ADR-313: Counterfactual inference — generative spatial reasoning

- **Status**: Accepted — initial implementation (ADR-300 phase 3)
- **Date**: 2026-08-11
- **Deciders**: ruv
- **Tags**: inference, generative, counterfactual, rf-twin, fusion, uncertainty, phase-3

## Context

This ADR is a child of **ADR-300** (perception substrate program) and owns
primitive #13, *counterfactual inference*. In the ADR-300 DAG it is a phase-3,
research-forward primitive that sits on top of the fused world state: it
**consumes ADR-311** (real sensor fusion) for the current fused estimate and
**ADR-315** (digital RF twin) for the twin's expected measurement
distributions. It is design intent, authored as Proposed, and is expected to be
revised as the phase-1 spine and the phase-2 fusion layer land.

RuView today reasons discriminatively: a task head maps measurements to a label
or a pose. That answers "what does the classifier say?" but not the questions an
operator actually asks — *would these RF measurements still make sense if nobody
were present? Does one person explain the observation better than two?* Those
are counterfactual questions, and a classifier cannot answer them because it has
no model of what a measurement *should* look like under a hypothesized world
state. A discriminative head asked about an empty room simply emits its
best-effort label; it cannot say "the observation is better explained by
absence."

The step this ADR proposes is toward a **generative spatial model**: given a
hypothesized scene state (occupancy, count, coarse positions) and the ADR-315
twin's propagation model for the deployment, predict the *expected* measurement
distribution, then score how well each hypothesis explains the observed
measurement. The best-explaining hypothesis — including the *nobody-present*
null hypothesis — is the answer, and the margin between hypotheses is a
first-class uncertainty signal.

Relevant existing assets to build on rather than duplicate:

- **ADR-311** (fusion) already produces the fused world estimate and its
  covariance; the counterfactual layer scores hypotheses *relative to* that
  estimate rather than re-fusing raw measurements.
- **ADR-315** (RF twin) is the generative forward model — per-deployment
  geometry, radio locations, and expected measurement distributions. This ADR
  is a *consumer* of the twin's forward simulator, not a second simulator.
- **ADR-302** (OOD/observability) already owns the `UNKNOWN` verdict; the
  null-hypothesis ("nobody present better explains this than any occupancy
  hypothesis") and the "no hypothesis explains this" case route through ADR-302,
  not a parallel gate.
- `frame::EvidenceLevel` L0–L5 (ADR-282) and the ADR-304 evidence engine
  account for the resulting confidence.

## Options considered

1. **Keep only discriminative heads.** Rejected: cannot express absence,
   cannot compare "one person vs. two" as competing explanations, and gives a
   confident label even when no world state explains the data.
2. **A second, independently trained generative network with its own forward
   model.** Rejected for the default path: duplicates the ADR-315 twin's
   propagation model, invites the two models to disagree, and multiplies the
   surface that must be validated. Reserved only if the twin's analytic forward
   model proves insufficient for a phenomenon.
3. **A hypothesis-scoring layer that uses the ADR-315 twin as the forward model
   and the ADR-311 fused state as the hypothesis prior, routing low-margin and
   null-dominant cases to the ADR-302 UNKNOWN verdict.** Chosen.

## Decision

Define a **counterfactual inference layer** that scores a small set of scene
hypotheses against observed measurements using the digital RF twin as the
generative forward model.

### 1. Hypothesis set

- Hypotheses are drawn from the ADR-311 fused state and its neighbourhood: the
  current estimate, the **null hypothesis** (nobody present), and a bounded set
  of nearby alternatives (±1 occupant, shifted position). The fused estimate
  supplies the prior so the search stays small and grounded rather than
  enumerating an open world.
- The hypothesis space is expressed over the **ADR-306** canonical ontology
  (`Space`/`Zone`, occupant count, coarse position), so a counterfactual result
  is a governed spatial statement, not an opaque score.

### 2. Forward model and scoring

- For each hypothesis, query the **ADR-315 twin** for the expected measurement
  distribution given that scene state and the deployment's propagation model.
  Score the observed measurement's likelihood under each hypothesis's expected
  distribution.
- The answer is the maximum-likelihood hypothesis; the **margin** between the
  top hypotheses (and between the top hypothesis and the null) is the
  confidence signal, carried into the ADR-304 evidence engine.

### 3. Routing to UNKNOWN

- When the null hypothesis dominates, the layer reports *absence*, not a
  low-confidence occupancy label.
- When **no** hypothesis explains the observation well (all likelihoods low, or
  the winning margin below threshold), the result routes to the **ADR-302**
  `UNKNOWN` verdict — the observation is outside what the twin can explain, and
  the honest output is "I cannot account for this," never a forced label.

### Evidence discipline

- Twin-predicted distributions are a **simulation** (evidence level L0 per
  ADR-282) labelled `SYNTHETIC`; a counterfactual verdict inherits the evidence
  level of its weakest input and is never presented as camera-grade ground
  truth (CLAUDE.md honesty rule).
- Any accuracy statement about counterfactual discrimination (e.g. "distinguishes
  one occupant from two") requires the mean-pose-style baseline discipline of
  CLAUDE.md, a leakage-free held-out split, and a reproducer before it may be
  tagged `MEASURED`. This ADR asserts **no** such number.

## Consequences

- RuView gains the ability to answer absence and "which explanation is better"
  questions that discriminative heads structurally cannot — a step toward
  generative spatial reasoning and a differentiator for security and
  facility-monitoring applications where *absence* is the valuable signal.
- Quality is bounded by the fidelity of the ADR-315 twin's forward model and the
  ADR-311 fused prior; the layer reports margins and defers to ADR-302 UNKNOWN
  rather than overstating a coarse model.
- Hard dependency on ADR-311 (fused state and covariance) and ADR-315 (forward
  model); this ADR builds neither a fusion engine nor a propagation simulator of
  its own.
- Being phase 3, this is design intent sitting on the fused world state; it is
  expected to be revised as ADR-311 and ADR-315 land, and it is not implemented
  by the phase-1 swarm.

## Validation

- Unit tests: hypothesis likelihood scoring is a deterministic function of
  observed measurement + hypothesis + twin parameters; the null hypothesis wins
  on a synthesized empty-room measurement; a two-occupant measurement scores the
  two-occupant hypothesis above the one-occupant hypothesis on a controlled
  synthetic case.
- Integration test: measurements the twin cannot explain (out-of-model
  scattering) drive the layer to the ADR-302 UNKNOWN verdict rather than a
  forced occupancy label; margins propagate into the ADR-304 evidence engine.
- Held-out discrimination (deferred, real-silicon): one-vs-two and
  presence-vs-absence discrimination on a leakage-free held-out split with a
  mean-pose baseline, reported as `MEASURED` with a reproducer. Until then all
  counterfactual output is `SYNTHETIC`/L0. No discrimination accuracy number is
  asserted by this ADR.
