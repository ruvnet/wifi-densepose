# ADR 357: Raw CSI calibration sample integrity

Status: Accepted

Date: 2026-09-06

## Context

The empty room field model requires 1,000 accepted observations over at least
600 seconds. The sensing server previously invoked calibration from both the
raw CSI path and the one second edge vital path. Edge vital packets did not add
a CSI observation, so that path repeatedly fed the existing history tail. A
counter could therefore satisfy the frame gate without representing 1,000
independent CSI packets.

Node liveness also combined edge vital and CSI traffic. A source emitting only
edge vital summaries could appear live enough to start calibration even though
it had no fresh raw CSI. Finally, cloning the complete per node history before
each feed copied as much as 204,800 bytes for a 100 frame, 256 subcarrier
window even though calibration consumes only the current observation.

The earlier single installation result recorded in ADR 355 predates this
admission boundary. Its frame count is not accepted as evidence of independent
CSI coverage. It remains useful only as diagnosis of the failure mode and must
not be promoted as a production bootstrap prior.

## Decision

1. Only a grid admitted raw CSI packet may advance field calibration. Edge
   vital packets update their own health and presentation state but never feed
   the field model.
2. Each calibration session stores the last admitted wrapping sequence for
   every contributing node. Duplicate sequences are ignored. Normal forward
   progress and the `u32::MAX` to zero wrap are accepted.
3. A packet one through four sequence values behind the current high water mark
   is bounded UDP reordering. It is recorded and dropped without feeding the
   model or moving the high water mark. A backward depth above four, or an
   ambiguous half range sequence, latches that node as faulted for the remainder
   of the session. Status exposes the policy, reorder window, per node reorder
   count, maximum observed depth, and `sequence_fault_node_ids`. Finalization
   returns `calibration_sequence_discontinuity` after a fault. The operator must
   cancel and restart the empty room capture.
4. A requested source may start calibration only when it has a grid admitted
   raw CSI packet newer than five seconds. General node liveness is not enough.
   The node endpoint reports both `csi_status` and `csi_last_seen_ms` so clients
   can explain the gate before the operator begins. Calibration status reports
   `last_sequence_by_node` so admission growth and sequence identity can be
   observed atomically without exposing CSI values.
5. Start, cancel, reset, and process initialization clear sequence admission
   state. A failed feed does not advance the sequence cursor, so the same
   packet may be retried after a transient model input failure.
6. The field bridge receives the current amplitude slice rather than a cloned
   history. Canonical grid normalization remains unchanged.

## Consequences

Calibration frame count now means unique, forward raw CSI observations within
one uninterrupted sequence epoch. A vitals only node cannot produce a
bootstrap model. Late packets never become calibration evidence.

A physical Node 5 capture observed sequence 4,119,566 followed by 4,119,563,
4,119,564, and 4,119,565 in one receive batch, then normal forward progress.
The window of four covers that measured depth of three with one frame of
margin. At the firmware acceptance ceiling of 50 Hz, the window represents no
more than 80 milliseconds. Larger regressions remain failed closed.

The current wire format still has no authenticated boot epoch or stable device
identity. A restart very early in a session or a same endpoint adversary cannot
be distinguished cryptographically. A future ADR 018 revision must carry an
authenticated device identifier and a random per boot identifier before the
server claims exact restart or identity collision detection.

No new raw CSI is persisted. Sequence cursors and fault markers remain bounded
in memory to active calibration sources. The persisted bootstrap format and
its negative only authority are unchanged.

## Implementation

1. `NodeState` records a separate accepted CSI arrival clock.
2. `AppStateInner::maybe_feed_calibration_frame` owns source binding, sequence
   ordering, fault latching, model feed, and cursor commit.
3. `field_bridge::maybe_feed_calibration` accepts exactly one current amplitude
   slice and rejects an empty observation.
4. Calibration start, status, stop, cancel, reset, and node status expose and
   enforce the new admission contract.

## Acceptance test

Unit tests must prove first packet acceptance, forward progress, duplicate
rejection, source isolation, maximum counter wrap, empty input retry, the
measured depth three reorder pattern without a model feed, depth four
acceptance, depth five latching, half range rejection, session clear, receipt
visibility, and typed stop refusal.

On physical hardware, begin a Node 5 empty room capture and observe a fresh
sixty second interval. Calibration frame growth must match unique grid admitted
Node 5 CSI sequences. Edge vital packets alone must produce zero growth. The
source must retain `csi_status=active`, sequence faults must remain empty, and
numeric vital output must remain absent. Any bounded late packet must appear in
the reorder receipts while leaving the model count unchanged. Complete both the
1,000 frame and 600 second gates, then run the twelve sample server controlled
bootstrap holdout. Do not claim human detection improvement until a separate
consented occupied holdout demonstrates retained recall.
