# ADR 347: Rate aware ESP32 temporal sensing

## Status

Accepted. Implemented in firmware 0.8.8. The timing and transport path is
physically qualified on ESP32 C6. The independent raw transport path is also
physically qualified on an ESP32 S3 running Tier 0. Held out inference accuracy
remains required.

## Context

The ESP32 firmware creates CSI opportunities by sending one byte ICMP probes to
the connected access point. The traffic source is configured for 50 Hz, but the
delivered CSI cadence varies with channel contention and callback safety gates.
A physical ESP32 C6 produced 28 to 37 callbacks per second during the baseline
capture. Firmware 0.8.5 then exposed that the old per-interval estimator saw
only 12 to 16 Hz because WiFi replies arrived in short bursts separated by
longer gaps. The filters still consumed those burst frames, so excluding them
from the clock estimate was incorrect.

The edge DSP estimates its sample rate from timestamps so that breathing,
heartbeat, motion, and future Doppler features stay in physical Hertz. That
estimator was capped at 30 Hz. Once the actual cadence exceeded the cap, every
temporal feature was scaled against the wrong clock.

Physical firmware 0.8.5 validation corrected that initial diagnosis. Although
the callback path received 26 to 40 frames per second, Tier 2 on the unicore C6
processed an irregular subset that converged toward the 8 Hz estimator floor.
The right design is not to force the edge DSP to match raw capture. The paths
need independent, explicit cadence contracts.

The device free gesture preprint at
`https://www.preprints.org/manuscript/202602.0018` reinforces the importance of
timestamp correct Doppler features, but its 100 Hz controlled link is not a
safe firmware default for RuView. Existing S3 and C6 evidence records WiFi ISR
and packet buffer failures under sustained callback pressure above 50 Hz.

## Decision

1. Make the connected STA probe rate a build time setting from 10 through 50
   Hz, with a default and hard ceiling of 50 Hz.

2. Track the delivered DSP cadence by counting every processed frame interval
   over one second timestamp windows, then smooth successive windows in an 8
   through 60 Hz estimator range. The 60 Hz estimator ceiling accommodates
   timestamp jitter; it does not authorize more than 50 Hz callback processing.

3. Reject incomplete windows below one second and stalled windows above three
   seconds. Do not discard valid burst frames from the estimated clock.

4. Surface the DSP rate in the one second controller diagnostic so hardware
   validation can compare callback yield with the clock used by temporal
   filters.

5. Keep raw CSI on the wire at the independent network cadence. Rate-limit the
   C6 on-device Tier 1 and Tier 2 DSP input to a uniform 8 Hz. Physical 0.8.7
   evidence showed that a requested 10 Hz input still converged to 8.0 through
   8.4 Hz under Tier 2 load, while raw delivery remained 30 through 40 pps.
   Eight hertz retains a 4 Hz Nyquist limit for the 0.1 through 2.0 Hz vital
   bands without creating a backlog. The S3 default remains 20 Hz.

6. STFT, spectrogram gating, and learned temporal
   classification remain host or iPhone responsibilities where memory,
   rollback, and held out evaluation are stronger.

## Consequences

Heartbeat, respiration, and motion features receive a stable timestamped clock
instead of an accidental subset determined by C6 backlog. Operators can lower
the probe or DSP load for constrained networks without editing source. The host
still receives the higher-rate raw stream for richer Doppler processing.

This does not prove vital sign accuracy or gesture recognition. Higher temporal
fidelity only improves the representation available to a separately validated
model. The 50 Hz ceiling also means the paper's 100 Hz results are not directly
transferable.

## Acceptance test

On a physical C6, run at least five minutes after flashing. Pass when the boot
log reports the configured probe and DSP rates, the controller converges within
one hertz of the configured DSP cadence, raw callback yield remains at least 20
pps, no steady-state ENOMEM, watchdog, panic, or reboot occurs, and the fail
closed occupancy invariant remains zero contradictions for at least 30 absent
packets.
