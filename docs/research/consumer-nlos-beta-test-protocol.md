# Consumer NLOS iPhone beta test protocol

Status: Draft for internal validation

This protocol validates onboarding, direct-depth acquisition, privacy-bounded
diagnostics, and optional external NLOS presentation. It does not validate
iPhone-only around-the-corner sensing. Public ARKit scene depth is processed
direct line-of-sight geometry; physical NLOS remains blocked without raw
transient access and the live evidence required by ADR-330 and ADR-331.

## Test inputs, outputs, and assumptions

Inputs are an approved TestFlight build, one named physical iPhone or iPad, a
plain wall with safe walking clearance, and a tester who has accepted the beta
notice. An external RuView endpoint is optional and belongs to a separate test
mode.

Outputs are completion timing, aggregate quality metrics, bounded failure codes,
the displayed capability labels, and an optional diagnostic JSON file no larger
than 64 KiB. No raw image, depth map, point cloud, CSI, transient, location, or
person trajectory is collected by this protocol.

Assume the tester can install from TestFlight but does not have Xcode or sensing
expertise. Run indoors with adequate light, a charged device above 30 percent,
and at least one metre of clear space. Do not scan bystanders or private areas.

## Release prerequisites

1. Record app version, build number, reviewed commit, Xcode version, privacy
   manifest review, and release approver.
2. Confirm pull request CI passed Swift tests, simulator build, unsigned archive
   dry gate, workflow policy checks, and diagnostic contract validation.
3. Confirm the selected TestFlight build came from the approved release path.
4. Confirm test notes say that direct depth is not physical NLOS.
5. Confirm the feedback destination and explainer link are current.

## Tester procedure

1. Install the build from TestFlight and open it. Start the setup timer at first
   launch. The full path must finish in under five minutes.
2. Read the purpose and privacy screen. Confirm it says TestFlight installed the
   app and the RuView assistant is configuring it after installation.
3. Open the linked explainer, return to the app, and continue.
4. Confirm the capability screen identifies the device as supported or
   unsupported. An unsupported device must show useful next steps and must not
   display simulated data as live.
5. Grant camera permission when asked. Local-network permission should appear
   only if external live mode was deliberately selected. Precise location should
   not be requested in the default flow.
6. Point the rear sensor at a plain wall from roughly one to two metres away.
   Hold the device steadily and complete the 15 second calibration.
7. Follow the on-screen movement guide and complete the 30 second wall scan.
   Keep the wall visible. This measures direct depth and pose stability, not a
   hidden object.
8. Review the summary. Confirm it shows frame rate, depth coverage, pose
   stability, thermal state, interruptions, and the exact capability label.
9. Confirm the result says `direct_depth` and physical NLOS says
   `blocked_raw_transients_unavailable`, unless a separately approved external
   live source was used.
10. Open diagnostic preview. Confirm consent says raw capture is false, then
    choose share or discard. Sharing is optional and must never start
    automatically.
11. Stop the setup timer and record elapsed time. Submit the short feedback form
    without names, room details, images, network names, or location.

## Troubleshooting

| Symptom | Likely cause | Safe recovery | Expected evidence state |
|---|---|---|---|
| Device unsupported | No public scene-depth capability | Use a LiDAR-capable supported device or test replay explicitly | `unavailable`, never live |
| Camera permission denied | Permission was declined or restricted | Open system Settings if the tester chooses, then start a new attempt | `permission_denied` |
| Calibration fails | Too little valid depth, fast motion, interruption | Face a plain wall, improve lighting, hold steady, retry once | failed attempt remains failed |
| Depth coverage is low | Reflective, transparent, very dark, near, or distant surface | Use a matte wall at about one to two metres | `direct_depth` only |
| Pose stability is low | Fast movement or visually sparse environment | Move slowly and keep wall edges or room features visible | degraded or failed |
| App pauses during scan | Screen lock, call, app switch, thermal pressure | Return, cool device if needed, start a new 30 second scan | partial scan discarded |
| External track disconnects | Auth, network, expiry, tenant, or server failure | Check endpoint status and pair again; do not use replay as a live fallback | disconnected and stale tracks cleared |
| Diagnostic will not export | Size or schema guard rejected it | Capture the public failure code and app build; do not attach logs or raw files | export blocked |
| TestFlight build unavailable | Invitation, beta review, expiry, or build removal | Contact the beta coordinator; do not seek an unsigned IPA | not installed |

After three failed attempts, stop. Record the public failure code and build
number rather than repeatedly granting permissions or collecting more data.

## Negative tests for internal testers

1. Deny camera permission. Verify the app stops before capture and reaches useful
   guidance in at most two taps.
2. Background the app during calibration and during wall scan. Verify each
   partial result is discarded and cannot show complete.
3. Turn on Low Power Mode and warm the device through ordinary use. Verify coarse
   thermal state is reported without inventing an accuracy claim.
4. Disconnect WiFi during external live mode. Verify tracks expire and the UI
   does not substitute replay.
5. Attempt to encode diagnostics containing `image`, `depth_map`, `point_cloud`,
   `csi`, `transient`, `token`, `latitude`, `longitude`, or `trajectory`. Verify
   local rejection.
6. Construct canonical diagnostics of 65,536 and 65,537 bytes. Verify the first
   is permitted and the second is rejected.
7. Inspect the exported JSON. Verify there are no free-form logs, URLs, IP
   addresses, WiFi names, precise device identifiers, or user-entered notes.

## Feedback request

Ask only:

1. Did setup complete: yes or no?
2. Total elapsed seconds.
3. Which screen was confusing, if any?
4. Public failure code, if any.
5. Did the direct-depth and physical-NLOS distinction make sense: yes or no?
6. Optional aggregate diagnostic attachment after preview.

Do not request a video of the room, screenshot containing a person, raw capture,
precise location, Apple account, device serial, or network credentials.

## Pilot scorecard

Run at least 20 compatible-device attempts before external expansion. Report:

1. completion rate, target at least 90 percent;
2. median completion time, target at most 180 seconds;
3. p95 completion time, target at most 300 seconds;
4. calibration and scan failure rates by public code;
5. permission-denial recovery rate;
6. diagnostic size distribution and forbidden-field count, target zero; and
7. proportion correctly understanding that direct depth is not NLOS, target at
   least 90 percent.

## Acceptance test

One nontechnical tester must complete steps 1 through 10 on a named physical
device in under five minutes. Calibration must run for 15 seconds, wall scan for
30 seconds, the result must remain `direct_depth` with physical NLOS blocked,
and any exported diagnostic must be at most 64 KiB with zero forbidden raw or
identifying fields.
