# 04 — Compliance and Regulatory Line

**Non-negotiable:** VEIL uses compliant waveform controls and **never jams.**
This file states the legal basis for that line and why every VEIL control falls
on the compliant side of it. It is engineering analysis, not legal advice; a
deployment in a given jurisdiction needs its own regulatory review.

---

## 1. The statutory line (United States)

The prohibition is on **interfering with others' transmissions**, not on how you
shape **your own** signal.

| Authority | What it prohibits |
|---|---|
| **47 U.S.C. §333** | *Willful or malicious interference* with any licensed/authorized radio station or U.S. Government station |
| **47 U.S.C. §302a(b)** | Manufacture, import, marketing, sale, or *operation* of non-compliant devices (jammers cannot be certified — their sole purpose is interference) |
| **47 U.S.C. §301** | Requires a license/authorization to transmit; a jammer can never be authorized |
| **47 U.S.C. §501 / §503** | Criminal penalties and forfeitures; FCC cites fines up to $112,500 per violation, **no exemptions** for business/residence/vehicle |

The distinguishing element of jamming is **intent to interfere plus effect on a
third party's link.** A device that shapes its own standards-conformant emission
— staying within transmit-power and spectral-mask limits, still type-certifiable
— is not a jammer.

---

## 2. Why each VEIL control is compliant

| Control | Compliance argument |
|---|---|
| **Keyed precoder rotation** | Orthogonal ⇒ preserves the report's energy exactly ⇒ **adds no power on top of anyone's signal.** It is still a valid precoder within the 802.11 feedback format. Machine-checked: energy ratio = 1.000000 (`compliance` module) |
| **Feedback quantization / dither** | Reports angles the standard already allows, at the standard's resolution; sub-step dither stays within the quantization envelope. No emission change beyond the node's own frame |
| **Sounding-cadence randomization** | Chooses *when* the node sends its own NDP soundings, within permitted timing. Sending fewer/jittered soundings never interferes with another station |
| **MU-group / stream-mapping shuffle** | Rearranges the node's own spatial mapping; a normal in-spec transmit choice |

None of the four transmits *to prevent* another station from communicating; none
adds out-of-mask energy; each passes normal type certification. Contrast a
jammer, whose defining purpose is to emit energy that denies others service.

---

## 3. The energy-conservation proof as a compliance artifact

VEIL turns "not jamming" from a promise into a **checked property.** The
`compliance::ComplianceReport` audits each protection step:

```
input_energy   = ‖report_before‖²
output_energy  = ‖report_after‖²
energy_ratio   = output_energy / input_energy          # ≈ 1.0 for a rotation
energy_conserving = |energy_ratio − 1| ≤ 1e-2
adds_interfering_energy = false                          # by construction
is_compliant  = energy_conserving ∧ ¬adds_interfering_energy
```

A regulator, an auditor, or the runtime attestation layer (ADR-141) can read the
report and verify the shield is a waveform-shaping control, not an interference
source. On the reference experiment the measured ratio is **1.000000**.

---

## 4. Jurisdictional notes

- **EU (GDPR framing).** Covert WiFi body-sensing of vital signs is sensitive
  health data and "almost certainly illegal under GDPR," but effectively
  unenforceable (receivers are undetectable) — which is precisely why a
  *technical* control is needed. VEIL as a transmit-side control does not itself
  raise GDPR issues; it reduces the personal data an attacker can derive.
- **RF-emission rules are jurisdiction-specific.** The energy-preserving property
  is the portable core of the compliance argument, but power/mask/timing limits
  differ by region and band; a deployment must confirm local rules.
- **Deliberate transmit-nulling toward a *located* sniffer** (steering a spatial
  null at a known passive receiver) is still the node's own emission and adds no
  interference, but is more aggressive and should get explicit regulatory review
  before field use. It is not part of the default VEIL profile.

---

## Sources

- 47 U.S.C. §333: https://www.law.cornell.edu/uscode/text/47/333
- 47 U.S.C. §302a: https://www.law.cornell.edu/uscode/text/47/302a
- FCC Jammer Enforcement: https://www.fcc.gov/general/jammer-enforcement · https://www.fcc.gov/enforcement/areas/jammers
- FCC Cell/GPS Jamming guidance: https://www.fcc.gov/general/cell-phone-and-gps-jamming
- FCC 14-92 enforcement order: https://docs.fcc.gov/public/attachments/FCC-14-92A1.pdf

*Caveat: FCC pages were cross-verified against Cornell LII; this is engineering
analysis, not legal advice.*
