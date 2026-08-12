# 06 — Market and Buyers

Facts are tagged **VERIFIED** (from a cited source), **CLAIMED** (asserted by a
vendor/analyst/press source), or **SPECULATIVE** (our inference). Market figures
are third-party projections, not independent measurements.

---

## 1. Why now

- **The threat is standardized and commercializing (VERIFIED/CLAIMED).** IEEE
  802.11bf was published Sep 2025; silicon (Infineon AIROC Wi-Fi 7 ACW741x,
  Qualcomm Dragonwing) lists 802.11bf sensing in 2026 briefs; Origin AI's
  embedded-sensing program targets late-2026 deployment; Plume/Cognitive Systems
  WiFi Motion is the largest deployed sensing footprint today.
- **The standards body declined to fix privacy (VERIFIED).** The BFI
  "secure transmission mechanism" proposal (802.11-23/0782) was **withdrawn**;
  802.11bf shipped with no privacy protections. This is the strongest demand
  signal — the gap is structural and acknowledged.
- **No targeted anti-sensing product ships (VERIFIED by absence).** Every
  countermeasure (IRShield, PhyCloak, MIMOCrypt, DP-Givens, ScatterShield) is
  research-stage. The claim "no obvious shipping product protects rooms from this
  inference" **holds** as of 2026, with one caveat below.

---

## 2. First buyers, ranked by procurement readiness

| Segment | Driver | Readiness |
|---|---|---|
| **Defence / government** | ICD 705 / DoD EMSEC already mandate RF attenuation in classified spaces; budgets and mandates exist | **Strongest beachhead (VERIFIED)** — but today they buy broadband shielding, not a sensing-specific control |
| **Corporate boardrooms / counter-espionage** | TSCM firms (Bastille, Murray Associates) now include WiFi audits and rogue-AP detection; CSI keystroke/gesture inference makes a boardroom shield a natural extension | **VERIFIED demand, EMERGING WiFi-specific** |
| **Hospitals** | RF-derived behavioral/vital data is HIPAA PHI; exam rooms, psychiatric units where inference is unwanted | **VERIFIED regulatory hook** — but the hook drives privacy-preserving *sensing* more than a *shield* |
| **Hotels** | Documented guest backlash against in-room sensors; privacy as differentiation | **SPECULATIVE** — narrative-led, not procurement-led today |
| **Router / AP manufacturers** | Ship opt-out/obfuscation as a firmware feature anticipating regulation | **SPECULATIVE** — no vendor has announced this |

---

## 3. Competitive landscape

- **Direct competitors:** none shipping. All targeted anti-sensing is academic.
- **The real substitute (VERIFIED):** broadband RF shielding — SCIF/TEMPEST
  window film, paint, panels (Signals Defense SD2500: >40 dB, 30 MHz–6 GHz, ICD
  705 / ASTM F3057-14). It defeats WiFi sensing as a side effect but is **blunt**:
  it kills *all* RF and cannot coexist with wanted WiFi.
- **TSCM services (VERIFIED):** detect, don't prevent.

**VEIL's differentiation** is exactly what the substitute lacks: **selective and
coexisting** — it removes identity/activity leakage while keeping the room's WiFi
working at ≥95% throughput, with a machine-checkable compliance artifact.

---

## 4. Market size (third-party projections, cite with care)

- **CLAIMED:** ABI Research — North American WiFi-sensing-compatible CPE install
  base to **112M by 2030 (51.6% CAGR)**.
- **CLAIMED:** Global WiFi sensing market ~$402M (2024) → ~$2.13B (2033)
  (MarketIntelo).

Implication: a shield must **coexist** with a large installed sensing base, not
assume RF denial — reinforcing the selective-coexistence positioning.

---

## 5. Where VEIL fits RuView's positioning

VEIL pairs with BFLD to make RuView the *both-sides* RF-perception platform:
BFLD/AETHER do sensing responsibly and detect leakage; VEIL is the customer-
facing **privacy firewall** that protects a room from *others'* sensing. That is a
defensible, standards-anchored, gap-filling story: the standards body left the
door open, the threat is shipping, and no one else sells the selective lock.

---

## Sources

- IEEE 802.11bf privacy-proposal withdrawal (802.11-23/0782), summarized: https://pascalpiron.substack.com/p/wifi-sensing-and-the-privacy-fix
- NIST 802.11bf: https://www.nist.gov/publications/ieee-80211bf-enabling-widespread-adoption-wi-fi-sensing
- IRShield: https://arxiv.org/abs/2112.01967 · MIMOCrypt: https://arxiv.org/pdf/2309.00250 · ScatterShield: https://dl.acm.org/doi/abs/10.1145/3770653 · WiShield JSAC 2024: https://dl.acm.org/doi/abs/10.1109/JSAC.2024.3414597
- Signals Defense TEMPEST/SCIF film: https://signalsdefense.com/tempest-and-scif-design/ · https://signalsdefense.com/shielding-films/
- National Shielding SCIF/ICD-705: https://www.national-shielding.com/pages/scif-icd-705-secure-facility-shielding
- Bastille TSCM: https://bastille.net/centers-of-excellence/tscm/ · IntellSIG TSCM overview: https://www.intellsig.com/2025/07/20/modern-eavesdropping-threats-a-tscm-overview/
- Origin AI program: https://www.prnewswire.com/news-releases/origin-ai-launches-compatible-with-origin-program-to-meet-industry-demand-for-scalable-wifi-sensing-and-accelerate-integration-across-global-soc-platforms-302650963.html
- MIT Tech Review, WiFi sensing: https://www.technologyreview.com/2024/02/27/1088154/wifi-sensing-tracking-movements/
- ABI Research 112M forecast: https://www.abiresearch.com/press/north-american-wi-fi-sensing-cpe-installations-to-surge-to-112-million-by-2030-as-the-technologys-maturing-unleashes-new-business-and-service-models
- MarketIntelo WiFi sensing market: https://marketintelo.com/report/wi-fi-sensing-market
- HIPAA/PHI RF-sensing context (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC11939480/

*Caveat: market figures are analyst/vendor projections; the "no shipping product"
finding reflects absence of evidence in these searches and should be confirmed
with a patent/vendor scan before anchoring a go-to-market claim.*
