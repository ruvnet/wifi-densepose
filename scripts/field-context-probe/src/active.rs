//! SYNTHETIC active evidence replay. All authority lives in this bounded wrapper.
use super::*;
use std::io::Read;

const SOURCES: [&str; 3] = ["wifi", "radar", "door"];
const RANK_BYTES: u64 = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State { Empty, Occupied, Unknown }
impl State {
    fn name(self) -> &'static str { match self { Self::Empty => "empty", Self::Occupied => "occupied", Self::Unknown => "unknown" } }
}
#[derive(Clone)]
struct Root { event: FieldEvent, node: WorldId, state: State, expires: u64 }
struct Active {
    world: WorldGraph,
    roots: BTreeMap<String, Root>,
    hashes: BTreeSet<String>,
    latest: BTreeMap<String, String>,
    clock: u64,
    exhausted: bool,
    beliefs: usize,
    last_published: Option<WorldId>, // Historical handle, never a current-state read.
    budget: Budget,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Budget { packets: u32, airtime_us: u32, energy_uj: u32 }
impl Budget {
    fn json(self) -> Value { json!({"packets":self.packets,"airtime_us":self.airtime_us,"energy_uj":self.energy_uj}) }
    fn reserve(&mut self, cost: Self) -> bool {
        if cost.packets > self.packets || cost.airtime_us > self.airtime_us || cost.energy_uj > self.energy_uj { return false; }
        self.packets -= cost.packets; self.airtime_us -= cost.airtime_us; self.energy_uj -= cost.energy_uj;
        true
    }
}
fn cost(source: &str) -> Budget {
    match source {
        "wifi" => Budget { packets: 2, airtime_us: 1000, energy_uj: 100 },
        "radar" => Budget { packets: 1, airtime_us: 2000, energy_uj: 200 },
        "door" => Budget { packets: 1, airtime_us: 100, energy_uj: 10 },
        _ => unreachable!("source is checked before costing"),
    }
}
fn active_fixture(id: &str, source: &str, now: u64, state: State) -> FieldEvent {
    let mut event = fixture(id, now);
    event.sensor.device_id = source.into();
    // Modalities remain wifi_csi to exercise the pinned fixture rule for every
    // logical source; these are NOT real radar or door device adapters.
    let feature = match state { State::Empty => 0.1, State::Occupied => 0.9, State::Unknown => 0.5 };
    event.observation.features.insert("presence".into(), feature);
    event.provenance.raw_hash = sha256_hex(format!("SYNTHETIC active {id} {source} {now} {}", state.name()).as_bytes());
    Signer::from_seed(&[17;32]).sign_event(&mut event).unwrap();
    event
}
impl Active {
    fn new() -> Self {
        Self { world: WorldGraph::new(Default::default()), roots: BTreeMap::new(), hashes: BTreeSet::new(),
            latest: BTreeMap::new(), clock: 0, exhausted: false, beliefs: 0, last_published: None,
            budget: Budget { packets: 2, airtime_us: 2200, energy_uj: 220 } }
    }
    fn tick(&mut self, now: u64) -> Result<(), String> {
        if now < self.clock { return Err("clock_regression".into()); }
        self.clock = now; Ok(())
    }
    fn ingest(&mut self, event: FieldEvent, dependencies: &[String], now: u64) -> Result<(), String> {
        if now < self.clock { return Err("clock_regression".into()); }
        if self.exhausted { return Err("capacity_fail_closed".into()); }
        if !dependencies.is_empty() { return Err("derived_evidence_not_an_independent_root".into()); }
        if !event.provenance.synthetic { return Err("synthetic_only".into()); }
        let source = event.sensor.device_id.clone();
        if !SOURCES.contains(&source.as_str()) { return Err("unknown_source".into()); }
        if event.timestamp_ns > now || now - event.timestamp_ns >= MAX_AGE { return Err("stale_or_future".into()); }
        if self.roots.contains_key(&event.event_id) || self.hashes.contains(&event.provenance.raw_hash) { return Err("duplicate_root".into()); }
        if let Some(previous) = self.latest.get(&source) {
            if event.timestamp_ns <= self.roots[previous].event.timestamp_ns { return Err("source_clock_nonmonotonic".into()); }
        }
        if self.roots.len() >= MAX_EVENTS { self.exhausted = true; return Err("capacity_fail_closed".into()); }
        let value = event.observation.features.get("presence").copied().ok_or("missing_explicit_presence")?;
        if !value.is_finite() || !(0.0..=1.0).contains(&value) { return Err("invalid_presence_feature".into()); }
        let state = if value >= 0.7 { State::Occupied } else if value <= 0.3 { State::Empty } else { State::Unknown };
        let expires = event.timestamp_ns.checked_add(MAX_AGE).ok_or("expiry_overflow")?;
        verify_event(&event).map_err(|_| "signature_invalid")?;
        // Real pinned library verification/ingestion before mutating WorldGraph.
        let mut fusion = RuFieldFusion::with_rules(RuleSet::from_toml(RULES).unwrap());
        fusion.ingest_at(event.clone(), now).map_err(|e| e.to_string())?;
        let node = self.world.upsert_node(WorldNode::Event { id: WorldId::UNASSIGNED,
            event_type: format!("SYNTHETIC immutable root {}", event.event_id),
            at_unix_ms: i64::try_from(event.timestamp_ns / 1_000_000).map_err(|_| "timestamp_overflow")?, located_in: None });
        self.latest.insert(source, event.event_id.clone());
        self.hashes.insert(event.provenance.raw_hash.clone());
        self.roots.insert(event.event_id.clone(), Root { event, node, state, expires });
        self.clock = now;
        Ok(())
    }
    fn snapshot(&mut self, now: u64, publish: bool) -> Result<Value, String> {
        self.tick(now)?;
        let live: Vec<_> = self.latest.values().filter_map(|id| {
            let root = &self.roots[id]; (root.expires > now).then_some(root.clone())
        }).collect();
        let occupied = live.iter().any(|r| r.state == State::Occupied);
        let empty = live.iter().any(|r| r.state == State::Empty);
        let uncertain = live.iter().any(|r| r.state == State::Unknown);
        let contradiction = occupied && empty;
        let state = if self.exhausted || live.is_empty() || contradiction || uncertain { State::Unknown }
            else if occupied { State::Occupied } else { State::Empty };
        let expires = live.iter().map(|r| r.expires).min().unwrap_or(now);
        // Fuse one latest root per source, never repeated correlated samples.
        let mut ordered = live.clone(); ordered.sort_by_key(|r| r.event.timestamp_ns);
        let mut fusion = RuFieldFusion::with_rules(RuleSet::from_toml(RULES).unwrap());
        for root in &ordered { fusion.ingest_at(root.event.clone(), now).map_err(|e| e.to_string())?; }
        let inference = if ordered.is_empty() { None } else { fusion.infer(&InferenceQuery::all()).map_err(|e| e.to_string())?.into_iter().next() };
        let heuristic = inference.map(|i| i.confidence).filter(|s| s.is_finite() && (0.0..=1.0).contains(s));
        let handles: Vec<_> = live.iter().map(|r| r.event.event_id.clone()).collect();
        let mut belief = None;
        if publish && !live.is_empty() && !self.exhausted {
            if self.beliefs >= MAX_EVENTS { self.exhausted = true; return self.snapshot(now, false); }
            let nodes: Vec<_> = live.iter().map(|r| r.node).collect();
            let id = self.world.add_semantic_state(format!("SYNTHETIC {}; wrapper expiry {expires}", state.name()),
                heuristic.unwrap_or(0.0), (now / 1_000_000) as i64,
                SemanticProvenance { evidence: handles.clone(), model_version: "synthetic_active_wrapper_v1".into(),
                    calibration_version: "uncalibrated".into(), privacy_decision: "SYNTHETIC local replay only".into() }, &nodes);
            self.beliefs += 1;
            self.last_published = Some(id);
            belief = Some(json!({"ancestryEdges":self.world.neighbors(id).len(),"expiresNs":expires}));
        }
        Ok(json!({"atNs":now,"state":state.name(),"contradiction":contradiction,"expiresNs":expires,
            "liveSourceCount":live.len(),"rootIds":handles,"fusionHeuristicScore":heuristic,
            "scoreIsProbability":false,"capacityExhausted":self.exhausted,"belief":belief}))
    }
    fn propose(&mut self, ranking: &[(String, f64)], state: &Value) -> Value {
        if state["state"] != "unknown" || self.exhausted { return json!({"requested":false,"reason":"not_needed_or_capacity","actuated":false}); }
        for (source, score) in ranking {
            let required = cost(source);
            if self.budget.reserve(required) {
                return json!({"requested":true,"sensorId":source,"rankingScore":score,"scoreIsProbability":false,
                    "proposalOnly":true,"actuated":false,"reservedAuthoredCost":required.json(),"remaining":self.budget.json()});
            }
        }
        json!({"requested":false,"reason":"budget_exhausted","actuated":false,"remaining":self.budget.json()})
    }
}

// serde_json::Value accepts duplicate object keys; reject them before advice admission.
fn reject_duplicate_keys(text: &str) -> Result<(), String> {
    let bytes=text.as_bytes(); let mut objects: Vec<BTreeSet<String>>=Vec::new(); let mut i=0;
    while i<bytes.len() {
        match bytes[i] {
            b'{' => { objects.push(BTreeSet::new()); i+=1; },
            b'}' => { objects.pop(); i+=1; },
            b'"' => {
                let start=i; i+=1;
                while i<bytes.len() {
                    if bytes[i]==b'\\' { i+=2; }
                    else if bytes[i]==b'"' { i+=1; break; }
                    else { i+=1; }
                }
                let end=i;
                while i<bytes.len() && bytes[i].is_ascii_whitespace() { i+=1; }
                if i<bytes.len() && bytes[i]==b':' {
                    let key: String=serde_json::from_str(&text[start..end]).map_err(|_| "ranking_key")?;
                    if let Some(keys)=objects.last_mut() { if !keys.insert(key) { return Err("duplicate_ranking_key".into()); } }
                }
            },
            _ => i+=1,
        }
    }
    Ok(())
}
pub fn parse_ranking(text: &str) -> Result<Vec<(String,f64)>, String> {
    if text.len() as u64 > RANK_BYTES { return Err("ranking_byte_limit".into()); }
    let value: Value = serde_json::from_str(text).map_err(|_| "invalid_ranking_json")?;
    reject_duplicate_keys(text)?;
    let object = value.as_object().ok_or("ranking_object_required")?;
    if object.len() != 3 || value["schemaVersion"] != 1 || value["evidence"] != "SYNTHETIC" { return Err("ranking_contract".into()); }
    let candidates = value["candidates"].as_array().ok_or("candidates_array_required")?;
    if candidates.is_empty() || candidates.len() > SOURCES.len() { return Err("ranking_candidate_limit".into()); }
    let mut seen = BTreeSet::new(); let mut result = Vec::new();
    for candidate in candidates {
        let fields = candidate.as_object().ok_or("candidate_object_required")?;
        if fields.len() != 2 { return Err("candidate_fields".into()); }
        let source = candidate["sensorId"].as_str().ok_or("sensor_id_required")?;
        let score = candidate["score"].as_f64().ok_or("score_required")?;
        if !SOURCES.contains(&source) || !seen.insert(source.to_string()) || !score.is_finite() { return Err("invalid_ranking_candidate".into()); }
        result.push((source.to_string(), score));
    }
    result.sort_by(|a,b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0))); Ok(result)
}
// Linux x86_64/aarch64 UAPI fcntl flags. No fallback that follows symlinks
// or blocks opening a FIFO. The opened descriptor is checked before reading.
#[cfg(all(target_os="linux", any(target_arch="x86_64",target_arch="aarch64")))]
fn ranking_file(path: &str) -> Result<std::fs::File,String> {
    use std::os::unix::fs::OpenOptionsExt;
    const O_NOFOLLOW: i32=0x20000;
    const O_NONBLOCK: i32=0x800;
    let file=std::fs::OpenOptions::new().read(true).custom_flags(O_NOFOLLOW|O_NONBLOCK)
        .open(path).map_err(|_| "ranking_file_open")?;
    let metadata=file.metadata().map_err(|_| "ranking_file_metadata")?;
    if !metadata.is_file() || metadata.len()>RANK_BYTES { return Err("ranking_file_limit".into()); }
    Ok(file)
}
#[cfg(not(all(target_os="linux", any(target_arch="x86_64",target_arch="aarch64"))))]
fn ranking_file(_path: &str) -> Result<std::fs::File,String> {
    Err("ranking_file_platform_unsupported_use_stdin".into())
}
pub fn read_ranking(path: &str) -> Result<Vec<(String,f64)>, String> {
    let mut data = Vec::new();
    if path == "-" { std::io::stdin().take(RANK_BYTES+1).read_to_end(&mut data).map_err(|_| "ranking_read")?; }
    else {
        ranking_file(path)?.take(RANK_BYTES+1).read_to_end(&mut data).map_err(|_| "ranking_read")?;
    }
    parse_ranking(std::str::from_utf8(&data).map_err(|_| "ranking_utf8")?)
}
pub fn demo(ranking: Option<Vec<(String,f64)>>) -> Value {
    let ranking = ranking.unwrap_or_else(|| vec![("radar".into(),3.0),("door".into(),2.0),("wifi".into(),1.0)]);
    let mut runtime = Active::new(); let mut steps = Vec::new();
    for (label, source, state, delta) in [
        ("explicit_empty", "wifi", State::Empty, 0),
        ("person_observed", "wifi", State::Occupied, 100_000_000),
        ("conflicting_source", "radar", State::Empty, 200_000_000),
        ("radar_agrees", "radar", State::Occupied, 300_000_000),
    ] {
        runtime.ingest(active_fixture(label,source,NOW+delta,state), &[], NOW+delta).unwrap();
        let mut snapshot = runtime.snapshot(NOW+delta,true).unwrap();
        snapshot["scenario"] = json!(label); snapshot["proposal"] = runtime.propose(&ranking,&snapshot); steps.push(snapshot);
    }
    let mut expired = runtime.snapshot(NOW+MAX_AGE+300_000_000,true).unwrap();
    expired["scenario"] = json!("motionless_or_missing_updates_expire_to_unknown");
    expired["proposal"] = runtime.propose(&ranking,&expired); steps.push(expired);
    let final_state = runtime.snapshot(NOW+MAX_AGE+300_000_001,false).unwrap();
    let budget_stop = runtime.propose(&ranking,&final_state);
    let wire = runtime.world.to_json().unwrap();
    let restored = WorldGraph::from_json(&wire).unwrap();
    json!({"schemaVersion":1,"evidence":"SYNTHETIC","steps":steps,"budgetStop":budget_stop,
        "worldGraph":{"nodes":runtime.world.node_count(),"edges":runtime.world.edge_count(),
            "roundTripPreserved":restored.node_count()==runtime.world.node_count() && restored.edge_count()==runtime.world.edge_count()},
        "authority":{"rankingChangesEvidence":false,"actuation":false,"promotion":false},
        "scope":"Logical wifi/radar/door sources use signed WiFi feature fixtures, not physical modality adapters. Scores and costs are authored, not calibrated or measured sensing values.",
        "expiryPolicy":"Wrapper recomputes live beliefs; persisted WorldGraph history is not a current authorization or occupancy API.",
        "ruFieldRevision":"b74bc035e4a3d4023665aa5479b41e938facd113","worldGraphRevision":"9b1c79c836cdacfb7b44f058c593157bac4c1dab"})
}

pub fn benchmark_active() -> Value {
    let event = active_fixture("benchmark-active","wifi",NOW,State::Occupied);
    let active_once = || { let mut runtime=Active::new(); runtime.ingest(event.clone(), &[], NOW).unwrap(); std::hint::black_box(runtime.snapshot(NOW,true).unwrap()); };
    let mut rounds = Vec::new();
    for round in 0..3 {
        for _ in 0..100 { native_once(&event); active_once(); }
        let mut baseline=Vec::new(); let mut active=Vec::new();
        for iteration in 0..1000 {
            let mut measure = |full: bool| { let start=std::time::Instant::now(); if full { active_once(); } else { native_once(&event); }
                let ms=start.elapsed().as_secs_f64()*1000.; if full { active.push(ms); } else { baseline.push(ms); } };
            if iteration%2==0 { measure(false); measure(true); } else { measure(true); measure(false); }
        }
        let extra: Vec<_> = active.iter().zip(&baseline).map(|(a,b)|a-b).collect();
        rounds.push(json!({"round":round+1,"baseline_ms":{"p50":quantile(&baseline,0.5),"p95":quantile(&baseline,0.95)},
            "active_ms":{"p50":quantile(&active,0.5),"p95":quantile(&active,0.95)},"paired_extra_ms":{"p50":quantile(&extra,0.5),"p95":quantile(&extra,0.95)}}));
    }
    json!({"evidence":"MEASURED software timing on SYNTHETIC fixture","rounds":rounds,"iterationsPerRound":1000,"warmupsPerRound":100,
        "scope":"Fresh engines per iteration; baseline verifies, ingests and fuses one signed root. Active adds bounded root intake, latest-source selection, refusion, WorldGraph evidence/belief ancestry and receipt. Excludes signing, sensing, ranking I/O and hardware. One logical WiFi source, not persistent-stream throughput.","profile":"release required","scoreIsProbability":false})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn evolving_states_and_expiry() {
        let report=demo(None);
        let states: Vec<_>=report["steps"].as_array().unwrap().iter().map(|s|s["state"].as_str().unwrap()).collect();
        assert_eq!(states,vec!["empty","occupied","unknown","occupied","unknown"]);
        assert_eq!(report["worldGraph"]["roundTripPreserved"],true);
        assert_eq!(report["budgetStop"]["requested"],false);
        assert_eq!(report["steps"][1]["liveSourceCount"],1);
    }
    #[test]
    fn ancestry_and_no_evidence_feedback() {
        let mut active=Active::new(); active.ingest(active_fixture("a","wifi",NOW,State::Occupied),&[],NOW).unwrap();
        assert!(active.ingest(active_fixture("b","radar",NOW+1,State::Occupied),&["a".into()],NOW+1).is_err());
        let snapshot=active.snapshot(NOW+1,true).unwrap(); assert_eq!(snapshot["belief"]["ancestryEdges"],1);
        assert_eq!(active.world.edge_count(),1);
        assert_eq!(active.world.node_count(),2);
        let ancestry=active.world.neighbors(active.last_published.unwrap());
        assert_eq!(ancestry[0].0,active.roots["a"].node);
        assert!(matches!(&ancestry[0].1,WorldEdge::DerivedFrom { evidence } if evidence=="a"));
    }
    #[test]
    fn guards_capacity_clock_signature_replay() {
        let mut active=Active::new(); let event=active_fixture("a","wifi",NOW,State::Occupied);
        active.ingest(event.clone(),&[],NOW).unwrap(); let before=active.world.node_count();
        assert!(active.ingest(event,&[],NOW).is_err()); assert!(active.snapshot(NOW-1,false).is_err());
        let mut tampered=active_fixture("b","wifi",NOW+1,State::Empty); tampered.tensor.values[0]=99.;
        assert!(active.ingest(tampered,&[],NOW+1).is_err()); assert_eq!(active.world.node_count(),before);
        for i in 1..MAX_EVENTS { active.ingest(active_fixture(&format!("root{i}"),"wifi",NOW+i as u64,State::Occupied),&[],NOW+i as u64).unwrap(); }
        assert!(active.ingest(active_fixture("overflow","wifi",NOW+100,State::Occupied),&[],NOW+100).is_err());
        assert_eq!(active.snapshot(NOW+100,false).unwrap()["state"],"unknown");
    }
    #[test]
    fn ranking_is_bounded_untrusted_advice() {
        let good=r#"{"schemaVersion":1,"evidence":"SYNTHETIC","candidates":[{"sensorId":"wifi","score":100.0}]}"#;
        assert_eq!(parse_ranking(good).unwrap()[0].1,100.0);
        for bad in [good.replace("wifi","shell"),good.replace("100.0","1e999"),good.replace("100.0","null"),
            good.replace("SYNTHETIC","REAL"),good.replace("\"score\":100.0","\"score\":1,\"score\":100.0"),good.replace("\"score\":100.0","\"score\":100.0,\"actuate\":true"),
            good.replace("[{", "[{\"sensorId\":\"wifi\",\"score\":1},{")," ".repeat(4097)] { assert!(parse_ranking(&bad).is_err(),"{bad}"); }
        let default=demo(None); let ranked=demo(Some(parse_ranking(good).unwrap()));
        for (a,b) in default["steps"].as_array().unwrap().iter().zip(ranked["steps"].as_array().unwrap()) { assert_eq!(a["state"],b["state"]); assert_eq!(a["fusionHeuristicScore"],b["fusionHeuristicScore"]); }
    }
    #[test]
    fn freshness_unknown_and_hash_alias_rejected() {
        let mut active=Active::new();
        assert!(active.ingest(active_fixture("stale","wifi",NOW-MAX_AGE,State::Empty),&[],NOW).is_err());
        assert!(active.ingest(active_fixture("future","wifi",NOW+1,State::Empty),&[],NOW).is_err());
        let original=active_fixture("original","wifi",NOW,State::Unknown);
        active.ingest(original.clone(),&[],NOW).unwrap();
        let before=active.world.to_json().unwrap();
        let mut alias=active_fixture("alias","radar",NOW+1,State::Occupied);
        alias.provenance.raw_hash=original.provenance.raw_hash.clone();
        Signer::from_seed(&[17;32]).sign_event(&mut alias).unwrap();
        assert!(active.ingest(alias,&[],NOW+1).is_err());
        assert_eq!(active.world.to_json().unwrap(),before);
        assert_eq!(active.snapshot(NOW+1,false).unwrap()["state"],"unknown");
        assert_eq!(active.snapshot(NOW+MAX_AGE,false).unwrap()["liveSourceCount"],0);
    }
    #[test]
    fn rejected_ingest_does_not_advance_clock() {
        let mut active=Active::new();
        active.ingest(active_fixture("accepted","wifi",NOW,State::Occupied),&[],NOW).unwrap();
        let graph=active.world.to_json().unwrap();
        let mut malformed=active_fixture("bad","radar",NOW+MAX_AGE,State::Empty);
        malformed.tensor.values[0]=999.;
        assert!(active.ingest(malformed,&[],NOW+MAX_AGE).is_err());
        assert_eq!(active.clock,NOW);
        assert_eq!(active.world.to_json().unwrap(),graph);
        assert_eq!(active.snapshot(NOW,false).unwrap()["state"],"occupied");
    }
    #[cfg(all(target_os="linux",any(target_arch="x86_64",target_arch="aarch64")))]
    #[test]
    fn ranking_opened_descriptor_guards() {
        use std::os::unix::fs::symlink;
        let dir=std::env::temp_dir().join(format!("ruview-ranking-{}-{}",std::process::id(),NOW));
        std::fs::create_dir(&dir).unwrap();
        let file=dir.join("ranking.json"); let link=dir.join("link"); let fifo=dir.join("fifo");
        std::fs::write(&file,br#"{"schemaVersion":1,"evidence":"SYNTHETIC","candidates":[{"sensorId":"wifi","score":1}]}"#).unwrap();
        symlink(&file,&link).unwrap();
        assert!(read_ranking(file.to_str().unwrap()).is_ok());
        assert!(read_ranking(link.to_str().unwrap()).is_err());
        assert!(read_ranking(dir.to_str().unwrap()).is_err());
        // mkfifo is only a local test fixture; the production reader spawns nothing.
        assert!(std::process::Command::new("mkfifo").arg(&fifo).status().unwrap().success());
        assert!(read_ranking(fifo.to_str().unwrap()).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn budget_atomic_in_all_units() {
        let initial=Budget{packets:2,airtime_us:2,energy_uj:2};
        for bad in [Budget{packets:3,..initial},Budget{airtime_us:3,..initial},Budget{energy_uj:3,..initial}] {
            let mut budget=initial; assert!(!budget.reserve(bad)); assert_eq!(budget,initial);
        }
        let mut budget=initial; assert!(budget.reserve(initial)); assert_eq!(budget,Budget{packets:0,airtime_us:0,energy_uj:0});
    }
}
