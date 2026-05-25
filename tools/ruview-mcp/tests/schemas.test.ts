/**
 * ADR-124 §4.1 / §4.1a schema coverage tests.
 *
 * Guards:
 *  1. Every catalogued tool name appears in TOOL_NAMES and TOOL_INPUT_SCHEMAS.
 *  2. TOOL_INPUT_SCHEMAS has no extra (undocumented) keys.
 *  3. Each schema accepts its documented happy-path input without throwing.
 *  4. Each schema rejects structurally invalid input (Zod parse failure).
 *  5. Shared sub-schemas (NodeId, DurationS, SemanticPrimitiveKind) enforce
 *     their documented constraints.
 */

import {
  TOOL_NAMES,
  TOOL_INPUT_SCHEMAS,
  SemanticPrimitiveKindSchema,
  DurationSSchema,
  NodeIdSchema,
  PosePersonResultSchema,
  PresenceNowInputSchema,
  VitalsGetBreathingInputSchema,
  PrimitivesGetInputSchema,
  BfldLastScanInputSchema,
  NodeStatusInputSchema,
  VectorSearchPoseInputSchema,
  VectorStorePoseInputSchema,
  PolicyCanAccessVitalsInputSchema,
  PolicyCanSubscribeInputSchema,
  PolicyRedactIdentityFieldsInputSchema,
} from "../src/schemas/index.js";

// ── 1. Catalog completeness ────────────────────────────────────────────────

describe("TOOL_NAMES catalog (ADR-124 §4.1 + §4.1a)", () => {
  const EXPECTED_COUNT = 20; // 15 sensing + 5 policy

  it("contains exactly 20 tools", () => {
    expect(TOOL_NAMES).toHaveLength(EXPECTED_COUNT);
  });

  it("contains all 15 §4.1 sensing tool names", () => {
    const sensing = [
      "ruview.presence.now",
      "ruview.vitals.get_breathing",
      "ruview.vitals.get_heart_rate",
      "ruview.vitals.get_all",
      "ruview.pose.latest",
      "ruview.pose.subscribe",
      "ruview.primitives.get",
      "ruview.primitives.list_active",
      "ruview.primitives.subscribe",
      "ruview.bfld.last_scan",
      "ruview.bfld.subscribe",
      "ruview.node.list",
      "ruview.node.status",
      "ruview.vector.search_pose",
      "ruview.vector.store_pose",
    ];
    for (const name of sensing) {
      expect(TOOL_NAMES).toContain(name);
    }
  });

  it("contains all 5 §4.1a policy tool names", () => {
    const policy = [
      "ruview.policy.can_access_vitals",
      "ruview.policy.can_query_presence",
      "ruview.policy.can_subscribe",
      "ruview.policy.redact_identity_fields",
      "ruview.policy.audit_log",
    ];
    for (const name of policy) {
      expect(TOOL_NAMES).toContain(name);
    }
  });

  it("TOOL_INPUT_SCHEMAS has a schema for every catalogued tool name", () => {
    for (const name of TOOL_NAMES) {
      // Use Object.prototype.hasOwnProperty to avoid Jest's dotted-path
      // interpretation of toHaveProperty (dots = nested path in Jest).
      expect(Object.prototype.hasOwnProperty.call(TOOL_INPUT_SCHEMAS, name)).toBe(true);
      expect(TOOL_INPUT_SCHEMAS[name]).toBeDefined();
    }
  });

  it("TOOL_INPUT_SCHEMAS has no extra keys beyond the catalog", () => {
    const schemaKeys = Object.keys(TOOL_INPUT_SCHEMAS).sort();
    const catalogKeys = [...TOOL_NAMES].sort();
    expect(schemaKeys).toEqual(catalogKeys);
  });
});

// ── 2. Happy-path parse ────────────────────────────────────────────────────

describe("Schema happy-path acceptance", () => {
  it("PresenceNow — accepts empty object (node_id optional)", () => {
    expect(() => PresenceNowInputSchema.parse({})).not.toThrow();
  });

  it("PresenceNow — accepts object with node_id", () => {
    const r = PresenceNowInputSchema.parse({ node_id: "node-abc" });
    expect(r.node_id).toBe("node-abc");
  });

  it("VitalsGetBreathing — accepts window_s and node_id", () => {
    const r = VitalsGetBreathingInputSchema.parse({ window_s: 30, node_id: "n1" });
    expect(r.window_s).toBe(30);
  });

  it("PrimitivesGet — accepts valid primitive kind", () => {
    const r = PrimitivesGetInputSchema.parse({ primitive: "fall_detected" });
    expect(r.primitive).toBe("fall_detected");
  });

  it("BfldLastScan — accepts empty object", () => {
    expect(() => BfldLastScanInputSchema.parse({})).not.toThrow();
  });

  it("NodeStatus — accepts node_id string", () => {
    const r = NodeStatusInputSchema.parse({ node_id: "cognitum-seed-1" });
    expect(r.node_id).toBe("cognitum-seed-1");
  });

  it("VectorSearchPose — applies default k=10", () => {
    const r = VectorSearchPoseInputSchema.parse({ query_embedding: [0.1, 0.2, 0.3] });
    expect(r.k).toBe(10);
  });

  it("VectorStorePose — accepts a valid 17-keypoint pose", () => {
    const kpts = Array.from({ length: 17 }, (_, i) => [i * 0.05, i * 0.03] as [number, number]);
    const r = VectorStorePoseInputSchema.parse({
      pose: { keypoints: kpts, confidence: 0.92 },
      node_id: "node-x",
    });
    expect(r.pose.keypoints).toHaveLength(17);
  });

  it("PolicyCanAccessVitals — accepts valid vital value", () => {
    const r = PolicyCanAccessVitalsInputSchema.parse({
      agent_id: "agent-007",
      node_id: "node-1",
      vital: "heart_rate",
    });
    expect(r.vital).toBe("heart_rate");
  });

  it("PolicyCanSubscribe — accepts valid duration_s", () => {
    const r = PolicyCanSubscribeInputSchema.parse({
      agent_id: "agent-007",
      topic: "ruview.vitals.get_all",
      duration_s: 300,
    });
    expect(r.duration_s).toBe(300);
  });

  it("PolicyRedactIdentityFields — accepts arbitrary payload record", () => {
    const r = PolicyRedactIdentityFieldsInputSchema.parse({
      payload: { sta_mac: "AA:BB:CC:DD:EE:FF", n_persons: 2 },
      agent_id: "agent-007",
    });
    expect(r.payload).toHaveProperty("sta_mac");
  });
});

// ── 3. Constraint rejection ────────────────────────────────────────────────

describe("Schema constraint enforcement", () => {
  it("NodeIdSchema — rejects empty string", () => {
    expect(() => NodeIdSchema.parse("")).toThrow();
  });

  it("DurationSSchema — rejects zero", () => {
    expect(() => DurationSSchema.parse(0)).toThrow();
  });

  it("DurationSSchema — rejects value > 3600", () => {
    expect(() => DurationSSchema.parse(3601)).toThrow();
  });

  it("SemanticPrimitiveKind — rejects unknown primitive", () => {
    expect(() => SemanticPrimitiveKindSchema.parse("unknown_primitive")).toThrow();
  });

  it("PosePersonResult — rejects keypoints array with wrong length", () => {
    const badKpts = Array.from({ length: 5 }, () => [0, 0] as [number, number]);
    expect(() => PosePersonResultSchema.parse({ keypoints: badKpts, confidence: 0.9 })).toThrow();
  });

  it("VectorSearchPose — rejects k > 100", () => {
    expect(() =>
      VectorSearchPoseInputSchema.parse({ query_embedding: [0.1], k: 101 })
    ).toThrow();
  });

  it("PolicyCanAccessVitals — rejects unknown vital value", () => {
    expect(() =>
      PolicyCanAccessVitalsInputSchema.parse({
        agent_id: "a",
        node_id: "n",
        vital: "temperature",
      })
    ).toThrow();
  });

  it("NodeStatus — rejects missing node_id", () => {
    expect(() => NodeStatusInputSchema.parse({})).toThrow();
  });
});
