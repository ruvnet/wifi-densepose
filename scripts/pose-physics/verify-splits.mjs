import fs from "node:fs";
import crypto from "node:crypto";

const path = process.argv[2];
if (!path) throw new Error("usage: verify-pose-physics-splits.sh <manifest.json>");
const bytes = fs.readFileSync(path);
const manifest = JSON.parse(bytes);
const dimensions = ["sequence", "take", "subject", "room", "calibration_session"];
if (!Array.isArray(manifest.records) || manifest.records.length === 0) throw new Error("manifest.records must be non-empty");
const ownership = new Map();
for (const record of manifest.records) {
  if (!record.split || !["train", "validation", "test"].includes(record.split)) throw new Error("invalid split");
  for (const dimension of dimensions) {
    const value = record[dimension];
    if (typeof value !== "string" || value.length === 0) throw new Error(`missing ${dimension}`);
    const key = `${dimension}:${value}`;
    const prior = ownership.get(key);
    if (prior && prior !== record.split) throw new Error(`leakage: ${key} crosses ${prior}/${record.split}`);
    ownership.set(key, record.split);
  }
}
const digest = crypto.createHash("sha256").update(bytes).digest("hex");
console.log(JSON.stringify({ verdict: "PASS", records: manifest.records.length, sha256: digest }));
