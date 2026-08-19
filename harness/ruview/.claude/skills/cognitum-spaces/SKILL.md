# Cognitum Spaces OAuth activation

Use this playbook to activate and inspect the tenant-scoped Cognitum Spaces
projection without giving an agent a bearer token or API key.

## Boundary

- This is a read-only P2/P3 semantic projection. HomeCore Edge remains
  authoritative.
- Raw CSI, CIR, RF tensors, recordings, pose frames, vital waveforms, and
  identity observations are prohibited.
- `spaces:read` grants no pairing, publication, write, command, policy approval,
  spending, or actuator authority.
- A read may refresh an expiring OAuth session and atomically rotate the local
  credential file.

## Activate OAuth explicitly

Install or build the `wifi-densepose` CLI, then request the additional scope:

```bash
wifi-densepose login --spaces
```

For a terminal without a browser:

```bash
wifi-densepose login --spaces --no-browser
```

Confirm that the account reports `spaces:read`, then list through the
metaharness:

```bash
wifi-densepose whoami
npx @ruvnet/ruview spaces
npx @ruvnet/ruview spaces --resource sites
npx @ruvnet/ruview spaces --resource events --limit 25
```

The versioned collections are `sites`, `buildings`, `floors`, `spaces`,
`zones`, `entities`, `events`, and `alerts`. Continue a page with the returned
opaque `nextCursor`; do not decode or reuse a cursor for another collection.

Use `--credentials-path <private-file>` only from the human-invoked CLI when a
non-default credential store is intentional. Never put a bearer token or API
key on the command line.

## MCP

The tool is `ruview_spaces_list`. It is denied by default even though the cloud
operation is read-only, because it consumes a local identity credential and
contacts an external service. The MCP server operator must grant that capability
and may bind the credential path in the server environment:

```bash
RUVIEW_MCP_GRANTS=credential-use \
RUVIEW_CREDENTIALS_PATH=/private/ruview/credentials.json \
npx @ruvnet/ruview mcp start
```

MCP calls cannot choose a credential path and the tool schema has no token or
API-key, workspace override, or base-URL field. The API origin is fixed to
`https://api.cognitum.one`, the adapter requires an installed
`wifi-densepose` binary, and the child environment excludes
`COGNITUM_SPACES_API`, so this
surface verifies the OAuth path rather than silently taking the compatibility
API-key path.

## Interpret results honestly

An empty `data` list can be a valid authenticated tenant result. It proves the
read path and isolation behavior, not sensing quality. Every accepted response
must declare `HomeCore Edge` as authoritative and carry the complete prohibited
field list. Parent lineage, schema version, anonymous person/track identity,
event/alert fields, confidence, and cursor bounds are independently checked.
Any malformed, oversized, non-semantic, or raw-field response fails closed.
