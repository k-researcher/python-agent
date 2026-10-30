# App-server contract fixture

Source: unmodified installed CLI 0.164.0, generated without authentication or model calls:

```sh
codex --version
codex app-server generate-json-schema --out <temporary-directory>
codex app-server generate-json-schema --experimental --out <temporary-directory>
```

`schema.json` retains the needed request/response/notification schemas and their referenced
definitions from the experimental bundle. Only descriptions, titles and schema dialect labels
were removed. Experimental fields are necessary for active permission profile provenance and
bounded command permissions. No schema generation takes place during tests.

`events.json` is a synthetic, anonymized JSON-RPC transcript checked against those schemas.
It contains no captured account data, credentials, prompts or output from a real model turn.
The Python peer records schema checks for client parameters and its replies; the parent harness
validates them with the generated schemas after reaping the peer. This keeps third-party imports
outside protocol deadlines without relaxing time bounds.
Additional request/response transcripts are assembled by the contract tests, including deliberate
invalid packets. The peer receives its scenario through a temporary file, communicates over real
stdin/stdout pipes and writes its received packets to a temporary log. Its executable is the only
allowed CLI; socket creation is refused. Every process is started under the shared guard and reaped.
