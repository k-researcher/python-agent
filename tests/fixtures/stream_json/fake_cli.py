"""Synthetic protocol peer. Never imports or launches a native engine; no sockets."""

import json
import os
import sys


def forbid_network(event, args):
    if event.startswith("socket."):
        raise AssertionError("network forbidden")


sys.addaudithook(forbid_network)
with open(sys.argv[1]) as stream:
    scenario = json.load(stream)
log = sys.argv[2]
with open(sys.argv[3]) as stream:
    fixture = json.load(stream)
args = sys.argv[4:]


def record(packet):
    with open(log, "a") as stream:
        stream.write(json.dumps(packet) + "\n")


def send(packet):
    print(json.dumps(packet), flush=True)


def read():
    raw = sys.stdin.readline()
    if not raw:
        sys.exit(0)
    packet = json.loads(raw)
    record(packet)
    return packet


def reply(packet, response):
    send(
        {
            "type": "control_response",
            "response": {
                "subtype": "success",
                "request_id": packet["request_id"],
                "response": response,
            },
        }
    )


# Test-only startup barrier, consumed by the peer fixture before the adapter
# receives the pipes. Guard readiness only means the child was created.
print('{"peer_ready":true}', flush=True)
record({"argv": args, "env": dict(os.environ), "cwd": os.getcwd(), "pid": os.getpid()})
if args == ["--version"]:
    print(scenario.get("version", fixture["version"]) + fixture["version_suffix"])
    sys.exit(0)
if args == ["auth", "status", "--json"]:
    if "auth_raw" in scenario:
        print(scenario["auth_raw"])
    else:
        send(scenario.get("auth", fixture["auth"]))
    sys.exit(scenario.get("auth_exit", 0))

expected = [
    "-p",
    "--input-format",
    "stream-json",
    "--output-format",
    "stream-json",
    "--verbose",
    "--model",
    "configured-model",
    "--permission-prompt-tool",
    "stdio",
    "--permission-mode",
    "default",
    "--setting-sources",
    "",
    "--strict-mcp-config",
    "--disable-slash-commands",
    "--settings",
]
assert args[: len(expected)] == expected, args
assert len(args) in (len(expected) + 1, len(expected) + 3)
settings = json.loads(args[len(expected)])
record({"settings": settings})
p = read()
assert p["type"] == "control_request" and p["request"]["subtype"] == "initialize"
assert p["request"]["hooks"] is None
if scenario.get("handshake_raw"):
    print(scenario["handshake_raw"], flush=True)
else:
    reply(p, scenario.get("initialize", fixture["initialize"]))
# The recorded contract does NOT attest isolation. Tests that exercise the transport
# inject a trusted gate in the parent, never a made-up effectiveSettings wire field.
# Timing of this pre-user init is synthetic, not evidence of native pre-turn support.
init = {**fixture["init"], "cwd": os.getcwd(), **scenario.get("init", {})}
if "--resume" in args:
    init["session_id"] = args[args.index("--resume") + 1]
if scenario.get("emit_init", True):
    send(init)
while True:
    p = read()
    if p.get("type") == "control_request":
        assert p["request"]["subtype"] == "interrupt", p
        reply(p, {"still_queued": []})
        continue
    assert p["type"] == "user", p
    for step in scenario.get("transcript", []):
        if "send" in step:
            send(step["send"])
        elif "raw" in step:
            print(step["raw"], flush=True)
        elif "exit" in step:
            sys.exit(step["exit"])
        elif "receive" in step:
            received = read()
            expected = step["receive"]
            if expected == "interrupt":
                assert received["request"]["subtype"] == "interrupt", received
                reply(received, {"still_queued": []})
            else:
                assert received["response"]["response"] == expected, received
    if scenario.get("exit_after_transcript"):
        sys.exit(0)
