"""Strict local peer: real JSON-RPC pipes, generated schema, no sockets or native CLI."""

import json
import os
import socket
import sys
import tomllib
from pathlib import Path

scenario = json.loads(Path(sys.argv[1]).read_text())
log = Path(sys.argv[2])
args = sys.argv[3:]


def forbidden(*args, **kwargs):
    raise AssertionError("network forbidden")


def record(value):
    with log.open("a") as stream:
        stream.write(json.dumps(value) + "\n")


record({"argv": args, "env": dict(os.environ), "pid": os.getpid(), "cwd": str(Path.cwd())})
if args == ["--version"]:
    print("codex-cli " + scenario.get("version", "0.164.0"))
    sys.exit(0)


socket.socket = forbidden


def validate(value, schema):
    # Validate in the parent harness, whose imports are outside RPC deadlines.
    record({"validate": schema, "value": value})


assert args[:3] == ["app-server", "--listen", "stdio://"]
config = {}
for index, arg in enumerate(args):
    if arg == "-c":

        def merge(target, patch):
            for key, value in patch.items():
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    merge(target[key], value)
                else:
                    target[key] = value

        merge(config, tomllib.loads(args[index + 1]))


def send(packet):
    print(json.dumps(packet), flush=True)


def read(method=None, schema=None):
    line = sys.stdin.readline()
    if not line:
        sys.exit(0)
    packet = json.loads(line)
    record(packet)
    if method:
        assert packet["method"] == method, packet
    if schema:
        validate(packet["params"], schema)
    return packet


def reply(packet, result, schema=None):
    if schema:
        validate(result, schema)
    send({"id": packet["id"], "result": result})


p = read("initialize", "InitializeParams")
if scenario.get("handshake") == "malformed":
    reply(p, {})
else:
    reply(
        p,
        {
            "userAgent": "codex-cli/0.164.0",
            "platformFamily": "unix",
            "platformOs": "macos",
            "codexHome": "/private/opaque",
        },
        "InitializeResponse",
    )
read("initialized")
p = read("account/read", "GetAccountParams")
assert p["params"] == {"refreshToken": False}
auth = scenario.get("auth", {"type": "chatgpt", "email": None, "planType": "plus"})
reply(p, {"account": auth, "requiresOpenaiAuth": True})
while True:
    p = read()
    method = p.get("method")
    if method == "config/read":
        validate(p["params"], "ConfigReadParams")
        effective = dict(config)
        if scenario.get("isolation") == "missing":
            effective.pop("default_permissions", None)
        if scenario.get("isolation") == "legacy":
            effective["sandbox_mode"] = "workspace-write"
        reply(p, {"config": effective, "origins": {}}, "ConfigReadResponse")
    elif method in ("thread/start", "thread/resume"):
        validate(
            p["params"],
            "ThreadStartParams" if method.endswith("start") else "ThreadResumeParams",
        )
        root = str(Path.cwd())
        thread_id = p["params"].get("threadId", "thread-1")
        thread = {
            "id": thread_id,
            "cliVersion": "0.164.0",
            "createdAt": 1,
            "updatedAt": 1,
            "cwd": root,
            "ephemeral": False,
            "modelProvider": "native",
            "preview": "",
            "projectId": None,
            "sessionId": "session-1",
            "source": "appServer",
            "status": {"type": "idle"},
            "turns": [],
        }
        profile = config["default_permissions"]
        sandbox = {
            "type": "workspaceWrite"
            if config["permissions"][profile]["extends"] == ":workspace"
            else "readOnly",
            "networkAccess": False,
        }
        result = {
            "thread": thread,
            "model": p["params"]["model"],
            "modelProvider": "native",
            "cwd": root,
            "approvalPolicy": "on-request",
            "approvalsReviewer": "user",
            "sandbox": sandbox,
            "activePermissionProfile": {"id": profile},
        }
        result.update(scenario.get("thread_overrides", {}))
        reply(
            p, result, "ThreadStartResponse" if method.endswith("start") else "ThreadResumeResponse"
        )
    elif method == "account/rateLimits/read":
        reply(p, {"rateLimits": scenario.get("limits", {})}, "GetAccountRateLimitsResponse")
    elif method == "turn/start":
        validate(p["params"], "TurnStartParams")
        reply(
            p, {"turn": {"id": "turn-1", "items": [], "status": "inProgress"}}, "TurnStartResponse"
        )
        for step in scenario.get("transcript", []):
            if "receive" in step:
                received = read()
                expected = step["receive"]
                if "id" in expected:
                    assert received["id"] == expected["id"], received
                if "result" in expected:
                    assert received["result"] == expected["result"], received
                if "error" in expected:
                    assert received["error"]["code"] == expected["error"]["code"], received
            elif "raw" in step:
                sys.stdout.write(step["raw"] + "\n")
                sys.stdout.flush()
            elif "exit" in step:
                sys.exit(step["exit"])
            else:
                send(step["send"])
        if scenario.get("wait_interrupt"):
            p = read("turn/interrupt", "TurnInterruptParams")
            reply(p, {}, "TurnInterruptResponse")
            send(
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": thread_id,
                        "turn": {"id": "turn-1", "items": [], "status": "interrupted"},
                    },
                }
            )
    elif method == "turn/interrupt":
        validate(p["params"], "TurnInterruptParams")
        reply(p, {}, "TurnInterruptResponse")
    else:
        raise AssertionError(p)
