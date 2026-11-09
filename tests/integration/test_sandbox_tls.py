"""Check native certificate verification without an external service or trust-store changes."""

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from agent.sandbox import SandboxPolicy, available, prepare


@pytest.mark.skipif(not available(), reason="no usable sandbox backend")
def test_network_policy_allows_native_certificate_verification(tmp_path: Path) -> None:
    project = (tmp_path / "project").resolve()
    project.mkdir()
    config = project / "certificate.cnf"
    config.write_text(
        "[req]\nprompt=no\ndistinguished_name=subject\nx509_extensions=extensions\n"
        "[subject]\nCN=fixture.test\n[extensions]\nsubjectAltName=DNS:fixture.test\n"
        "basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\n"
        "extendedKeyUsage=serverAuth\n",
        encoding="utf-8",
    )
    policy = SandboxPolicy(mode="required", temp_root=tmp_path / "private")

    def run(network: bool, *command: str) -> subprocess.CompletedProcess[str]:
        prepared = prepare(replace(policy, allow_network=network), project, "tls")
        assert prepared is not None
        prefix, environment = prepared
        environment["OPENSSL_CONF"] = str(config)
        return subprocess.run(
            [*prefix, *command],
            cwd=project,
            env=environment,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

    generated = run(
        False,
        "/usr/bin/openssl",
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        "key.pem",
        "-out",
        "certificate.pem",
        "-days",
        "2",
        "-config",
        str(config),
    )
    assert generated.returncode == 0, generated.stderr
    base = ["/usr/bin/security", "verify-cert", "-c", "certificate.pem", "-p", "ssl", "-L", "-N"]
    # An explicit anchor applies only to this verification, never to a stored keychain.
    trusted = run(True, *base, "-r", "certificate.pem", "-n", "fixture.test")
    assert trusted.returncode == 0, trusted.stderr
    for network, arguments in (
        (False, ["-r", "certificate.pem", "-n", "fixture.test"]),
        (True, ["-r", "certificate.pem", "-n", "wrong.test"]),
        (True, ["-n", "fixture.test"]),
    ):
        rejected = run(network, *base, *arguments)
        assert rejected.returncode != 0, (arguments, rejected.stdout, rejected.stderr)
