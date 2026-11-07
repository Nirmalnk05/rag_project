"""The deployment manifests carry two load-bearing claims.

1. The identity headers are caller-supplied, so ENABLE_ACL asserts that a
   gateway rebuilt them from verified claims. networkpolicy.yaml is what makes
   that assertion true: default-deny, with only the gateway namespace allowed
   to reach port 8000. Without it, anything with cluster network access can
   send `X-Roles: admin` straight past the gateway.
2. The Retriever is built in the FastAPI lifespan hook (~90s at 100k chunks),
   so liveness alone kills the container at ~55s — a crash loop. A startupProbe
   suspends liveness and readiness until it succeeds.
"""
from pathlib import Path

import pytest
import yaml

K8S = Path("deploy/k8s")
APP_LABEL = {"app": "rag-app"}
APP_PORT = 8000


def _documents(name: str) -> list[dict]:
    text = (K8S / name).read_text(encoding="utf-8")
    return [d for d in yaml.safe_load_all(text) if d]


def _of_kind(name: str, kind: str) -> list[dict]:
    return [d for d in _documents(name) if d.get("kind") == kind]


def _policies() -> list[dict]:
    return _of_kind("networkpolicy.yaml", "NetworkPolicy")


def _selects(policy: dict, labels: dict) -> bool:
    return policy["spec"].get("podSelector", {}).get("matchLabels") == labels


def _rules_for_port(policy: dict, port: int) -> list[dict]:
    out = []
    for rule in policy["spec"].get("ingress") or []:
        ports = rule.get("ports") or []
        if any(p.get("port") == port for p in ports) or not ports:
            out.append(rule)
    return out


# ---------- every manifest has to be loadable ----------

@pytest.mark.parametrize("name", sorted(p.name for p in K8S.glob("*.yaml")))
def test_manifest_parses(name):
    assert _documents(name), f"{name} parsed to nothing"


# ---------- default deny ----------

def test_a_default_deny_ingress_policy_exists():
    deny = [p for p in _policies()
            if _selects(p, {}) or p["spec"].get("podSelector") == {}]
    assert deny, "no policy selects every pod"
    assert any("Ingress" in p["spec"].get("policyTypes", [])
               and not p["spec"].get("ingress")
               for p in deny), "the catch-all policy still allows ingress"


# ---------- only the gateway namespace reaches the app ----------

def test_the_app_port_is_opened_to_the_gateway_namespace():
    allowed = [p for p in _policies() if _selects(p, APP_LABEL)
               and _rules_for_port(p, APP_PORT)]
    assert allowed, f"nothing opens port {APP_PORT} to the gateway"
    peers = [peer for p in allowed for rule in _rules_for_port(p, APP_PORT)
             for peer in (rule.get("from") or [])]
    assert peers, "the rule has no peers, which allows every source"
    assert all("namespaceSelector" in peer for peer in peers), \
        "a peer reaches the app without being scoped to a namespace"


def test_nothing_opens_the_app_port_to_every_source():
    """An ingress rule with no `from` allows the whole cluster."""
    for policy in _policies():
        for rule in _rules_for_port(policy, APP_PORT):
            if policy["spec"].get("podSelector") in ({}, None):
                continue                      # the default-deny policy has no rules
            assert rule.get("from"), \
                f"{policy['metadata']['name']} opens {APP_PORT} to everything"


# ---------- default-deny must not break the app's own dependencies ----------

@pytest.mark.parametrize("labels,port", [({"app": "qdrant"}, 6333),
                                         ({"app": "redis"}, 6379)])
def test_the_app_can_still_reach_its_backends(labels, port):
    """Default-deny applies to the backends too, so the app needs an explicit
    way in or retrieval fails closed on every request."""
    policies = [p for p in _policies() if _selects(p, labels)]
    assert policies, f"nothing lets the app reach {labels}"
    peers = [peer for p in policies for rule in _rules_for_port(p, port)
             for peer in (rule.get("from") or [])]
    assert any(peer.get("podSelector", {}).get("matchLabels") == APP_LABEL
               for peer in peers), f"the app is not allowed to reach {labels}"


# ---------- the cold start needs a startupProbe ----------

def _app_container() -> dict:
    deployment = _of_kind("app-deployment.yaml", "Deployment")[0]
    return deployment["spec"]["template"]["spec"]["containers"][0]


def test_the_deployment_has_a_startup_probe():
    assert "startupProbe" in _app_container(), \
        "liveness alone kills the container at ~55s while BM25 is still building"


def test_the_startup_probe_budget_covers_the_measured_cold_start():
    probe = _app_container()["startupProbe"]
    budget = probe.get("periodSeconds", 10) * probe.get("failureThreshold", 3)
    assert budget >= 90, f"startup budget {budget}s is under the measured ~90s"
