"""The monitoring stack has to query metrics that actually exist.

`/metrics` has always exposed Prometheus text format, but nothing scraped it.
Adding a scraper and a dashboard introduces a failure mode that is invisible at
deploy time: a panel whose query names a metric the app never emits renders an
empty graph forever, and looks exactly like "no traffic". Nothing in Prometheus
or Grafana rejects it.

So these tests check the queries against the registry in src/metrics.py:

1. every `rag_*` series named in a rule or a panel is one the code can emit;
2. nothing asks for a quantile. The latency metric is exposed as a count+sum
   summary with no buckets and no quantiles, so `histogram_quantile()` returns
   nothing at all rather than failing;
3. the files the compose profile mounts exist, and the dashboard's datasource
   uid matches the provisioned one -- a dangling uid renders every panel as
   "Datasource not found".
"""
import json
import re
from pathlib import Path

import pytest
import yaml

from src import metrics as m

MON = Path("deploy/monitoring")
DASHBOARD = MON / "grafana/dashboards/rag-overview.json"
DATASOURCES = MON / "grafana/provisioning/datasources/prometheus.yml"
PROVIDERS = MON / "grafana/provisioning/dashboards/dashboards.yml"
APP_PORT = 8000
METRICS_PATH = "/metrics"

# Series Prometheus itself synthesises, so they are legitimate in a query even
# though the app never exposes them.
BUILTIN_SERIES = {"up", "scrape_duration_seconds", "scrape_samples_scraped"}


def _yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _dashboard() -> dict:
    return json.loads(DASHBOARD.read_text(encoding="utf-8"))


def _exposed_series() -> set[str]:
    """What src/metrics.py can put on the wire.

    A summary exposes `_count` and `_sum`, never the bare family name, which is
    the detail a hand-written query gets wrong.
    """
    out = set()
    for metric in m._ALL:
        if isinstance(metric, m.Histogram):
            out |= {f"{metric.name}_count", f"{metric.name}_sum"}
        else:
            out.add(metric.name)
    return out


def _queries() -> list[tuple[str, str]]:
    """(where it came from, PromQL) for every expression in the stack."""
    found: list[tuple[str, str]] = []
    rules = _yaml(MON / "rules.yml")
    for group in rules["groups"]:
        for rule in group["rules"]:
            found.append((f"rules.yml:{rule['alert']}", rule["expr"]))
    for panel in _dashboard()["panels"]:
        for target in panel.get("targets", []):
            found.append((f"panel:{panel['title']}", target["expr"]))
    for var in _dashboard()["templating"]["list"]:
        found.append((f"variable:{var['name']}", var["query"]))
    return found


def _rag_series(expr: str) -> set[str]:
    return set(re.findall(r"\brag_[a-z0-9_]+", expr))


# ---------- the queries have to name real series ----------

def test_there_are_queries_to_check():
    """Guards against a regex or a schema change silently finding nothing."""
    queries = _queries()
    assert len(queries) >= 10, f"only found {len(queries)} expressions"
    assert any(_rag_series(expr) for _, expr in queries)


@pytest.mark.parametrize("source,expr", _queries(), ids=lambda v: str(v)[:40])
def test_every_queried_series_is_one_the_app_exposes(source, expr):
    exposed = _exposed_series()
    for series in _rag_series(expr):
        assert series in exposed, (
            f"{source} queries {series}, which src/metrics.py never emits "
            f"(it exposes {sorted(exposed)})")


def test_the_latency_metric_is_queried_as_a_summary_not_a_histogram():
    """The exposition is count+sum only. `histogram_quantile()` over it returns
    nothing -- an empty panel, not an error."""
    for source, expr in _queries():
        assert "histogram_quantile" not in expr, \
            f"{source} asks for a quantile the exposition cannot provide"
        assert f"{m.ANSWER_LATENCY.name}_bucket" not in expr, \
            f"{source} queries a bucket series that does not exist"


def test_latency_is_averaged_from_both_halves_of_the_summary():
    """rate(_sum)/rate(_count). Using _sum alone reports throughput-weighted
    seconds-per-second, which looks like a latency but is not one."""
    latency = [(s, e) for s, e in _queries() if m.ANSWER_LATENCY.name in e]
    assert latency, "nothing charts answer latency"
    for source, expr in latency:
        assert f"{m.ANSWER_LATENCY.name}_sum" in expr, f"{source} has no _sum"
        assert f"{m.ANSWER_LATENCY.name}_count" in expr, f"{source} has no _count"


def test_a_counter_is_never_graphed_raw():
    """A monotonic counter plotted as-is only ever slopes upward, and resets to
    zero on restart. Every counter panel has to go through rate()/increase()."""
    counters = {metric.name for metric in m._ALL if isinstance(metric, m.Counter)}
    for panel in _dashboard()["panels"]:
        for target in panel.get("targets", []):
            expr = target["expr"]
            if counters & _rag_series(expr):
                assert re.search(r"\b(rate|irate|increase)\s*\(", expr), \
                    f"panel {panel['title']!r} graphs a counter without rate()"


# ---------- a scraper has to accept what /metrics emits ----------

def _scrape() -> dict[str, set[tuple[str, ...]]]:
    """Parse a populated /metrics with the real Prometheus parser.

    src/metrics.py writes the text format by hand to stay dependency-free at
    runtime, so nothing else proves a scraper can read it. {series name: set of
    label-key tuples}.
    """
    from prometheus_client.parser import text_string_to_metric_families

    m.reset()
    try:
        m.REQUESTS.inc(endpoint="ask", status="ok")
        m.REQUESTS.inc(endpoint="ask_stream", status="error")
        m.REQUESTS.inc(endpoint="ask", status="rejected")
        m.CACHE_HITS.inc()
        m.TOOL_CALLS.inc(tool="search")
        m.ANSWER_LATENCY.observe(1.25, endpoint="ask")
        out: dict[str, set[tuple[str, ...]]] = {}
        for family in text_string_to_metric_families(m.render()):
            for sample in family.samples:
                out.setdefault(sample.name, set()).add(tuple(sorted(sample.labels)))
        return out
    finally:
        m.reset()


def test_the_exposition_parses_as_prometheus_text():
    scraped = _scrape()
    assert scraped, "the parser found no samples in /metrics"
    assert scraped["rag_requests_total"] == {("endpoint", "status")}


def test_an_empty_registry_still_parses():
    """A pod's first scrape happens before any request. A malformed empty body
    makes the target look down."""
    from prometheus_client.parser import text_string_to_metric_families

    m.reset()
    families = list(text_string_to_metric_families(m.render()))
    assert [f.name for f in families], "nothing is exposed before the first request"


def test_every_series_the_dashboard_queries_appears_in_a_real_scrape():
    """The end of the chain: the names in the panels are the names a scraper
    actually sees, suffixes and all."""
    scraped = _scrape()
    queried = {s for _, expr in _queries() for s in _rag_series(expr)}
    assert queried, "no series to check"
    missing = queried - set(scraped)
    assert not missing, f"queried but never scraped: {sorted(missing)}"


def test_the_labels_the_queries_group_by_are_really_emitted():
    """`sum by (endpoint)` over a series carrying no endpoint label collapses
    every value into one line instead of erroring."""
    scraped = _scrape()
    assert "endpoint" in set().union(*scraped["rag_requests_total"])
    assert "tool" in set().union(*scraped["rag_tool_calls_total"])
    assert "endpoint" in set().union(*scraped["rag_answer_latency_seconds_count"])


# ---------- the scrape config has to reach the app ----------

def test_prometheus_scrapes_the_app_metrics_endpoint():
    cfg = _yaml(MON / "prometheus.yml")
    jobs = {j["job_name"]: j for j in cfg["scrape_configs"]}
    assert "rag-app" in jobs, "no job scrapes the app"
    job = jobs["rag-app"]
    assert job.get("metrics_path", "/metrics") == METRICS_PATH
    targets = [t for sc in job["static_configs"] for t in sc["targets"]]
    assert any(t.endswith(f":{APP_PORT}") for t in targets), \
        f"the job does not target port {APP_PORT}: {targets}"


def test_every_rule_file_prometheus_loads_is_mounted():
    """A `rule_files` glob that matches nothing is not an error: Prometheus
    starts clean and every alert silently does not exist."""
    cfg = _yaml(MON / "prometheus.yml")
    mounts = _compose_mounts("prometheus")
    for pattern in cfg.get("rule_files") or []:
        assert pattern in mounts, f"{pattern} is not mounted into the container"
        assert Path(mounts[pattern]).is_file(), f"{mounts[pattern]} does not exist"


def test_the_alert_rules_parse_and_are_scoped():
    rules = _yaml(MON / "rules.yml")
    assert rules["groups"], "no rule groups"
    for group in rules["groups"]:
        for rule in group["rules"]:
            assert rule.get("alert"), f"unnamed rule in {group['name']}"
            assert rule.get("for"), f"{rule['alert']} fires on a single sample"
            assert rule["annotations"].get("summary"), \
                f"{rule['alert']} has no summary"


# ---------- provisioning ----------

def _compose(path: str = "docker-compose.yml") -> dict:
    return _yaml(Path(path))


def _compose_mounts(service: str) -> dict[str, str]:
    """{container path: host path} for the service's bind mounts."""
    out = {}
    for vol in _compose()["services"][service].get("volumes", []):
        host, _, rest = vol.partition(":")
        container = rest.split(":")[0]
        if host.startswith("."):          # a bind mount, not a named volume
            out[container] = host
    return out


@pytest.mark.parametrize("service", ["prometheus", "grafana"])
def test_the_monitoring_services_are_behind_a_profile(service):
    """They must not start with a plain `docker compose up`."""
    assert _compose()["services"][service]["profiles"] == ["monitoring"]


@pytest.mark.parametrize("service", ["prometheus", "grafana"])
def test_every_mounted_path_exists(service):
    for container, host in _compose_mounts(service).items():
        assert Path(host).exists(), \
            f"{service} mounts {host} at {container}, which is not in the repo"


@pytest.mark.parametrize("service", ["prometheus", "grafana"])
def test_the_monitoring_ports_are_not_published_to_the_network(service):
    """`/metrics` is unauthenticated and the Grafana instance allows anonymous
    read, so neither belongs on a LAN-reachable port."""
    for mapping in _compose()["services"][service]["ports"]:
        assert str(mapping).startswith("127.0.0.1:"), \
            f"{service} publishes {mapping} on every interface"


def _datasource_uids(node) -> set[str]:
    """Every uid referenced as a datasource, at any depth.

    Deliberately not a regex over the file: the dashboard's own top-level
    "uid" is its identity, not a datasource reference.
    """
    found = set()
    if isinstance(node, dict):
        ds = node.get("datasource")
        if isinstance(ds, dict) and "uid" in ds:
            found.add(ds["uid"])
        for value in node.values():
            found |= _datasource_uids(value)
    elif isinstance(node, list):
        for item in node:
            found |= _datasource_uids(item)
    return found


def test_the_dashboard_uses_the_provisioned_datasource_uid():
    """A uid that does not match renders every panel as 'Datasource not found'."""
    uid = _yaml(DATASOURCES)["datasources"][0]["uid"]
    referenced = _datasource_uids(_dashboard())
    assert referenced, "no panel declares a datasource"
    assert referenced == {uid}, f"dashboard points at {referenced}, provisioned is {uid}"


def test_every_panel_and_variable_names_its_datasource():
    """An omitted datasource falls back to Grafana's default, which makes the
    dashboard depend on instance state instead of the provisioning file."""
    dash = _dashboard()
    for panel in dash["panels"]:
        assert isinstance(panel.get("datasource"), dict), \
            f"panel {panel['title']!r} has no explicit datasource"
        for target in panel.get("targets", []):
            assert isinstance(target.get("datasource"), dict), \
                f"a target of {panel['title']!r} has no explicit datasource"
    for var in dash["templating"]["list"]:
        assert isinstance(var.get("datasource"), dict), \
            f"variable {var['name']!r} has no explicit datasource"


def test_grafana_finds_the_dashboard_where_the_provider_looks():
    provider_path = _yaml(PROVIDERS)["providers"][0]["options"]["path"]
    mounts = _compose_mounts("grafana")
    assert provider_path in mounts, \
        f"the provider reads {provider_path}, which is not mounted"
    assert (Path(mounts[provider_path]) / DASHBOARD.name).is_file()


def test_grafana_runs_sqlite_in_wal_mode():
    """Observed on a first boot of grafana 11.5.2 without it: provisioning
    races the startup migrations and loses -- `failed to save dashboard ...
    database is locked`. The file provisioner then only retries when the
    file's mtime changes, so Grafana serves no dashboard at all, across
    restarts, with nothing in the UI to say why."""
    env = _compose()["services"]["grafana"]["environment"]
    assert "GF_DATABASE_WAL=true" in env


def test_only_the_provisioning_subdirectories_we_provide_are_mounted():
    """A read-only bind over /etc/grafana/provisioning hides the image's own
    plugins/ and alerting/ directories, and Grafana logs an error for each
    missing one on every boot. Error lines in a monitoring stack train people
    to ignore error lines."""
    mounts = _compose_mounts("grafana")
    assert "/etc/grafana/provisioning" not in mounts, \
        "mounting the whole provisioning tree shadows the image's own subdirectories"
    for sub in ("datasources", "dashboards"):
        assert f"/etc/grafana/provisioning/{sub}" in mounts


def test_the_home_dashboard_path_points_at_the_dashboard_file():
    env = _compose()["services"]["grafana"]["environment"]
    home = [v.split("=", 1)[1] for v in env
            if v.startswith("GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=")]
    assert home, "no default home dashboard"
    provider_path = _yaml(PROVIDERS)["providers"][0]["options"]["path"]
    assert home[0] == f"{provider_path}/{DASHBOARD.name}"


def test_every_panel_has_a_query():
    for panel in _dashboard()["panels"]:
        assert panel.get("targets"), f"panel {panel['title']!r} charts nothing"
        for target in panel["targets"]:
            assert target.get("expr", "").strip(), \
                f"panel {panel['title']!r} has an empty expression"


# ---------- kubernetes ----------

def test_the_deployment_is_annotated_for_scraping():
    dep = _yaml(Path("deploy/k8s/app-deployment.yaml"))
    ann = dep["spec"]["template"]["metadata"].get("annotations") or {}
    assert ann.get("prometheus.io/scrape") == "true", \
        "the pods are not advertised for scraping"
    assert ann.get("prometheus.io/path") == METRICS_PATH
    container = dep["spec"]["template"]["spec"]["containers"][0]
    port = container["ports"][0]["containerPort"]
    assert int(ann["prometheus.io/port"]) == port, \
        f"the annotation points at {ann['prometheus.io/port']}, the container listens on {port}"


def test_the_monitoring_namespace_is_allowed_through_default_deny():
    """Default-deny ingress drops the scrape too, and a dropped scrape is
    indistinguishable from an idle app."""
    docs = [d for d in yaml.safe_load_all(
        Path("deploy/k8s/networkpolicy.yaml").read_text(encoding="utf-8")) if d]
    policies = [d for d in docs if d.get("kind") == "NetworkPolicy"]
    peers = [peer
             for p in policies
             if p["spec"].get("podSelector", {}).get("matchLabels") == {"app": "rag-app"}
             for rule in p["spec"].get("ingress") or []
             for peer in rule.get("from") or []]
    labels = [peer.get("namespaceSelector", {}).get("matchLabels", {})
              for peer in peers]
    assert {"rag-monitoring": "true"} in labels, \
        "no policy lets a monitoring namespace reach the app"
