from src.access import AccessContext, acl_visible


def test_from_headers_parses_csv_and_admin():
    ctx = AccessContext.from_headers({"X-User-Id": "u1", "X-Roles": "tech, admin",
                                      "X-Sites": "siteA"})
    assert ctx.user_id == "u1" and ctx.is_admin is True
    assert ctx.roles == frozenset({"tech", "admin"}) and ctx.sites == frozenset({"siteA"})


def test_public_chunk_visible_to_all():
    ctx = AccessContext(user_id="u", roles=frozenset({"tech"}))
    assert acl_visible({}, ctx) is True
    assert acl_visible({"acl_roles": [], "acl_sites": []}, ctx) is True


def test_none_or_admin_sees_everything():
    assert acl_visible({"acl_roles": ["secret"]}, None) is True
    assert acl_visible({"acl_roles": ["secret"]}, AccessContext.admin()) is True


def test_role_gate():
    ctx = AccessContext(roles=frozenset({"tech"}))
    assert acl_visible({"acl_roles": ["tech", "eng"]}, ctx) is True
    assert acl_visible({"acl_roles": ["eng"]}, ctx) is False


def test_site_gate():
    ctx = AccessContext(roles=frozenset({"tech"}), sites=frozenset({"siteA"}))
    assert acl_visible({"acl_sites": ["siteA"]}, ctx) is True
    assert acl_visible({"acl_sites": ["siteB"]}, ctx) is False


def test_both_dimensions_must_pass():
    ctx = AccessContext(roles=frozenset({"tech"}), sites=frozenset({"siteA"}))
    # right role, wrong site → blocked
    assert acl_visible({"acl_roles": ["tech"], "acl_sites": ["siteB"]}, ctx) is False
    # right role, right site → visible
    assert acl_visible({"acl_roles": ["tech"], "acl_sites": ["siteA"]}, ctx) is True
