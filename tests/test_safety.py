from src.safety import assess_risk, safety_notice


def test_detects_electrical_and_earth_fault():
    cats = assess_risk("What caused the earth fault on the 440V winch motor?")
    assert "electrical" in cats


def test_detects_refrigerant():
    assert "refrigerant" in assess_risk("The CH-200 threw an R-410A leak fault E-24")


def test_detects_lockout_and_pressure():
    assert "lockout_tagout" in assess_risk("apply lockout tagout before service")
    assert "pressure" in assess_risk("depressurize the hydraulic accumulator")


def test_no_risk_for_routine_query():
    assert assess_risk("how often should the condenser fins be cleaned?") == []


def test_safety_notice_assembles():
    notice = safety_notice(["electrical"])
    assert notice.startswith("⚠ Safety notice:")
    assert "lockout/tagout" in notice
    assert "supervisor" in notice


def test_empty_categories_no_notice():
    assert safety_notice([]) == ""
