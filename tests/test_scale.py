import pytest

from triage_desk import scale as sc

NORMAL = {"mobility": "walking", "rr": 12, "hr": 70, "sbp": 120, "temp_c": 37.0, "avpu": "A", "trauma": False}

# Every boundary of SATS-2012 Fig.14 (adult TEWS), checked one parameter at a time.
BOUNDARIES = [
    ("rr", 8, 2), ("rr", 9, 0), ("rr", 14, 0), ("rr", 15, 1), ("rr", 20, 1), ("rr", 21, 2), ("rr", 29, 2), ("rr", 30, 3),
    ("hr", 40, 2), ("hr", 41, 1), ("hr", 50, 1), ("hr", 51, 0), ("hr", 100, 0), ("hr", 101, 1), ("hr", 110, 1),
    ("hr", 111, 2), ("hr", 129, 2), ("hr", 130, 3),
    ("sbp", 70, 3), ("sbp", 71, 2), ("sbp", 80, 2), ("sbp", 81, 1), ("sbp", 100, 1), ("sbp", 101, 0), ("sbp", 199, 0), ("sbp", 200, 2),
    ("temp_c", 34.9, 2), ("temp_c", 35.0, 0), ("temp_c", 38.4, 0), ("temp_c", 38.5, 2),
    ("avpu", "A", 0), ("avpu", "C", 2), ("avpu", "V", 1), ("avpu", "P", 2), ("avpu", "U", 3),
    ("mobility", "walking", 0), ("mobility", "with_help", 1), ("mobility", "stretcher_immobile", 2),
    ("trauma", False, 0), ("trauma", True, 1),
]


@pytest.mark.parametrize("param,value,expected", BOUNDARIES)
def test_tews_boundaries(param, value, expected):
    assert sc.score_param(param, value) == expected


@pytest.mark.parametrize("tews,colour", [(0, "GREEN"), (2, "GREEN"), (3, "YELLOW"), (4, "YELLOW"), (5, "ORANGE"),
                                         (6, "ORANGE"), (7, "RED"), (14, "RED")])
def test_bands(tews, colour):
    assert sc.tews_band(tews)["colour"] == colour


def test_normal_is_green():
    r = sc.score(NORMAL)
    assert r["colour"] == "GREEN" and r["tews"] == 0 and r["complete"]


def test_emergency_sign_is_red_regardless_of_tews():
    r = sc.score(NORMAL, ["seizure_current"])
    assert r["colour"] == "RED" and r["emergency_sign"]


def test_highest_of_sign_and_tews():
    v = {**NORMAL, "rr": 30, "hr": 130, "sbp": 70}  # 3+3+3 = 9
    assert sc.score(v, ["moderate_pain"])["colour"] == "RED"
    assert sc.score(NORMAL, ["chest_pain"])["colour"] == "ORANGE"


def test_missing_is_reported_not_assumed():
    v = {k: x for k, x in NORMAL.items() if k != "sbp"}
    r = sc.score(v)
    assert not r["complete"] and r["missing"] == ["sbp"] and r["tews"] is None
    assert r["tews_max"] == 3 and "UNKNOWN: SBP" in r["reason"]


def test_confused_derives_very_urgent():
    r = sc.score({**NORMAL, "avpu": "C"})
    assert r["colour"] == "ORANGE"
    assert any(d["id"] == "reduced_loc_confused" for d in r["drivers"])


def test_glucose_rules():
    assert sc.score({**NORMAL, "glucose_mmol": 2.9})["colour"] == "RED"
    assert sc.score({**NORMAL, "glucose_mmol": 12, "ketonuria": True})["colour"] == "ORANGE"
    assert sc.score({**NORMAL, "glucose_mmol": 18, "ketonuria": False})["colour"] == "YELLOW"
    assert sc.score({**NORMAL, "glucose_mmol": 12})["colour"] == "GREEN"


def test_pregnancy_abdominal_pain():
    assert sc.score(NORMAL, ["abdominal_pain"], pregnant=True)["colour"] == "ORANGE"
    assert sc.score(NORMAL, ["abdominal_pain"], pregnant=False)["colour"] == "YELLOW"


def test_spo2_local_policy():
    assert sc.score({**NORMAL, "spo2": 91})["colour"] == "RED"
    assert sc.score({**NORMAL, "spo2": 92})["colour"] == "GREEN"


def test_child_out_of_scope():
    r = sc.score(NORMAL, age_years=8)
    assert r["out_of_scope"] and r["colour"] is None


def test_every_driver_is_cited():
    r = sc.score({**NORMAL, "rr": 25, "spo2": 90}, ["chest_pain"])
    assert all(d["cite"] for d in r["drivers"])


def test_bad_values_dropped_with_reason():
    r = sc.score({**NORMAL, "avpu": "drowsy"}, ["made_up_sign"])
    assert "avpu" in r["missing"]
    assert any("AVPU" in p for p in r["problems"]) and any("made_up_sign" in p for p in r["problems"])
