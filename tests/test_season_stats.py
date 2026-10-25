from stuffplus.season_stats import parse_innings


def test_baseball_innings_are_thirds_not_decimals():
    assert parse_innings("162.1") == 162 + 1 / 3
    assert parse_innings("0.2") == 2 / 3
    assert parse_innings("10.0") == 10
    assert parse_innings("7") == 7
