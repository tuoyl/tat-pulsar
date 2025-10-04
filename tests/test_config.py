from pathlib import Path

from tatpulsar import config


def test_jpleph_returns_existing_ephemeris_paths():
    de200_path = Path(config.jpleph("DE200"))
    de421_path = Path(config.jpleph("de421"))

    assert de200_path.name == "de200.bsp"
    assert de421_path.name == "de421.bsp"

    assert de200_path.is_file()
    assert de421_path.is_file()
