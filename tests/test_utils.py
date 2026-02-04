import datetime

from kpower_forecast.utils import calculate_solar_elevation


def test_calculate_solar_elevation():
    # Test location: Equator, Prime Meridian
    lat, lon = 0.0, 0.0
    
    # Test time: Noon at Equinox (approx) -> Sun should be high (90 deg ideally)
    # March 21st, 12:00 UTC
    times = [datetime.datetime(2024, 3, 21, 12, 0, tzinfo=datetime.timezone.utc)]
    
    elevations = calculate_solar_elevation(lat, lon, times)
    assert len(elevations) == 1
    # Should be close to 90 degrees (zenith)
    assert 85 < elevations[0] < 95

def test_night_mask():
    # Test midnight
    lat, lon = 0.0, 0.0
    times = [datetime.datetime(2024, 3, 21, 0, 0, tzinfo=datetime.timezone.utc)]
    elevations = calculate_solar_elevation(lat, lon, times)
    assert elevations[0] < 0
