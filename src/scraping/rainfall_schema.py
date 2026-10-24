"""Independent five-minute NEA gauge observations, never radar-derived truth."""
from datetime import datetime, timedelta
import math
from zoneinfo import ZoneInfo

SGT = ZoneInfo('Asia/Singapore')
URL = 'https://api-open.data.gov.sg/v2/real-time/api/rainfall'


def parse_pages(pages, day):
    stations, observations = {}, {}
    missing = 0
    for page in pages:
        if page.get('code') != 0:
            raise ValueError('Rainfall API returned an error')
        data = page['data']
        if data['readingUnit'] != 'mm' or data['readingType'] != 'TB1 Rainfall 5 Minute Total F':
            raise ValueError('Unknown rainfall units or accumulation period')
        for station in data['stations']:
            sid = station['id']
            value = dict(name=station['name'], **station['location'])
            if not all(math.isfinite(float(value[k])) for k in ('latitude', 'longitude')):
                raise ValueError('Invalid station location')
            if sid in stations and stations[sid] != value:
                raise ValueError('Station metadata changed within day')
            stations[sid] = value
        for reading in data['readings']:
            stamp = datetime.fromisoformat(reading['timestamp'])
            if stamp.tzinfo is None:
                raise ValueError('Gauge timestamp needs timezone')
            stamp = stamp.astimezone(SGT)
            if stamp.strftime('%Y%m%d') != day or stamp.second or stamp.microsecond or stamp.minute % 5:
                raise ValueError('Observation is outside requested day or five-minute grid')
            key = stamp.strftime('%Y%m%d%H%M')
            for row in reading['data']:
                sid, amount = row['stationId'], row['value']
                if sid not in stations:
                    raise ValueError('Observation lacks station metadata')
                if amount is None:
                    missing += 1
                    continue
                if isinstance(amount, bool) or not isinstance(amount, (float, int)) or not math.isfinite(amount) or amount < 0:
                    raise ValueError('Invalid rainfall amount; never replace it with zero')
                record = (sid, key)
                if record in observations and observations[record] != amount:
                    raise ValueError('Conflicting duplicate rainfall observation')
                observations[record] = float(amount)
    return stations, observations, missing
