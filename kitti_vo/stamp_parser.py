
import re

class StampParser(object):

    def __init__(self):
        self.sec_minute = 60
        self.sec_hour = 60 * 60
        self.sec_day = 60 * 60 * 24
        self.sec_month = 60 * 60 * 24 * 30
        self.sec_year = 60 * 60 * 24 * 30 * 12
        self.iso_re = re.compile('(\\d{4})-(\\d{2})-(\\d{2})\s(\\d{2}):(\\d{2}):(\\d{2}(?:\\.?\\d+))')

    def parse(self, time_str):
        match = self.iso_re.match(time_str)
        year, month, day, hour, minute, second = map(float, match.groups())
        return (
            second                   +
            minute * self.sec_minute +
            hour   * self.sec_hour   +
            day    * self.sec_day    +
            month  * self.sec_month  +
            year   * self.sec_year
        )
