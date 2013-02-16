# -*- coding: utf-8 -*-

import os
import io

from ConfigParser import SafeConfigParser, NoSectionError, NoOptionError

class Inifile(object):
    def __init__(self, inistr=None):
        self._cp = cp = SafeConfigParser()

        # Preserve case
        cp.optionxform = str

        if inistr:
            cp.readfp(io.BytesIO(inistr))

    @staticmethod
    def load(file):
        if isinstance(file, basestring):
            file = open(file)

        return Inifile(file.read())

    def set(self, section, option, value):
        value = str(value)

        try:
            self._cp.set(section, option, value)
        except NoSectionError:
            self._cp.add_section(section)
            self._cp.set(section, option, value)

    def get(self, section, option, default=None, raw=False, vars=None):
        try:
            return self._cp.get(section, option, raw=raw, vars=vars)
        except NoSectionError:
            if default is None:
                raise

            self._cp.add_section(section)
            return self.get(section, option, default, raw, vars)
        except NoOptionError:
            if default is None:
                raise

            self._cp.set(section, option, str(default))
            return self._cp.get(section, option, raw=raw, vars=vars)

    def getpath(self, section, option, default=None, vars=None):
        path = self.get(section, option, default, vars)
        path = os.path.expandvars(path)
        path = os.path.expanduser(path)
        path = os.path.abspath(path)

        return path

    def getfloat(self, section, option, default=None):
        return float(self.get(section, option, default))

    def getint(self, section, option, default=None):
        return int(self.get(section, option, default))

    _bool_states = {'1': True, 'yes': True, 'true': True, 'on': True,
                    '0': False, 'no': False, 'false': False, 'off': False}

    def getbool(self, section, option, default=None):
        v = self.get(section, option, default)
        return self._bool_states[v.lower()]

    def tostr(self):
        buf = io.BytesIO()
        self._cp.write(buf)
        return buf.getvalue()