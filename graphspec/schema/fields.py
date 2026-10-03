def fields(mapping, names, where):
    if not isinstance(mapping, dict):
        raise ValueError("%s is stated as a mapping, found %r" % (where, mapping))
    lacking = [name for name in names if name not in mapping]
    if lacking:
        raise ValueError("%s lacks %s" % (where, ", ".join(lacking)))
    unknown = sorted(map(str, set(mapping) - set(names)))
    if unknown:
        raise ValueError("%s holds fields it does not know: %s" % (where, ", ".join(unknown)))
    return mapping


def named(name, where):
    if type(name) is not str or not name or ":" in name:
        raise ValueError("%s needs a name without colons, found %r" % (where, name))
    return name
