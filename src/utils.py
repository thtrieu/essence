
def extract(name, dfault, **kwargs):
    kw = dict(kwargs)
    val = dfault
    if name in kw:
        val = kw[name]
        del kw[name]
    return val, kw