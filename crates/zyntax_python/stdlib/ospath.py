# The functions of `os.path` that work on the text of a path. The ones
# that look at the file system are the host's.


def isabs(p):
    return p.startswith("/")


def join(first, *rest):
    path = first
    for part in rest:
        if part.startswith("/"):
            path = part
        elif path == "" or path.endswith("/"):
            path = path + part
        else:
            path = path + "/" + part
    return path


def split(p):
    i = p.rfind("/") + 1
    head = p[:i]
    tail = p[i:]
    if head != "" and head != "/" * len(head):
        head = head.rstrip("/")
    return head, tail


def splitext(p):
    sep = p.rfind("/")
    dot = p.rfind(".")
    if dot <= sep:
        return p, ""
    # A leading dot of the name is part of the name.
    i = sep + 1
    while i < dot and p[i] == ".":
        i += 1
    if i == dot:
        return p, ""
    return p[:dot], p[dot:]


def normpath(p):
    if p == "":
        return "."
    rooted = p.startswith("/")
    leading = 0
    if rooted:
        leading = 1
        if p.startswith("//") and not p.startswith("///"):
            leading = 2
    parts = []
    for part in p.split("/"):
        if part == "" or part == ".":
            continue
        if part != ".." or (not rooted and (len(parts) == 0 or parts[len(parts) - 1] == "..")):
            parts.append(part)
        elif len(parts) > 0:
            parts.pop()
    out = "/".join(parts)
    if rooted:
        out = "/" * leading + out
    if out == "":
        return "."
    return out


def commonprefix(paths):
    if len(paths) == 0:
        return ""
    low = min(paths)
    high = max(paths)
    i = 0
    while i < len(low) and i < len(high) and low[i] == high[i]:
        i += 1
    return low[:i]
