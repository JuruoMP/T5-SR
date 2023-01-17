from typing import List


def tuple_to_list(obj):
    if isinstance(obj, dict):
        for k in obj:
            obj[k] = tuple_to_list(obj[k])
        return obj
    elif isinstance(obj, List):
        obj = [tuple_to_list(x) for x in obj]
        return obj
    elif isinstance(obj, tuple):
        obj = [tuple_to_list(x) for x in obj]
        return obj
    else:
        return obj


def list_to_set(obj):
    if isinstance(obj, dict):
        for k in obj:
            obj[k] = list_to_set(obj[k])
        return obj
    elif isinstance(obj, List):
        return list(sorted(obj, key=lambda x: str(x)))
    else:
        return obj