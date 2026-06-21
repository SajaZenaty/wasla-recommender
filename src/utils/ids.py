"""Helpers for matching user/post ids across str and int representations."""


def id_variants(value):
    """Return the value and its common alternate representation."""
    variants = [value]
    if isinstance(value, str) and value.isdigit():
        variants.append(int(value))
    elif isinstance(value, int):
        variants.append(str(value))
    return variants


def filter_by_id(df, column, value):
    """Return rows whose ``column`` matches ``value`` (str/int tolerant)."""
    for variant in id_variants(value):
        matches = df[df[column] == variant]
        if not matches.empty:
            return matches
    return df.iloc[0:0]


def lookup_in_mapping(mapping, value, default=None):
    """Look up ``value`` in a dict keyed by int or str ids."""
    for variant in id_variants(value):
        if variant in mapping:
            return mapping[variant]
    return default
