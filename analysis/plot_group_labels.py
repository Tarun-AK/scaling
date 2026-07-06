from __future__ import annotations

import re


def _clean_label(text: str) -> str:
    return text.strip(" _-./:")


def _strip_common_token_parts(a: str, b: str) -> tuple[str, str] | None:
    a_tokens = [token for token in re.split(r"[_\-./:]+", a) if token]
    b_tokens = [token for token in re.split(r"[_\-./:]+", b) if token]
    if len(a_tokens) <= 1 and len(b_tokens) <= 1:
        return None

    prefix_len = 0
    while (
        prefix_len < len(a_tokens)
        and prefix_len < len(b_tokens)
        and a_tokens[prefix_len] == b_tokens[prefix_len]
    ):
        prefix_len += 1

    suffix_len = 0
    while (
        suffix_len < len(a_tokens) - prefix_len
        and suffix_len < len(b_tokens) - prefix_len
        and a_tokens[-(suffix_len + 1)] == b_tokens[-(suffix_len + 1)]
    ):
        suffix_len += 1

    a_rem = a_tokens[prefix_len : len(a_tokens) - suffix_len or None]
    b_rem = b_tokens[prefix_len : len(b_tokens) - suffix_len or None]
    a_label = _clean_label("_".join(a_rem))
    b_label = _clean_label("_".join(b_rem))
    if not a_label or not b_label or a_label == b_label:
        return None
    return a_label, b_label


def _strip_common_char_parts(a: str, b: str) -> tuple[str, str]:
    max_prefix = min(len(a), len(b))
    prefix_len = 0
    while prefix_len < max_prefix and a[prefix_len] == b[prefix_len]:
        prefix_len += 1

    a_rem = a[prefix_len:]
    b_rem = b[prefix_len:]
    max_suffix = min(len(a_rem), len(b_rem))
    suffix_len = 0
    while (
        suffix_len < max_suffix
        and a_rem[-(suffix_len + 1)] == b_rem[-(suffix_len + 1)]
    ):
        suffix_len += 1

    if suffix_len > 0:
        a_rem = a_rem[:-suffix_len]
        b_rem = b_rem[:-suffix_len]

    a_label = _clean_label(a_rem) or a
    b_label = _clean_label(b_rem) or b
    return a_label, b_label


def distinct_group_labels(groups: list[str]) -> dict[str, str]:
    if not groups:
        return {}
    if len(groups) != 2:
        return {group: group for group in groups}

    a, b = groups
    stripped = _strip_common_token_parts(a, b)
    if stripped is None:
        stripped = _strip_common_char_parts(a, b)
    a_label, b_label = stripped
    if a_label == b_label:
        return {a: a, b: b}
    return {a: a_label, b: b_label}
