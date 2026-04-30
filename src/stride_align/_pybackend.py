"""Pure Python fallback backend matching the native API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    score: int
    query_start: int
    query_end: int
    target_start: int
    target_end: int
    aligned_query: object
    aligned_target: object
    operations: str


@dataclass(frozen=True, slots=True)
class AlignmentPath:
    score: int
    query_start: int
    query_end: int
    target_start: int
    target_end: int
    operations: str
    cigar: str
    matches: int
    mismatches: int
    insertions: int
    deletions: int
    aligned_length: int


@dataclass(frozen=True, slots=True)
class _PreparedAlignment:
    output_kind: str
    kernel_bits: int
    query_tokens: list[int]
    target_tokens: list[int]
    query_items: list[object]
    target_items: list[object]


def _score_bound(
    query_size: int,
    target_size: int,
    match_score: int,
    mismatch_score: int,
    gap_score: int,
    gap_extend_score: int | None = None,
) -> int:
    gap_extend = gap_score if gap_extend_score is None else gap_extend_score
    return (query_size + target_size) * max(
        abs(match_score),
        abs(mismatch_score),
        abs(gap_score),
        abs(gap_extend),
    )


def _resolve_gap_scores(
    gap_score: int,
    gap_open_score: int | None,
    gap_extend_score: int | None,
) -> tuple[int, int]:
    gap_open = gap_score if gap_open_score is None else gap_open_score
    gap_extend = gap_open if gap_extend_score is None else gap_extend_score
    return gap_open, gap_extend


def _select_kernel_bits(symbol_limit: int, score_bound: int) -> int:
    if score_bound <= 127 and symbol_limit <= 0xFF:
        return 8
    if score_bound <= 32_767 and symbol_limit <= 0xFFFF:
        return 16
    if score_bound <= 2_147_483_647 and symbol_limit <= 0xFFFFFFFF:
        return 32
    if score_bound > 9_223_372_036_854_775_807:
        raise OverflowError("alignment score bound exceeds the 64-bit kernel capacity")
    return 64


def _select_score_bits(score_bound: int) -> int:
    if score_bound <= 127:
        return 8
    if score_bound <= 32_767:
        return 16
    if score_bound <= 2_147_483_647:
        return 32
    if score_bound > 9_223_372_036_854_775_807:
        raise OverflowError("alignment score bound exceeds the 64-bit kernel capacity")
    return 64


def _apply_forced_width(automatic: int, width: int | None) -> int:
    if width is None or width == 0:
        return automatic
    if width not in {8, 16, 32, 64}:
        raise ValueError("width must be None, 0, 8, 16, 32, or 64")
    if width < automatic:
        raise ValueError("width cannot be narrower than the automatically selected kernel width")
    return width


def _materialize_sequence(source: object, label: str) -> list[object]:
    try:
        return list(source)
    except TypeError as exc:
        raise TypeError(
            f"{label} must be bytes, str, or a sequence of immutable hashable objects"
        ) from exc


def _prepare_alignment(
    query: object,
    target: object,
    *,
    match_score: int,
    mismatch_score: int,
    gap_score: int,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> _PreparedAlignment:
    query_is_bytes = isinstance(query, bytes)
    target_is_bytes = isinstance(target, bytes)
    query_is_str = isinstance(query, str)
    target_is_str = isinstance(target, str)

    if (query_is_bytes and target_is_str) or (query_is_str and target_is_bytes):
        raise TypeError("bytes and str inputs cannot be aligned directly against each other")

    if query_is_bytes and target_is_bytes:
        query_tokens = list(query)
        target_tokens = list(target)
        return _PreparedAlignment(
            output_kind="bytes",
            kernel_bits=_apply_forced_width(
                _select_kernel_bits(
                    max(query_tokens + target_tokens, default=0),
                    _score_bound(
                        len(query_tokens),
                        len(target_tokens),
                        match_score,
                        mismatch_score,
                        gap_score,
                        gap_extend_score,
                    ),
                ),
                width,
            ),
            query_tokens=query_tokens,
            target_tokens=target_tokens,
            query_items=[],
            target_items=[],
        )

    if query_is_str and target_is_str:
        query_tokens = [ord(character) for character in query]
        target_tokens = [ord(character) for character in target]
        return _PreparedAlignment(
            output_kind="unicode",
            kernel_bits=_apply_forced_width(
                _select_kernel_bits(
                    max(query_tokens + target_tokens, default=0),
                    _score_bound(
                        len(query_tokens),
                        len(target_tokens),
                        match_score,
                        mismatch_score,
                        gap_score,
                        gap_extend_score,
                    ),
                ),
                width,
            ),
            query_tokens=query_tokens,
            target_tokens=target_tokens,
            query_items=[],
            target_items=[],
        )

    query_items = _materialize_sequence(query, "query")
    target_items = _materialize_sequence(target, "target")

    token_map: dict[object, int] = {}
    next_token = 0

    def encode(items: list[object]) -> list[int]:
        nonlocal next_token
        encoded: list[int] = []
        for item in items:
            token = token_map.get(item)
            if token is None:
                next_token += 1
                token = next_token
                token_map[item] = token
            encoded.append(token)
        return encoded

    query_tokens = encode(query_items)
    target_tokens = encode(target_items)

    return _PreparedAlignment(
        output_kind="sequence",
        kernel_bits=_apply_forced_width(
            _select_kernel_bits(
                next_token,
                _score_bound(
                    len(query_tokens),
                    len(target_tokens),
                    match_score,
                    mismatch_score,
                    gap_score,
                    gap_extend_score,
                ),
            ),
            width,
        ),
        query_tokens=query_tokens,
        target_tokens=target_tokens,
        query_items=query_items,
        target_items=target_items,
    )


def _prepare_farrar_alignment(
    query: object,
    target: object,
    *,
    match_score: int,
    mismatch_score: int,
    gap_score: int,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> _PreparedAlignment:
    query_is_bytes = isinstance(query, bytes)
    target_is_bytes = isinstance(target, bytes)
    query_is_str = isinstance(query, str)
    target_is_str = isinstance(target, str)

    if (query_is_bytes and target_is_str) or (query_is_str and target_is_bytes):
        raise TypeError("bytes and str inputs cannot be aligned directly against each other")

    if query_is_bytes and target_is_bytes:
        query_tokens = list(query)
        target_tokens = list(target)
    elif query_is_str and target_is_str:
        token_map: dict[str, int] = {}

        def encode_str(value: str) -> list[int]:
            encoded: list[int] = []
            for character in value:
                if character not in token_map:
                    if len(token_map) >= 256:
                        raise ValueError(
                            "Farrar token channel supports at most 256 distinct symbols"
                        )
                    token_map[character] = len(token_map)
                encoded.append(token_map[character])
            return encoded

        query_tokens = encode_str(query)
        target_tokens = encode_str(target)
    else:
        query_items = _materialize_sequence(query, "query")
        target_items = _materialize_sequence(target, "target")
        token_map: dict[object, int] = {}

        def encode_items(items: list[object]) -> list[int]:
            encoded: list[int] = []
            for item in items:
                if item not in token_map:
                    if len(token_map) >= 256:
                        raise ValueError(
                            "Farrar token channel supports at most 256 distinct symbols"
                        )
                    token_map[item] = len(token_map)
                encoded.append(token_map[item])
            return encoded

        query_tokens = encode_items(query_items)
        target_tokens = encode_items(target_items)

    score_bound = _score_bound(
        len(query_tokens),
        len(target_tokens),
        match_score,
        mismatch_score,
        gap_score,
        gap_extend_score,
    )
    return _PreparedAlignment(
        output_kind="farrar",
        kernel_bits=_apply_forced_width(_select_score_bits(score_bound), width),
        query_tokens=query_tokens,
        target_tokens=target_tokens,
        query_items=[],
        target_items=[],
    )


def _substitution_score(
    query_base: int,
    target_base: int,
    match_score: int,
    mismatch_score: int,
) -> int:
    return match_score if query_base == target_base else mismatch_score


def _token_independent_global_score(
    query_size: int,
    target_size: int,
    substitution_score: int,
    gap_score: int,
) -> int:
    diagonal_count = min(query_size, target_size) if substitution_score >= 2 * gap_score else 0
    return diagonal_count * substitution_score + (
        query_size + target_size - 2 * diagonal_count
    ) * gap_score


def _token_independent_local_score(
    query_size: int,
    target_size: int,
    substitution_score: int,
    gap_score: int,
) -> int:
    if query_size == 0 or target_size == 0:
        return 0
    if substitution_score <= 0 and gap_score <= 0:
        return 0
    if gap_score <= 0:
        return min(query_size, target_size) * substitution_score if substitution_score > 0 else 0
    if substitution_score <= 0:
        return (query_size + target_size - 1) * gap_score
    if substitution_score >= 2 * gap_score:
        diagonal_count = min(query_size, target_size)
        return diagonal_count * substitution_score + (
            query_size + target_size - 2 * diagonal_count
        ) * gap_score
    return max(
        (query_size + target_size - 1) * gap_score,
        substitution_score + (query_size + target_size - 2) * gap_score,
    )


def _lcs_length(query: list[int], target: list[int]) -> int:
    masks: dict[int, int] = {}
    for index, token in enumerate(query):
        masks[token] = masks.get(token, 0) | (1 << index)

    row = 0
    for token in target:
        x = masks.get(token, 0) | row
        y = (row << 1) | 1
        row = x & ~(x - y)
    return row.bit_count()


def _fast_score_only(
    query: list[int],
    target: list[int],
    *,
    local_alignment: bool,
    match_score: int,
    mismatch_score: int,
    gap_score: int,
) -> int | None:
    if not query or not target:
        return 0 if local_alignment else (len(query) + len(target)) * gap_score

    if local_alignment and match_score <= 0 and mismatch_score <= 0 and gap_score <= 0:
        return 0

    if match_score == mismatch_score:
        if local_alignment:
            return _token_independent_local_score(
                len(query),
                len(target),
                match_score,
                gap_score,
            )
        return _token_independent_global_score(
            len(query),
            len(target),
            match_score,
            gap_score,
        )

    if (
        gap_score == 0
        and match_score > 0
        and mismatch_score <= 0
        and mismatch_score < match_score
    ):
        return _lcs_length(query, target) * match_score

    return None


def _cigar_from_operations(operations: str) -> str:
    if not operations:
        return ""

    pieces: list[str] = []
    current = operations[0]
    count = 0
    for operation in operations:
        if operation == current:
            count += 1
            continue
        pieces.append(f"{count}{current}")
        current = operation
        count = 1
    pieces.append(f"{count}{current}")
    return "".join(pieces)


def _alignment_path_from_result(result: AlignmentResult) -> AlignmentPath:
    operations = result.operations
    return AlignmentPath(
        score=result.score,
        query_start=result.query_start,
        query_end=result.query_end,
        target_start=result.target_start,
        target_end=result.target_end,
        operations=operations,
        cigar=_cigar_from_operations(operations),
        matches=operations.count("M"),
        mismatches=operations.count("X"),
        insertions=operations.count("I"),
        deletions=operations.count("D"),
        aligned_length=len(operations),
    )


def _score_only(
    query: list[int],
    target: list[int],
    *,
    local_alignment: bool,
    match_score: int,
    mismatch_score: int,
    gap_score: int,
) -> int:
    fast_score = _fast_score_only(
        query,
        target,
        local_alignment=local_alignment,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
    )
    if fast_score is not None:
        return fast_score

    previous = [0] * (len(target) + 1)
    current = [0] * (len(target) + 1)

    if not local_alignment:
        for column in range(len(target) + 1):
            previous[column] = column * gap_score

    best_score = 0 if local_alignment else previous[-1]

    for row, query_base in enumerate(query, start=1):
        current[0] = 0 if local_alignment else row * gap_score

        for column, target_base in enumerate(target, start=1):
            diagonal = previous[column - 1] + _substitution_score(
                query_base,
                target_base,
                match_score,
                mismatch_score,
            )
            up = previous[column] + gap_score
            left = current[column - 1] + gap_score
            cell = max(diagonal, up, left)

            if local_alignment:
                cell = max(0, cell)
                best_score = max(best_score, cell)

            current[column] = cell

        previous, current = current, previous

    return best_score if local_alignment else previous[-1]


_NEGATIVE_INFINITY = -(1 << 120)


def _gap_cost(length: int, gap_open_score: int, gap_extend_score: int) -> int:
    if length == 0:
        return 0
    return gap_open_score + (length - 1) * gap_extend_score


def _affine_score_only(
    query: list[int],
    target: list[int],
    *,
    local_alignment: bool,
    match_score: int,
    mismatch_score: int,
    gap_open_score: int,
    gap_extend_score: int,
) -> int:
    previous_h = [0] * (len(target) + 1)
    current_h = [0] * (len(target) + 1)
    previous_e = [_NEGATIVE_INFINITY] * (len(target) + 1)
    current_e = [_NEGATIVE_INFINITY] * (len(target) + 1)

    if not local_alignment:
        for column in range(1, len(target) + 1):
            previous_h[column] = _gap_cost(column, gap_open_score, gap_extend_score)

    best_score = 0 if local_alignment else previous_h[-1]

    for row, query_base in enumerate(query, start=1):
        if local_alignment:
            current_h[0] = 0
            current_e[0] = _NEGATIVE_INFINITY
        else:
            current_h[0] = _gap_cost(row, gap_open_score, gap_extend_score)
            current_e[0] = current_h[0]

        f_score = _NEGATIVE_INFINITY
        for column, target_base in enumerate(target, start=1):
            e_score = max(
                previous_h[column] + gap_open_score,
                previous_e[column] + gap_extend_score,
            )
            f_score = max(
                current_h[column - 1] + gap_open_score,
                f_score + gap_extend_score,
            )
            diagonal = previous_h[column - 1] + _substitution_score(
                query_base,
                target_base,
                match_score,
                mismatch_score,
            )
            cell = max(diagonal, e_score, f_score)
            if local_alignment:
                cell = max(0, cell)
                best_score = max(best_score, cell)

            current_h[column] = cell
            current_e[column] = e_score

        previous_h, current_h = current_h, previous_h
        previous_e, current_e = current_e, previous_e

    return best_score if local_alignment else previous_h[-1]


def _affine_traceback(
    prepared: _PreparedAlignment,
    *,
    local_alignment: bool,
    match_score: int,
    mismatch_score: int,
    gap_open_score: int,
    gap_extend_score: int,
) -> AlignmentResult:
    query = prepared.query_tokens
    target = prepared.target_tokens
    rows = len(query) + 1
    columns = len(target) + 1
    h = [[0 if local_alignment else _NEGATIVE_INFINITY] * columns for _ in range(rows)]
    e = [[_NEGATIVE_INFINITY] * columns for _ in range(rows)]
    f = [[_NEGATIVE_INFINITY] * columns for _ in range(rows)]
    h[0][0] = 0

    if not local_alignment:
        for row in range(1, rows):
            score = _gap_cost(row, gap_open_score, gap_extend_score)
            h[row][0] = score
            e[row][0] = score
        for column in range(1, columns):
            score = _gap_cost(column, gap_open_score, gap_extend_score)
            h[0][column] = score
            f[0][column] = score

    best_score = 0
    best_row = 0
    best_column = 0

    for row, query_base in enumerate(query, start=1):
        for column, target_base in enumerate(target, start=1):
            e[row][column] = max(
                h[row - 1][column] + gap_open_score,
                e[row - 1][column] + gap_extend_score,
            )
            f[row][column] = max(
                h[row][column - 1] + gap_open_score,
                f[row][column - 1] + gap_extend_score,
            )
            diagonal = h[row - 1][column - 1] + _substitution_score(
                query_base,
                target_base,
                match_score,
                mismatch_score,
            )
            cell = max(diagonal, e[row][column], f[row][column])
            if local_alignment:
                cell = max(0, cell)
            h[row][column] = cell

            if local_alignment and cell > best_score:
                best_score = cell
                best_row = row
                best_column = column

    if not local_alignment:
        best_row = len(query)
        best_column = len(target)
        best_score = h[best_row][best_column]

    operations: list[str] = []
    row = best_row
    column = best_column
    state = "h"

    while row > 0 or column > 0:
        if local_alignment and state == "h" and h[row][column] <= 0:
            break

        if state == "h":
            if row > 0 and column > 0:
                diagonal = h[row - 1][column - 1] + _substitution_score(
                    query[row - 1],
                    target[column - 1],
                    match_score,
                    mismatch_score,
                )
                if h[row][column] == diagonal:
                    operations.append("M" if query[row - 1] == target[column - 1] else "X")
                    row -= 1
                    column -= 1
                    continue
            if row > 0 and h[row][column] == e[row][column]:
                state = "e"
                continue
            if column > 0 and h[row][column] == f[row][column]:
                state = "f"
                continue
            break

        if state == "e":
            operations.append("D")
            continues_gap = row > 1 and e[row][column] == e[row - 1][column] + gap_extend_score
            row -= 1
            state = "e" if continues_gap else "h"
            continue

        operations.append("I")
        continues_gap = column > 1 and f[row][column] == f[row][column - 1] + gap_extend_score
        column -= 1
        state = "f" if continues_gap else "h"

    operations.reverse()
    operations_text = "".join(operations)

    return AlignmentResult(
        score=best_score,
        query_start=row,
        query_end=best_row,
        target_start=column,
        target_end=best_column,
        aligned_query=_materialize_output(
            prepared,
            side="query",
            start=row,
            operations=operations_text,
        ),
        aligned_target=_materialize_output(
            prepared,
            side="target",
            start=column,
            operations=operations_text,
        ),
        operations=operations_text,
    )


def _traceback(
    prepared: _PreparedAlignment,
    *,
    local_alignment: bool,
    match_score: int,
    mismatch_score: int,
    gap_score: int,
) -> AlignmentResult:
    query = prepared.query_tokens
    target = prepared.target_tokens
    rows = len(query) + 1
    columns = len(target) + 1
    scores = [[0] * columns for _ in range(rows)]
    directions = [["stop"] * columns for _ in range(rows)]

    if not local_alignment:
        for row in range(1, rows):
            scores[row][0] = row * gap_score
            directions[row][0] = "up"
        for column in range(1, columns):
            scores[0][column] = column * gap_score
            directions[0][column] = "left"

    best_score = 0
    best_row = 0
    best_column = 0

    for row, query_base in enumerate(query, start=1):
        for column, target_base in enumerate(target, start=1):
            diagonal = scores[row - 1][column - 1] + _substitution_score(
                query_base,
                target_base,
                match_score,
                mismatch_score,
            )
            up = scores[row - 1][column] + gap_score
            left = scores[row][column - 1] + gap_score

            cell = diagonal
            direction = "diagonal"
            if up > cell:
                cell = up
                direction = "up"
            if left > cell:
                cell = left
                direction = "left"
            if local_alignment and cell <= 0:
                cell = 0
                direction = "stop"

            scores[row][column] = cell
            directions[row][column] = direction

            if local_alignment and cell > best_score:
                best_score = cell
                best_row = row
                best_column = column

    if not local_alignment:
        best_row = len(query)
        best_column = len(target)
        best_score = scores[best_row][best_column]

    operations: list[str] = []
    row = best_row
    column = best_column

    while row > 0 or column > 0:
        direction = directions[row][column]
        if direction == "stop":
            break
        if direction == "diagonal":
            operations.append("M" if query[row - 1] == target[column - 1] else "X")
            row -= 1
            column -= 1
            continue
        if direction == "up":
            operations.append("D")
            row -= 1
            continue

        operations.append("I")
        column -= 1

    operations.reverse()
    operations_text = "".join(operations)

    return AlignmentResult(
        score=best_score,
        query_start=row,
        query_end=best_row,
        target_start=column,
        target_end=best_column,
        aligned_query=_materialize_output(
            prepared,
            side="query",
            start=row,
            operations=operations_text,
        ),
        aligned_target=_materialize_output(
            prepared,
            side="target",
            start=column,
            operations=operations_text,
        ),
        operations=operations_text,
    )


def _materialize_output(
    prepared: _PreparedAlignment,
    *,
    side: str,
    start: int,
    operations: str,
) -> object:
    is_query = side == "query"

    if prepared.output_kind == "bytes":
        tokens = prepared.query_tokens if is_query else prepared.target_tokens
        gap_operation = "I" if is_query else "D"
        output: list[int] = []
        index = start
        for operation in operations:
            if operation == gap_operation:
                output.append(ord("-"))
            else:
                output.append(tokens[index])
                index += 1
        return bytes(output)

    if prepared.output_kind == "unicode":
        tokens = prepared.query_tokens if is_query else prepared.target_tokens
        gap_operation = "I" if is_query else "D"
        characters: list[str] = []
        index = start
        for operation in operations:
            if operation == gap_operation:
                characters.append("-")
            else:
                characters.append(chr(tokens[index]))
                index += 1
        return "".join(characters)

    items = prepared.query_items if is_query else prepared.target_items
    gap_operation = "I" if is_query else "D"
    output_items: list[Any] = []
    index = start
    for operation in operations:
        if operation == gap_operation:
            output_items.append(None)
        else:
            output_items.append(items[index])
            index += 1
    return tuple(output_items)


def smith_waterman_score(
    query: object,
    target: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> int:
    gap_open, gap_extend = _resolve_gap_scores(gap_score, gap_open_score, gap_extend_score)
    prepared = _prepare_alignment(
        query,
        target,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_open,
        gap_extend_score=gap_extend,
        width=width,
    )
    if gap_open != gap_extend:
        return _affine_score_only(
            prepared.query_tokens,
            prepared.target_tokens,
            local_alignment=True,
            match_score=match_score,
            mismatch_score=mismatch_score,
            gap_open_score=gap_open,
            gap_extend_score=gap_extend,
        )
    return _score_only(
        prepared.query_tokens,
        prepared.target_tokens,
        local_alignment=True,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_open,
    )


def smith_waterman_path(
    query: object,
    target: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> AlignmentResult:
    gap_open, gap_extend = _resolve_gap_scores(gap_score, gap_open_score, gap_extend_score)
    prepared = _prepare_alignment(
        query,
        target,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_open,
        gap_extend_score=gap_extend,
        width=width,
    )
    if gap_open != gap_extend:
        return _affine_traceback(
            prepared,
            local_alignment=True,
            match_score=match_score,
            mismatch_score=mismatch_score,
            gap_open_score=gap_open,
            gap_extend_score=gap_extend,
        )
    return _traceback(
        prepared,
        local_alignment=True,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_open,
    )


def smith_waterman_path_info(
    query: object,
    target: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> AlignmentPath:
    return _alignment_path_from_result(
        smith_waterman_path(
            query,
            target,
            match_score=match_score,
            mismatch_score=mismatch_score,
            gap_score=gap_score,
            gap_open_score=gap_open_score,
            gap_extend_score=gap_extend_score,
            width=width,
        )
    )


def smith_waterman_cigar(
    query: object,
    target: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> str:
    return smith_waterman_path_info(
        query,
        target,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
        width=width,
    ).cigar


smith_waterman_trace_cigar = smith_waterman_cigar
smith_waterman_trade_cigar = smith_waterman_cigar


def smith_waterman_farrar_score(
    query: object,
    target: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> int:
    gap_open, gap_extend = _resolve_gap_scores(gap_score, gap_open_score, gap_extend_score)
    prepared = _prepare_farrar_alignment(
        query,
        target,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_open,
        gap_extend_score=gap_extend,
        width=width,
    )
    if gap_open != gap_extend:
        return _affine_score_only(
            prepared.query_tokens,
            prepared.target_tokens,
            local_alignment=True,
            match_score=match_score,
            mismatch_score=mismatch_score,
            gap_open_score=gap_open,
            gap_extend_score=gap_extend,
        )
    return _score_only(
        prepared.query_tokens,
        prepared.target_tokens,
        local_alignment=True,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_open,
    )


def needleman_wunsch_score(
    query: object,
    target: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> int:
    gap_open, gap_extend = _resolve_gap_scores(gap_score, gap_open_score, gap_extend_score)
    prepared = _prepare_alignment(
        query,
        target,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_open,
        gap_extend_score=gap_extend,
        width=width,
    )
    if gap_open != gap_extend:
        return _affine_score_only(
            prepared.query_tokens,
            prepared.target_tokens,
            local_alignment=False,
            match_score=match_score,
            mismatch_score=mismatch_score,
            gap_open_score=gap_open,
            gap_extend_score=gap_extend,
        )
    return _score_only(
        prepared.query_tokens,
        prepared.target_tokens,
        local_alignment=False,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_open,
    )


def needleman_wunsch_path(
    query: object,
    target: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> AlignmentResult:
    gap_open, gap_extend = _resolve_gap_scores(gap_score, gap_open_score, gap_extend_score)
    prepared = _prepare_alignment(
        query,
        target,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_open,
        gap_extend_score=gap_extend,
        width=width,
    )
    if gap_open != gap_extend:
        return _affine_traceback(
            prepared,
            local_alignment=False,
            match_score=match_score,
            mismatch_score=mismatch_score,
            gap_open_score=gap_open,
            gap_extend_score=gap_extend,
        )
    return _traceback(
        prepared,
        local_alignment=False,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_open,
    )


def needleman_wunsch_path_info(
    query: object,
    target: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> AlignmentPath:
    return _alignment_path_from_result(
        needleman_wunsch_path(
            query,
            target,
            match_score=match_score,
            mismatch_score=mismatch_score,
            gap_score=gap_score,
            gap_open_score=gap_open_score,
            gap_extend_score=gap_extend_score,
            width=width,
        )
    )


def needleman_wunsch_cigar(
    query: object,
    target: object,
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    gap_open_score: int | None = None,
    gap_extend_score: int | None = None,
    width: int | None = None,
) -> str:
    return needleman_wunsch_path_info(
        query,
        target,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
        gap_open_score=gap_open_score,
        gap_extend_score=gap_extend_score,
        width=width,
    ).cigar


needleman_wunsch_trace_cigar = needleman_wunsch_cigar
needleman_wunsch_trade_cigar = needleman_wunsch_cigar
