"""Property-generated adversarial inputs for final correctness checks.

The public numeric stress suite scales whole tensors by fixed constants. That
is useful, but it makes the final distributions easy to recognize. This module
uses Hypothesis to vary structural properties which a correct kernel must
preserve: long KDA history, ragged MoE routing, concentrated top-k values,
short paged-attention sequences, and input storage after graph warmup.

Strategies generate compact mutation specifications rather than tensors.
Tensor allocation remains under the checker so examples stay on the intended
device and do not enter Hypothesis' example database.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import torch
from hypothesis import HealthCheck, Phase, example, given, settings
from hypothesis import seed as hypothesis_seed
from hypothesis import strategies as st

from src.eval.correctness import check_correctness


@dataclass(frozen=True)
class KDALongMemoryCase:
    split_chunks: int
    key_millivalue: int
    query_scale_percent: int
    value_scale_percent: int
    decay_micros: int
    beta_percent: int


@dataclass(frozen=True)
class SonicRaggedMixedCase:
    receiver: int
    donor: int
    row_delta: int
    prefix_rows: int
    suffix_scale: int


@dataclass(frozen=True)
class TopKClusterCase:
    partition: int
    offset: int
    extra_values: int
    baseline_quarters: int


@dataclass(frozen=True)
class PagedShortSequenceCase:
    divisor: int
    tail: int
    stride: int


PropertyCase: TypeAlias = (
    KDALongMemoryCase | SonicRaggedMixedCase | TopKClusterCase | PagedShortSequenceCase
)

_CANONICAL_CASES: dict[str, PropertyCase] = {
    "02_kda_cutlass": KDALongMemoryCase(
        split_chunks=8,
        key_millivalue=40,
        query_scale_percent=800,
        value_scale_percent=800,
        decay_micros=3_000,
        beta_percent=80,
    ),
    "03_paged_attention": PagedShortSequenceCase(divisor=16, tail=1, stride=3),
    "05_topk_bitonic": TopKClusterCase(
        partition=3,
        offset=17,
        extra_values=8,
        baseline_quarters=4,
    ),
    "06_sonic_moe_swiglu": SonicRaggedMixedCase(
        receiver=0,
        donor=1,
        row_delta=129,
        prefix_rows=64,
        suffix_scale=8,
    ),
}

_PROPERTY_SHAPES = {
    "02_kda_cutlass": 0,
    "03_paged_attention": 0,
    "05_topk_bitonic": 0,
    "06_sonic_moe_swiglu": 1,
}

_GENERATED_EXAMPLES = {
    # Hypothesis deliberately tries one simplest generated example before its
    # seeded exploration. Keep two KDA examples so the low-cost plan contains
    # at least one seed-dependent case rather than the same minimum every run.
    "02_kda_cutlass": 2,
    "03_paged_attention": 2,
    "05_topk_bitonic": 3,
    "06_sonic_moe_swiglu": 2,
}

_PROPERTY_SEED_ENV = "KBH_PROPERTY_SEED"
_MAX_PROPERTY_SEED = (1 << 64) - 1


def property_shape_index(problem_name: str) -> int | None:
    """Return the canonical shape index used for bounded property checks."""
    return _PROPERTY_SHAPES.get(problem_name)


def tolerance_for_property(problem_name: str, base: dict | None) -> dict | None:
    """Return a tolerance calibrated for the generated value regime."""
    if problem_name == "06_sonic_moe_swiglu":
        merged = dict(base or {})
        merged["bfloat16"] = {"atol": 1e-1, "rtol": 5e-2}
        return merged
    return base


def canonical_property_case(problem_name: str) -> PropertyCase:
    """Return the fixed regression example paired with generated examples."""
    try:
        return _CANONICAL_CASES[problem_name]
    except KeyError as exc:
        raise ValueError(f"no property strategy for {problem_name!r}") from exc


def property_case_strategy(problem_name: str) -> st.SearchStrategy[PropertyCase]:
    """Build the Hypothesis strategy for one problem's semantic invariants."""
    if problem_name == "02_kda_cutlass":
        return st.builds(
            KDALongMemoryCase,
            split_chunks=st.integers(min_value=6, max_value=8),
            key_millivalue=st.integers(min_value=35, max_value=50),
            query_scale_percent=st.integers(min_value=800, max_value=1_000),
            value_scale_percent=st.integers(min_value=800, max_value=1_000),
            decay_micros=st.integers(min_value=2_000, max_value=4_000),
            beta_percent=st.integers(min_value=75, max_value=85),
        )
    if problem_name == "03_paged_attention":
        return st.builds(
            PagedShortSequenceCase,
            divisor=st.integers(min_value=4, max_value=32),
            tail=st.integers(min_value=1, max_value=15),
            stride=st.integers(min_value=1, max_value=7),
        )
    if problem_name == "05_topk_bitonic":
        return st.builds(
            TopKClusterCase,
            partition=st.integers(min_value=0, max_value=127),
            offset=st.integers(min_value=0, max_value=127),
            extra_values=st.integers(min_value=1, max_value=32),
            baseline_quarters=st.integers(min_value=2, max_value=6),
        )
    if problem_name == "06_sonic_moe_swiglu":
        return st.builds(
            SonicRaggedMixedCase,
            receiver=st.integers(min_value=0, max_value=127),
            donor=st.integers(min_value=0, max_value=127),
            row_delta=st.sampled_from((1, 7, 63, 127, 129, 257)),
            prefix_rows=st.sampled_from((1, 16, 32, 64, 127)),
            suffix_scale=st.integers(min_value=4, max_value=12),
        ).filter(lambda case: case.receiver != case.donor)
    raise ValueError(f"no property strategy for {problem_name!r}")


def run_property_cases(
    problem_name: str,
    check: Callable[[PropertyCase], None],
    *,
    seed: int | None = None,
    max_examples: int | None = None,
) -> int:
    """Run one fixed regression and generated examples through a callback.

    The random seed is printed before any candidate code runs. A failed log is
    therefore reproducible, while successive official checks do not expose one
    permanent set of values that a submission can memorize.
    """
    actual_seed = _property_seed(seed)
    example_count = _GENERATED_EXAMPLES[problem_name] if max_examples is None else max_examples
    print(f"PROPERTY_SEED: {actual_seed}", flush=True)
    strategy = property_case_strategy(problem_name)
    canonical = canonical_property_case(problem_name)

    @hypothesis_seed(actual_seed)
    @settings(
        max_examples=example_count,
        deadline=None,
        database=None,
        phases=(Phase.explicit, Phase.generate),
        suppress_health_check=(HealthCheck.filter_too_much, HealthCheck.too_slow),
    )
    @example(case=canonical)
    @given(case=strategy)
    def property_check(case: PropertyCase) -> None:
        check(case)

    property_check()
    return actual_seed


def generate_property_cases(
    problem_name: str,
    *,
    seed: int | None = None,
    max_examples: int | None = None,
) -> tuple[int, tuple[PropertyCase, ...]]:
    """Generate a frozen plan before importing untrusted candidate code."""
    cases: list[PropertyCase] = []
    actual_seed = run_property_cases(
        problem_name,
        cases.append,
        seed=seed,
        max_examples=max_examples,
    )
    return actual_seed, tuple(cases)


def apply_property_case(
    problem_name: str,
    inputs: Sequence[object],
    case: PropertyCase,
) -> list[object]:
    """Clone and mutate inputs according to a generated property case."""
    if problem_name == "02_kda_cutlass" and isinstance(case, KDALongMemoryCase):
        return _apply_kda(inputs, case)
    if problem_name == "03_paged_attention" and isinstance(case, PagedShortSequenceCase):
        return _apply_paged_attention(inputs, case)
    if problem_name == "05_topk_bitonic" and isinstance(case, TopKClusterCase):
        return _apply_topk(inputs, case)
    if problem_name == "06_sonic_moe_swiglu" and isinstance(case, SonicRaggedMixedCase):
        return _apply_sonic(inputs, case)
    raise TypeError(f"case {case!r} does not belong to {problem_name!r}")


def prime_replay_state(
    model: torch.nn.Module,
    inputs: Sequence[object],
    *,
    repeats: int = 11,
) -> None:
    """Cross the benchmark's ten warmups before a different-storage call."""
    with torch.no_grad():
        for _ in range(repeats):
            model(*inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def check_tensor_properties(
    problem_name: str,
    reference_model: torch.nn.Module,
    solution_model: torch.nn.Module,
    base_inputs: Sequence[object],
    *,
    tolerance: dict | None,
    cases: Sequence[PropertyCase] | None = None,
) -> int:
    """Compare tensor outputs for generated structural cases."""

    # Candidate warmup runs before the same-storage probe so CUDA graphs and
    # pointer-keyed paths are hot. Preserve a trusted copy first: a candidate
    # must not be able to poison the inputs during warmup and thereby choose
    # the values used to build both the mutation and the reference oracle.
    pristine_inputs = _clone_inputs(base_inputs)
    pristine_metadata = _input_metadata(base_inputs)

    def verify(
        case: PropertyCase,
        reference_inputs: Sequence[object],
        solution_inputs: Sequence[object],
    ) -> None:
        with torch.no_grad():
            reference_out = reference_model(*reference_inputs)
            solution_out = solution_model(*solution_inputs)
        if type(reference_out) is not torch.Tensor:
            raise AssertionError(
                f"property case {case!r}: reference returned "
                f"{type(reference_out)!r}, expected torch.Tensor"
            )
        _validate_solution_tensor(case, reference_out, solution_out, "output")
        ok, message = check_correctness(
            reference_out,
            solution_out,
            dtype=reference_out.dtype,
            override=tolerance,
        )
        if not ok:
            raise AssertionError(f"property case {case!r}: {message}")

    def check(case: PropertyCase) -> None:
        inputs = apply_property_case(problem_name, pristine_inputs, case)
        verify(case, _clone_inputs(inputs), _clone_inputs(inputs))

    # Exercise the exact storage that a version-aware output cache just saw.
    # Mutating through a detached storage alias deliberately preserves both the
    # data pointer and Tensor._version, so a real kernel recomputes while a
    # cached result cannot hide behind PyTorch's normal in-place invalidation.
    prime_replay_state(solution_model, base_inputs)
    _assert_input_metadata_unchanged(base_inputs, pristine_metadata)
    _overwrite_inputs_without_version(base_inputs, pristine_inputs)
    canonical = canonical_property_case(problem_name)
    same_storage_inputs = list(base_inputs)
    mutated = apply_property_case(problem_name, pristine_inputs, canonical)
    _overwrite_inputs_without_version(same_storage_inputs, mutated)
    try:
        verify(canonical, _clone_inputs(mutated), same_storage_inputs)
    finally:
        # Generated cases must start from the nominal distribution, not from a
        # canonical mutation compounded a second time.
        _overwrite_inputs_without_version(same_storage_inputs, pristine_inputs)

    if cases is None:
        return run_property_cases(problem_name, check)
    for case in cases:
        check(case)
    return 0


def check_topk_properties(
    reference_model: torch.nn.Module,
    solution_model: torch.nn.Module,
    base_inputs: Sequence[object],
    *,
    k: int,
    tolerance: dict | None,
    cases: Sequence[PropertyCase] | None = None,
) -> int:
    """Check clustered values after warming any pointer-specific replay state."""
    pristine_inputs = _clone_inputs(base_inputs)
    pristine_metadata = _input_metadata(base_inputs)
    prime_replay_state(solution_model, base_inputs)
    _assert_input_metadata_unchanged(base_inputs, pristine_metadata)
    _overwrite_inputs_without_version(base_inputs, pristine_inputs)

    def verify(
        case: PropertyCase,
        reference_inputs: Sequence[object],
        solution_inputs: Sequence[object],
    ) -> None:
        x = _tensor(reference_inputs[0], 0)
        with torch.no_grad():
            reference_values, reference_indices = reference_model(*reference_inputs)
            solution_out = solution_model(*solution_inputs)
        if not (isinstance(solution_out, (tuple, list)) and len(solution_out) == 2):
            raise AssertionError(f"property case {case!r}: solution must return (values, indices)")
        solution_values, solution_indices = solution_out
        if (
            type(reference_values) is not torch.Tensor
            or type(reference_indices) is not torch.Tensor
        ):
            raise AssertionError(
                f"property case {case!r}: reference must return plain tensor values and indices"
            )
        _validate_solution_tensor(case, reference_values, solution_values, "values")
        _validate_solution_tensor(case, reference_indices, solution_indices, "indices")
        expected_shape = (x.shape[0], k)
        if tuple(solution_values.shape) != expected_shape:
            raise AssertionError(
                f"property case {case!r}: values shape {tuple(solution_values.shape)} "
                f"!= {expected_shape}"
            )
        if tuple(solution_indices.shape) != expected_shape:
            raise AssertionError(
                f"property case {case!r}: indices shape {tuple(solution_indices.shape)} "
                f"!= {expected_shape}"
            )
        indices = solution_indices.to(torch.int64)
        if indices.numel() and (indices.min() < 0 or indices.max() >= x.shape[-1]):
            raise AssertionError(f"property case {case!r}: indices are out of range")
        for row in indices:
            if torch.unique(row).numel() != k:
                raise AssertionError(f"property case {case!r}: duplicate top-k indices")

        ok, message = check_correctness(
            reference_values.float(),
            solution_values.float(),
            dtype=torch.float32,
            override=tolerance,
        )
        if not ok:
            raise AssertionError(f"property case {case!r} values: {message}")
        gathered = torch.gather(x, dim=-1, index=indices)
        ok, message = check_correctness(
            reference_values.float(),
            gathered.float(),
            dtype=torch.float32,
            override=tolerance,
        )
        if not ok:
            raise AssertionError(f"property case {case!r} indices: {message}")

    def check(case: PropertyCase) -> None:
        inputs = apply_property_case("05_topk_bitonic", pristine_inputs, case)
        verify(case, _clone_inputs(inputs), _clone_inputs(inputs))

    # First preserve the warmed pointer and version while changing its contents
    # in place. Then generated checks use fresh storage. Real CUDA-graph replay
    # recomputes; pointer/version-keyed output memoization does not.
    canonical = canonical_property_case("05_topk_bitonic")
    same_storage_inputs = list(base_inputs)
    mutated = apply_property_case("05_topk_bitonic", pristine_inputs, canonical)
    _overwrite_inputs_without_version(same_storage_inputs, mutated)
    try:
        verify(canonical, _clone_inputs(mutated), same_storage_inputs)
    finally:
        _overwrite_inputs_without_version(same_storage_inputs, pristine_inputs)

    if cases is None:
        return run_property_cases("05_topk_bitonic", check)
    for case in cases:
        check(case)
    return 0


def _tensor(value: object, index: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"input {index} is not a tensor")
    return value


def _property_seed(seed: int | None) -> int:
    """Resolve an explicit/replay seed, otherwise draw a fresh 64-bit seed."""
    if seed is not None:
        actual_seed = seed
    else:
        configured = os.environ.get(_PROPERTY_SEED_ENV)
        if configured is None or not configured.strip():
            return secrets.randbits(64)
        try:
            actual_seed = int(configured, 0)
        except ValueError as exc:
            raise ValueError(
                f"{_PROPERTY_SEED_ENV} must be a decimal or 0x-prefixed integer"
            ) from exc
    if not isinstance(actual_seed, int) or isinstance(actual_seed, bool):
        raise TypeError("property seed must be an integer")
    if not 0 <= actual_seed <= _MAX_PROPERTY_SEED:
        raise ValueError(
            f"property seed must be between 0 and {_MAX_PROPERTY_SEED}, got {actual_seed}"
        )
    return actual_seed


def _clone_inputs(inputs: Sequence[object]) -> list[object]:
    return [value.clone() if isinstance(value, torch.Tensor) else value for value in inputs]


def _input_metadata(inputs: Sequence[object]) -> tuple[tuple[object, ...], ...]:
    """Snapshot candidate-visible tensor identity and layout information."""
    metadata = []
    for value in inputs:
        if not isinstance(value, torch.Tensor):
            metadata.append((type(value), id(value)))
            continue
        storage = value.untyped_storage()
        metadata.append(
            (
                type(value),
                id(value),
                value.data_ptr(),
                storage.data_ptr(),
                storage.nbytes(),
                value._version,
                value.storage_offset(),
                value.shape,
                value.stride(),
                value.dtype,
                value.device,
                value.layout,
            )
        )
    return tuple(metadata)


def _assert_input_metadata_unchanged(
    inputs: Sequence[object],
    expected: tuple[tuple[object, ...], ...],
) -> None:
    """Reject candidates that resize, restride, or replace warmed inputs."""
    actual = _input_metadata(inputs)
    if actual == expected:
        return
    for index, (before, after) in enumerate(zip(expected, actual, strict=False)):
        if before != after:
            raise AssertionError(
                f"candidate changed input {index} metadata during warmup: {before!r} -> {after!r}"
            )
    raise AssertionError(
        f"candidate changed the number of inputs during warmup: {len(expected)} -> {len(actual)}"
    )


def _validate_solution_tensor(
    case: PropertyCase,
    reference: torch.Tensor,
    solution: object,
    label: str,
) -> None:
    """Reject deferred Tensor subclasses and output contract mismatches."""
    if type(solution) is not torch.Tensor:
        raise AssertionError(
            f"property case {case!r}: solution {label} must be a plain torch.Tensor; "
            f"got {type(solution)!r}"
        )
    if solution.shape != reference.shape:
        raise AssertionError(
            f"property case {case!r}: solution {label} shape {tuple(solution.shape)} "
            f"!= {tuple(reference.shape)}"
        )
    if solution.dtype != reference.dtype:
        raise AssertionError(
            f"property case {case!r}: solution {label} dtype {solution.dtype} != {reference.dtype}"
        )
    if solution.device != reference.device:
        raise AssertionError(
            f"property case {case!r}: solution {label} device {solution.device} "
            f"!= {reference.device}"
        )


def _overwrite_inputs_without_version(
    destinations: Sequence[object],
    sources: Sequence[object],
) -> None:
    """Copy tensor bytes while preserving identity, storage, and version."""
    if len(destinations) != len(sources):
        raise ValueError(f"cannot overwrite {len(destinations)} inputs with {len(sources)} values")
    for index, (destination, source) in enumerate(zip(destinations, sources, strict=True)):
        if not isinstance(destination, torch.Tensor) or not isinstance(source, torch.Tensor):
            if destination != source:
                raise TypeError(f"input {index} is not a matching tensor value")
            continue
        before = (
            id(destination),
            destination.data_ptr(),
            destination._version,
            destination.storage_offset(),
            destination.shape,
            destination.stride(),
            destination.dtype,
            destination.device,
        )
        if destination is not source:
            # `.data` is intentional here: its alias has an independent version
            # counter, unlike destination.copy_(...), which increments the
            # candidate-visible destination._version and lets memoizers evade.
            storage_alias = destination.data
            with torch.no_grad():
                storage_alias.copy_(source)
        after = (
            id(destination),
            destination.data_ptr(),
            destination._version,
            destination.storage_offset(),
            destination.shape,
            destination.stride(),
            destination.dtype,
            destination.device,
        )
        if after != before:
            raise AssertionError(
                f"same-storage property mutation changed input {index} metadata: "
                f"{before!r} -> {after!r}"
            )


def _apply_kda(inputs: Sequence[object], case: KDALongMemoryCase) -> list[object]:
    if len(inputs) != 5:
        raise ValueError(f"KDA expects five inputs, got {len(inputs)}")
    q, k, v, g, beta = (_tensor(value, i).clone() for i, value in enumerate(inputs))
    chunk_size = 64
    chunks = q.shape[1] // chunk_size
    split_chunks = min(max(1, case.split_chunks), max(1, chunks - 1))
    split = split_chunks * chunk_size

    channel_sign = torch.where(
        torch.arange(k.shape[-1], device=k.device) % 2 == 0,
        1.0,
        -1.0,
    ).to(k.dtype)
    token_sign = torch.where(
        torch.arange(k.shape[1], device=k.device) // chunk_size % 2 == 0,
        1.0,
        -1.0,
    ).to(k.dtype)
    structured_k = token_sign[None, :, None, None] * channel_sign[None, None, None, :]
    k.copy_(structured_k * (case.key_millivalue / 1_000.0))

    q[:, :split].zero_()
    q[:, split:].mul_(case.query_scale_percent / 100.0)
    v[:, :split].mul_(case.value_scale_percent / 100.0)
    v[:, split:].zero_()
    # Keep the first-token probe nominal while later tokens carry long memory.
    k[:, :1].mul_(0.05)
    g.fill_(-case.decay_micros / 1_000_000.0)
    beta.fill_(case.beta_percent / 100.0)
    return [q, k, v, g, beta]


def _apply_sonic(inputs: Sequence[object], case: SonicRaggedMixedCase) -> list[object]:
    if len(inputs) != 2:
        raise ValueError(f"Sonic MoE expects two inputs, got {len(inputs)}")
    hidden = _tensor(inputs[0], 0).clone()
    offsets = _tensor(inputs[1], 1)
    counts = (offsets[1:] - offsets[:-1]).clone()
    experts = counts.numel()
    receiver = case.receiver % experts
    donor = case.donor % experts
    if receiver == donor:
        donor = (donor + 1) % experts
    # Pull from consecutive donors until the requested skew is reached while
    # keeping every donor nonempty. On the bounded property shape every expert
    # starts with 256 rows; the canonical 129-row transfer makes the receiver
    # cross an average-plus-one-128-row-block launch cap (384 -> 385). Pooling
    # also lets the generated 257-row case cross two block boundaries.
    remaining = min(case.row_delta, int(counts.sum().item()) - experts)
    moved = 0
    for step in range(experts):
        current_donor = (donor + step) % experts
        if current_donor == receiver:
            continue
        available = max(0, int(counts[current_donor].item()) - 1)
        delta = min(remaining, available)
        if delta:
            counts[current_donor] -= delta
            moved += delta
            remaining -= delta
        if not remaining:
            break
    if moved:
        counts[receiver] += moved
    ragged_offsets = torch.zeros_like(offsets)
    ragged_offsets[1:] = torch.cumsum(counts, dim=0)

    probe_rows = (65_536 + hidden.shape[1] - 1) // hidden.shape[1]
    prefix = min(max(64, probe_rows, case.prefix_rows), hidden.shape[0] - 1)
    hidden[prefix:].mul_(case.suffix_scale)
    return [hidden, ragged_offsets]


def _apply_topk(inputs: Sequence[object], case: TopKClusterCase) -> list[object]:
    if len(inputs) != 1:
        raise ValueError(f"Top-k expects one input, got {len(inputs)}")
    x = _tensor(inputs[0], 0).clone()
    _, n = x.shape
    k = min(64, n)
    spike_count = min(n, k + case.extra_values)
    baseline = case.baseline_quarters / 4.0
    x.fill_(baseline)

    partition_size = min(1024, n)
    partitions = max(1, n // partition_size)
    start = (case.partition % partitions) * partition_size
    room = max(1, partition_size - spike_count + 1)
    start += case.offset % room
    spikes = torch.linspace(
        baseline + 1.0,
        baseline + 0.25,
        spike_count,
        dtype=x.dtype,
        device=x.device,
    )
    x[:, start : start + spike_count] = spikes
    return [x]


def _apply_paged_attention(
    inputs: Sequence[object],
    case: PagedShortSequenceCase,
) -> list[object]:
    if len(inputs) != 4:
        raise ValueError(f"paged attention expects four inputs, got {len(inputs)}")
    out = list(inputs)
    query = _tensor(inputs[0], 0).clone()
    kv_cache = _tensor(inputs[1], 1).clone()
    block_table = _tensor(inputs[2], 2)
    seq_lens = _tensor(inputs[3], 3).clone()
    full_length = int(seq_lens.max().item())
    for batch_idx in range(seq_lens.numel()):
        divisor = case.divisor + batch_idx * case.stride
        tail = ((case.tail - 1 + batch_idx * case.stride) % 15) + 1
        length = max(1, full_length // divisor)
        length = max(1, min(full_length, length - (length % 16) + tail))
        seq_lens[batch_idx] = length
    query.zero_()
    head_dim = kv_cache.shape[-1] // 2
    kv_cache[..., :head_dim].zero_()
    kv_cache[..., head_dim:].fill_(-1)
    page_size = kv_cache.shape[1]
    for batch_idx, length_tensor in enumerate(seq_lens):
        length = int(length_tensor.item())
        page_count = (length + page_size - 1) // page_size
        for page_idx, page_tensor in enumerate(block_table[batch_idx, :page_count]):
            valid = min(page_size, length - page_idx * page_size)
            kv_cache[int(page_tensor.item()), :valid, :, head_dim:].fill_(1)
    out[0] = query
    out[1] = kv_cache
    out[3] = seq_lens
    return out
