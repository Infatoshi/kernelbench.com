import runpy
from inspect import signature
from pathlib import Path

import pytest
import torch
from hypothesis import given, settings

from src.eval.property_stress import (
    KDALongMemoryCase,
    PagedShortSequenceCase,
    SonicRaggedMixedCase,
    TopKClusterCase,
    apply_property_case,
    canonical_property_case,
    check_tensor_properties,
    check_topk_properties,
    generate_property_cases,
    prime_replay_state,
    property_case_strategy,
    property_shape_index,
    run_property_cases,
)
from src.eval.timing import time_variant

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
PROPERTY_PROBLEMS = (
    "02_kda_cutlass",
    "03_paged_attention",
    "05_topk_bitonic",
    "06_sonic_moe_swiglu",
)


@settings(max_examples=20, deadline=None)
@given(case=property_case_strategy("02_kda_cutlass"))
def test_kda_properties_preserve_shapes_and_create_long_memory(case) -> None:
    q = torch.ones(1, 256, 1, 8)
    k = torch.ones_like(q)
    v = torch.ones_like(q)
    g = torch.zeros_like(q)
    beta = torch.zeros(1, 256, 1)

    mutated = apply_property_case("02_kda_cutlass", [q, k, v, g, beta], case)

    assert all(new.shape == old.shape for new, old in zip(mutated, [q, k, v, g, beta], strict=True))
    assert torch.all(mutated[3] < 0)
    assert torch.all(mutated[4] >= 0.75)
    assert mutated[1][0, 0].abs().mean() < k[0, 0].abs().mean()


def test_kda_canonical_case_exercises_high_order_recurrence_terms() -> None:
    case = canonical_property_case("02_kda_cutlass")

    assert isinstance(case, KDALongMemoryCase)
    assert case.split_chunks >= 8
    assert case.key_millivalue >= 40


@settings(max_examples=20, deadline=None)
@given(case=property_case_strategy("06_sonic_moe_swiglu"))
def test_sonic_properties_keep_total_rows_but_make_routing_ragged(case) -> None:
    hidden = torch.ones(1024, 16)
    offsets = torch.arange(0, 1025, 128, dtype=torch.int32)

    mutated_hidden, mutated_offsets = apply_property_case(
        "06_sonic_moe_swiglu", [hidden, offsets], case
    )
    counts = mutated_offsets[1:] - mutated_offsets[:-1]

    assert int(mutated_offsets[-1]) == hidden.shape[0]
    assert torch.all(counts > 0)
    assert torch.unique(counts).numel() > 1
    assert mutated_hidden.shape == hidden.shape
    assert mutated_hidden[-1].abs().mean() > mutated_hidden[0].abs().mean()


def test_sonic_canonical_case_crosses_one_block_average_capacity() -> None:
    hidden, offsets, average_rows = _sonic_property_inputs()

    _, mutated_offsets = apply_property_case(
        "06_sonic_moe_swiglu",
        [hidden, offsets],
        canonical_property_case("06_sonic_moe_swiglu"),
    )
    counts = mutated_offsets[1:] - mutated_offsets[:-1]

    assert average_rows == 256
    assert int(counts.max()) == average_rows + 129
    assert torch.all(counts > 0)


def test_sonic_property_guard_rejects_average_plus_one_block_truncation() -> None:
    class Reference(torch.nn.Module):
        def forward(self, hidden, _offsets):
            return hidden.clone()

    class Truncated(torch.nn.Module):
        def forward(self, hidden, offsets):
            out = torch.zeros_like(hidden)
            experts = offsets.numel() - 1
            average_rows = hidden.shape[0] // experts
            launch_capacity = average_rows + 128
            for expert in range(experts):
                start = int(offsets[expert])
                end = min(int(offsets[expert + 1]), start + launch_capacity)
                out[start:end] = hidden[start:end]
            return out

    hidden, offsets, _ = _sonic_property_inputs()
    with pytest.raises(AssertionError, match="property case"):
        check_tensor_properties(
            "06_sonic_moe_swiglu",
            Reference(),
            Truncated(),
            [hidden, offsets],
            tolerance={"float32": 1e-4},
            cases=(),
        )


def test_sonic_property_guard_accepts_complete_implementation() -> None:
    class Reference(torch.nn.Module):
        def forward(self, hidden, _offsets):
            return hidden.clone()

    hidden, offsets, _ = _sonic_property_inputs()
    assert (
        check_tensor_properties(
            "06_sonic_moe_swiglu",
            Reference(),
            Reference(),
            [hidden, offsets],
            tolerance={"float32": 1e-4},
            cases=(),
        )
        == 0
    )


@settings(max_examples=20, deadline=None)
@given(case=property_case_strategy("05_topk_bitonic"))
def test_topk_properties_concentrate_more_than_k_values_in_one_partition(case) -> None:
    x = torch.randn(2, 4096)
    (mutated,) = apply_property_case("05_topk_bitonic", [x], case)
    top_values = torch.topk(mutated, k=64, dim=-1).values

    assert mutated.shape == x.shape
    assert torch.all(top_values[:, :-1] >= top_values[:, 1:])
    assert torch.all(top_values[:, -1] > mutated.amin(dim=-1))


@settings(max_examples=20, deadline=None)
@given(case=property_case_strategy("03_paged_attention"))
def test_paged_properties_generate_valid_nonuniform_short_lengths(case) -> None:
    query = torch.empty(4, 8, 16)
    cache = torch.empty(64, 16, 2, 32)
    table = torch.arange(64, dtype=torch.int32).reshape(4, 16)
    lengths = torch.full((4,), 256, dtype=torch.int32)

    *_, mutated_lengths = apply_property_case(
        "03_paged_attention", [query, cache, table, lengths], case
    )

    assert torch.all(mutated_lengths > 0)
    assert torch.all(mutated_lengths <= lengths)
    assert torch.unique(mutated_lengths).numel() > 1
    assert torch.all(mutated_lengths % 16 != 0)


def test_property_runner_includes_fixed_and_generated_examples() -> None:
    seen: list[TopKClusterCase] = []
    seed = run_property_cases(
        "05_topk_bitonic",
        lambda case: seen.append(case),
        seed=12345,
        max_examples=3,
    )

    assert seed == 12345
    assert len(seen) >= 4
    assert any(case.partition == 3 and case.offset == 17 for case in seen)


def test_kda_generated_plan_changes_with_seed() -> None:
    _, first = generate_property_cases("02_kda_cutlass", seed=1)
    _, second = generate_property_cases("02_kda_cutlass", seed=2)

    assert first[0] == canonical_property_case("02_kda_cutlass")
    assert second[0] == canonical_property_case("02_kda_cutlass")
    assert first[1:] != second[1:]


def test_property_seed_environment_replays_the_generated_plan(monkeypatch) -> None:
    logged_seed = 14_457_558_819_567_596_466
    monkeypatch.setenv("KBH_PROPERTY_SEED", str(logged_seed))

    actual_seed, replayed = generate_property_cases("02_kda_cutlass")
    _, explicit = generate_property_cases("02_kda_cutlass", seed=logged_seed)

    assert actual_seed == logged_seed
    assert replayed == explicit


def test_explicit_property_seed_overrides_environment(monkeypatch) -> None:
    monkeypatch.setenv("KBH_PROPERTY_SEED", "123")

    actual_seed, _ = generate_property_cases("03_paged_attention", seed=456)

    assert actual_seed == 456


@pytest.mark.parametrize("value", ["not-an-int", "-1", str(1 << 64)])
def test_invalid_property_seed_environment_is_rejected(monkeypatch, value) -> None:
    monkeypatch.setenv("KBH_PROPERTY_SEED", value)

    with pytest.raises((TypeError, ValueError), match="property seed|KBH_PROPERTY_SEED"):
        generate_property_cases("03_paged_attention")


def test_topk_replay_guard_rejects_cached_output_on_same_storage() -> None:
    class Reference(torch.nn.Module):
        def forward(self, x):
            return torch.topk(x, k=64, dim=-1)

    class Cached(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cached = None

        def forward(self, x):
            if self.cached is None:
                self.cached = torch.topk(x, k=64, dim=-1)
            return self.cached

    case = canonical_property_case("05_topk_bitonic")
    with pytest.raises(AssertionError, match="property case"):
        check_topk_properties(
            Reference(),
            Cached(),
            [torch.randn(1, 128)],
            k=64,
            tolerance={"float32": 1e-4},
            cases=(case,),
        )


def test_topk_replay_guard_rejects_version_keyed_cache() -> None:
    class Reference(torch.nn.Module):
        def forward(self, x):
            return torch.topk(x, k=64, dim=-1)

    class VersionCached(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.key = None
            self.cached = None

        def forward(self, x):
            key = (x.data_ptr(), x._version)
            if key != self.key:
                self.key = key
                self.cached = torch.topk(x, k=64, dim=-1)
            return self.cached

    case = canonical_property_case("05_topk_bitonic")
    with pytest.raises(AssertionError, match="property case"):
        check_topk_properties(
            Reference(),
            VersionCached(),
            [torch.randn(1, 128)],
            k=64,
            tolerance={"float32": 1e-4},
            cases=(case,),
        )


def test_topk_replay_guard_crosses_benchmark_warmup_activation() -> None:
    benchmark_warmups = signature(time_variant).parameters["warmup"].default
    replay_calls = signature(prime_replay_state).parameters["repeats"].default
    assert replay_calls > benchmark_warmups

    class Reference(torch.nn.Module):
        def forward(self, x):
            return torch.topk(x, k=64, dim=-1)

    class LateCached(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.cached = None

        def forward(self, x):
            self.calls += 1
            current = torch.topk(x, k=64, dim=-1)
            if self.calls <= benchmark_warmups:
                return current
            if self.cached is None:
                self.cached = current
            return self.cached

    candidate = LateCached()
    with pytest.raises(AssertionError, match="property case"):
        check_topk_properties(
            Reference(),
            candidate,
            [torch.randn(1, 128)],
            k=64,
            tolerance={"float32": 1e-4},
            cases=(),
        )
    # The benchmark warmups plus one first timed-style call all see nominal
    # storage; the following mutated replay must expose the cache.
    assert candidate.calls == replay_calls + 1


def test_topk_replay_guard_accepts_recomputing_kernel() -> None:
    class Reference(torch.nn.Module):
        def forward(self, x):
            return torch.topk(x, k=64, dim=-1)

    case = canonical_property_case("05_topk_bitonic")
    inputs = [torch.randn(1, 128)]
    original = inputs[0].clone()
    assert (
        check_topk_properties(
            Reference(),
            Reference(),
            inputs,
            k=64,
            tolerance={"float32": 1e-4},
            cases=(case,),
        )
        == 0
    )
    assert torch.equal(inputs[0], original)


def test_topk_property_guard_rejects_warmup_metadata_mutation() -> None:
    class Reference(torch.nn.Module):
        def forward(self, x):
            return torch.topk(x, k=64, dim=-1)

    class Reshaping(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            result = torch.topk(x, k=64, dim=-1)
            self.calls += 1
            if self.calls == 11:
                # Keep enough columns for top-k so a checker that snapshots
                # shape only after warmup would accept the candidate's shape.
                x.resize_(2, 64)
            return result

    with pytest.raises(AssertionError, match="changed input 0 metadata"):
        check_topk_properties(
            Reference(),
            Reshaping(),
            [torch.randn(1, 128)],
            k=64,
            tolerance={"float32": 1e-4},
            cases=(),
        )


def test_tensor_replay_guard_rejects_version_keyed_cache() -> None:
    class Reference(torch.nn.Module):
        def forward(self, q, _k, v, _g, _beta):
            return q + v

    class VersionCached(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.key = None
            self.cached = None

        def forward(self, q, _k, v, _g, _beta):
            key = (q.data_ptr(), q._version, v.data_ptr(), v._version)
            if key != self.key:
                self.key = key
                self.cached = q + v
            return self.cached

    with pytest.raises(AssertionError, match="property case"):
        check_tensor_properties(
            "02_kda_cutlass",
            Reference(),
            VersionCached(),
            _small_kda_inputs(),
            tolerance={"float32": 1e-4},
            cases=(),
        )


def test_tensor_property_guard_rejects_warmup_input_mutation() -> None:
    class Reference(torch.nn.Module):
        def forward(self, q, _k, v, _g, _beta):
            return q + v

    class MutatingZero(torch.nn.Module):
        def forward(self, q, _k, v, _g, _beta):
            # Mutate through version-independent aliases so the metadata guard
            # cannot catch this; the pristine value snapshot still must.
            q.data.zero_()
            v.data.zero_()
            return torch.zeros_like(q)

    with pytest.raises(AssertionError, match="property case"):
        check_tensor_properties(
            "02_kda_cutlass",
            Reference(),
            MutatingZero(),
            _small_kda_inputs(),
            tolerance={"float32": 1e-4},
            cases=(canonical_property_case("02_kda_cutlass"),),
        )


def test_tensor_property_guard_rejects_warmup_metadata_mutation() -> None:
    class Reference(torch.nn.Module):
        def forward(self, q, _k, v, _g, _beta):
            return q + v

    class Transposing(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, q, _k, v, _g, _beta):
            result = q + v
            self.calls += 1
            if self.calls == 11:
                q.transpose_(1, 2)
            return result

    with pytest.raises(AssertionError, match="changed input 0 metadata"):
        check_tensor_properties(
            "02_kda_cutlass",
            Reference(),
            Transposing(),
            _small_kda_inputs(),
            tolerance={"float32": 1e-4},
            cases=(),
        )


def test_tensor_property_guard_accepts_recomputing_kernel() -> None:
    class Reference(torch.nn.Module):
        def forward(self, q, _k, v, _g, _beta):
            return q + v

    case = canonical_property_case("02_kda_cutlass")
    inputs = _small_kda_inputs()
    originals = [value.clone() for value in inputs]
    assert (
        check_tensor_properties(
            "02_kda_cutlass",
            Reference(),
            Reference(),
            inputs,
            tolerance={"float32": 1e-4},
            cases=(case,),
        )
        == 0
    )
    assert all(
        torch.equal(value, original) for value, original in zip(inputs, originals, strict=True)
    )


def test_tensor_property_guard_rejects_tensor_subclass_output() -> None:
    class DeferredTensor(torch.Tensor):
        pass

    class Reference(torch.nn.Module):
        def forward(self, q, _k, v, _g, _beta):
            return q + v

    class Deferred(torch.nn.Module):
        def forward(self, q, _k, v, _g, _beta):
            return (q + v).as_subclass(DeferredTensor)

    with pytest.raises(AssertionError, match="plain torch.Tensor"):
        check_tensor_properties(
            "02_kda_cutlass",
            Reference(),
            Deferred(),
            _small_kda_inputs(),
            tolerance={"float32": 1e-4},
            cases=(),
        )


def test_case_types_are_problem_specific() -> None:
    assert isinstance(_draw_cases("02_kda_cutlass")[0], KDALongMemoryCase)
    assert isinstance(_draw_cases("03_paged_attention")[0], PagedShortSequenceCase)
    assert isinstance(_draw_cases("05_topk_bitonic")[0], TopKClusterCase)
    assert isinstance(_draw_cases("06_sonic_moe_swiglu")[0], SonicRaggedMixedCase)


def test_all_gpu_decks_freeze_property_plan_before_importing_solution() -> None:
    for deck in ("problems-rtxpro6000", "problems-h100", "problems-b200"):
        for problem in PROPERTY_PROBLEMS:
            checker = (ROOT / deck / problem / "check.py").read_text()
            assert "generate_property_cases" in checker
            assert "property stress" in checker
            plan = checker.index("_, property_cases = generate_property_cases")
            assert plan < checker.index("import solution")


def test_all_property_checker_copies_are_identical() -> None:
    environment = REPO_ROOT / "environments" / "kernel_hard" / "_bench"
    canonical_helper = (ROOT / "src" / "eval" / "property_stress.py").read_bytes()
    assert (environment / "src" / "eval" / "property_stress.py").read_bytes() == canonical_helper

    for problem in PROPERTY_PROBLEMS:
        canonical_checker = (ROOT / "problems-rtxpro6000" / problem / "check.py").read_bytes()
        for deck in ("problems-h100", "problems-b200"):
            assert (ROOT / deck / problem / "check.py").read_bytes() == canonical_checker
        assert (environment / "problems" / problem / "check.py").read_bytes() == canonical_checker


def _draw_cases(problem_name: str):
    seen = []
    run_property_cases(problem_name, seen.append, seed=7, max_examples=1)
    return seen


def _small_kda_inputs() -> list[torch.Tensor]:
    q = torch.randn(1, 256, 1, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.randn_like(q)
    beta = torch.rand(1, 256, 1)
    return [q, k, v, g, beta]


def _sonic_property_inputs() -> tuple[torch.Tensor, torch.Tensor, int]:
    scope = runpy.run_path(ROOT / "problems-rtxpro6000" / "06_sonic_moe_swiglu" / "shapes.py")
    shape = scope["SHAPES"][property_shape_index("06_sonic_moe_swiglu")]
    total_rows = shape["T_total"] * shape["K"]
    average_rows, remainder = divmod(total_rows, shape["E"])
    assert remainder == 0
    hidden = torch.ones(total_rows, 1)
    offsets = torch.arange(
        0,
        total_rows + 1,
        average_rows,
        dtype=torch.int32,
    )
    return hidden, offsets, average_rows
