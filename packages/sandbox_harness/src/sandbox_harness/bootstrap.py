from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PythonRepoTaskSpec:
    task_id: str
    prompt: str
    starter_code: str
    visible_tests: str
    hidden_tests: str
    source_path: str = "solution.py"
    metadata: dict[str, str] = field(default_factory=dict)


def python_repo_initial_files(spec: PythonRepoTaskSpec) -> dict[str, str]:
    return {
        "README.md": f"# Task\n\n{spec.prompt.strip()}\n",
        spec.source_path: spec.starter_code,
        "tests/test_visible.py": spec.visible_tests,
        "hidden_tests/test_hidden.py": spec.hidden_tests,
    }


def toy_addition_task() -> PythonRepoTaskSpec:
    return PythonRepoTaskSpec(
        task_id="toy-addition",
        prompt="Implement add(a, b) in solution.py.",
        starter_code="def add(a: int, b: int) -> int:\n    raise NotImplementedError\n",
        visible_tests=(
            "from solution import add\n\n"
            "def test_add_positive_numbers():\n"
            "    assert add(2, 3) == 5\n"
        ),
        hidden_tests=(
            "from solution import add\n\n"
            "def test_add_negative_numbers():\n"
            "    assert add(-4, 3) == -1\n"
        ),
        metadata={"source": "toy"},
    )
