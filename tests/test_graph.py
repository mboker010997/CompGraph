import pathlib
import json
import typing as tp

from compgraph import Graph


class _Key:
    def __init__(self, *args: str) -> None:
        self._items = args

    def __call__(self, d: tp.Mapping[str, tp.Any]) -> tuple[str, ...]:
        return tuple(str(d.get(key)) for key in self._items)


def test_graph_from_file(tmp_path: pathlib.Path) -> None:
    table = [
        {"key": "aboba", "value": 18},
        {"key": "mboker", "value": 23},
    ]

    dir = tmp_path / 'resources'
    dir.mkdir()
    input_file = dir / "input.txt"

    input_file.write_text('\n'.join([json.dumps(row) for row in table]))

    graph = Graph.graph_from_file(str(input_file), lambda line: json.loads(line))

    actual = []
    for row in graph.run():
        actual.append(row)

    key_func = _Key("key", "value")
    assert sorted(actual, key=key_func) == sorted(table, key=key_func)
