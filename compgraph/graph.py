import typing as tp

from . import operations as ops
from . import external_sort as ext_sort


class JoinGraph(ops.Join):
    def __init__(self, joiner: ops.Joiner, keys: tp.Sequence[str], other_graph: 'Graph'):
        super().__init__(joiner, keys)
        self.other_graph = other_graph

    def __call__(self, rows: ops.TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> ops.TRowsGenerator:
        return super().__call__(rows, self.other_graph.run(**kwargs))


class Graph:
    """Computational graph implementation"""

    def __init__(self) -> None:
        self.operations: list[ops.Operation] = []

    @staticmethod
    def graph_from_iter(name: str) -> 'Graph':
        """Construct new graph which reads data from row iterator (in form of sequence of Rows
        from 'kwargs' passed to 'run' method) into graph data-flow
        Use ops.ReadIterFactory
        :param name: name of kwarg to use as data source
        """
        graph = Graph()
        graph.operations.append(ops.ReadIterFactory(name))
        return graph

    @staticmethod
    def graph_from_file(filename: str, parser: tp.Callable[[str], ops.TRow]) -> 'Graph':
        """Construct new graph extended with operation for reading rows from file
        Use ops.Read
        :param filename: filename to read from
        :param parser: parser from string to Row
        """
        graph = Graph()
        graph.operations.append(ops.Read(filename, parser))
        return graph

    def map(self, mapper: ops.Mapper) -> 'Graph':
        """Construct new graph extended with map operation with particular mapper
        :param mapper: mapper to use
        """
        graph = Graph()
        graph.operations = self.operations.copy()
        graph.operations.append(ops.Map(mapper))
        return graph

    def reduce(self, reducer: ops.Reducer, keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with reduce operation with particular reducer
        :param reducer: reducer to use
        :param keys: keys for grouping
        """
        graph = Graph()
        graph.operations = self.operations.copy()
        graph.operations.append(ops.Reduce(reducer, keys))
        return graph

    def sort(self, keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with sort operation
        :param keys: sorting keys (typical is tuple of strings)
        """
        graph = Graph()
        graph.operations = self.operations.copy()
        graph.operations.append(ext_sort.ExternalSort(keys))
        return graph

    def join(self, joiner: ops.Joiner, join_graph: 'Graph', keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with join operation with another graph
        :param joiner: join strategy to use
        :param join_graph: other graph to join with
        :param keys: keys for grouping
        """
        graph = Graph()
        graph.operations = self.operations.copy()
        graph.operations.append(JoinGraph(joiner, keys, join_graph))
        return graph

    def run(self, **kwargs: tp.Any) -> ops.TRowsIterable:
        """Single method to start execution; data sources passed as kwargs"""
        it: ops.TRowsIterable = [{}]
        for op in self.operations:
            it = op(it, **kwargs)
        return it
