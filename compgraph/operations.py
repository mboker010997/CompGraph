import string
from abc import abstractmethod, ABC
import typing as tp
import re
from collections import defaultdict
from itertools import groupby, tee
from datetime import datetime
import math

TRow = dict[str, tp.Any]
TRowsIterable = tp.Iterable[TRow]
TRowsGenerator = tp.Generator[TRow, None, None]


class Operation(ABC):
    @abstractmethod
    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        pass


class Read(Operation):
    def __init__(self, filename: str, parser: tp.Callable[[str], TRow]) -> None:
        self.filename = filename
        self.parser = parser

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        with open(self.filename) as f:
            for line in f:
                yield self.parser(line)


class ReadIterFactory(Operation):
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for row in kwargs[self.name]():
            yield row


# Operations


class Mapper(ABC):
    """Base class for mappers"""
    @abstractmethod
    def __call__(self, row: TRow) -> TRowsGenerator:
        """
        :param row: one table row
        """
        pass


class Map(Operation):
    def __init__(self, mapper: Mapper) -> None:
        self.mapper = mapper

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for row in rows:
            for res_row in self.mapper(row):
                yield res_row


class Reducer(ABC):
    """Base class for reducers"""
    @abstractmethod
    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        """
        :param rows: table rows
        """
        pass


def _get_keys_proj(row: TRow, keys: tp.Sequence[str]) -> TRow:
    res = {}
    for column in keys:
        res[column] = row[column]
    return res


class Reduce(Operation):
    def __init__(self, reducer: Reducer, keys: tp.Sequence[str]) -> None:
        self.reducer = reducer
        self.keys = keys

    def _reduce_table(self, table: TRowsIterable) -> TRowsGenerator:
        res = self.reducer(tuple(self.keys), table)
        for row in res:
            yield row

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for _, group in groupby(rows, lambda x: _get_keys_proj(x, self.keys)):
            for row in self.reducer(tuple(self.keys), group):
                yield row


class Joiner(ABC):
    """Base class for joiners"""
    def __init__(self, suffix_a: str = '_1', suffix_b: str = '_2') -> None:
        self._a_suffix = suffix_a
        self._b_suffix = suffix_b

    @abstractmethod
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        """
        :param keys: join keys
        :param rows_a: left table rows
        :param rows_b: right table rows
        """
        pass


def _get_iter_val(it: tp.Iterator[tp.Any]) -> tp.Any:
    try:
        return next(it)
    except StopIteration:
        return None


def compare_dicts(keys: tp.Sequence[str], fst: TRow, snd: TRow) -> int:
    for key in keys:
        if fst[key] < snd[key]:
            return -1
        elif fst[key] > snd[key]:
            return 1
    return 0


class Join(Operation):
    def __init__(self, joiner: Joiner, keys: tp.Sequence[str]):
        self.keys = keys
        self.joiner = joiner

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        fst = rows
        snd = args[0]
        fst_group_by = groupby(fst, lambda x: _get_keys_proj(x, self.keys))
        snd_group_by = groupby(snd, lambda x: _get_keys_proj(x, self.keys))

        fst_cur = _get_iter_val(fst_group_by)
        snd_cur = _get_iter_val(snd_group_by)

        while not (fst_cur is None and snd_cur is None):
            if snd_cur is None or (fst_cur is not None and compare_dicts(self.keys, fst_cur[0], snd_cur[0]) == -1):
                res = self.joiner(self.keys, fst_cur[1], {})
                for row in res:
                    yield row
                fst_cur = _get_iter_val(fst_group_by)
            elif fst_cur is None or (snd_cur is not None and compare_dicts(self.keys, fst_cur[0], snd_cur[0]) == 1):
                res = self.joiner(self.keys, {}, snd_cur[1])
                for row in res:
                    yield row
                snd_cur = _get_iter_val(snd_group_by)
            else:
                res = self.joiner(self.keys, fst_cur[1], snd_cur[1])
                for row in res:
                    yield row
                fst_cur = _get_iter_val(fst_group_by)
                snd_cur = _get_iter_val(snd_group_by)


# Dummy operators


class DummyMapper(Mapper):
    """Yield exactly the row passed"""
    def __call__(self, row: TRow) -> TRowsGenerator:
        yield row


class FirstReducer(Reducer):
    """Yield only first row from passed ones"""
    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        for row in rows:
            yield row
            break


# Mappers


class FilterPunctuation(Mapper):
    """Left only non-punctuation symbols"""
    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    @staticmethod
    def _remove_punct(txt: str) -> str:
        def remove_punctuation(char: str) -> bool:
            return char not in string.punctuation
        return ''.join(filter(remove_punctuation, txt))

    def __call__(self, row: TRow) -> TRowsGenerator:
        try:
            res = row.copy()
        except Exception:
            print('mbo', row)
            raise
        res[self.column] = self._remove_punct(res[self.column])
        yield res


class FuncMapper(Mapper):
    def __init__(self, column: str, func: tp.Callable[[tp.Any], tp.Any]):
        self.column = column
        self.func = func

    def __call__(self, row: TRow) -> TRowsGenerator:
        res = row.copy()
        res[self.column] = self.func(res[self.column])
        yield res


class DateMapper(Mapper):
    WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    def __init__(self, enter_column: str, leave_column: str,
                 hour_column: str = 'hour',
                 weekday_column: str = 'weekday',
                 delta_column: str = 'delta'):
        self.hour_column = hour_column
        self.weekday_column = weekday_column
        self.enter_column = enter_column
        self.leave_column = leave_column
        self.delta_column = delta_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        res = row.copy()
        try:
            enter_date = datetime.strptime(row[self.enter_column], '%Y%m%dT%H%M%S.%f')
        except Exception:
            enter_date = datetime.strptime(row[self.enter_column], '%Y%m%dT%H%M%S')
        try:
            leave_date = datetime.strptime(row[self.leave_column], '%Y%m%dT%H%M%S.%f')
        except Exception:
            leave_date = datetime.strptime(row[self.leave_column], '%Y%m%dT%H%M%S')
        res[self.weekday_column] = self.WEEKDAYS[enter_date.weekday()]
        res[self.hour_column] = enter_date.hour
        res[self.delta_column] = leave_date.timestamp() - enter_date.timestamp()
        yield res


class DistanceMapper(Mapper):
    def __init__(self, start_column: str = 'start', end_column: str = 'end', dist_column: str = 'dist'):
        self.start_column = start_column
        self.end_column = end_column
        self.dist_column = dist_column

    @staticmethod
    def get_distance(fst: tuple[float, float], snd: tuple[float, float]) -> float:
        dlat = (snd[1] - fst[1]) * math.pi / 180.0
        dlon = (snd[0] - fst[0]) * math.pi / 180.0

        lat1 = fst[1] * math.pi / 180.0
        lat2 = snd[1] * math.pi / 180.0

        a = math.sin(dlat / 2)**2 + math.sin(dlon / 2)**2 * math.cos(lat1) * math.cos(lat2)
        rad = 6373
        c = 2 * math.asin(math.sqrt(a))
        return rad * c

    def __call__(self, row: TRow, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        res = row.copy()
        res[self.dist_column] = self.get_distance(row[self.start_column], row[self.end_column])
        yield res


class LowerCase(Mapper):
    """Replace column value with value in lower case"""
    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    @staticmethod
    def _lower_case(txt: str) -> str:
        return txt.lower()

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = self._lower_case(row[self.column])
        yield row


class Split(Mapper):
    """Split row on multiple rows by separator"""
    def __init__(self, column: str, separator: str | None = None) -> None:
        """
        :param column: name of column to split
        :param separator: string to separate by
        """
        self.column = column
        self.separator = separator

    def __call__(self, row: TRow) -> TRowsGenerator:
        string = row[self.column]
        if self.separator is not None:
            it = re.finditer(self.separator, string)
        else:
            it = re.finditer(r'\s+', string)

        prev = 0
        for match in it:
            left, right = match.span()
            res = row.copy()
            res[self.column] = string[prev:left]
            prev = right
            yield res

        if len(string) - prev > 0:
            res = row.copy()
            res[self.column] = string[prev:]
            yield res


class Product(Mapper):
    """Calculates product of multiple columns"""
    def __init__(self, columns: tp.Sequence[str], result_column: str = 'product') -> None:
        """
        :param columns: column names to product
        :param result_column: column name to save product in
        """
        self.columns = columns
        self.result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        res = 1
        for column in self.columns:
            res *= row[column]
        row[self.result_column] = res
        yield row


class Filter(Mapper):
    """Remove records that don't satisfy some condition"""
    def __init__(self, condition: tp.Callable[[TRow], bool]) -> None:
        """
        :param condition: if condition is not true - remove record
        """
        self.condition = condition

    def __call__(self, row: TRow) -> TRowsGenerator:
        if self.condition(row):
            yield row


class Project(Mapper):
    """Leave only mentioned columns"""
    def __init__(self, columns: tp.Sequence[str]) -> None:
        """
        :param columns: names of columns
        """
        self.columns = columns

    def __call__(self, row: TRow) -> TRowsGenerator:
        res = {}
        for column in self.columns:
            res[column] = row[column]
        yield res


# Reducers
class TopN(Reducer):
    """Calculate top N by value"""
    def __init__(self, column: str, n: int) -> None:
        """
        :param column: column name to get top by
        :param n: number of top values to extract
        """
        self.column = column
        self.n = n

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        values = []
        for row in rows:
            values.append(row)
            i = len(values) - 1
            while i > 0 and values[i][self.column] > values[i - 1][self.column]:
                values[i], values[i - 1] = values[i - 1], values[i]
                i -= 1
            if len(values) > self.n:
                values.pop()
        for res in values:
            yield res


class TermFrequency(Reducer):
    """Calculate frequency of values in column"""
    def __init__(self, words_column: str, result_column: str = 'tf') -> None:
        """
        :param words_column: name for column with words
        :param result_column: name for result column
        """
        self.words_column = words_column
        self.result_column = result_column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        cnt: dict[tp.Any, int] = defaultdict(int)
        total = 0
        any_row = {}
        for row in rows:
            cnt[row[self.words_column]] += 1
            any_row = row
            total += 1
        for key, val in cnt.items():
            res = {self.words_column: key, self.result_column: val / total}
            for column in group_key:
                res[column] = any_row[column]
            yield res


class Count(Reducer):
    """
    Count records by key
    Example for group_key=('a',) and column='d'
        {'a': 1, 'b': 5, 'c': 2}
        {'a': 1, 'b': 6, 'c': 1}
        =>
        {'a': 1, 'd': 2}
    """
    def __init__(self, column: str) -> None:
        """
        :param column: name for result column
        """
        self.column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        res = {}
        cnt = 0
        for row in rows:
            cnt += 1
            if cnt == 1:
                for column in group_key:
                    res[column] = row[column]
        res[self.column] = cnt
        yield res


class Sum(Reducer):
    """
    Sum values aggregated by key
    Example for key=('a',) and column='b'
        {'a': 1, 'b': 2, 'c': 4}
        {'a': 1, 'b': 3, 'c': 5}
        =>
        {'a': 1, 'b': 5}
    """
    def __init__(self, column: str) -> None:
        """
        :param column: name for sum column
        """
        self.column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        res = {self.column: 0}
        for row in rows:
            res[self.column] += row[self.column]
            for column in group_key:
                res[column] = row[column]
        yield res


# Joiners


class InnerJoiner(Joiner):
    """Join with inner strategy"""
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        snd_rows = [row for row in rows_b]

        counter = 0
        for fst in rows_a:
            for snd in snd_rows:
                res = {}
                for column, val in fst.items():
                    if column not in keys and column in snd:
                        res[column + self._a_suffix] = val
                    else:
                        res[column] = val
                for column, val in snd.items():
                    if column not in keys and column in fst:
                        res[column + self._b_suffix] = val
                    else:
                        res[column] = val
                yield res
            counter += 1


def _get_iter_len(it: tp.Iterator[tp.Any]) -> int:
    res = 0
    for _ in it:
        res += 1
    return res


class OuterJoiner(Joiner):
    """Join with outer strategy"""
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        fst_dup, fst_rows, fst_dup2 = tee(rows_a, 3)
        fst_len = _get_iter_len(fst_dup)
        snd_dup = tee(rows_b, fst_len + 5)

        counter = 0
        for fst in fst_rows:
            for snd in snd_dup[counter]:
                res = {}
                for column, val in fst.items():
                    if column not in keys and column in snd:
                        res[column + self._a_suffix] = val
                    else:
                        res[column] = val
                for column, val in snd.items():
                    if column not in keys and column in fst:
                        res[column + self._b_suffix] = val
                    else:
                        res[column] = val
                yield res
            counter += 1

        snd_len = _get_iter_len(snd_dup[counter])
        counter += 1
        if fst_len == 0:
            for row in snd_dup[counter]:
                yield row
        if snd_len == 0:
            for row in fst_dup2:
                yield row


class LeftJoiner(Joiner):
    """Join with left strategy"""
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        fst_dup, fst_rows, fst_dup2 = tee(rows_a, 3)
        fst_len = _get_iter_len(fst_dup)
        snd_dup = tee(rows_b, fst_len + 5)

        counter = 0
        for fst in fst_rows:
            for snd in snd_dup[counter]:
                res = {}
                for column, val in fst.items():
                    if column not in keys and column in snd:
                        res[column + self._a_suffix] = val
                    else:
                        res[column] = val
                for column, val in snd.items():
                    if column not in keys and column in fst:
                        res[column + self._b_suffix] = val
                    else:
                        res[column] = val
                yield res
            counter += 1

        snd_len = _get_iter_len(snd_dup[counter])
        counter += 1
        if snd_len == 0:
            for row in fst_dup2:
                yield row


class RightJoiner(Joiner):
    """Join with right strategy"""
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        fst_dup, fst_rows, fst_dup2 = tee(rows_a, 3)
        fst_len = _get_iter_len(fst_dup)
        snd_dup = tee(rows_b, fst_len + 5)

        counter = 0
        for fst in fst_rows:
            for snd in snd_dup[counter]:
                res = {}
                for column, val in fst.items():
                    if column not in keys and column in snd:
                        res[column + self._a_suffix] = val
                    else:
                        res[column] = val
                for column, val in snd.items():
                    if column not in keys and column in fst:
                        res[column + self._b_suffix] = val
                    else:
                        res[column] = val
                yield res
            counter += 1

        counter += 1
        if fst_len == 0:
            for row in snd_dup[counter]:
                yield row
