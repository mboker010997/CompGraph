import copy
import dataclasses
import typing as tp

import pytest
from pytest import approx

from compgraph import operations as ops


class _Key:
    def __init__(self, *args: str) -> None:
        self._items = args

    def __call__(self, d: tp.Mapping[str, tp.Any]) -> tuple[str, ...]:
        return tuple(str(d.get(key)) for key in self._items)


@dataclasses.dataclass
class MapCase:
    mapper: ops.Mapper
    data: list[ops.TRow]
    ground_truth: list[ops.TRow]
    cmp_keys: tuple[str, ...]
    mapper_item: int = 0
    mapper_ground_truth_items: tuple[int, ...] = (0,)


MAP_CASES = [
    MapCase(
        mapper=ops.DateMapper('enter_time', 'leave_time'),
        data=[
            {'leave_time': '20171020T112238.723000', 'enter_time': '20171020T112237.427000'},
        ],
        ground_truth=[
            {'hour': 11, 'weekday': 'Fri', 'delta': approx(1.3, 0.1),
             'enter_time': '20171020T112237.427000',
             'leave_time': '20171020T112238.723000'}
        ],
        cmp_keys=('hour', 'weekday', 'delta')
    ),
]


@pytest.mark.parametrize('case', MAP_CASES)
def test_new_mappers(case: MapCase) -> None:
    mapper_data_row = copy.deepcopy(case.data[case.mapper_item])
    mapper_ground_truth_rows = [copy.deepcopy(case.ground_truth[i]) for i in case.mapper_ground_truth_items]

    key_func = _Key(*case.cmp_keys)

    mapper_result = case.mapper(mapper_data_row)
    assert isinstance(mapper_result, tp.Iterator)
    assert sorted(mapper_result, key=key_func) == sorted(mapper_ground_truth_rows, key=key_func)

    result = ops.Map(case.mapper)(iter(case.data))
    assert isinstance(result, tp.Iterator)
    assert sorted(result, key=key_func) == sorted(case.ground_truth, key=key_func)
