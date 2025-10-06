import pytest
import pyarrow as pa
from datetime import datetime
from pydantic import BaseModel
from typing import List, Tuple
from asani.arrow.serializer import Serializer


class Person(BaseModel):
    name: str
    age: int
    birthday: datetime
    bank_balance: Tuple[float, str]
    monthly_income: List[float] = []
    expense_matrix: List[List[float]]
    investment_history: List[List[List[float]]]


person_serializer = Serializer[Person](Person)


@pytest.fixture
def alice():
    return Person(
        name="Alice",
        age=30,
        birthday=datetime(1990, 1, 1),
        bank_balance=(12500.50, "USD"),
        monthly_income=[5500.00, 5800.50, 6200.25, 5900.75],
        expense_matrix=[
            [1200.0, 800.0, 300.0],
            [1200.0, 750.0, 250.0],
            [1200.0, 900.0, 400.0],
        ],
        investment_history=[
            [
                [1000.0, 500.0, 200.0],
                [1100.0, 520.0, 180.0],
            ],
            [
                [1200.0, 540.0, 220.0],
                [1300.0, 560.0, 250.0],
            ],
        ],
    )


@pytest.fixture
def bob():
    return Person(
        name="Bob",
        age=25,
        birthday=datetime(1995, 5, 15),
        bank_balance=(8750.25, "EUR"),
        monthly_income=[4200.00, 4500.00, 4800.00],
        expense_matrix=[
            [900.0, 600.0, 200.0],
            [900.0, 650.0, 150.0],
            [900.0, 700.0, 300.0],
        ],
        investment_history=[
            [
                [800.0, 400.0, 100.0],
                [850.0, 420.0, 90.0],
            ],
            [
                [900.0, 440.0, 120.0],
                [950.0, 460.0, 140.0],
            ],
        ],
    )


@pytest.fixture
def sample_data(alice, bob):
    return [alice, bob]


def test_round_trip_preserves_all_data(sample_data, alice, bob):
    arrow_table = person_serializer.to_table(sample_data)
    result = person_serializer.from_table(arrow_table)

    assert len(result) == 2
    assert result[0] == alice
    assert result[1] == bob


def test_round_trip_preserves_scalar_fields(sample_data):
    arrow_table = person_serializer.to_table(sample_data)
    result = person_serializer.from_table(arrow_table)

    assert result[0].name == "Alice"
    assert result[0].age == 30
    assert result[0].birthday == datetime(1990, 1, 1)

    assert result[1].name == "Bob"
    assert result[1].age == 25
    assert result[1].birthday == datetime(1995, 5, 15)


def test_round_trip_preserves_tuple_fields(sample_data):
    arrow_table = person_serializer.to_table(sample_data)
    result = person_serializer.from_table(arrow_table)

    assert result[0].bank_balance == (12500.50, "USD")
    assert result[1].bank_balance == (8750.25, "EUR")


def test_round_trip_preserves_one_dimensional_lists(sample_data):
    arrow_table = person_serializer.to_table(sample_data)
    result = person_serializer.from_table(arrow_table)

    assert result[0].monthly_income == [5500.00, 5800.50, 6200.25, 5900.75]
    assert result[1].monthly_income == [4200.00, 4500.00, 4800.00]


def test_round_trip_preserves_two_dimensional_tensors(sample_data):
    arrow_table = person_serializer.to_table(sample_data)
    result = person_serializer.from_table(arrow_table)

    assert result[0].expense_matrix == [
        [1200.0, 800.0, 300.0],
        [1200.0, 750.0, 250.0],
        [1200.0, 900.0, 400.0],
    ]
    assert result[1].expense_matrix == [
        [900.0, 600.0, 200.0],
        [900.0, 650.0, 150.0],
        [900.0, 700.0, 300.0],
    ]


def test_round_trip_preserves_three_dimensional_tensors(sample_data):
    arrow_table = person_serializer.to_table(sample_data)
    result = person_serializer.from_table(arrow_table)

    assert result[0].investment_history == [
        [
            [1000.0, 500.0, 200.0],
            [1100.0, 520.0, 180.0],
        ],
        [
            [1200.0, 540.0, 220.0],
            [1300.0, 560.0, 250.0],
        ],
    ]
    assert result[1].investment_history == [
        [
            [800.0, 400.0, 100.0],
            [850.0, 420.0, 90.0],
        ],
        [
            [900.0, 440.0, 120.0],
            [950.0, 460.0, 140.0],
        ],
    ]


def test_to_table_creates_correct_schema(sample_data):
    arrow_table = person_serializer.to_table(sample_data)

    expected_fields = [
        "name",
        "age",
        "birthday",
        "bank_balance",
        "monthly_income",
        "expense_matrix",
        "investment_history",
    ]
    assert arrow_table.schema.names == expected_fields


def test_to_table_creates_correct_scalar_types(sample_data):
    arrow_table = person_serializer.to_table(sample_data)

    assert arrow_table.schema.field("name").type == pa.string()
    assert arrow_table.schema.field("age").type == pa.int64()
    assert arrow_table.schema.field("birthday").type == pa.timestamp("ms")


def test_to_table_creates_correct_tuple_types(sample_data):
    arrow_table = person_serializer.to_table(sample_data)

    expected_type = pa.struct(
        [pa.field("_0", pa.float64()), pa.field("_1", pa.string())]
    )
    assert arrow_table.schema.field("bank_balance").type == expected_type


def test_to_table_creates_correct_list_types(sample_data):
    arrow_table = person_serializer.to_table(sample_data)

    assert arrow_table.schema.field("monthly_income").type == pa.list_(pa.float64())


def test_to_table_creates_correct_row_count(sample_data):
    arrow_table = person_serializer.to_table(sample_data)

    assert arrow_table.num_rows == len(sample_data)


def test_to_table_converts_scalar_data_correctly(sample_data):
    arrow_table = person_serializer.to_table(sample_data)

    assert arrow_table.column("name").to_pylist() == ["Alice", "Bob"]
    assert arrow_table.column("age").to_pylist() == [30, 25]
    assert arrow_table.column("birthday").to_pylist() == [
        datetime(1990, 1, 1),
        datetime(1995, 5, 15),
    ]


def test_to_table_converts_one_dimensional_lists_correctly(sample_data):
    arrow_table = person_serializer.to_table(sample_data)

    monthly_income = arrow_table.column("monthly_income").to_pylist()
    assert monthly_income[0] == [5500.00, 5800.50, 6200.25, 5900.75]
    assert monthly_income[1] == [4200.00, 4500.00, 4800.00]


def test_to_table_converts_tuples_to_structs_correctly(sample_data):
    arrow_table = person_serializer.to_table(sample_data)

    bank_balance = arrow_table.column("bank_balance").to_pylist()
    assert bank_balance[0] == {"_0": 12500.50, "_1": "USD"}
    assert bank_balance[1] == {"_0": 8750.25, "_1": "EUR"}


def test_to_table_flattens_two_dimensional_tensors_correctly(sample_data):
    arrow_table = person_serializer.to_table(sample_data)

    expense_matrix = arrow_table.column("expense_matrix").to_pylist()

    expected_alice = [
        1200.0,
        800.0,
        300.0,
        1200.0,
        750.0,
        250.0,
        1200.0,
        900.0,
        400.0,
    ]
    expected_bob = [
        900.0,
        600.0,
        200.0,
        900.0,
        650.0,
        150.0,
        900.0,
        700.0,
        300.0,
    ]

    assert expense_matrix[0] == expected_alice
    assert expense_matrix[1] == expected_bob


def test_to_table_flattens_three_dimensional_tensors_correctly(sample_data):
    arrow_table = person_serializer.to_table(sample_data)

    investment_history = arrow_table.column("investment_history").to_pylist()

    expected_alice = [
        1000.0,
        500.0,
        200.0,
        1100.0,
        520.0,
        180.0,
        1200.0,
        540.0,
        220.0,
        1300.0,
        560.0,
        250.0,
    ]
    expected_bob = [
        800.0,
        400.0,
        100.0,
        850.0,
        420.0,
        90.0,
        900.0,
        440.0,
        120.0,
        950.0,
        460.0,
        140.0,
    ]

    assert investment_history[0] == expected_alice
    assert investment_history[1] == expected_bob


def test_get_arrow_type_maps_primitive_types_correctly():
    assert person_serializer._get_arrow_type(int) == pa.int64()
    assert person_serializer._get_arrow_type(str) == pa.string()
    assert person_serializer._get_arrow_type(float) == pa.float64()
    assert person_serializer._get_arrow_type(bool) == pa.bool_()
    assert person_serializer._get_arrow_type(bytes) == pa.binary()
    assert person_serializer._get_arrow_type(datetime) == pa.timestamp("ms")


def test_get_arrow_type_maps_tuple_to_struct():
    expected = pa.struct([pa.field("_0", pa.float64()), pa.field("_1", pa.string())])
    assert person_serializer._get_arrow_type(Tuple[float, str]) == expected


def test_get_arrow_type_maps_one_dimensional_list():
    assert person_serializer._get_arrow_type(List[int]) == pa.list_(pa.int64())


def test_get_arrow_type_infers_two_dimensional_tensor_shape():
    tensor_type = List[List[int]]
    tensor_data = [[1, 2, 3], [4, 5, 6]]

    expected = pa.fixed_shape_tensor(pa.int64(), [2, 3])
    assert person_serializer._get_arrow_type(tensor_type, tensor_data) == expected


def test_get_arrow_type_infers_three_dimensional_tensor_shape():
    tensor_type = List[List[List[int]]]
    tensor_data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]

    expected = pa.fixed_shape_tensor(pa.int64(), [2, 2, 2])
    assert person_serializer._get_arrow_type(tensor_type, tensor_data) == expected


def test_get_arrow_type_raises_error_for_unsupported_types():
    with pytest.raises(ValueError, match="Unsupported field type"):
        person_serializer._get_arrow_type(dict)


if __name__ == "__main__":
    pytest.main()
