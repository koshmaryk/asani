import pytest
import pyarrow as pa
from datetime import datetime
from pydantic import BaseModel
from typing import Annotated, List, Tuple
from asani.arrow.serializer import Serializer


# Sample Pydantic model
class Person(BaseModel):
    name: str
    age: int
    birthday: datetime
    bank_balance: Tuple[float, str]
    monthly_income: List[float] = []  # 1D list
    expense_matrix: Annotated[List[List[float]], "tensor", (3, 3)]  # 2D tensor
    investment_history: Annotated[
        List[List[List[float]]], "tensor", (2, 2, 3)
    ]  # 3D tensor


# Initialize Serializer for Person
person_serializer = Serializer[Person](Person)


@pytest.fixture
def sample_data():
    return [
        Person(
            name="Alice",
            age=30,
            birthday=datetime(1990, 1, 1),
            bank_balance=(12500.50, "USD"),
            monthly_income=[5500.00, 5800.50, 6200.25, 5900.75],
            expense_matrix=[
                [1200.0, 800.0, 300.0],  # Month 1: [Housing, Food, Entertainment]
                [1200.0, 750.0, 250.0],  # Month 2
                [1200.0, 900.0, 400.0],  # Month 3
            ],
            investment_history=[
                # Year 1
                [
                    [1000.0, 500.0, 200.0],  # Q1: [Stocks, Bonds, Crypto]
                    [1100.0, 520.0, 180.0],  # Q2
                ],
                # Year 2
                [
                    [1200.0, 540.0, 220.0],  # Q1
                    [1300.0, 560.0, 250.0],  # Q2
                ],
            ],
        ),
        Person(
            name="Bob",
            age=25,
            birthday=datetime(1995, 5, 15),
            bank_balance=(8750.25, "EUR"),
            monthly_income=[4200.00, 4500.00, 4800.00],
            expense_matrix=[
                [900.0, 600.0, 200.0],  # Month 1: [Housing, Food, Entertainment]
                [900.0, 650.0, 150.0],  # Month 2
                [900.0, 700.0, 300.0],  # Month 3
            ],
            investment_history=[
                # Year 1
                [
                    [800.0, 400.0, 100.0],  # Q1: [Stocks, Bonds, Crypto]
                    [850.0, 420.0, 90.0],  # Q2
                ],
                # Year 2
                [
                    [900.0, 440.0, 120.0],  # Q1
                    [950.0, 460.0, 140.0],  # Q2
                ],
            ],
        ),
    ]


def test_from_table(sample_data):
    # Convert sample data to PyArrow Table
    arrow_table = person_serializer.to_table(sample_data)

    # Use the from_table method to convert back to Pydantic models
    result = person_serializer.from_table(arrow_table)

    # Test if the length of the result is correct
    assert len(result) == len(sample_data)

    # Test if the fields are correctly populated
    assert result[0].name == "Alice"
    assert result[1].age == 25
    assert result[1].birthday == datetime(1995, 5, 15)

    # Test tuple and
    assert result[0].bank_balance == (12500.50, "USD")
    assert result[1].bank_balance == (8750.25, "EUR")

    # Test list fields
    assert result[0].monthly_income == [5500.00, 5800.50, 6200.25, 5900.75]
    assert result[1].monthly_income == [4200.00, 4500.00, 4800.00]

    # Test 2D tensor
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

    # Test 3D tensor
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


def test_to_table(sample_data):
    # Convert sample data to PyArrow Table
    arrow_table = person_serializer.to_table(sample_data)

    # Check the schema of the resulting table
    expected_names = [
        "name",
        "age",
        "birthday",
        "bank_balance",
        "monthly_income",
        "expense_matrix",
        "investment_history",
    ]
    assert arrow_table.schema.names == expected_names

    # Check if the data types match
    assert arrow_table.schema.field("name").type == pa.string()
    assert arrow_table.schema.field("age").type == pa.int64()
    assert arrow_table.schema.field("birthday").type == pa.timestamp("ms")

    # Check list and tuple field types
    assert arrow_table.schema.field("monthly_income").type == pa.list_(pa.float64())
    assert arrow_table.schema.field("bank_balance").type == pa.struct(
        [pa.field("_0", pa.float64()), pa.field("_1", pa.string())]
    )

    # Check if the data is correctly converted
    assert arrow_table.num_rows == len(sample_data)
    assert arrow_table.column("name").to_pylist() == ["Alice", "Bob"]
    assert arrow_table.column("age").to_pylist() == [30, 25]
    assert arrow_table.column("birthday").to_pylist() == [
        datetime(1990, 1, 1),
        datetime(1995, 5, 15),
    ]

    # Test list and tuple data conversion
    monthly_income_data = arrow_table.column("monthly_income").to_pylist()
    assert monthly_income_data[0] == [5500.00, 5800.50, 6200.25, 5900.75]
    assert monthly_income_data[1] == [4200.00, 4500.00, 4800.00]

    bank_balance_data = arrow_table.column("bank_balance").to_pylist()
    assert bank_balance_data[0] == {"_0": 12500.50, "_1": "USD"}
    assert bank_balance_data[1] == {"_0": 8750.25, "_1": "EUR"}

    # Test 2D tensor (compare flattened)
    expense_matrix_data = arrow_table.column("expense_matrix").to_pylist()
    expected_expense_matrix_0 = [
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
    expected_expense_matrix_1 = [
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
    assert expense_matrix_data[0] == expected_expense_matrix_0
    assert expense_matrix_data[1] == expected_expense_matrix_1

    # Test 3D tensor (compare flattened)
    investment_history_data = arrow_table.column("investment_history").to_pylist()
    expected_investment_history_0 = [
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
    expected_investment_history_1 = [
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
    assert investment_history_data[0] == expected_investment_history_0
    assert investment_history_data[1] == expected_investment_history_1


def test_get_arrow_type():
    # Test supported field types
    assert person_serializer._get_arrow_type(int) == pa.int64()
    assert person_serializer._get_arrow_type(str) == pa.string()
    assert person_serializer._get_arrow_type(float) == pa.float64()
    assert person_serializer._get_arrow_type(bool) == pa.bool_()
    assert person_serializer._get_arrow_type(bytes) == pa.binary()
    assert person_serializer._get_arrow_type(datetime) == pa.timestamp("ms")

    # Test tuple type
    assert person_serializer._get_arrow_type(Tuple[float, str]) == pa.struct(
        [pa.field("_0", pa.float64()), pa.field("_1", pa.string())]
    )

    # Test list type
    assert person_serializer._get_arrow_type(List[int]) == pa.list_(pa.int64())

    # Test 2D tensor type
    tensor_2d_type = Annotated[List[List[float]], "tensor", (3, 3)]
    assert person_serializer._get_arrow_type(tensor_2d_type) == pa.fixed_shape_tensor(
        pa.float64(), [3, 3]
    )

    # Test 3D tensor type
    tensor_3d_type = Annotated[List[List[List[float]]], "tensor", (2, 2, 3)]
    assert person_serializer._get_arrow_type(tensor_3d_type) == pa.fixed_shape_tensor(
        pa.float64(), [2, 2, 3]
    )

    # Test unsupported field type
    with pytest.raises(ValueError):
        person_serializer._get_arrow_type(dict)


if __name__ == "__main__":
    pytest.main()
