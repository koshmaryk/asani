from datetime import datetime
from typing import Any, Optional, Type, TypeVar, Generic, List, get_origin, get_args
from pydantic import BaseModel
import pyarrow as pa

T = TypeVar("T", bound=BaseModel)


class Serializer(Generic[T]):
    def __init__(self, model: Type[T]):
        self.model = model

    def from_table(self, table: pa.Table) -> List[T]:
        """
        Convert a PyArrow Table (VectorSchemaRoot) to a list of Pydantic models.
        """
        # Get the schema (fields) from the Pydantic model
        schema = self.model.__annotations__

        data_dict = {}
        for field_name in table.schema.names:
            field_data = table.column(field_name)
            field_type = schema.get(field_name)

            if isinstance(field_data.type, pa.FixedShapeTensorType):
                # Extract shape from the Arrow type itself
                shape = list(field_data.type.shape)
                data_dict[field_name] = [
                    self._unflatten_tensor(tensor, shape)
                    for tensor in field_data.to_pylist()
                ]
            elif get_origin(field_type) is tuple:
                data_dict[field_name] = [
                    tuple(row[field] for field in field_data.type.names)
                    for row in field_data.to_pylist()
                ]
            else:
                data_dict[field_name] = field_data.to_pylist()

        records = [
            {field: data_dict[field][i] for field in data_dict}
            for i in range(table.num_rows)
        ]

        return [self.model(**record) for record in records]

    def to_table(self, data: List[T]) -> pa.Table:
        """
        Convert a list of Pydantic models to a PyArrow Table (VectorSchemaRoot).
        Handles both regular fields and FixedShapeTensorType fields.
        """
        # Get the schema (fields) from the Pydantic model
        schema = self.model.__annotations__

        # Prepare a list of Arrow Field Vectors
        vectors = []
        arrow_fields = []

        for field, field_type in schema.items():
            # Initialize an empty list to collect values for this field
            field_data = [getattr(item, field) for item in data]

            # Check if it's a 2D+ list (tensor)
            shape = self.infer_shape(field_data[0])
            if len(shape) > 1:
                element_type = self.get_tensor_element_type(field_type)
                arrow_type = pa.fixed_shape_tensor(
                    self._get_arrow_type(element_type), shape
                )

                # Flatten each tensor and create the array
                flattened_data = [
                    self.flatten_tensor(tensor, shape) for tensor in field_data
                ]
                vector = pa.array(flattened_data, type=arrow_type)
            else:
                # Determine Arrow type based on the Pydantic model field type
                arrow_type = self._get_arrow_type(field_type)
                vector = pa.array(field_data, type=arrow_type)

            vectors.append(vector)
            arrow_fields.append(pa.field(field, arrow_type))

        # Create a schema based on the fields
        arrow_schema = pa.schema(arrow_fields)

        # Create a Table from the vectors and schema
        return pa.table(vectors, schema=arrow_schema)

    def _get_arrow_type(
        self, field_type: Type, field_data: Optional[Any] = None
    ) -> pa.DataType:
        if field_type is int:
            return pa.int64()
        elif field_type is str:
            return pa.string()
        elif field_type is float:
            return pa.float64()
        elif field_type is bool:
            return pa.bool_()
        elif field_type is bytes:
            return pa.binary()
        elif field_type is datetime:
            return pa.timestamp("ms")
        elif get_origin(field_type) is tuple:
            args = get_args(field_type)
            if not args:
                return pa.struct([])

            fields = []
            for i, arg_type in enumerate(args):
                field_name = f"_{i}"
                field_arrow_type = self._get_arrow_type(arg_type)
                fields.append(pa.field(field_name, field_arrow_type))
            return pa.struct(fields)
        elif get_origin(field_type) is list:
            shape = self.infer_shape(field_data)
            if len(shape) > 1:
                # 2D+ list (tensor)
                element_type = self.get_tensor_element_type(field_type)
                return pa.fixed_shape_tensor(self._get_arrow_type(element_type), shape)
            elif len(shape) == 1:
                # 1D list
                element_type = self.get_tensor_element_type(field_type)
                return pa.list_(self._get_arrow_type(element_type))
            else:
                # Fallback: use type annotation
                element_type = get_args(field_type)[0]
                return pa.list_(self._get_arrow_type(element_type))
        else:
            raise ValueError(f"Unsupported field type: {field_type}")

    def infer_shape(self, field_data: Any) -> list:
        if not isinstance(field_data, list):
            return []
        shape = [len(field_data)]
        if len(field_data) > 0 and isinstance(field_data[0], list):
            shape += self.infer_shape(field_data[0])
        return shape

    def get_tensor_element_type(self, field_type: Type) -> Type:
        curr_type = field_type
        while get_origin(curr_type) is list:
            args = get_args(curr_type)
            if not args:
                raise ValueError(f"Cannot extract element type from {field_type}")
            curr_type = args[0]

        return curr_type

    def flatten_tensor(self, data: Any, shape: List[int]) -> List:
        if not isinstance(data, list):
            return [data]

        result = []
        for item in data:
            result.extend(self.flatten_tensor(item, shape))

        return result

    def _unflatten_tensor(self, flat_data: List, shape: List[int]) -> Any:
        total_size = 1
        for dim in shape:
            total_size *= dim

        if len(flat_data) != total_size:
            raise ValueError(f"Data size {len(flat_data)} does not match shape {shape}")

        if len(shape) == 1:
            return flat_data

        chunk_size = total_size // shape[0]
        result = []

        for i in range(shape[0]):
            start = i * chunk_size
            end = start + chunk_size
            chunk = flat_data[start:end]

            if len(shape) > 2:
                result.append(self._unflatten_tensor(chunk, shape[1:]))
            else:
                result.append(chunk)

        return result
