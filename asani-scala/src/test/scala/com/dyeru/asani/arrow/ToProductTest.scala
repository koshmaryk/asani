package com.dyeru.asani.arrow

import com.dyeru.asani.arrow.createVectorSchemaRoot
import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector.VectorSchemaRoot
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll
import org.apache.arrow.vector.types.pojo.{ArrowType, Field, FieldType, Schema}
import org.apache.arrow.vector.types.Types.MinorType
import org.apache.arrow.vector.types.FloatingPointPrecision
import org.apache.arrow.vector.IntVector

import java.time.Instant
import scala.jdk.CollectionConverters.*

class ToProductTest extends AnyFunSuite with BeforeAndAfterAll {

  override def beforeAll(): Unit = {
    FixedShapeTensor.register()
  }

  test("ToProduct for case class with Int, String, Boolean") {
    case class TestClass(id: Int, name: String, active: Boolean)

    val fields = Seq(
      new Field("id", FieldType.nullable(MinorType.INT.getType), null),
      new Field("name", FieldType.nullable(MinorType.VARCHAR.getType), null),
      new Field("active", FieldType.nullable(MinorType.BIT.getType), null)
    )
    val data = Seq(
      Seq(1, "Alice", true),
      Seq(2, "Bob", false)
    )
    val root = createVectorSchemaRoot(fields, data)

    val result: List[TestClass] = root.toProducts

    assert(result == List(TestClass(1, "Alice", true), TestClass(2, "Bob", false)))

    root.close()
  }

  test("ToProduct for case class with Double and Instant") {
    case class TestClass(score: Double, timestamp: Instant)

    val fields = Seq(
      new Field("score", FieldType.nullable(MinorType.FLOAT8.getType), null),
      new Field("timestamp", FieldType.nullable(MinorType.TIMESTAMPMILLI.getType), null)
    )
    val data = Seq(
      Seq(95.5, Instant.ofEpochMilli(1633089600000L)),
      Seq(88.0, Instant.ofEpochMilli(1633176000000L))
    )
    val root = createVectorSchemaRoot(fields, data)

    val result: List[TestClass] = root.toProducts

    assert(result == List(
      TestClass(95.5, Instant.ofEpochMilli(1633089600000L)),
      TestClass(88.0, Instant.ofEpochMilli(1633176000000L))
    ))

    root.close()
  }

  test("ToProduct for case class with Array[Byte]") {
    case class TestClass(data: Array[Byte])

    val fields = Seq(
      new Field("data", FieldType.nullable(MinorType.VARBINARY.getType), null)
    )
    val data = Seq(
      Seq(Array[Byte](1, 2, 3)),
      Seq(Array[Byte](4, 5, 6))
    )
    val root = createVectorSchemaRoot(fields, data)

    val result: List[TestClass] = root.toProducts

    assert(result.map(_.data.toList) == List(List(1, 2, 3), List(4, 5, 6)))

    root.close()
  }

  test("ToProduct for case class with empty result") {
    case class TestClass(data: Array[Byte])

    val fields = Seq(
      new Field("data", FieldType.nullable(MinorType.VARBINARY.getType), null)
    )
    val data = Seq()
    val root = createVectorSchemaRoot(fields, data)

    val result: List[TestClass] = root.toProducts

    assert(result == List())

    root.close()
  }

  test("ToProduct for case class with Seq[Float]") {
    case class TestClass(data: Seq[Float])

    val data = Seq(
      TestClass(Seq[Float](1, 2, 3)),
      TestClass(Seq[Float](4, 5, 6))
    )

    val allocator = new RootAllocator(Long.MaxValue)
    val root = VectorSchemaRoot.create(ArrowSchema.derived[TestClass].schema, allocator)

    val vectorRoot = data.toArrowVector(root)

    val result: List[TestClass] = root.toProducts

    assert(result.map(_.data.toList) == List(List(1, 2, 3), List(4, 5, 6)))

    root.close()
  }

  test("ToProduct for case class with Seq[Option[Float]]") {
    case class TestClass(data: Seq[Option[Float]])

    val data = Seq(
      TestClass(Seq(None, Some(2), Some(3))),
      TestClass(Seq(Some(4), None, None))
    )

    val allocator = new RootAllocator(Long.MaxValue)
    val root = VectorSchemaRoot.create(ArrowSchema.derived[TestClass].schema, allocator)

    data.toArrowVector(root)

    val result: List[TestClass] = root.toProducts

    assert(result == List(
      TestClass(Seq(None, Some(2), Some(3))),
      TestClass(Seq(Some(4), None, None)))
    )

    root.close()
  }

  // Test for 2D Float Tensor
  test("ToProduct for case class with 2D Float Tensor (Seq[Seq[Float]])") {
    case class DataWith2DTensor(id: Int, matrix: Seq[Seq[Float]])

    val data = Seq(
      DataWith2DTensor(1, Seq(Seq(1.0f, 2.0f), Seq(3.0f, 4.0f))),
      DataWith2DTensor(2, Seq(Seq(5.0f, 6.0f), Seq(7.0f, 8.0f)))
    )

    val allocator = new RootAllocator(Long.MaxValue)
    
    // Create tensor vectors manually
    val valueType = new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)
    val tensorType = FixedShapeTensor(valueType, Seq(2, 2))
    
    val idVector = new IntVector("id", allocator)
    val matrixVector = new Tensor("matrix", allocator, tensorType)
    
    val fields = List(idVector.getField, matrixVector.getField)
    val schema = new Schema(fields.asJava)
    
    val root = new VectorSchemaRoot(schema, List(idVector, matrixVector).asJava, 0)
    data.toArrowVector(root)

    // Convert back to products
    val result: List[DataWith2DTensor] = root.toProducts
    
    assert(result.length == 2)
    assert(result(0).id == 1)
    assert(result(0).matrix == Seq(Seq(1.0f, 2.0f), Seq(3.0f, 4.0f)))
    assert(result(1).id == 2)
    assert(result(1).matrix == Seq(Seq(5.0f, 6.0f), Seq(7.0f, 8.0f)))

    root.close()
    idVector.close()
    matrixVector.close()
  }

  // Test for 3D Double Tensor
  test("ToProduct for case class with 3D Double Tensor (Seq[Seq[Seq[Double]]])") {
    case class DataWith3DTensor(id: Int, tensor3d: Seq[Seq[Seq[Double]]])

    val data = Seq(
      DataWith3DTensor(1, Seq(
        Seq(Seq(1.0, 2.0), Seq(3.0, 4.0)),
        Seq(Seq(5.0, 6.0), Seq(7.0, 8.0))
      ))
    )

    val allocator = new RootAllocator(Long.MaxValue)
    
    // Create tensor vectors manually
    val valueType = new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)
    val tensorType = FixedShapeTensor(valueType, Seq(2, 2, 2))
    
    val idVector = new IntVector("id", allocator)
    val tensorVector3d = new Tensor("tensor3d", allocator, tensorType)
    
    val fields = List(idVector.getField, tensorVector3d.getField)
    val schema = new Schema(fields.asJava)
    
    val root = new VectorSchemaRoot(schema, List(idVector, tensorVector3d).asJava, 0)
    data.toArrowVector(root)

    // Convert back to products
    val result: List[DataWith3DTensor] = root.toProducts
    
    assert(result.length == 1)
    assert(result(0).id == 1)
    assert(result(0).tensor3d == Seq(
      Seq(Seq(1.0, 2.0), Seq(3.0, 4.0)),
      Seq(Seq(5.0, 6.0), Seq(7.0, 8.0))
    ))

    root.close()
    idVector.close()
    tensorVector3d.close()
  }

  // Test for 2D Int Tensor
  test("ToProduct for case class with 2D Int Tensor (Seq[Seq[Int]])") {
    case class DataWithIntTensor(name: String, matrix: Seq[Seq[Int]])

    val data = Seq(
      DataWithIntTensor("Alice", Seq(Seq(10, 20), Seq(30, 40))),
      DataWithIntTensor("Bob", Seq(Seq(50, 60), Seq(70, 80)))
    )

    val allocator = new RootAllocator(Long.MaxValue)
    
    // Create tensor vectors manually
    val valueType = new ArrowType.Int(32, true)
    val tensorType = FixedShapeTensor(valueType, Seq(2, 2))
    
    import org.apache.arrow.vector.VarCharVector
    val nameVector = new VarCharVector("name", allocator)
    val matrixVector = new Tensor("matrix", allocator, tensorType)
    
    val fields = List(nameVector.getField, matrixVector.getField)
    val schema = new Schema(fields.asJava)
    
    val root = new VectorSchemaRoot(schema, List(nameVector, matrixVector).asJava, 0)
    data.toArrowVector(root)

    // Convert back to products
    val result: List[DataWithIntTensor] = root.toProducts
    
    assert(result.length == 2)
    assert(result(0).name == "Alice")
    assert(result(0).matrix == Seq(Seq(10, 20), Seq(30, 40)))
    assert(result(1).name == "Bob")
    assert(result(1).matrix == Seq(Seq(50, 60), Seq(70, 80)))

    root.close()
    nameVector.close()
    matrixVector.close()
  }

  // Test for multiple tensors in same case class
  test("ToProduct for case class with multiple Tensors") {
    case class DataWithMultipleTensors(id: Int, matrix1: Seq[Seq[Float]], matrix2: Seq[Seq[Double]])

    val data = Seq(
      DataWithMultipleTensors(1, Seq(Seq(1.0f, 2.0f)), Seq(Seq(10.0, 20.0), Seq(30.0, 40.0)))
    )

    val allocator = new RootAllocator(Long.MaxValue)
    
    // Create tensor vectors manually
    val valueType1 = new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)
    val tensorType1 = FixedShapeTensor(valueType1, Seq(1, 2))
    
    val valueType2 = new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)
    val tensorType2 = FixedShapeTensor(valueType2, Seq(2, 2))
    
    val idVector = new IntVector("id", allocator)
    val matrix1Vector = new Tensor("matrix1", allocator, tensorType1)
    val matrix2Vector = new Tensor("matrix2", allocator, tensorType2)
    
    val fields = List(idVector.getField, matrix1Vector.getField, matrix2Vector.getField)
    val schema = new Schema(fields.asJava)
    
    val root = new VectorSchemaRoot(schema, List(idVector, matrix1Vector, matrix2Vector).asJava, 0)
    data.toArrowVector(root)

    // Convert back to products
    val result: List[DataWithMultipleTensors] = root.toProducts
    
    assert(result.length == 1)
    assert(result(0).id == 1)
    assert(result(0).matrix1 == Seq(Seq(1.0f, 2.0f)))
    assert(result(0).matrix2 == Seq(Seq(10.0, 20.0), Seq(30.0, 40.0)))

    root.close()
    idVector.close()
    matrix1Vector.close()
    matrix2Vector.close()
  }

  // Test for round-trip conversion
  test("ToProduct round-trip: write and read back 2D tensor") {
    case class DataWithMatrix(id: Int, values: Seq[Seq[Float]])

    val original = Seq(
      DataWithMatrix(1, Seq(Seq(1.0f, 2.0f, 3.0f), Seq(4.0f, 5.0f, 6.0f))),
      DataWithMatrix(2, Seq(Seq(7.0f, 8.0f, 9.0f), Seq(10.0f, 11.0f, 12.0f)))
    )

    val allocator = new RootAllocator(Long.MaxValue)
    
    // Create tensor vectors manually
    val valueType = new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)
    val tensorType = FixedShapeTensor(valueType, Seq(2, 3))
    
    val idVector = new IntVector("id", allocator)
    val valuesVector = new Tensor("values", allocator, tensorType)
    
    val fields = List(idVector.getField, valuesVector.getField)
    val schema = new Schema(fields.asJava)
    
    val root = new VectorSchemaRoot(schema, List(idVector, valuesVector).asJava, 0)
    
    // Write data
    original.toArrowVector(root)

    // Read back
    val result: List[DataWithMatrix] = root.toProducts
    
    assert(result == original)

    root.close()
    idVector.close()
    valuesVector.close()
  }
}