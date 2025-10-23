package com.dyeru.asani.arrow

import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector.*
import org.apache.arrow.vector.complex.ListVector
import org.apache.arrow.vector.types.pojo.Schema
import org.apache.arrow.vector.types.FloatingPointPrecision
import org.apache.arrow.vector.types.pojo.ArrowType
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll

import java.nio.ByteOrder
import scala.runtime.stdLibPatches.Predef.assert
import scala.jdk.CollectionConverters.*

case class EmployeeWithAddress(name: String, position: String, address: Address)

class ToVectorTest extends AnyFunSuite with BeforeAndAfterAll {

  override def beforeAll(): Unit = {
    FixedShapeTensor.register()
  }

  // Test case for Person case class
  test("toArrowVector should convert Person case class to Arrow Vector") {
    val people = Seq(Person("Alice", 30), Person("Bob", 25))

    val allocator = new RootAllocator(Long.MaxValue)
    val root = VectorSchemaRoot.create(ArrowSchema.derived[Person].schema, allocator)

    val vectorRoot = people.toArrowVector(root)

    // Assertions to check if the Arrow Vector contains expected values
    val nameVector = vectorRoot.getVector("name").asInstanceOf[VarCharVector]
    assert(new String(nameVector.get(0)) == "Alice")
    assert(new String(nameVector.get(1)) == "Bob")

    val ageVector = vectorRoot.getVector("age").asInstanceOf[IntVector]
    assert(ageVector.get(0) == 30)
    assert(ageVector.get(1) == 25)

    vectorRoot.close()
  }

  // Test case for Employee case class
  test("toArrowVector should convert Employee case class to Arrow Vector") {
    val employees = Seq(Employee("Bob", "Developer", 100000.0), Employee("Alice", "Manager", 120000.0))

    val allocator = new RootAllocator(Long.MaxValue)
    val root = VectorSchemaRoot.create(ArrowSchema.derived[Employee].schema, allocator)

    val vectorRoot = employees.toArrowVector(root)

    val nameVector = vectorRoot.getVector("name").asInstanceOf[VarCharVector]
    assert(new String(nameVector.get(0)) == "Bob")
    assert(new String(nameVector.get(1)) == "Alice")

    val positionVector = vectorRoot.getVector("position").asInstanceOf[VarCharVector]
    assert(new String(positionVector.get(0)) == "Developer")
    assert(new String(positionVector.get(1)) == "Manager")

    val salaryVector = vectorRoot.getVector("salary").asInstanceOf[Float8Vector]
    assert(salaryVector.get(0) == 100000.0)
    assert(salaryVector.get(1) == 120000.0)

    vectorRoot.close()
  }

  // Test case for Address case class
  test("toArrowVector should convert Address case class to Arrow Vector") {
    val addresses = Seq(Address("123 Main St", "Metropolis", "12345"), Address("456 Oak St", "Smalltown", "67890"))

    val allocator = new RootAllocator(Long.MaxValue)
    val root = VectorSchemaRoot.create(ArrowSchema.derived[Address].schema, allocator)

    val vectorRoot = addresses.toArrowVector(root)

    val streetVector = vectorRoot.getVector("street").asInstanceOf[VarCharVector]
    assert(new String(streetVector.get(0)) == "123 Main St")
    assert(new String(streetVector.get(1)) == "456 Oak St")

    val cityVector = vectorRoot.getVector("city").asInstanceOf[VarCharVector]
    assert(new String(cityVector.get(0)) == "Metropolis")
    assert(new String(cityVector.get(1)) == "Smalltown")

    val zipVector = vectorRoot.getVector("zip").asInstanceOf[VarCharVector]
    assert(new String(zipVector.get(0)) == "12345")
    assert(new String(zipVector.get(1)) == "67890")

    vectorRoot.close()
  }

  // Test case for empty list
  test("toArrowVector should return an empty vector for an empty list") {
    val emptyList = Seq.empty[Person]

    val allocator = new RootAllocator(Long.MaxValue)
    val root = VectorSchemaRoot.create(ArrowSchema.derived[Person].schema, allocator)

    val vectorRoot = emptyList.toArrowVector(root)

    assert(vectorRoot.getRowCount == 0)

    vectorRoot.close()
  }

  // Test case for Option type
  test("toArrowVector should handle Option type correctly") {
    case class PersonOpt(name: String, age: Option[Int])

    val people = Seq(PersonOpt("Alice", Some(30)), PersonOpt("Bob", None))

    val allocator = new RootAllocator(Long.MaxValue)
    val root = VectorSchemaRoot.create(ArrowSchema.derived[PersonOpt].schema, allocator)

    val vectorRoot = people.toArrowVector(root)

    val nameVector = vectorRoot.getVector("name").asInstanceOf[VarCharVector]
    assert(new String(nameVector.get(0)) == "Alice")
    assert(new String(nameVector.get(1)) == "Bob")

    // Handling the Option type for age
    val ageVector = vectorRoot.getVector("age").asInstanceOf[IntVector]
    assert(ageVector.get(0) == 30)
    assert(ageVector.isNull(1)) // Check that the second element is null (because age is None)

    vectorRoot.close()
  }

  // Test case for List[String] in a case class (no nesting)
  test("toArrowVector should handle List[String] correctly in a case class") {
    case class PersonWithNumbers(name: String, age: Int, phoneNumbers: List[String])

    // Sample data with List[String] field
    val people = Seq(
      PersonWithNumbers("Alice", 30, List("123-456-7890", "234-567-8901")),
      PersonWithNumbers("Bob", 25, List("987-654-3210"))
    )

    // Convert to Arrow Vector
    val allocator = new RootAllocator(Long.MaxValue)
    val root = VectorSchemaRoot.create(ArrowSchema.derived[PersonWithNumbers].schema, allocator)

    val vectorRoot = people.toArrowVector(root)

    // Access the 'phoneNumbers' field vector which will be a ListVector
    val phoneNumbersVector = vectorRoot.getVector("phoneNumbers").asInstanceOf[ListVector]

    // Verify the value count (the number of rows)
    assert(phoneNumbersVector.getValueCount == 2)

    // Test first record: Alice (with 2 phone numbers)
    val firstPhoneNumbers = phoneNumbersVector.getObject(0).toArray.map(_.toString).toList
    assert(firstPhoneNumbers == List("123-456-7890", "234-567-8901"))

    // Test second record: Bob (with 1 phone number)
    val secondPhoneNumbers = phoneNumbersVector.getObject(1).toArray.map(_.toString).toList
    assert(secondPhoneNumbers == List("987-654-3210"))

    vectorRoot.close()
  }

  // Test case for Array[Byte] type
  test("toArrowVector should handle Array[Byte] correctly in a case class") {
    case class PersonWithBytes(name: String, data: Array[Byte])

    // Sample data with Array[Byte] field
    val people = Seq(
      PersonWithBytes("Alice", Array(1, 2, 3, 4)),
      PersonWithBytes("Bob", Array(5, 6, 7))
    )

    // Convert to Arrow Vector
    val allocator = new RootAllocator(Long.MaxValue)
    val root = VectorSchemaRoot.create(ArrowSchema.derived[PersonWithBytes].schema, allocator)

    val vectorRoot = people.toArrowVector(root)

    // Access the 'data' field vector which will be a ListVector containing VarBinary
    val dataVector = vectorRoot.getVector("data").asInstanceOf[LargeVarBinaryVector]

    // Verify the value count (the number of rows)
    assert(dataVector.getValueCount == 2)

    // Test first record: Alice (with Array[Byte](1, 2, 3, 4))
    val firstData = dataVector.getObject(0)
    assert(firstData.sorted sameElements Array(1, 2, 3, 4).sorted)

    // Test second record: Bob (with Array[Byte](5, 6, 7))
    val secondData = dataVector.getObject(1)
    assert(secondData.sorted sameElements Array(5, 6, 7).sorted)

    vectorRoot.close()
  }

  // Test case for 1D Float Tensor (Seq[Float] is ListVector, not Tensor)
  test("toArrowVector should handle 1D Seq[Float] as ListVector") {
    case class DataWith1DSeq(id: Int, values: Seq[Float])

    val data = Seq(
      DataWith1DSeq(1, Seq(1.0f, 2.0f, 3.0f)),
      DataWith1DSeq(2, Seq(4.0f, 5.0f, 6.0f))
    )

    val allocator = new RootAllocator(Long.MaxValue)
    val root = VectorSchemaRoot.create(ArrowSchema.derived[DataWith1DSeq].schema, allocator)

    val vectorRoot = data.toArrowVector(root)

    // Verify the ID field
    val idVector = vectorRoot.getVector("id").asInstanceOf[IntVector]
    assert(idVector.get(0) == 1)
    assert(idVector.get(1) == 2)

    // Verify the list field (not a tensor, just a regular list)
    val listVector = vectorRoot.getVector("values").asInstanceOf[ListVector]
    assert(listVector.getValueCount == 2)

    // Test first record
    val firstValues = listVector.getObject(0).toArray.map(_.asInstanceOf[Float]).toList
    assert(firstValues == List(1.0f, 2.0f, 3.0f))

    // Test second record
    val secondValues = listVector.getObject(1).toArray.map(_.asInstanceOf[Float]).toList
    assert(secondValues == List(4.0f, 5.0f, 6.0f))

    vectorRoot.close()
  }

  // Test case for 2D Float Tensor with manual schema
  test("toArrowVector should handle 2D Float Tensor correctly") {
    case class DataWith2DTensor(id: Int, matrix: Seq[Seq[Float]])

    val data = Seq(
      DataWith2DTensor(1, Seq(Seq(1.0f, 2.0f), Seq(3.0f, 4.0f))),
      DataWith2DTensor(2, Seq(Seq(5.0f, 6.0f), Seq(7.0f, 8.0f)))
    )

    val allocator = new RootAllocator(Long.MaxValue)
    
    // Create tensor vectors manually and add them to VectorSchemaRoot
    val valueType = new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)
    val tensorType = FixedShapeTensor(valueType, Seq(2, 2))
    
    val idVector = new IntVector("id", allocator)
    val matrixVector = new Tensor("matrix", allocator, tensorType)
    
    import org.apache.arrow.vector.types.pojo.{Field, FieldType, Schema}
    val fields = List(idVector.getField, matrixVector.getField)
    val schema = new Schema(fields.asJava)
    
    val root = new VectorSchemaRoot(schema, List(idVector, matrixVector).asJava, 0)
    val vectorRoot = data.toArrowVector(root)

    // Verify the ID field
    val idVec = vectorRoot.getVector("id").asInstanceOf[IntVector]
    assert(idVec.get(0) == 1)
    assert(idVec.get(1) == 2)

    // Verify the tensor field
    val tensorVector = vectorRoot.getVector("matrix").asInstanceOf[Tensor]
    assert(tensorVector.getValueCount == 2)

    // Check tensor metadata - shape should be [2, 2]
    assert(tensorVector.extensionType.getShape == Vector(2, 2))
    assert(tensorVector.extensionType.getListSize == 4)

    // Read first tensor data (flattened row-major order: 1, 2, 3, 4)
    val buffer0 = tensorVector.getTensorBuffer(0)
    val bb0 = buffer0.nioBuffer().order(ByteOrder.LITTLE_ENDIAN)
    assert(bb0.getFloat(0 * 4) == 1.0f)
    assert(bb0.getFloat(1 * 4) == 2.0f)
    assert(bb0.getFloat(2 * 4) == 3.0f)
    assert(bb0.getFloat(3 * 4) == 4.0f)

    // Read second tensor data (flattened row-major order: 5, 6, 7, 8)
    val buffer1 = tensorVector.getTensorBuffer(1)
    val bb1 = buffer1.nioBuffer().order(ByteOrder.LITTLE_ENDIAN)
    assert(bb1.getFloat(0 * 4) == 5.0f)
    assert(bb1.getFloat(1 * 4) == 6.0f)
    assert(bb1.getFloat(2 * 4) == 7.0f)
    assert(bb1.getFloat(3 * 4) == 8.0f)

    vectorRoot.close()
    idVector.close()
    matrixVector.close()
  }

  // Test case for 3D Double Tensor with manual schema
  test("toArrowVector should handle 3D Double Tensor correctly") {
    case class DataWith3DTensor(id: String, tensor3d: Seq[Seq[Seq[Double]]])

    val data = Seq(
      DataWith3DTensor("A", Seq(
        Seq(Seq(1.0, 2.0), Seq(3.0, 4.0)),
        Seq(Seq(5.0, 6.0), Seq(7.0, 8.0))
      ))
    )

    val allocator = new RootAllocator(Long.MaxValue)
    
    // Create tensor vectors manually
    val valueType = new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)
    val tensorType = FixedShapeTensor(valueType, Seq(2, 2, 2))
    
    import org.apache.arrow.vector.types.pojo.Schema
    val idVector = new VarCharVector("id", allocator)
    val tensorVector3d = new Tensor("tensor3d", allocator, tensorType)
    
    val fields = List(idVector.getField, tensorVector3d.getField)
    val schema = new Schema(fields.asJava)
    
    val root = new VectorSchemaRoot(schema, List(idVector, tensorVector3d).asJava, 0)
    val vectorRoot = data.toArrowVector(root)

    // Verify the tensor field
    val tensorVector = vectorRoot.getVector("tensor3d").asInstanceOf[Tensor]
    assert(tensorVector.getValueCount == 1)

    // Check tensor metadata - shape should be [2, 2, 2]
    assert(tensorVector.extensionType.getShape == Vector(2, 2, 2))
    assert(tensorVector.extensionType.getListSize == 8)

    // Read tensor data (flattened row-major order: 1, 2, 3, 4, 5, 6, 7, 8)
    val buffer0 = tensorVector.getTensorBuffer(0)
    val bb0 = buffer0.nioBuffer().order(ByteOrder.LITTLE_ENDIAN)
    assert(bb0.getDouble(0 * 8) == 1.0)
    assert(bb0.getDouble(1 * 8) == 2.0)
    assert(bb0.getDouble(2 * 8) == 3.0)
    assert(bb0.getDouble(3 * 8) == 4.0)
    assert(bb0.getDouble(4 * 8) == 5.0)
    assert(bb0.getDouble(5 * 8) == 6.0)
    assert(bb0.getDouble(6 * 8) == 7.0)
    assert(bb0.getDouble(7 * 8) == 8.0)

    vectorRoot.close()
    idVector.close()
    tensorVector3d.close()
  }

  // Test case for Int Seq as ListVector (1D sequence)
  test("toArrowVector should handle Seq[Int] as ListVector") {
    case class DataWithIntSeq(name: String, values: Seq[Int])

    val data = Seq(
      DataWithIntSeq("Alice", Seq(10, 20, 30, 40)),
      DataWithIntSeq("Bob", Seq(50, 60, 70, 80))
    )

    val allocator = new RootAllocator(Long.MaxValue)
    val root = VectorSchemaRoot.create(ArrowSchema.derived[DataWithIntSeq].schema, allocator)

    val vectorRoot = data.toArrowVector(root)

    // Verify the list field (not a tensor, just a regular list)
    val listVector = vectorRoot.getVector("values").asInstanceOf[ListVector]
    assert(listVector.getValueCount == 2)

    // Read first list data
    val firstValues = listVector.getObject(0).toArray.map(_.asInstanceOf[Int]).toList
    assert(firstValues == List(10, 20, 30, 40))

    vectorRoot.close()
  }

  // Test case for multiple tensors in same case class with manual schema
  test("toArrowVector should handle multiple Tensors in same case class") {
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
    
    import org.apache.arrow.vector.types.pojo.Schema
    val idVector = new IntVector("id", allocator)
    val matrix1Vector = new Tensor("matrix1", allocator, tensorType1)
    val matrix2Vector = new Tensor("matrix2", allocator, tensorType2)
    
    val fields = List(idVector.getField, matrix1Vector.getField, matrix2Vector.getField)
    val schema = new Schema(fields.asJava)
    
    val root = new VectorSchemaRoot(schema, List(idVector, matrix1Vector, matrix2Vector).asJava, 0)
    val vectorRoot = data.toArrowVector(root)

    // Verify first tensor field
    val tensorVector1 = vectorRoot.getVector("matrix1").asInstanceOf[Tensor]
    assert(tensorVector1.extensionType.getShape == Vector(1, 2))
    val buffer1 = tensorVector1.getTensorBuffer(0)
    val bb1 = buffer1.nioBuffer().order(ByteOrder.LITTLE_ENDIAN)
    assert(bb1.getFloat(0 * 4) == 1.0f)
    assert(bb1.getFloat(1 * 4) == 2.0f)

    // Verify second tensor field
    val tensorVector2 = vectorRoot.getVector("matrix2").asInstanceOf[Tensor]
    assert(tensorVector2.extensionType.getShape == Vector(2, 2))
    val buffer2 = tensorVector2.getTensorBuffer(0)
    val bb2 = buffer2.nioBuffer().order(ByteOrder.LITTLE_ENDIAN)
    assert(bb2.getDouble(0 * 8) == 10.0)
    assert(bb2.getDouble(1 * 8) == 20.0)
    assert(bb2.getDouble(2 * 8) == 30.0)
    assert(bb2.getDouble(3 * 8) == 40.0)

    vectorRoot.close()
    idVector.close()
    matrix1Vector.close()
    matrix2Vector.close()
  }
}