package com.dyeru.asani.arrow

import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector.Float4Vector
import org.apache.arrow.vector.types.FloatingPointPrecision
import org.apache.arrow.vector.types.pojo.ArrowType
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers
import java.nio.ByteBuffer
import java.nio.ByteOrder

class FixedShapeTensorVectorTest extends AnyFunSuite with Matchers with BeforeAndAfterAll {

  val allocator = new RootAllocator(Long.MaxValue)

  override def beforeAll(): Unit = {
    // Register the extension type for all tests in this suite
    FixedShapeTensor.register()
  }

  override def afterAll(): Unit = {
    allocator.close()
  }

  test("Vector: should write and read tensor data") {
    val tensorType = FixedShapeTensor(
      new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE),
      Seq(2, 3) // Each tensor is a 2x3 matrix of floats
    )
    val tensorVector = new FixedShapeTensorVector("tensor_vec", allocator, tensorType)

    val listSize = tensorType.getListSize // 2 * 3 = 6
    val elementSize = 4 // sizeof(float)

    // --- Write Data ---
    tensorVector.allocateNew()
    val underlyingList = tensorVector.getUnderlyingVector
    val underlyingValues = underlyingList.getDataVector.asInstanceOf[Float4Vector]

    // Write tensor for index 0
    underlyingList.setNotNull(0) // FIX: Mark index 0 as valid in the parent vector
    (0 until listSize).foreach { i => underlyingValues.set(i, i.toFloat) }

    // Write tensor for index 1
    underlyingList.setNotNull(1) // FIX: Mark index 1 as valid in the parent vector
    (0 until listSize).foreach { i => underlyingValues.set(listSize + i, (i + 10).toFloat) }
    tensorVector.setValueCount(2)

    // --- Read and Verify Data ---
    tensorVector.isNull(0) shouldBe false
    tensorVector.isNull(1) shouldBe false

    // Verify tensor 0 via getTensorBuffer
    val buffer0 = tensorVector.getTensorBuffer(0)
    buffer0.capacity() shouldBe (listSize * elementSize)
    val bb0 = buffer0.nioBuffer().order(ByteOrder.LITTLE_ENDIAN)
    bb0.getFloat(0 * elementSize) shouldBe 0.0f
    bb0.getFloat(5 * elementSize) shouldBe 5.0f

    // Verify tensor 1 via getObject
    val obj1 = tensorVector.getObject(1).asInstanceOf[ByteBuffer].order(ByteOrder.LITTLE_ENDIAN)
    obj1.capacity() shouldBe (listSize * elementSize)
    obj1.getFloat(0 * elementSize) shouldBe 10.0f
    obj1.getFloat(5 * elementSize) shouldBe 15.0f

    tensorVector.close()
  }

  test("Vector: should handle null values") {
    val tensorType = FixedShapeTensor(new ArrowType.Int(32, true), Seq(4))
    val tensorVector = new FixedShapeTensorVector("tensor_vec_nulls", allocator, tensorType)

    tensorVector.allocateNew()
    val underlyingList = tensorVector.getUnderlyingVector

    // A new vector starts as all-null
    underlyingList.setNotNull(0)
    // index 1 remains null
    underlyingList.setNotNull(2)

    tensorVector.setValueCount(3) // We have values at 0 (valid), 1 (null), 2 (valid)

    tensorVector.isNull(0) shouldBe false
    tensorVector.isNull(1) shouldBe true
    tensorVector.getObject(1) shouldBe null
    tensorVector.getTensorBuffer(1).capacity() shouldBe 0

    tensorVector.close()
  }

  test("Vector: should calculate hash codes") {
    val tensorType = FixedShapeTensor(new ArrowType.Int(32, true), Seq(2))
    val tensorVector = new FixedShapeTensorVector("tensor_vec_hash", allocator, tensorType)

    tensorVector.allocateNew()
    val underlyingList = tensorVector.getUnderlyingVector
    val underlying = underlyingList.getDataVector.asInstanceOf[org.apache.arrow.vector.IntVector]

    // Two identical tensors at index 0 and 2
    underlyingList.setNotNull(0)
    underlying.set(0, 10)
    underlying.set(1, 20)

    // Different tensor at index 1
    underlyingList.setNotNull(1)
    underlying.set(2, 100)
    underlying.set(3, 200)

    underlyingList.setNotNull(2)
    underlying.set(4, 10)
    underlying.set(5, 20)

    tensorVector.setValueCount(3)

    val hash0 = tensorVector.hashCode(0)
    val hash1 = tensorVector.hashCode(1)
    val hash2 = tensorVector.hashCode(2)

    hash0 should not be 0
    hash1 should not be 0

    hash0 shouldEqual hash2  // Identical tensors should have the same hash code
    hash0 should not equal hash1 // Different tensors should have different hash codes

    tensorVector.close()
  }
}