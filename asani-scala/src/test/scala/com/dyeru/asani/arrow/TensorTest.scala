package com.dyeru.asani.arrow

import org.apache.arrow.memory.{ArrowBuf, RootAllocator}
import org.apache.arrow.vector.Float4Vector
import org.apache.arrow.vector.types.FloatingPointPrecision
import org.apache.arrow.vector.types.pojo.ArrowType
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

import java.nio.ByteOrder

class TensorTest extends AnyFunSuite with Matchers with BeforeAndAfterAll {

  val allocator = new RootAllocator(Long.MaxValue)

  override def beforeAll(): Unit = {
    FixedShapeTensor.register()
  }

  override def afterAll(): Unit = {
    allocator.close()
  }

  test("Vector: should write and read tensor data") {
    val tensorType = FixedShapeTensor(
      new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE),
      Seq(2, 3)
    )
    val tensorVector = new Tensor("tensor_vec", allocator, tensorType)

    val listSize = tensorType.getListSize
    val elementSize = 4

    tensorVector.allocateNew()
    val underlyingList = tensorVector.getUnderlyingVector
    val underlyingValues = underlyingList.getDataVector.asInstanceOf[Float4Vector]

    underlyingList.setNotNull(0)
    (0 until listSize).foreach { i => underlyingValues.set(i, i.toFloat) }

    underlyingList.setNotNull(1)
    (0 until listSize).foreach { i => underlyingValues.set(listSize + i, (i + 10).toFloat) }
    tensorVector.setValueCount(2)

    tensorVector.isNull(0) shouldBe false
    tensorVector.isNull(1) shouldBe false

    val buffer0 = tensorVector.getTensorBuffer(0)
    buffer0.capacity() shouldBe (listSize * elementSize)
    val bb0 = buffer0.nioBuffer().order(ByteOrder.LITTLE_ENDIAN)
    bb0.getFloat(0 * elementSize) shouldBe 0.0f
    bb0.getFloat(5 * elementSize) shouldBe 5.0f

    val obj1 = tensorVector.getObject(1).asInstanceOf[ArrowBuf]
    val bb1 = obj1.nioBuffer().order(ByteOrder.LITTLE_ENDIAN)
    bb1.capacity() shouldBe (listSize * elementSize)
    bb1.getFloat(0 * elementSize) shouldBe 10.0f
    bb1.getFloat(5 * elementSize) shouldBe 15.0f

    tensorVector.close()
  }

  test("Vector: should handle null values") {
    val tensorType = FixedShapeTensor(new ArrowType.Int(32, true), Seq(4))
    val tensorVector = new Tensor("tensor_vec_nulls", allocator, tensorType)

    tensorVector.allocateNew()
    val underlyingList = tensorVector.getUnderlyingVector

    underlyingList.setNotNull(0)
    // index 1 remains null
    underlyingList.setNotNull(2)

    tensorVector.setValueCount(3)

    tensorVector.isNull(0) shouldBe false
    tensorVector.isNull(1) shouldBe true
    tensorVector.getObject(1) shouldBe null

    an[IllegalStateException] should be thrownBy {
      tensorVector.getTensorBuffer(1)
    }

    tensorVector.close()
  }

  test("Vector: should calculate hash codes") {
    val tensorType = FixedShapeTensor(new ArrowType.Int(32, true), Seq(2))
    val tensorVector = new Tensor("tensor_vec_hash", allocator, tensorType)

    tensorVector.allocateNew()
    val underlyingList = tensorVector.getUnderlyingVector
    val underlying = underlyingList.getDataVector.asInstanceOf[org.apache.arrow.vector.IntVector]

    underlyingList.setNotNull(0)
    underlying.set(0, 10)
    underlying.set(1, 20)

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

    hash0 shouldEqual hash2
    hash0 should not equal hash1

    tensorVector.close()
  }
}