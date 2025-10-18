package com.dyeru.asani.arrow

import org.apache.arrow.vector.types.FloatingPointPrecision
import org.apache.arrow.vector.types.pojo.ArrowType
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

class FixedShapeTensorTest extends AnyFunSuite with Matchers {

  test("Serialization: should serialize and deserialize metadata") {
    val tensor = FixedShapeTensor(
      new ArrowType.Int(32, true),
      Seq(2, 3, 4),
      Some(Seq("x", "y", "z")),
      None
    )

    val serialized = tensor.serialize()
    serialized should include("\"valueType\":\"int32\"")
    serialized should include("\"shape\":[2,3,4]")
    serialized should include("\"dimNames\":[\"x\",\"y\",\"z\"]")

    val deserialized = tensor.deserialize(tensor.storageType(), serialized).asInstanceOf[FixedShapeTensor]

    deserialized.getValueType shouldEqual new ArrowType.Int(32, true)
    deserialized.getShape shouldEqual Vector(2, 3, 4)
    deserialized.getDimNames shouldEqual Some(Vector("x", "y", "z"))
    deserialized.getPermutation shouldEqual None
    deserialized.getListSize shouldEqual 24
  }

  test("Serialization: should handle permutation and different value types") {
    val tensor = FixedShapeTensor(
      new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE),
      Seq(10, 20),
      None,
      Some(Seq(1, 0)) // Column-major permutation
    )

    val serialized = tensor.serialize()
    serialized should include("\"valueType\":\"float\"")
    serialized should include("\"shape\":[10,20]")
    serialized should include("\"permutation\":[1,0]")
    serialized should not include "dimNames"

    val deserialized = tensor.deserialize(tensor.storageType(), serialized).asInstanceOf[FixedShapeTensor]

    deserialized.getValueType shouldEqual new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)
    deserialized.getShape shouldEqual Vector(10, 20)
    deserialized.getDimNames shouldEqual None
    deserialized.getPermutation shouldEqual Some(Vector(1, 0))
    deserialized.getListSize shouldEqual 200
  }

  test("Validation: should reject invalid parameters") {
    // Empty shape
    assertThrows[IllegalArgumentException] {
      FixedShapeTensor(new ArrowType.Int(32, true), Seq.empty)
    }
    // Mismatched dim_names length
    assertThrows[IllegalArgumentException] {
      FixedShapeTensor(new ArrowType.Int(32, true), Seq(2, 3), Some(Seq("x")))
    }
    // Invalid permutation (duplicate)
    assertThrows[IllegalArgumentException] {
      FixedShapeTensor(new ArrowType.Int(32, true), Seq(2, 3), None, Some(Seq(0, 0)))
    }
    // Invalid permutation (out of range)
    assertThrows[IllegalArgumentException] {
      FixedShapeTensor(new ArrowType.Int(32, true), Seq(2, 3), None, Some(Seq(0, 2)))
    }
  }

  test("Logic: should calculate strides correctly") {
    val floatType = new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)
    val elementSize = 4 // sizeof(float)

    // Standard row-major layout
    val rowMajorTensor = FixedShapeTensor(floatType, Seq(10, 20))
    // Strides for shape (10, 20) should be [20 * 4, 4]
    rowMajorTensor.getStrides shouldEqual Vector(20 * elementSize, elementSize)

    // Permuted layout (column-major)
    val colMajorTensor = FixedShapeTensor(floatType, Seq(10, 20), None, Some(Seq(1, 0)))
    // Physical shape is (20, 10). Strides should be [10 * 4, 4]
    colMajorTensor.getStrides shouldEqual Vector(10 * elementSize, elementSize)

    // 3D tensor
    val tensor3D = FixedShapeTensor(floatType, Seq(2, 3, 5))
    // Strides for shape (2, 3, 5) should be [3 * 5 * 4, 5 * 4, 4]
    tensor3D.getStrides shouldEqual Vector(15 * elementSize, 5 * elementSize, elementSize)
  }
}
