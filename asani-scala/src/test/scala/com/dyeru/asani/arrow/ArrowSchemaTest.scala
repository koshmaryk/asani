package com.dyeru.asani.arrow

import org.apache.arrow.vector.types.pojo.{ArrowType, Field, FieldType, Schema}
import org.apache.arrow.vector.types.Types
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers
import scala.jdk.CollectionConverters.*

class ArrowSchemaTest extends AnyFunSuite with Matchers {

  test("schema should handle int type") {
    case class TestCase(id: Int)
    val schema: Schema = ArrowSchema.derived[TestCase].schema
    val field          = schema.getFields.get(0)

    field.getName shouldBe "id"
    field.getType shouldBe Types.MinorType.INT.getType
    field.isNullable shouldBe false
  }

  test("schema should handle long type") {
    case class TestCase(id: Long)
    val schema: Schema = ArrowSchema.derived[TestCase].schema
    val field          = schema.getFields.get(0)

    field.getName shouldBe "id"
    field.getType shouldBe Types.MinorType.BIGINT.getType
    field.isNullable shouldBe false
  }

  test("schema should handle String type") {
    case class TestCase(name: String)
    val schema: Schema = ArrowSchema.derived[TestCase].schema
    val field          = schema.getFields.get(0)

    field.getName shouldBe "name"
    field.getType shouldBe Types.MinorType.VARCHAR.getType
    field.isNullable shouldBe false
  }

  test("schema should handle double type") {
    case class TestCase(value: Double)
    val schema: Schema = ArrowSchema.derived[TestCase].schema
    val field          = schema.getFields.get(0)

    field.getName shouldBe "value"
    field.getType shouldBe Types.MinorType.FLOAT8.getType
    field.isNullable shouldBe false
  }

  test("schema should handle float type") {
    case class TestCase(value: Float)
    val schema: Schema = ArrowSchema.derived[TestCase].schema
    val field          = schema.getFields.get(0)

    field.getName shouldBe "value"
    field.getType shouldBe Types.MinorType.FLOAT4.getType
    field.isNullable shouldBe false
  }

  test("schema should handle boolean type") {
    case class TestCase(flag: Boolean)
    val schema: Schema = ArrowSchema.derived[TestCase].schema
    val field          = schema.getFields.get(0)

    field.getName shouldBe "flag"
    field.getType shouldBe Types.MinorType.BIT.getType
    field.isNullable shouldBe false
  }

  test("schema should handle byte array type") {
    case class TestCase(data: Array[Byte])
    val schema: Schema = ArrowSchema.derived[TestCase].schema
    val field          = schema.getFields.get(0)

    field.getName shouldBe "data"
    field.getType shouldBe Types.MinorType.LARGEVARBINARY.getType
    field.isNullable shouldBe false
  }

  test("schema should handle Instant type") {
    case class TestCase(timestamp: java.time.Instant)
    val schema: Schema = ArrowSchema.derived[TestCase].schema
    val field          = schema.getFields.get(0)

    field.getName shouldBe "timestamp"
    field.getType shouldBe Types.MinorType.TIMESTAMPMILLI.getType
    field.isNullable shouldBe false
  }

  test("schema should handle Option[type] correctly") {
    case class TestCase(value: Option[Int])
    val schema: Schema = ArrowSchema.derived[TestCase].schema
    val field          = schema.getFields.get(0)

    field.getName shouldBe "value"
    field.isNullable shouldBe true
    field.getType shouldBe Types.MinorType.INT.getType
  }

  test("schema should handle List[String] correctly (with child check)") {
    case class TestCase(values: List[String])
    val schema: Schema = ArrowSchema.derived[TestCase].schema
    val field          = schema.getFields.get(0)

    field.getName shouldBe "values"
    field.isNullable shouldBe true // The list field itself is nullable
    field.getType shouldBe ArrowType.List()

    // Verify the child field
    field.getChildren.size() shouldBe 1
    val childField = field.getChildren.get(0)
    childField.getName shouldBe "item"
    childField.isNullable shouldBe true // Elements inside a list are always nullable
    childField.getType shouldBe Types.MinorType.VARCHAR.getType
  }

  test("schema should handle List[Int] correctly") {
    case class TestCase(values: List[Int])
    val schema: Schema = ArrowSchema.derived[TestCase].schema
    val field          = schema.getFields.get(0)

    field.getName shouldBe "values"
    field.isNullable shouldBe true
    field.getType shouldBe ArrowType.List()

    // Verify the child field
    field.getChildren.size() shouldBe 1
    val childField = field.getChildren.get(0)
    childField.getName shouldBe "item"
    childField.isNullable shouldBe true
    childField.getType shouldBe Types.MinorType.INT.getType
  }

  test("schema should handle List[Double] correctly") {
    case class TestCase(values: List[Double])
    val schema: Schema = ArrowSchema.derived[TestCase].schema
    val field          = schema.getFields.get(0)

    field.getName shouldBe "values"
    field.isNullable shouldBe true
    field.getType shouldBe ArrowType.List()

    // Verify the child field
    field.getChildren.size() shouldBe 1
    val childField = field.getChildren.get(0)
    childField.getName shouldBe "item"
    childField.isNullable shouldBe true
    childField.getType shouldBe Types.MinorType.FLOAT8.getType
  }

  test("schema should handle Option[List[Int]] correctly") {
    case class TestCase(values: Option[List[Int]])
    val schema: Schema = ArrowSchema.derived[TestCase].schema
    val field          = schema.getFields.get(0)

    // The list field itself is nullable
    field.getName shouldBe "values"
    field.isNullable shouldBe true
    field.getType shouldBe ArrowType.List()

    // Verify the child field
    field.getChildren.size() shouldBe 1
    val childField = field.getChildren.get(0)
    childField.getName shouldBe "item"
    childField.isNullable shouldBe true
    childField.getType shouldBe Types.MinorType.INT.getType
  }

  test("schema should handle List[Option[String]] correctly") {
    case class TestCase(values: List[Option[String]])
    val schema: Schema = ArrowSchema.derived[TestCase].schema
    val field          = schema.getFields.get(0)

    field.getName shouldBe "values"
    field.isNullable shouldBe true
    field.getType shouldBe ArrowType.List()

    // Verify the child field (which should be a nullable string)
    field.getChildren.size() shouldBe 1
    val childField = field.getChildren.get(0)
    childField.getName shouldBe "item"
    childField.isNullable shouldBe true
    childField.getType shouldBe Types.MinorType.VARCHAR.getType
  }

  test("schema should handle multiple fields correctly") {
    case class TestCase(id: Int, name: String, timestamp: java.time.Instant)
    val schema: Schema = ArrowSchema.derived[TestCase].schema

    // Verify the first field (id)
    val field1 = schema.getFields.get(0)
    field1.getName shouldBe "id"
    field1.getType shouldBe Types.MinorType.INT.getType

    // Verify the second field (name)
    val field2 = schema.getFields.get(1)
    field2.getName shouldBe "name"
    field2.getType shouldBe Types.MinorType.VARCHAR.getType

    // Verify the third field (timestamp)
    val field3 = schema.getFields.get(2)
    field3.getName shouldBe "timestamp"
    field3.getType shouldBe Types.MinorType.TIMESTAMPMILLI.getType
  }
}
