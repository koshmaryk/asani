package com.dyeru.asani.arrow

import java.time.Instant

import scala.compiletime.{constValueTuple, erasedValue, summonInline}
import scala.deriving.Mirror
import scala.jdk.CollectionConverters.*

import org.apache.arrow.vector.types.Types
import org.apache.arrow.vector.types.pojo.{ArrowType, Field, FieldType, Schema}

trait ArrowSchema[T] {
  def schema: Schema
}

object ArrowSchema {

  inline def derived[T](
    using p: Mirror.Of[T]
  ): ArrowSchema[T] =
    new ArrowSchema[T] {
      override def schema: Schema = {
        val labels = constValueTuple[p.MirroredElemLabels].toList.asInstanceOf[List[String]]
        val types  = getTypes[p.MirroredElemTypes]

        val arrowFields = labels.zip(types).map {
          case (name, tensor: FixedShapeTensor)    =>
            new Field(name, FieldType.nullable(tensor.storageType()), null)
          case (name, (list: ArrowType.List, tpe: ArrowType)) =>
            new Field(
              name,
              FieldType.nullable(list),
              List(new Field("item", FieldType.nullable(tpe), null)).asJava
            )
          case (name, (_: ArrowType.Null, tpe: ArrowType))    =>
            new Field(name, FieldType.nullable(tpe), null)
          case (name, tpe: ArrowType)              =>
            new Field(name, FieldType.notNullable(tpe), null)
          case _                                   =>
            throw new IllegalArgumentException("Not supported Arrow type.")
        }

        new Schema(arrowFields.asJava)
      }
    }

  type TensorMetadata = (ArrowType, List[Int]) // (ValueType, Shape)

  private inline def getTensorMetadata[T]: Option[TensorMetadata] =
    inline erasedValue[T] match {
      case _: Seq[t] =>
        getTensorMetadata[t] match {
          // It's a nested sequence, prepend a dimension
          case Some((valueType, shape)) => Some((valueType, 1 :: shape))
          // It's the innermost sequence, determine the primitive type
          case None => primitiveArrowType[t].map(t => (t, List(1)))
        }
      case _ => None
    }

  private inline def getTypes[T <: Tuple]: List[Any] =
    inline erasedValue[T] match {
      case _: EmptyTuple => Nil
      case _: (head *: tail) => arrowType[head] :: getTypes[tail]
    }

  private inline def primitiveArrowType[T]: Option[ArrowType] =
    inline erasedValue[T] match {
      case _: Int => Some(Types.MinorType.INT.getType)
      case _: Long => Some(Types.MinorType.BIGINT.getType)
      case _: Double => Some(Types.MinorType.FLOAT8.getType)
      case _: Float => Some(Types.MinorType.FLOAT4.getType)
      case _: Boolean => Some(Types.MinorType.BIT.getType)
      case _ => None
    }

  private inline def arrowType[T]: Any =
    inline erasedValue[T] match {
      // First, check if it's a tensor-like structure
      case _: Seq[_] =>
        getTensorMetadata[T] match {
          case Some((valueType, shape)) =>
            // We found a tensor! Return a FixedShapeTensor instance.
            // Note: The shape is a placeholder here; the actual shape will be data-dependent.
            FixedShapeTensor(valueType, shape.toSeq)
          case None => new ArrowType.List() // Fallback for non-primitive lists
        }
      case _: Option[t] => (new ArrowType.Null(), arrowType[t])
      case _ => primitiveArrowType[T].getOrElse {
        inline erasedValue[T] match {
          case _: String => Types.MinorType.VARCHAR.getType
          case _: Array[Byte] => Types.MinorType.LARGEVARBINARY.getType
          case _: Instant => Types.MinorType.TIMESTAMPMILLI.getType
          case t => throw new IllegalArgumentException(s"Unsupported type: $t")
        }
      }
    }

  inline given derivedArrowSchema[T](using m: Mirror.ProductOf[T]): ArrowSchema[T] = derived
}
