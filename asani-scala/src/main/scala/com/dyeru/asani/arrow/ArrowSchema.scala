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

  inline def derived[T](using p: Mirror.Of[T]): ArrowSchema[T] =
    new ArrowSchema[T] {
      override def schema: Schema = {
        val labels = constValueTuple[p.MirroredElemLabels].toList.asInstanceOf[List[String]]
        val types  = getTypes[p.MirroredElemTypes]

        val arrowFields = labels.zip(types).map {
          case (name, tensor: FixedShapeTensor)               =>
            new Field(name, FieldType.nullable(tensor), null)
          case (name, (list: ArrowType.List, tpe: ArrowType)) =>
            new Field(
              name,
              FieldType.nullable(list),
              List(new Field("item", FieldType.nullable(tpe), null)).asJava
            )
          case (name, (_: ArrowType.Null, tpe: ArrowType))    =>
            new Field(name, FieldType.nullable(tpe), null)
          case (name, tpe: ArrowType)                         =>
            new Field(name, FieldType.notNullable(tpe), null)
          case (name, other)                                  =>
            throw new IllegalArgumentException(
              s"Field '$name': Not a supported Arrow type: $other"
            )
        }

        new Schema(arrowFields.asJava)
      }
    }

  private inline def getNestedSeqBaseType[T]: (ArrowType, Int) =
    inline erasedValue[T] match {
      case _: Array[Byte] => (Types.MinorType.LARGEVARBINARY.getType, 0)
      case _: Seq[t] =>
        val (base, depth) = getNestedSeqBaseType[t]
        (base, depth + 1)
      case _: Option[t] =>
        getNestedSeqBaseType[t]
      case _: Int         => (Types.MinorType.INT.getType, 0)
      case _: Long        => (Types.MinorType.BIGINT.getType, 0)
      case _: Double      => (Types.MinorType.FLOAT8.getType, 0)
      case _: Float       => (Types.MinorType.FLOAT4.getType, 0)
      case _: Boolean     => (Types.MinorType.BIT.getType, 0)
      case _: String      => (Types.MinorType.VARCHAR.getType, 0)
      case _: Array[Byte] => (Types.MinorType.LARGEVARBINARY.getType, 0)
      case _: Instant     => (Types.MinorType.TIMESTAMPMILLI.getType, 0)
      case _ => throw new IllegalArgumentException(s"Unsupported base type for tensor")
    }

  private inline def getTypes[T <: Tuple]: List[Any] =
    inline erasedValue[T] match {
      case _: EmptyTuple     => Nil
      case _: (head *: tail) => arrowType[head] :: getTypes[tail]
    }

  private inline def arrowType[T]: Any =
    inline erasedValue[T] match {
      case _: Int         => Types.MinorType.INT.getType
      case _: Long        => Types.MinorType.BIGINT.getType
      case _: Double      => Types.MinorType.FLOAT8.getType
      case _: Float       => Types.MinorType.FLOAT4.getType
      case _: Boolean     => Types.MinorType.BIT.getType
      case _: String      => Types.MinorType.VARCHAR.getType
      case _: Array[Byte] => Types.MinorType.LARGEVARBINARY.getType
      case _: Instant     => Types.MinorType.TIMESTAMPMILLI.getType
      case _: Option[t]   => (new ArrowType.Null(), arrowType[t])
      case _: Seq[t] =>
        getNestedSeqBaseType[t] match {
          case (arrowType, 0) =>
            (new ArrowType.List(), arrowType)
          case (arrowType, _) =>
            FixedShapeTensor(arrowType, Seq(1))
        }
      case t              => throw new IllegalArgumentException(s"Unsupported type: $t")
    }

  inline given derivedArrowSchema[T](using m: Mirror.ProductOf[T]): ArrowSchema[T] = derived
}
