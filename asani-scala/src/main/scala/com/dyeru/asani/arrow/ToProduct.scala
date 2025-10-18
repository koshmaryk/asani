package com.dyeru.asani.arrow

import org.apache.arrow.vector.VectorSchemaRoot
import org.apache.arrow.vector.util.{JsonStringArrayList, Text}

import java.time.{LocalDateTime, ZoneOffset}
import scala.compiletime.erasedValue
import scala.deriving.Mirror
import scala.jdk.CollectionConverters.*

extension [T](root: VectorSchemaRoot)(using instance: ToProduct[T])
  def toProducts: List[T] = instance.toProducts(root)

trait ToProduct[T] {
  def toProducts(root: VectorSchemaRoot): List[T]
}

object ToProduct {
  def apply[T](using toProduct: ToProduct[T]): ToProduct[T] = toProduct

  private inline def derived[T](using p: Mirror.ProductOf[T]): ToProduct[T] = {
    (root: VectorSchemaRoot) =>
      (0 until root.getRowCount)
        .map { idx =>
          val values = root.getSchema.getFields.asScala
            .map(f => root.getVector(f.getName).getObject(idx))
            .toList

          p.fromProduct(listToTuple[p.MirroredElemTypes](values))
        }.toList
  }

  private inline def listToTuple[Tup <: Tuple](list: List[Any]): Tup =
    inline erasedValue[Tup] match {
      case _: EmptyTuple     => EmptyTuple.asInstanceOf[Tup]
      case _: (head *: tail) => (mapValue[head](list.head).asInstanceOf[head] *: listToTuple[tail](
          list.tail
        )).asInstanceOf[Tup]
    }

  private inline def mapValue[T](value: Any): Any =
    inline erasedValue[T] match {
      case _: Option[t] => if value != null then Some(mapValue(value).asInstanceOf[t]) else None
      case _: Seq[t]    => value.asInstanceOf[JsonStringArrayList[_]].toArray.toList.map(mapValue[t])
      case _: Any       => value match {
          case v: Text          => v.toString
          case v: LocalDateTime => v.toInstant(ZoneOffset.UTC)
          case v: Any           => v
        }
    }

  // Enable derivation for case classes
  inline given derivedToProduct[T](using m: Mirror.ProductOf[T]): ToProduct[T] = derived
}
