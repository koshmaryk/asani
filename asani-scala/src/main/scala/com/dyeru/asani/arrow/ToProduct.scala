package com.dyeru.asani.arrow

import org.apache.arrow.memory.ArrowBuf
import org.apache.arrow.vector.VectorSchemaRoot
import org.apache.arrow.vector.types.pojo.ArrowType
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
            .map { f =>
              val vector = root.getVector(f.getName)
              val obj = vector.getObject(idx)
              // For Tensor vectors, we need to pass additional metadata
              vector match {
                case tensor: Tensor =>
                  (obj, tensor.extensionType)
                case _ =>
                  obj
              }
            }
            .toList

          p.fromProduct(listToTuple[p.MirroredElemTypes](values))
        }.toList
  }

  /**
   * Reconstructs a nested Seq[...] from a flat ArrowBuf and a shape vector.
   */
  private def unflattenTensor(buffer: ArrowBuf, shape: List[Int], valueType: ArrowType): Any = {
    val elementSize = FixedShapeTensor.getElementSize(valueType)
    val totalElements = shape.product

    // Read all elements from the buffer into a 1D iterator
    val iter = (0 until totalElements).iterator.map { i =>
      val offset = i * elementSize
      valueType match {
        case _: ArrowType.Int if elementSize == 4 => buffer.getInt(offset)
        case _: ArrowType.Int if elementSize == 8 => buffer.getLong(offset)
        case _: ArrowType.FloatingPoint if elementSize == 4 => buffer.getFloat(offset)
        case _: ArrowType.FloatingPoint if elementSize == 8 => buffer.getDouble(offset)
        case _: ArrowType.Bool => buffer.getByte(offset) == 1
        case _ => throw new IllegalArgumentException(s"Unsupported tensor value type for unflattening: $valueType")
      }
    }

    def buildNested(currentShape: List[Int]): Any = {
      currentShape match {
        case Nil => // Base case: read a single element
          iter.next()
        case dim :: rest => // Recursive case: build a Seq of this dimension
          (0 until dim).map(_ => buildNested(rest))
      }
    }

    buildNested(shape)
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
      case _: Option[t] => if value != null then Some(mapValue[t](value).asInstanceOf[t]) else None
      case _: Seq[t]    => 
        value match {
          // Handle Tensor: value is (ArrowBuf, FixedShapeTensor)
          case (buffer: ArrowBuf, tensorType: FixedShapeTensor) =>
            unflattenTensor(buffer, tensorType.getShape.toList, tensorType.getValueType)
          // Handle regular lists
          case list: JsonStringArrayList[_] =>
            list.toArray.toList.map(mapValue[t])
          case other =>
            throw new IllegalArgumentException(s"Unexpected value type for Seq: ${other.getClass}")
        }
      case _: Any       => value match {
          case v: Text          => v.toString
          case v: LocalDateTime => v.toInstant(ZoneOffset.UTC)
          // Handle tuples from Tensor (strip metadata)
          case (buf: ArrowBuf, _: FixedShapeTensor) => buf
          case v: Any           => v
        }
    }

  // Enable derivation for case classes
  inline given derivedToProduct[T](using m: Mirror.ProductOf[T]): ToProduct[T] = derived
}
