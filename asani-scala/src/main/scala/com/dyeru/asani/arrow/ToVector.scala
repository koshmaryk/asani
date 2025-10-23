package com.dyeru.asani.arrow

import org.apache.arrow.vector.*
import org.apache.arrow.vector.complex.ListVector
import org.apache.arrow.vector.complex.impl.UnionListWriter
import org.apache.arrow.vector.types.pojo.{Field, FieldType}

import java.nio.charset.StandardCharsets
import java.time.Instant
import scala.annotation.tailrec
import scala.collection.mutable.ArrayBuffer
import scala.deriving.Mirror
import scala.jdk.CollectionConverters.*

extension [F[_] <: Seq[_], T: Mirror.ProductOf](values: F[T])(using instance: ToVector[T])
  def toArrowVector(root: VectorSchemaRoot): VectorSchemaRoot =
    instance.toVector(values.asInstanceOf[Seq[T]], root)

trait ToVector[T] {
  def toVector(values: Seq[T], root: VectorSchemaRoot): VectorSchemaRoot
}

object ToVector {
  def apply[T](using toVector: ToVector[T]): ToVector[T] = toVector

  private inline def derived[T](using p: Mirror.ProductOf[T]): ToVector[T] = {
    (values: Seq[T], root: VectorSchemaRoot) => {

      val records = values.map(implicitly[ToMap[T]].toMap)

      root.allocateNew()

      root
        .getFieldVectors
        .asScala
        .foreach(vector =>
          records
            .map(_(vector.getName))
            .zipWithIndex
            .foreach((value, index) => setField(vector, index, value))

          vector.setValueCount(records.length)
        )

      root.setRowCount(values.length)
      root
    }
  }

  //@tailrec
  private def getTensorShape(data: Any, shape: Vector[Int] = Vector.empty): (Vector[Int], Any) = {
    data match {
      case s: Seq[_] if s.nonEmpty =>
        val (innerShape, firstElement) = getTensorShape(s.head)
        s.tail.foreach { item =>
          val (shape, _) = getTensorShape(item)
          require(shape == innerShape, s"Tensor dimensions are not uniform. Expected shape $innerShape, got $shape")
        }
        (s.length +: innerShape, firstElement)
      case s: Seq[_] if s.isEmpty => (shape :+ 0, null)
      case other => (shape, other)
    }
  }

  private def flattenTensor(data: Any, buffer: ArrayBuffer[Any]): Unit = {
    data match {
      case seq: Seq[_] => seq.foreach(flattenTensor(_, buffer))
      case other => buffer += other
    }
  }

  //@tailrec
  private def setField(vector: FieldVector, index: Int, value: Any): Unit =
    value match {
      case v: Int => vector.asInstanceOf[IntVector].setSafe(index, v)
      case v: Long => vector.asInstanceOf[BigIntVector].setSafe(index, v)
      case v: String => vector.asInstanceOf[VarCharVector].setSafe(index, v.getBytes(StandardCharsets.UTF_8))
      case v: Double => vector.asInstanceOf[Float8Vector].setSafe(index, v)
      case v: Float => vector.asInstanceOf[Float4Vector].setSafe(index, v)
      case v: Boolean => vector.asInstanceOf[BitVector].setSafe(index, if v then 1 else 0)
      case v: Array[Byte] => vector.asInstanceOf[LargeVarBinaryVector].setSafe(index, v)
      case v: Instant => vector.asInstanceOf[TimeStampMilliVector].setSafe(index, v.toEpochMilli)

      case v: Seq[_] =>
        vector match {
          case listVector: ListVector =>
            val writer = listVector.getWriter
            writer.setPosition(index)
            writer.startList()
            v.foreach(element => writeListRecursive(writer, element))
            writer.endList()

          case tensorVector: Tensor =>
            val buffer = ArrayBuffer[Any]()
            flattenTensor(v, buffer)
            
            val underlyingList = tensorVector.getUnderlyingVector
            val dataVector = underlyingList.getDataVector
            
            underlyingList.setNotNull(index)
            
            // Calculate the starting offset for this tensor in the flat data vector
            val listSize = tensorVector.extensionType.getListSize
            val offset = index * listSize
            
            buffer.zipWithIndex.foreach { case (element, i) =>
              writeTensorElement(dataVector, offset + i, element)
            }

          case _ =>
            throw new IllegalArgumentException(s"Unsupported vector type ${vector.getClass.getName} for Seq data")
        }
      case v: Option[_] => v match
        case Some(iv) => setField(vector, index, iv)
        case None => vector.setNull(index)
    }
  
  @tailrec
  private def writeListRecursive(writer: UnionListWriter, value: Any): Unit = {
    value match {
      case v: Int => writer.writeInt(v)
      case v: Long => writer.writeBigInt(v)
      case v: Float => writer.writeFloat4(v)
      case v: Double => writer.writeFloat8(v)
      case v: Boolean => writer.writeBit(if v then 1 else 0)
      case v: String => writer.writeVarChar(v)
      case v: Instant => writer.writeBigInt(v.toEpochMilli)
      case v: Option[_] => v match
        case Some(inner) => writeListRecursive(writer, inner)
        case None => writer.writeNull()
      case other => throw new IllegalArgumentException(s"Unsupported type in list: ${other.getClass}")
    }
  }

  private def writeTensorElement(dataVector: FieldVector, index: Int, value: Any): Unit = {
    value match {
      case v: Int => dataVector.asInstanceOf[IntVector].set(index, v)
      case v: Long => dataVector.asInstanceOf[BigIntVector].set(index, v)
      case v: Float => dataVector.asInstanceOf[Float4Vector].set(index, v)
      case v: Double => dataVector.asInstanceOf[Float8Vector].set(index, v)
      case v: Boolean => dataVector.asInstanceOf[BitVector].set(index, if v then 1 else 0)
      case other => throw new IllegalArgumentException(s"Unsupported type for tensor element: ${other.getClass}")
    }
  }

  // Enable derivation for case classes
  inline given derivedToVector[T](using m: Mirror.ProductOf[T]): ToVector[T] = derived
}
