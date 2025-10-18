error id: file://<WORKSPACE>/src/main/scala/com/dyeru/asani/arrow/ToVector.scala:`<none>`.
file://<WORKSPACE>/src/main/scala/com/dyeru/asani/arrow/ToVector.scala
empty definition using pc, found symbol in pc: `<none>`.
empty definition using semanticdb
empty definition using fallback
non-local guesses:
	 -org/apache/arrow/vector/i.
	 -org/apache/arrow/vector/i#
	 -org/apache/arrow/vector/i().
	 -scala/jdk/CollectionConverters.i.
	 -scala/jdk/CollectionConverters.i#
	 -scala/jdk/CollectionConverters.i().
	 -i.
	 -i#
	 -i().
	 -scala/Predef.i.
	 -scala/Predef.i#
	 -scala/Predef.i().
offset: 3555
uri: file://<WORKSPACE>/src/main/scala/com/dyeru/asani/arrow/ToVector.scala
text:
```scala
package com.dyeru.asani.arrow

import com.dyeru.asani.AsaniException
import org.apache.arrow.vector.*
import org.apache.arrow.vector.complex.{FixedSizeListVector, ListVector}
import org.apache.arrow.vector.complex.impl.UnionListWriter

import java.nio.charset.StandardCharsets
import java.time.Instant
import scala.annotation.tailrec
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

  private def getShapeAndFlatten(data: Seq[Any]): (List[Int], List[Any]) = {
    def loop(current: Any, dims: List[Long]): (List[Int], List[Any]) = {
      current match {
        case s: Seq[_] if s.nonEmpty && s.head.isInstanceOf[Seq[_]] =>
          val innerShapes = s.map(loop(_, Nil)._1)
          // For FixedShapeTensor, all inner shapes must be identical.
          if (innerShapes.distinct.size > 1) throw new AsaniException("Ragged tensors are not supported.")
          (s.length :: innerShapes.head, s.flatMap(loop(_, Nil)._2).toList)
        case s: Seq[_] => (List(s.length), s.toList)
        case _ => (Nil, List(current))
      }
    }

    loop(data, Nil)
  }


  @tailrec
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
          // Handle FixedShapeTensorVector
          case tensorVector: FixedShapeTensorVector =>
            val (shape, flattened) = getShapeAndFlatten(v)
            val tensorType = tensorVector.extensionType
            val underlyingValues = tensorVector.getUnderlyingVector.getDataVector
            val offset = index * tensorType.getListSize

            tensorVector.getUnderlyingVector.setNotNull(index)

            flattened.zipWithIndex.foreach { case (elem, i) =>
              // REUSE setField for the primitive elements, as requested.
              // This is safe because `flattened` contains no Seqs.
              setField(underlyingValues, offset + i@@, elem)
            }

          // Handle standard ListVector
          case listVector: ListVector =>
            val writer = listVector.getWriter
            writer.setPosition(index)
            writer.startList()
            v.foreach(elem => writeListRecursive(writer, elem))
            writer.endList()
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

  // Enable derivation for case classes
  inline given derivedToVector[T](using m: Mirror.ProductOf[T]): ToVector[T] = derived
}

```


#### Short summary: 

empty definition using pc, found symbol in pc: `<none>`.