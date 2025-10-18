package com.dyeru.asani.arrow

import org.apache.arrow.memory.{ArrowBuf, BufferAllocator}
import org.apache.arrow.memory.util.hash.ArrowBufHasher
import org.apache.arrow.vector.ExtensionTypeVector
import org.apache.arrow.vector.complex.FixedSizeListVector
import org.apache.arrow.vector.types.pojo.{ArrowType, Field, FieldType}

import scala.jdk.CollectionConverters.*

class FixedShapeTensorVector(
    name: String,
    allocator: BufferAllocator,
    val extensionType: FixedShapeTensor
) extends ExtensionTypeVector[FixedSizeListVector](
      name,
      allocator,
      FixedShapeTensorVector
        .createUnderlyingVector(name, allocator, extensionType)
    ) {

  /** Provides access to the raw buffer for a single tensor at the given index.
    *
    * @param index
    *   The row index of the tensor.
    * @return
    *   An ArrowBuf slice containing the raw bytes of the tensor.
    */
  def getTensorBuffer(index: Int): ArrowBuf = {
    val underlying = getUnderlyingVector
    if (index < 0 || index >= underlying.getValueCount || isNull(index)) {
      allocator.getEmpty
    } else {
      val elementSize =
        FixedShapeTensor.getElementSize(extensionType.getValueType)

      val listSize = underlying.getListSize
      val offset = index * listSize * elementSize
      val bytesPerTensor = listSize * elementSize

      // Slice the data buffer of the *child* vector
      underlying.getDataVector.getDataBuffer.slice(offset, bytesPerTensor)
    }
  }

  override def getObject(index: Int): AnyRef = {
    if (isNull(index)) {
      null
    } else {
      getTensorBuffer(index).nioBuffer()
    }
  }

  override def hashCode(index: Int): Int = {
    getUnderlyingVector.hashCode(index)
  }

  override def hashCode(index: Int, hasher: ArrowBufHasher): Int = {
    getUnderlyingVector.hashCode(index, hasher)
  }
}

object FixedShapeTensorVector {

  /** A helper method to construct the underlying physical storage vector. A
    * tensor is stored as a `FixedSizeList` of its base `valueType`.
    */
  private def createUnderlyingVector(
      name: String,
      allocator: BufferAllocator,
      extensionType: FixedShapeTensor
  ): FixedSizeListVector = {
    val valueType = extensionType.getValueType
    val listSize = extensionType.getListSize

    val childField =
      new Field("element", new FieldType(true, valueType, null), null)

    val listType = new ArrowType.FixedSizeList(listSize)
    // IMPORTANT: The field metadata for the extension type must be set on the storage vector's field
    val metadata = Map(
      "ARROW:extension:name" -> extensionType.extensionName(),
      "ARROW:extension:metadata" -> extensionType.serialize()
    ).asJava

    val listFieldType = new FieldType(true, listType, null, metadata)
    val listField = new Field(name, listFieldType, List(childField).asJava)
    val vector =
      listField.createVector(allocator).asInstanceOf[FixedSizeListVector]
    vector
  }
}
