package com.dyeru.asani.arrow

import org.apache.arrow.memory.{ArrowBuf, BufferAllocator}
import org.apache.arrow.memory.util.hash.ArrowBufHasher
import org.apache.arrow.vector.ExtensionTypeVector
import org.apache.arrow.vector.complex.FixedSizeListVector
import org.apache.arrow.vector.types.pojo.{ArrowType, Field, FieldType}

import scala.jdk.CollectionConverters.*

class Tensor(
  name: String,
  allocator: BufferAllocator,
  val extensionType: FixedShapeTensor
) extends ExtensionTypeVector[FixedSizeListVector](
      name,
      allocator,
      Tensor
        .createUnderlyingVector(name, allocator, extensionType)
    ) {

  /**
   * Provides access to the raw buffer for a single tensor at the given index.
   *
   * @param index
   *   The row index of the tensor.
   * @return
   *   An ArrowBuf slice containing the raw bytes of the tensor.
   */
  def getTensorBuffer(index: Int): ArrowBuf = {
    val underlying = getUnderlyingVector

    if (index < 0 || index >= underlying.getValueCount) {
      throw new IndexOutOfBoundsException(
        s"Index $index out of bounds for vector with ${underlying.getValueCount} elements"
      )
    }

    if (isNull(index)) {
      throw new IllegalStateException(s"Tensor at index $index is null")
    }

    val elementSize = FixedShapeTensor.getElementSize(extensionType.getValueType)
    val dataVector  = underlying.getDataVector
    val dataBuffer  = dataVector.getDataBuffer

    val startOffset = underlying.getElementStartIndex(index)
    val endOffset   = underlying.getElementEndIndex(index)

    val byteOffset = startOffset * elementSize
    val byteLength = (endOffset - startOffset) * elementSize

    dataBuffer.slice(byteOffset, byteLength)
  }

  override def getObject(index: Int): AnyRef = {
    if (isNull(index)) {
      null
    } else {
      // Return ArrowBuf directly instead of wrapping in NIO buffer
      getTensorBuffer(index)
    }
  }

  override def hashCode(index: Int): Int = {
    getUnderlyingVector.hashCode(index)
  }

  override def hashCode(index: Int, hasher: ArrowBufHasher): Int = {
    getUnderlyingVector.hashCode(index, hasher)
  }
}

object Tensor {

  private val EXTENSION_NAME_KEY = "ARROW:extension:name"
  private val EXTENSION_METADATA_KEY = "ARROW:extension:metadata"

  /**
   * A helper method to construct the underlying physical storage vector. 
   * A tensor is stored as a FixedSizeList` of its base `valueType`.
   */
  private def createUnderlyingVector(
    name: String,
    allocator: BufferAllocator,
    extensionType: FixedShapeTensor
  ): FixedSizeListVector = {
    val valueType = extensionType.getValueType
    val listSize  = extensionType.getListSize

    val childField =
      new Field("element", new FieldType(true, valueType, null), null)

    val listType = new ArrowType.FixedSizeList(listSize)
    // IMPORTANT: The field metadata for the extension type must be set on the storage vector's field
    val metadata = Map(
      EXTENSION_NAME_KEY     -> extensionType.extensionName(),
      EXTENSION_METADATA_KEY -> extensionType.serialize()
    )

    val listFieldType = new FieldType(true, listType, null, metadata.asJava)
    val listField     = new Field(name, listFieldType, List(childField).asJava)
    val vector        =
      listField.createVector(allocator).asInstanceOf[FixedSizeListVector]
    vector
  }
}
