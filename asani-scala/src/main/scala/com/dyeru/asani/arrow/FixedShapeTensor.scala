package com.dyeru.asani.arrow

import com.github.plokhotnyuk.jsoniter_scala.core.*
import com.github.plokhotnyuk.jsoniter_scala.macros.*
import org.apache.arrow.memory.util.hash.ArrowBufHasher
import org.apache.arrow.memory.{ArrowBuf, BufferAllocator}
import org.apache.arrow.vector.{ExtensionTypeVector, FieldVector}
import org.apache.arrow.vector.complex.FixedSizeListVector
import org.apache.arrow.vector.types.FloatingPointPrecision
import org.apache.arrow.vector.types.pojo.{
  ArrowType,
  ExtensionTypeRegistry,
  Field,
  FieldType
}

import scala.jdk.CollectionConverters.*
import scala.util.{Failure, Success, Try}

class FixedShapeTensor private (
    private val valueType: ArrowType,
    private val shape: Vector[Int],
    private val dimNames: Option[Vector[String]] = None,
    private val permutation: Option[Vector[Int]] = None
) extends ArrowType.ExtensionType {

  import FixedShapeTensor.*

  require(shape.nonEmpty, "Shape must have at least one dimension")
  require(
    dimNames.forall(_.length == shape.length),
    "dim_names length must match shape length"
  )
  require(
    permutation.forall { p =>
      p.length == shape.length && p.sorted == shape.indices.toVector
    },
    "permutation must be a valid permutation of [0, 1, ..., N-1]"
  )

  def getValueType: ArrowType = valueType
  def getShape: Vector[Int] = shape
  def getDimNames: Option[Vector[String]] = dimNames
  def getPermutation: Option[Vector[Int]] = permutation
  def getNdim: Int = shape.length
  def getListSize: Int = shape.product

  def getStrides: Vector[Int] = {
    val elementSize = getElementSize(valueType)
    val physicalShape = permutation match {
      case Some(perm) =>
        // Apply permutation to get a physical layout
        val permuted = Array.ofDim[Int](shape.length)
        perm.zipWithIndex.foreach { case (logicalIdx, physicalIdx) =>
          permuted(physicalIdx) = shape(logicalIdx)
        }
        permuted.toVector
      case None => shape
    }

    // Compute row-major strides for physical shape
    val strides = Array.ofDim[Int](physicalShape.length)
    if (strides.nonEmpty) {
      strides(strides.length - 1) = elementSize
      for (i <- (strides.length - 2) to 0 by -1) {
        strides(i) = strides(i + 1) * physicalShape(i + 1)
      }
    }
    strides.toVector
  }

  override def storageType(): ArrowType =
    ArrowType.FixedSizeList(getListSize)

  override def extensionName(): String = "arrow.fixed_shape_tensor"

  override def extensionEquals(other: ArrowType.ExtensionType): Boolean = {
    other match {
      case that: FixedShapeTensor =>
        this.valueType == that.valueType &&
        this.shape == that.shape &&
        this.dimNames == that.dimNames &&
        this.permutation == that.permutation
      case _ => false
    }
  }

  override def toString: String = {
    val permStr = permutation
      .map(p => s", permutation=${p.mkString("[", ",", "]")}")
      .getOrElse("")
    val namesStr = dimNames
      .map(n => s", dim_names=${n.mkString("[", ",", "]")}")
      .getOrElse("")
    s"${extensionName()}[value_type=$valueType, shape=${shape.mkString("[", ",", "]")}$namesStr$permStr]"
  }

  override def serialize(): String = {
    val valueTypeStr = arrowTypeToString(valueType)
    val metadata =
      TensorMetadata(valueTypeStr, shape, dimNames, permutation)
    writeToString(metadata)
  }

  override def deserialize(
      storageType: ArrowType,
      serializedData: String
  ): ArrowType.ExtensionType = {
    Try(readFromString[TensorMetadata](serializedData)) match {
      case Success(metadata) =>
        storageType match {
          case fs: ArrowType.FixedSizeList =>
            val valueType = stringToArrowType(metadata.valueType)
            val tensor = FixedShapeTensor(
              valueType,
              metadata.shape.toSeq,
              metadata.dimNames.map(_.toSeq),
              metadata.permutation.map(_.toSeq)
            )
            require(
              fs.getListSize == tensor.getListSize,
              s"Storage list size ${fs.getListSize} does not match tensor element count ${tensor.getListSize}"
            )
            tensor
          case _ =>
            throw new IllegalArgumentException(
              s"Expected FixedSizeList storage type, got $storageType"
            )
        }
      case Failure(e) =>
        throw new IllegalArgumentException(
          s"Failed to parse metadata JSON: ${e.getMessage}",
          e
        )
    }
  }

  override def getNewVector(
      name: String,
      fieldType: FieldType,
      allocator: BufferAllocator
  ): FieldVector = {
    new Tensor(name, allocator, this)
  }
}

object FixedShapeTensor {

  private case class TensorMetadata(
      valueType: String,
      shape: Vector[Int],
      dimNames: Option[Vector[String]] = None,
      permutation: Option[Vector[Int]] = None
  )

  private given JsonValueCodec[TensorMetadata] = JsonCodecMaker.make

  def apply(
      valueType: ArrowType,
      shape: Seq[Int],
      dimNames: Option[Seq[String]] = None,
      permutation: Option[Seq[Int]] = None
  ): FixedShapeTensor = {
    new FixedShapeTensor(
      valueType,
      shape.toVector,
      dimNames.map(_.toVector),
      permutation.map(_.toVector)
    )
  }

  def register(): Unit = {
    // Register a dummy instance so Arrow can find the type and call its deserialize method.
    val dummyInstance = FixedShapeTensor(new ArrowType.Int(32, true), Seq(1))
    if (ExtensionTypeRegistry.lookup(dummyInstance.extensionName()) != null) {
      ExtensionTypeRegistry.register(dummyInstance)
    }
  }

  def getElementSize(arrowType: ArrowType): Int = arrowType match {
    case i: ArrowType.Int => i.getBitWidth / 8
    case f: ArrowType.FloatingPoint =>
      f.getPrecision match {
        case FloatingPointPrecision.HALF   => 2
        case FloatingPointPrecision.SINGLE => 4
        case FloatingPointPrecision.DOUBLE => 8
      }
    case _: ArrowType.Bool              => 1
    case fsb: ArrowType.FixedSizeBinary => fsb.getByteWidth
    case d: ArrowType.Decimal           => d.getBitWidth / 8
    case t: ArrowType.Timestamp         => 8 // Timestamps are typically 64-bit
    case _: ArrowType.Utf8 | _: ArrowType.Binary =>
      throw new IllegalArgumentException(
        "Variable-length types not supported for tensor elements"
      )
    case other =>
      throw new IllegalArgumentException(
        s"Unsupported value type for tensor: $other"
      )
  }

  private def arrowTypeToString(arrowType: ArrowType): String =
    arrowType match {
      case s: ArrowType.Bool => "bool"
      case i: ArrowType.Int => s"int${i.getBitWidth}"
      case f: ArrowType.FloatingPoint =>
        f.getPrecision match {
          case FloatingPointPrecision.SINGLE => "float"
          case FloatingPointPrecision.DOUBLE => "double"
          case other =>
            throw new IllegalArgumentException(
              s"Unsupported floating point precision: $other"
            )
        }
      case _: ArrowType.Utf8 => "str"
      case fsb: ArrowType.FixedSizeBinary =>
        s"fixedsizebinary[${fsb.getByteWidth}]"
      case _ =>
        throw new IllegalArgumentException(
          s"Unsupported ArrowType for tensor serialization: $arrowType"
        )
    }

  private def stringToArrowType(typeStr: String): ArrowType =
    typeStr match {
      // Signed integers
      case "int8"  => new ArrowType.Int(8, true)
      case "int16" => new ArrowType.Int(16, true)
      case "int32" => new ArrowType.Int(32, true)
      case "int64" => new ArrowType.Int(64, true)
      // Floating point
      case "float" => new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)
      case "double" =>
        new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)
      case "bool" => new ArrowType.Bool()
      case s if s.startsWith("fixedsizebinary[") && s.endsWith("]") =>
        val width = s.stripPrefix("fixedsizebinary[").stripSuffix("]").toInt
        new ArrowType.FixedSizeBinary(width)
      case _ =>
        throw new IllegalArgumentException(s"Unknown type string: $typeStr")
    }
}
