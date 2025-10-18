package com.dyeru.asani.arrow

object ArrayUtils {
  def inferShape(arr: Any): Seq[Long] = {
    def loop(a: Any): List[Long] = a match {
      case arr: Array[_] if arr.isEmpty => List(0)
      case arr: Array[_] =>
        arr.length.toLong :: loop(arr.head)
      case _ => Nil
    }
    loop(arr)
  }

  def isMultiDimensional(arr: Any): Boolean = arr match {
    case a: Array[_] if a.nonEmpty =>
      a.head match {
        case _: Array[_] => true
        case _ => false
      }
    case _ => false
  }

  def flattenArray(arr: Any): Array[_] = {
    def loop(a: Any): List[Any] = a match {
      case arr: Array[_] => arr.flatMap(loop).toList
      case leaf => List(leaf)
    }
    loop(arr).toArray
  }
}
