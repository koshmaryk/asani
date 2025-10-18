ThisBuild / organization := "com.dyeru"
ThisBuild / version := "0.1.8"
ThisBuild / scalaVersion := "3.3.4"
ThisBuild / publishMavenStyle := true
ThisBuild / description := "Lightning-fast communication framework based on Apache Arrow and gRPC.\n\nThere are 2 implementation planned:\n\nIPC using Apache Arrow memory-mapped files\nRPC using Arrow Flight"
ThisBuild / licenses := Seq(
  "Apache-2.0" -> url("https://www.apache.org/licenses/LICENSE-2.0")
)
ThisBuild / versionScheme := Some("semver-spec")

ThisBuild / homepage := Some(url("https://github.com/DmitryYerusalimtsev/asani"))

ThisBuild / developers := List(
  Developer(
    id = "DmitryYerusalimtsev",
    name = "Dmitry Yerusalimtsev",
    email = "dima.yerusalimtsev@gmail.com",
    url = url("https://www.linkedin.com/in/dmytro-yerusalimtsev-9bba8b58/")
  )
)

ThisBuild / scmInfo := Some(
  ScmInfo(
    url("https://github.com/DmitryYerusalimtsev/asani"),
    "scm:git@github.com:DmitryYerusalimtsev/asani.git"
  )
)

ThisBuild / publishTo := Some(
  "GitHub DmitryYerusalimtsev Apache Maven Packages" at "https://maven.pkg.github.com/DmitryYerusalimtsev/asani"
)

ThisBuild / credentials += Credentials(
  "GitHub Package Registry",
  "maven.pkg.github.com",
  "DmitryYerusalimtsev",
  System.getenv("GITHUB_TOKEN")
)

Test / fork := true
Test / javaOptions += "--add-opens=java.base/java.nio=org.apache.arrow.memory.core,ALL-UNNAMED"

val testing = Seq(
  "org.scalatest" %% "scalatest" % V.scalatest % Test
)

val arrow = Seq(
  "org.apache.arrow" % "arrow-vector" % V.arrow,
  "org.apache.arrow" % "flight-core" % V.arrow
)

val jsoniter = Seq(
  "com.github.plokhotnyuk.jsoniter-scala" %% "jsoniter-scala-core" % V.jsoniter,
  "com.github.plokhotnyuk.jsoniter-scala" %% "jsoniter-scala-macros" % V.jsoniter
)

lazy val root = (project in file("."))
  .settings(
    name := "asani-scala",
    libraryDependencies ++= arrow ++ jsoniter ++ testing
  )