package syntheticExamplesWoFixes.safeCast

fun main() {
    val obj: Any = "String"
    val num: Int? = obj as? Int
    println(num)
}
