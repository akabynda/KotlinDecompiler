package syntheticExamplesWoFixes.higherOrderFunction

fun main() {
    val square: (Int) -> Int = { it * it }
    val result = square(5)
    println(result)
}
