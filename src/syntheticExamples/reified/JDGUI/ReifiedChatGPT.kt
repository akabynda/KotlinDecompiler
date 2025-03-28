package syntheticExamples.reified.JDGUI

inline fun <reified T> getTypeName(): String {
    return T::class.simpleName ?: "Unknown"
}

fun main() {
    val typeName = getTypeName<String>()
    println(typeName)
}
