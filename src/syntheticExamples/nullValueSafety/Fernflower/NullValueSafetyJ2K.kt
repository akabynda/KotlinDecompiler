package syntheticExamples.nullValueSafety.Fernflower

/*
@Metadata(
    mv = [2, 0, 0],
    k = 2,
    xi = 48,
    d1 = ["\u0000\b\n\u0000\n\u0002\u0010\u0002\n\u0000\u001a\u0006\u0010\u0000\u001a\u00020\u0001¨\u0006\u0002"],
    d2 = ["main", "", "KotlinDecompiler"]
)
 */
object NullValueSafetyJ2K {
    fun main() {
        val text: String? = null
        val var1: Any? = null
        println(var1)
    }

    // $FF: synthetic method
    @JvmStatic
    fun main(args: Array<String>) {
        main()
    }
}