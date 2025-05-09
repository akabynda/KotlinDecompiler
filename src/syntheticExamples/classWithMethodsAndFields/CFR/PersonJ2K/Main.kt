package syntheticExamples.classWithMethodsAndFields.CFR.PersonJ2K

import syntheticExamples.classWithMethodsAndFields.Person

/*
@Metadata(
    mv = [2, 0, 0],
    k = 2,
    xi = 82,
    d1 = ["\u0000\b\n\u0000\n\u0002\u0010\u0002\n\u0000\u001a\u0006\u0010\u0000\u001a\u00020\u0001\u00a8\u0006\u0002"],
    d2 = ["main", "", "KotlinDecompiler"]
)
 */
object Main {
    fun main() {
        val person = Person("Alice", 29)
        person.sayHello()
    }

    @JvmStatic
    fun main(args: Array<String>) {
        main()
    }
}

