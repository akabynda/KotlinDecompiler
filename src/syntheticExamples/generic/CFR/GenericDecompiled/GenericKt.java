package syntheticExamples.generic.CFR.GenericDecompiled;

import kotlin.Metadata;
import syntheticExamples.generic.Box;

@Metadata(mv = {2, 0, 0}, k = 2, xi = 82, d1 = {"\u0000\b\n\u0000\n\u0002\u0010\u0002\n\u0000\u001a\u0006\u0010\u0000\u001a\u00020\u0001\u00a8\u0006\u0002"}, d2 = {"main", "", "KotlinDecompiler"})
public final class GenericKt {
    public static void main() {
        Box<String> box = new Box<String>("Hello");
        System.out.println((Object) box.getValue());
    }

    public static /* synthetic */ void main(String[] args) {
        GenericKt.main();
    }
}
