package syntheticExamplesWoFixes.labeledBlocks.JDGUI.Decompilation;

import kotlin.Metadata;

@Metadata(mv = {1, 9, 0}, k = 2, xi = 48, d1 = {"\000\b\n\000\n\002\020\002\n\000\032\006\020\000\032\0020\001¨\006\002"}, d2 = {"main", "", "KotlinDecompiler"})
public final class LabeledBlocksKt {
    public static void main() {
        int i;
        label16:
        for (i = 1; i < 6; i++) {
            for (int j = 1; j < 6; j++) {
                if (i * j > 6) {
                    System.out.println("Breaking out of the outer loop at i = " + i + ", j = " + j);
                    break label16;
                }
                System.out.println("i = " + i + ", j = " + j);
            }
        }
    }
}


/* Location:              /Users/akabynda/KotlinDecompiler/out/production/KotlinDecompiler/!/syntheticExamplesWoFixes/labeledBlocks/LabeledBlocksKt.class
 * Java compiler version: 8 (52.0)
 * JD-Core Version:       1.1.3
 */