/*
 * Decompiled with CFR 0.152.
 *
 * Could not load the following classes:
 *  kotlin.Metadata
 *  kotlin.jvm.internal.Intrinsics
 *  org.jetbrains.annotations.NotNull
 */
package syntheticExamplesWoFixes.classDelegation.CFR.Decompilation;

import kotlin.Metadata;
import kotlin.jvm.internal.Intrinsics;
import org.jetbrains.annotations.NotNull;

@Metadata(mv = {1, 9, 0}, k = 1, xi = 48, d1 = {"\u0000\u0012\n\u0002\u0018\u0002\n\u0002\u0018\u0002\n\u0002\b\u0003\n\u0002\u0010\u0002\n\u0000\u0018\u00002\u00020\u0001B\r\u0012\u0006\u0010\u0002\u001a\u00020\u0001\u00a2\u0006\u0002\u0010\u0003J\t\u0010\u0004\u001a\u00020\u0005H\u0096\u0001\u00a8\u0006\u0006"}, d2 = {"LsyntheticExamplesWoFixes/classDelegation/Derived;", "LsyntheticExamplesWoFixes/classDelegation/Base;", "base", "(LsyntheticExamplesWoFixes/classDelegation/Base;)V", "printMessage", "", "KotlinDecompiler"})
public final class Derived
        implements Base {
    private final /* synthetic */ Base $$delegate_0;

    public Derived(@NotNull Base base) {
        Intrinsics.checkNotNullParameter(base, "base");
        this.$$delegate_0 = base;
    }

    @Override
    public void printMessage() {
        this.$$delegate_0.printMessage();
    }
}
