package syntheticExamples.lazyInitialization.CFR.LazyInitializationDecompiled;

import kotlin.Lazy;
import kotlin.LazyKt;
import kotlin.Metadata;
import kotlin.jvm.functions.Function0;
import org.jetbrains.annotations.NotNull;

// @Metadata(mv={2, 0, 0}, k=1, xi=82, d1={"\u0000\u0014\n\u0002\u0018\u0002\n\u0002\u0010\u0000\n\u0002\b\u0003\n\u0002\u0010\u000e\n\u0002\b\u0005\u0018\u00002\u00020\u0001B\u0007\u00a2\u0006\u0004\b\u0002\u0010\u0003R\u001b\u0010\u0004\u001a\u00020\u00058FX\u0086\u0084\u0002\u00a2\u0006\f\n\u0004\b\b\u0010\t\u001a\u0004\b\u0006\u0010\u0007\u00a8\u0006\n"}, d2={"LsyntheticExamples/lazyInitialization/LazyInitialization;", "", "<init>", "()V", "value", "", "getValue", "()Ljava/lang/String;", "value$delegate", "Lkotlin/Lazy;", "KotlinDecompiler"})
// not working
/*
public final class LazyInitialization {
    @NotNull
    private final Lazy value$delegate = LazyKt.lazy((Function0)value.2.INSTANCE);

    @NotNull
    public final String getValue() {
        Lazy lazy = this.value$delegate;
        return (String)lazy.getValue();
    }
}
 */