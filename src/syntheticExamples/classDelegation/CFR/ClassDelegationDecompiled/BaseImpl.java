package syntheticExamples.classDelegation.CFR.ClassDelegationDecompiled;

import kotlin.Metadata;
import kotlin.jvm.internal.Intrinsics;
import org.jetbrains.annotations.NotNull;

@Metadata(mv = {2, 0, 0}, k = 1, xi = 82, d1 = {"\u0000\u0018\n\u0002\u0018\u0002\n\u0002\u0018\u0002\n\u0000\n\u0002\u0010\u000e\n\u0002\b\u0005\n\u0002\u0010\u0002\n\u0000\u0018\u00002\u00020\u0001B\u000f\u0012\u0006\u0010\u0002\u001a\u00020\u0003\u00a2\u0006\u0004\b\u0004\u0010\u0005J\b\u0010\b\u001a\u00020\tH\u0016R\u0011\u0010\u0002\u001a\u00020\u0003\u00a2\u0006\b\n\u0000\u001a\u0004\b\u0006\u0010\u0007\u00a8\u0006\n"}, d2 = {"LsyntheticExamples/classDelegation/BaseImpl;", "LsyntheticExamples/classDelegation/Base;", "message", "", "<init>", "(Ljava/lang/String;)V", "getMessage", "()Ljava/lang/String;", "printMessage", "", "KotlinDecompiler"})
public final class BaseImpl
        implements Base {
    @NotNull
    private final String message;

    public BaseImpl(@NotNull String message) {
        Intrinsics.checkNotNullParameter(message, "message");
        this.message = message;
    }

    @NotNull
    public String getMessage() {
        return this.message;
    }

    @Override
    public void printMessage() {
        System.out.println((Object) this.message);
    }
}
