package syntheticExamples.scopeFunctions.CFR.ScopeFunctionsDecompiled;

import kotlin.Metadata;
import kotlin.jvm.internal.Intrinsics;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

@Metadata(mv={2, 0, 0}, k=1, xi=82, d1={"\u0000 \n\u0002\u0018\u0002\n\u0002\u0010\u0000\n\u0000\n\u0002\u0010\u000e\n\u0000\n\u0002\u0010\b\n\u0002\b\u0012\n\u0002\u0010\u000b\n\u0002\b\u0004\b\u0086\b\u0018\u00002\u00020\u0001B!\u0012\u0006\u0010\u0002\u001a\u00020\u0003\u0012\u0006\u0010\u0004\u001a\u00020\u0005\u0012\b\u0010\u0006\u001a\u0004\u0018\u00010\u0003\u00a2\u0006\u0004\b\u0007\u0010\bJ\t\u0010\u0013\u001a\u00020\u0003H\u00c6\u0003J\t\u0010\u0014\u001a\u00020\u0005H\u00c6\u0003J\u000b\u0010\u0015\u001a\u0004\u0018\u00010\u0003H\u00c6\u0003J)\u0010\u0016\u001a\u00020\u00002\b\b\u0002\u0010\u0002\u001a\u00020\u00032\b\b\u0002\u0010\u0004\u001a\u00020\u00052\n\b\u0002\u0010\u0006\u001a\u0004\u0018\u00010\u0003H\u00c6\u0001J\u0013\u0010\u0017\u001a\u00020\u00182\b\u0010\u0019\u001a\u0004\u0018\u00010\u0001H\u00d6\u0003J\t\u0010\u001a\u001a\u00020\u0005H\u00d6\u0001J\t\u0010\u001b\u001a\u00020\u0003H\u00d6\u0001R\u001a\u0010\u0002\u001a\u00020\u0003X\u0086\u000e\u00a2\u0006\u000e\n\u0000\u001a\u0004\b\t\u0010\n\"\u0004\b\u000b\u0010\fR\u001a\u0010\u0004\u001a\u00020\u0005X\u0086\u000e\u00a2\u0006\u000e\n\u0000\u001a\u0004\b\r\u0010\u000e\"\u0004\b\u000f\u0010\u0010R\u001c\u0010\u0006\u001a\u0004\u0018\u00010\u0003X\u0086\u000e\u00a2\u0006\u000e\n\u0000\u001a\u0004\b\u0011\u0010\n\"\u0004\b\u0012\u0010\f\u00a8\u0006\u001c"}, d2={"LsyntheticExamples/scopeFunctions/Person;", "", "name", "", "age", "", "email", "<init>", "(Ljava/lang/String;ILjava/lang/String;)V", "getName", "()Ljava/lang/String;", "setName", "(Ljava/lang/String;)V", "getAge", "()I", "setAge", "(I)V", "getEmail", "setEmail", "component1", "component2", "component3", "copy", "equals", "", "other", "hashCode", "toString", "KotlinDecompiler"})
public final class Person {
    @NotNull
    private String name;
    private int age;
    @Nullable
    private String email;

    public Person(@NotNull String name, int age, @Nullable String email) {
        Intrinsics.checkNotNullParameter((Object)name, (String)"name");
        this.name = name;
        this.age = age;
        this.email = email;
    }

    @NotNull
    public final String getName() {
        return this.name;
    }

    public final void setName(@NotNull String string) {
        Intrinsics.checkNotNullParameter((Object)string, (String)"<set-?>");
        this.name = string;
    }

    public final int getAge() {
        return this.age;
    }

    public final void setAge(int n) {
        this.age = n;
    }

    @Nullable
    public final String getEmail() {
        return this.email;
    }

    public final void setEmail(@Nullable String string) {
        this.email = string;
    }

    @NotNull
    public final String component1() {
        return this.name;
    }

    public final int component2() {
        return this.age;
    }

    @Nullable
    public final String component3() {
        return this.email;
    }

    @NotNull
    public final Person copy(@NotNull String name, int age, @Nullable String email) {
        Intrinsics.checkNotNullParameter((Object)name, (String)"name");
        return new Person(name, age, email);
    }

    public static /* synthetic */ Person copy$default(Person person, String string, int n, String string2, int n2, Object object) {
        if ((n2 & 1) != 0) {
            string = person.name;
        }
        if ((n2 & 2) != 0) {
            n = person.age;
        }
        if ((n2 & 4) != 0) {
            string2 = person.email;
        }
        return person.copy(string, n, string2);
    }

    @NotNull
    public String toString() {
        return "Person(name=" + this.name + ", age=" + this.age + ", email=" + this.email + ')';
    }

    public int hashCode() {
        int result = this.name.hashCode();
        result = result * 31 + Integer.hashCode(this.age);
        result = result * 31 + (this.email == null ? 0 : this.email.hashCode());
        return result;
    }

    public boolean equals(@Nullable Object other) {
        if (this == other) {
            return true;
        }
        if (!(other instanceof Person)) {
            return false;
        }
        Person person = (Person)other;
        if (!Intrinsics.areEqual((Object)this.name, (Object)person.name)) {
            return false;
        }
        if (this.age != person.age) {
            return false;
        }
        return Intrinsics.areEqual((Object)this.email, (Object)person.email);
    }
}