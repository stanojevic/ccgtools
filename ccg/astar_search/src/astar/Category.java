package astar;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

public abstract class Category {
    private final String asString;
    private final int id;
    private final static String WILDCARD_FEATURE = "X";
    private final static Set<String> bracketAndQuoteCategories =
        Collections.unmodifiableSet(new HashSet<>(Arrays.asList("LRB", "RRB", "LQU", "RQU")));

    private Category(String asString) {
        this.asString = asString;
        this.id = numCats.getAndIncrement();
    }

    public boolean featurelessMathces(Category other){
        if(this.isConj() && other.isConj()){
            ConjCategory left = (ConjCategory) this;
            ConjCategory right = (ConjCategory) other;
            return left.cat.featurelessMathces(right.cat);
        }else if(this.isFunctor() && other.isFunctor()){
            FunctorCategory left = (FunctorCategory) this;
            FunctorCategory right = (FunctorCategory) other;
            return left.left.featurelessMathces(right.left) && left.right.featurelessMathces(right.right) && left.slash.matches(right.slash);
        }else if((this instanceof AtomicCategory) && (other instanceof AtomicCategory)){
            AtomicCategory left = (AtomicCategory) this;
            AtomicCategory right = (AtomicCategory) other;
            return left.type == right.type;
        }else{
            return false;
        }
    }

    public enum Slash {
        FWD, BWD, EITHER;

        public String toString() {
            String result = "";

            switch (this) {
                case FWD:
                    result = "/";
                    break;
                case BWD:
                    result = "\\";
                    break;
                case EITHER:
                    result = "|";
                    break;

            }

            return result;
        }

        public static Slash fromString(String text) {
            if (text != null) {
                for (Slash slash : values()) {
                    if (text.equalsIgnoreCase(slash.toString())) {
                        return slash;
                    }
                }
            }
            throw new IllegalArgumentException("Invalid slash: " + text);
        }

        public boolean matches(Slash other) {
            return this == EITHER || this == other;
        }
    }

    public static int maxCatId(){ return numCats.get(); }

    private final static AtomicInteger numCats = new AtomicInteger();
    private final static ConcurrentHashMap<String, Category> cache = new ConcurrentHashMap<>();
    public static final Category COMMA = Category.valueOf(",");
    public static final Category SEMICOLON = Category.valueOf(";");
    public static final Category CONJ = Category.valueOf("conj");
    public final static Category N = valueOf("N");
    public static final Category LQU = Category.valueOf("LQU");
    public static final Category LRB = Category.valueOf("LRB");
    public static final Category NP_b_NP = Category.valueOf(("NP\\NP"));
    public static final Category GLUE = Category.valueOf("GLUE");

    public static Category valueOf(String cat) {
        Category result = cache.get(cat);
        if(result == null)
            result = Category.valueOfUncached(cat);
            cache.put(cat, result);
        return result;
    }

    /**
     * Builds a category from a string representation.
     */
    private static Category valueOfUncached(String source) {
        // Categories have the form: ((X/Y)\Z[feature]){ANNOTATION}
        String newSource = source;

        if (newSource.endsWith("}")) {
            int openIndex = newSource.lastIndexOf("{");
            newSource = newSource.substring(0, openIndex);
        }

        if (newSource.startsWith("(")) {
            int closeIndex = Util.findClosingBracket(newSource);

            if (Util.indexOfAny(newSource.substring(closeIndex), "/\\|") == -1) {
                // Simplify (X) to X
                newSource = newSource.substring(1, closeIndex);
                return valueOfUncached(newSource);
            }
        }

        int endIndex = newSource.length();

        int opIndex = Util.findNonNestedChar(newSource, "/\\|");

        if (opIndex == -1) {
            // Atomic Category
            int featureIndex = newSource.indexOf("[");
            List<String> features = new ArrayList<>();

            String base = (featureIndex == -1 ? newSource : newSource.substring(0, featureIndex));

            while (featureIndex > -1) {
                features.add(newSource.substring(featureIndex + 1, newSource.indexOf("]", featureIndex)));
                featureIndex = newSource.indexOf("[", featureIndex + 1);
            }

            if (features.size() > 1) {
                throw new RuntimeException("Can only handle single features: " + source);
            }

            return new AtomicCategory(base, features.size() == 0 ? null : features.get(0));
        } else {
            // Functor Category

            Category left = valueOf(newSource.substring(0, opIndex));
            Category right = valueOf(newSource.substring(opIndex + 1, endIndex));
            return new FunctorCategory(left,
                    Slash.fromString(newSource.substring(opIndex, opIndex + 1)),
                    right
            );
        }
    }

    public String toString() {
        return asString;
    }

    @Override
    public boolean equals(Object other) {
        return this == other;
    }

    @Override
    public int hashCode() {
        return id;
    }

    abstract boolean isTypeRaised();

    abstract boolean isForwardTypeRaised();

    abstract boolean isBackwardTypeRaised();

    public abstract boolean isModifier();

    public abstract boolean matches(Category other);

    public abstract Category getLeft();

    public abstract Category getRight();

    abstract Slash getSlash();

    abstract String getFeature();

    abstract String toStringWithBrackets();

    static class ConjCategory extends Category {

        public Category cat;

        private ConjCategory(Category cat) {
            super(cat.asString+"[conj]");
            this.cat = cat;
        }

        @Override
        boolean isTypeRaised() {
            return false;
        }

        @Override
        boolean isForwardTypeRaised() {
            return false;
        }

        @Override
        boolean isBackwardTypeRaised() {
            return false;
        }

        @Override
        public boolean isModifier() {
            return false;
        }

        @Override
        public boolean matches(Category other) {
            return other.isConj() && this.cat.matches(((ConjCategory) other).cat);
        }

        @Override
        public Category getLeft() {
            return null;
        }

        @Override
        public Category getRight() {
            return null;
        }

        @Override
        Slash getSlash() {
            return null;
        }

        @Override
        String getFeature() {
            return null;
        }

        @Override
        String toStringWithBrackets() {
            return null;
        }

        @Override
        String getSubstitution(Category other) {
            return null;
        }

        @Override
        String getType() {
            return null;
        }

        @Override
        boolean isFunctor() {
            return false;
        }

        @Override
        boolean isPunctuation() {
            return false;
        }

        @Override
        boolean isNounOrNP() {
            return false;
        }
    }

    static class FunctorCategory extends Category {
        private final Category left;
        private final Category right;
        private final Slash slash;
        private final boolean isMod;

        private FunctorCategory(Category left, Slash slash, Category right) {
            super(left.toStringWithBrackets() + slash + right.toStringWithBrackets());
            this.left = left;
            this.right = right;
            this.slash = slash;
            this.isMod = left.equals(right);

            // X|(X|Y)
            this.isTypeRaised = right.isFunctor() && right.getLeft().equals(left);
        }

        @Override
        public boolean isModifier() {
            return isMod;
        }

        @Override
        public boolean matches(Category other) {
            return other.isFunctor() && left.matches(other.getLeft()) && right.matches(other.getRight()) && slash.matches(other.getSlash());
        }

        @Override
        public Category getLeft() {
            return left;
        }

        @Override
        public Category getRight() {
            return right;
        }

        @Override
        Slash getSlash() {
            return slash;
        }

        @Override
        String getFeature() {
            throw new UnsupportedOperationException();
        }

        @Override
        String toStringWithBrackets() {
            return "(" + toString() + ")";
        }

        @Override
        public boolean isFunctor() {
            return true;
        }

        @Override
        public boolean isPunctuation() {
            return false;
        }

        @Override
        String getType() {
            throw new UnsupportedOperationException();
        }

        @Override
        String getSubstitution(Category other) {
            String result = getRight().getSubstitution(other.getRight());
            if (result == null) {
                // Bit of a hack, but seems to reproduce CCGBank in cases of clashing features.
                result = getLeft().getSubstitution(other.getLeft());
            }
            return result;
        }

        private final boolean isTypeRaised;

        @Override
        public boolean isTypeRaised() {
            return isTypeRaised;
        }

        @Override
        public boolean isForwardTypeRaised() {
            // X/(X\Y)
            return isTypeRaised() && getSlash() == Slash.FWD;
        }

        @Override
        public boolean isBackwardTypeRaised() {
            // X\(X/Y)
            return isTypeRaised() && getSlash() == Slash.BWD;
        }

        @Override
        boolean isNounOrNP() {
            return false;
        }

    }

    abstract String getSubstitution(Category other);


    static class AtomicCategory extends Category {

        private AtomicCategory(String type, String feature) {
            super(type + (feature == null ? "" : "[" + feature + "]"));
            this.type = type;
            this.feature = feature;
            isPunctuation = !type.matches("[A-Za-z]+") || bracketAndQuoteCategories.contains(type);
        }

        private final String type;
        private final String feature;

        @Override
        public boolean isModifier() {
            return false;
        }

        @Override
        public boolean matches(Category other) {
            return (other instanceof AtomicCategory) && type.equals(other.getType()) &&
                    (feature == null || feature.equals(other.getFeature()) || WILDCARD_FEATURE.equals(getFeature()) || WILDCARD_FEATURE.equals(other.getFeature())
                            || feature.equals("nb") // Ignoring the NP[nb] feature, which isn't very helpful. For example, it stops us coordinating "John and a girl",
                            // because "and a girl" ends up with a NP[nb]\NP[nb] tag.
                    );
        }

        @Override
        public Category getLeft() {
            throw new UnsupportedOperationException();
        }

        @Override
        public Category getRight() {
            throw new UnsupportedOperationException();
        }

        @Override
        Slash getSlash() {
            throw new UnsupportedOperationException();
        }

        @Override
        String getFeature() {
            return feature;
        }

        @Override
        String toStringWithBrackets() {
            return toString();
        }

        @Override
        public boolean isFunctor() {
            return false;
        }

        private final boolean isPunctuation;

        @Override
        public boolean isPunctuation() {
            return isPunctuation;
        }

        @Override
        String getType() {
            return type;
        }

        @Override
        String getSubstitution(Category other) {
            if (WILDCARD_FEATURE.equals(getFeature())) {
                return other.getFeature();
            } else if (WILDCARD_FEATURE.equals(other.getFeature())) {
                return feature;
            }
            return null;
        }

        @Override
        public boolean isTypeRaised() {
            return false;
        }

        public boolean isForwardTypeRaised() {
            return false;
        }

        public boolean isBackwardTypeRaised() {
            return false;
        }

        @Override
        boolean isNounOrNP() {
            return type.equals("N") || type.equals("NP");
        }

    }

    public static Category make(Category left, Slash op, Category right) {
        return valueOf(left.toStringWithBrackets() + op + right.toStringWithBrackets());
    }

    public static Category toConj(Category cat) {
        return new ConjCategory(cat);
    }

    abstract String getType();

    abstract boolean isFunctor();

    abstract boolean isPunctuation();

    boolean isConj(){
        return (this instanceof ConjCategory);
    }

    /**
     * Returns the Category created by substituting all [X] wildcard features with the supplied argument.
     */
    Category doSubstitution(String substitution) {
        if (substitution == null) return this;
        return valueOf(toString().replaceAll(WILDCARD_FEATURE, substitution));
    }

    /**
     * A unique numeric identifier for this category.
     */
    int getID() {
        return id;
    }

    abstract boolean isNounOrNP();

}