package astar;

import java.util.*;

import astar.Category.Slash;

public abstract class Combinator {
    public enum RuleType {
        FA, BA, FC, BX, GFC, GBX, CONJ_BOTTOM, CONJ_TOP, RP, LP, NOISE, UNARY, LEXICON
    }

    private Combinator(RuleType ruleType) {
        this.ruleType = ruleType;
    }

    static class RuleProduction {
        public RuleProduction(RuleType ruleType, Category result, final boolean headIsLeft) {
            this.ruleType = ruleType;
            this.category = result;
            this.headIsLeft = headIsLeft;
        }

        public final RuleType ruleType;
        public final Category category;
        public final boolean headIsLeft;
    }

    public abstract boolean headIsLeft(Category left, Category right);

    public final static Collection<Combinator> STANDARD_COMBINATORS = new ArrayList<>(Arrays.asList(
            new ForwardApplication(),
            new BackwardApplication(),
            new ForwardComposition(Slash.FWD, Slash.FWD, Slash.FWD),
            new BackwardComposition(Slash.FWD, Slash.BWD, Slash.FWD),
            new GeneralizedForwardComposition(Slash.FWD, Slash.FWD, Slash.FWD),
            new GeneralizedBackwardComposition(Slash.FWD, Slash.BWD, Slash.FWD),
            new ConjBottom(),
            new ConjTop(),
            new RemovePunctuation(false),
            new RemovePunctuationLeft()
    ));

    private final RuleType ruleType;

    public abstract boolean canApply(Category left, Category right);

    public abstract Category apply(Category left, Category right);

    /**
     * Makes sure wildcard features are correctly instantiated.
     * <p>
     * We want: S[X]/(S[X]\NP) and S[dcl]\NP to combine to S[dcl]. This is done by finding any wildcards that
     * need to be matched between S[X]\NP and S[dcl]\NP, and applying the substitution to S[dcl].
     */
    private static Category correctWildCardFeatures(Category toCorrect, Category match1, Category match2) {
        return toCorrect.doSubstitution(match1.getSubstitution(match2));
    }

    /**
     * Returns a set of rules that can be applied to a pair of categories.
     */
    static Collection<RuleProduction> getRules(Category left, Category right, Collection<Combinator> rules) {
        List<RuleProduction> result = new ArrayList<>(2);
        for (Combinator c : rules) {
            if (c.canApply(left, right)) {
                result.add(new RuleProduction(c.ruleType, c.apply(left, right), c.headIsLeft(left, right)));
            }
        }

        return Collections.unmodifiableList(result);
    }

    private static class ConjTop extends Combinator {

        private ConjTop() {
            super(RuleType.CONJ_TOP);
        }

        @Override
        public boolean canApply(Category left, Category right) {
            if(!right.isConj()){
                return false;
            }
            return ((Category.ConjCategory) right).cat.matches(left);
        }

        @Override
        public Category apply(Category left, Category right) {
            return left;
        }

        @Override
        public boolean headIsLeft(Category left, Category right) {
            return false;
        }
    }

    private static final class ConjBottom extends Combinator {

        private ConjBottom() {
            super(RuleType.CONJ_BOTTOM);
        }

        @Override
        public boolean canApply(Category left, Category right) {
            if (Category.NP_b_NP.matches(right)) {
                // C&C evaluation script doesn't let you do this, for some reason.
                return false;
            }

            return (left == Category.CONJ || left == Category.COMMA || left == Category.SEMICOLON)
                    && !right.isPunctuation() // Don't start making weird ,\, categories...
                    && !right.isTypeRaised()  // Improves coverage of C&C evaluation script. Categories can just conjoin first, then type-raise.

                    // Blocks noun conjunctions, which should normally be NP conjunctions.
                    // In a better world, conjunctions would have categories like (NP\NP/NP.
                    // Doesn't affect F-scopes, but makes output semantically nicer.
                    && !((right instanceof Category.AtomicCategory) && right.getType().equals("N"))
                    && !right.isConj();
        }

        @Override
        public Category apply(Category left, Category right) {
            return Category.toConj(right);
        }

        @Override
        public boolean headIsLeft(Category left, Category right) {
            return false;
        }
    }

    private static class RemovePunctuation extends Combinator {
        private final boolean punctuationIsLeft;

        private RemovePunctuation(boolean punctuationIsLeft) {
            super(RuleType.RP);
            this.punctuationIsLeft = punctuationIsLeft;
        }

        @Override
        public boolean canApply(Category left, Category right) {
            return punctuationIsLeft ? left.isPunctuation() :
                    right.isPunctuation() && !Category.N.matches(left); // Disallow punctuation combining with nouns, to avoid getting NPs like "Barack Obama ."
        }

        @Override
        public Category apply(Category left, Category right) {
            return punctuationIsLeft ? right : left;
        }

        @Override
        public boolean headIsLeft(Category left, Category right) {
            return !punctuationIsLeft;
        }
    }

    /**
     * Open Brackets and Quotations
     */
    private static class RemovePunctuationLeft extends Combinator {
        private RemovePunctuationLeft() {
            super(RuleType.LP);
        }

        @Override
        public boolean canApply(Category left, Category right) {
            return left == Category.LQU || left == Category.LRB;
        }

        @Override
        public Category apply(Category left, Category right) {
            return right;
        }

        @Override
        public boolean headIsLeft(Category left, Category right) {
            return false;
        }
    }

    private static class ForwardApplication extends Combinator {
        private ForwardApplication() {
            super(RuleType.FA);
        }

        @Override
        public boolean canApply(Category left, Category right) {
            return left.isFunctor() && left.getSlash() == Slash.FWD && left.getRight().matches(right);
        }

        @Override
        public Category apply(Category left, Category right) {
            if (left.isModifier()) return right;

            Category result = left.getLeft();

            result = correctWildCardFeatures(result, left.getRight(), right);

            return result;
        }

        @Override
        public boolean headIsLeft(Category left, Category right) {
            return !left.isModifier() && !left.isTypeRaised();
        }
    }

    private static class BackwardApplication extends Combinator {
        private BackwardApplication() {
            super(RuleType.BA);
        }

        @Override
        public boolean canApply(Category left, Category right) {
            return right.isFunctor() && right.getSlash() == Slash.BWD && right.getRight().matches(left);
        }

        @Override
        public Category apply(Category left, Category right) {
            Category result;
            if (right.isModifier()) {
                result = left;
            } else {
                result = right.getLeft();
            }

            return correctWildCardFeatures(result, right.getRight(), left);
        }

        @Override
        public boolean headIsLeft(Category left, Category right) {
            return right.isModifier() || right.isTypeRaised();
        }
    }

    private static class ForwardComposition extends Combinator {
        private final Slash leftSlash;
        private final Slash rightSlash;
        private final Slash resultSlash;

        private ForwardComposition(Slash left, Slash right, Slash result) {
            super(RuleType.FC);
            this.leftSlash = left;
            this.rightSlash = right;
            this.resultSlash = result;
        }

        @Override
        public boolean canApply(Category left, Category right) {
            return left.isFunctor() && right.isFunctor() && left.getRight().matches(right.getLeft()) && left.getSlash() == leftSlash && right.getSlash() == rightSlash;
        }

        @Override
        public Category apply(Category left, Category right) {
            Category result;
            if (left.isModifier()) {
                result = right;
            } else {
                result = Category.make(left.getLeft(), resultSlash, right.getRight());
            }

            return correctWildCardFeatures(result, right.getLeft(), left.getRight());
        }

        @Override
        public boolean headIsLeft(Category left, Category right) {
            return !left.isModifier() && !left.isTypeRaised();
        }
    }

    private static class GeneralizedForwardComposition extends Combinator {
        private final Slash leftSlash;
        private final Slash rightSlash;
        private final Slash resultSlash;

        private GeneralizedForwardComposition(Slash left, Slash right, Slash result) {
            super(RuleType.GFC);
            this.leftSlash = left;
            this.rightSlash = right;
            this.resultSlash = result;
        }

        @Override
        public boolean canApply(Category left, Category right) {
            if (left.isFunctor() && right.isFunctor() && right.getLeft().isFunctor()) {
                Category rightLeft = right.getLeft();
                return left.getRight().matches(rightLeft.getLeft()) && left.getSlash() == leftSlash && rightLeft.getSlash() == rightSlash;
            } else {
                return false;
            }
        }

        @Override
        public Category apply(Category left, Category right) {
            if (left.isModifier()) return right;

            Category rightLeft = right.getLeft();

            Category result = Category.make(Category.make(left.getLeft(), resultSlash, rightLeft.getRight()), right.getSlash(), right.getRight());

            result = correctWildCardFeatures(result, rightLeft.getLeft(), left.getRight());
            return result;
        }

        @Override
        public boolean headIsLeft(Category left, Category right) {
            return !left.isModifier() && !left.isTypeRaised();
        }
    }


    private static class GeneralizedBackwardComposition extends Combinator {
        private final Slash leftSlash;
        private final Slash rightSlash;
        private final Slash resultSlash;

        private GeneralizedBackwardComposition(Slash left, Slash right, Slash result) {
            super(RuleType.GBX);
            this.leftSlash = left;
            this.rightSlash = right;
            this.resultSlash = result;
        }

        @Override
        public boolean canApply(Category left, Category right) {
            if (left.isFunctor() && right.isFunctor() && left.getLeft().isFunctor()) {
                Category leftLeft = left.getLeft();
                return right.getRight().matches(leftLeft.getLeft()) && leftLeft.getSlash() == leftSlash && right.getSlash() == rightSlash
                        && !(left.getLeft().isNounOrNP()); // Additional constraint from Steedman (2000)
            } else {
                return false;
            }
        }

        @Override
        public Category apply(Category left, Category right) {
            if (right.isModifier()) return left;

            Category leftLeft = left.getLeft();

            Category result = Category.make(Category.make(right.getLeft(), resultSlash, leftLeft.getRight()), left.getSlash(), left.getRight());

            result = correctWildCardFeatures(result, leftLeft.getLeft(), right.getRight());
            return result;
        }

        @Override
        public boolean headIsLeft(Category left, Category right) {
            return right.isModifier() || right.isTypeRaised();
        }
    }


    private static class BackwardComposition extends Combinator {
        private final Slash leftSlash;
        private final Slash rightSlash;
        private final Slash resultSlash;

        private BackwardComposition(Slash left, Slash right, Slash result) {
            super(RuleType.BX);
            this.leftSlash = left;
            this.rightSlash = right;
            this.resultSlash = result;
        }

        @Override
        public boolean canApply(Category left, Category right) {
            return left.isFunctor() &&
                    right.isFunctor() &&
                    right.getRight().matches(left.getLeft()) &&
                    left.getSlash() == leftSlash && right.getSlash() == rightSlash &&
                    !(left.getLeft().isNounOrNP()); // Additional constraint from Steedman (2000)
        }

        @Override
        public Category apply(Category left, Category right) {
            Category result;
            if (right.isModifier()) {
                result = left;
            } else {
                result = Category.make(right.getLeft(), resultSlash, left.getRight());
            }

            return result.doSubstitution(left.getLeft().getSubstitution(right.getRight()));
        }

        @Override
        public boolean headIsLeft(Category left, Category right) {
            return (right.isModifier() || right.isTypeRaised());
        }
    }

}