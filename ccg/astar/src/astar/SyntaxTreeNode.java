package astar;

import java.util.Random;
import astar.Combinator.RuleType;

public abstract class SyntaxTreeNode implements Comparable<SyntaxTreeNode> {

    private final Category category;
    final double probability;
    final int hash;
    final int totalDependencyLength;
    private final int headIndex;

    private SyntaxTreeNode(
            Category category,
            double probability,
            int hash,
            int totalDependencyLength,
            int headIndex
    ) {
        this.category = category;
        this.probability = probability;
        this.hash = hash;
        this.totalDependencyLength = totalDependencyLength;
        this.headIndex = headIndex;
    }

    static class SyntaxTreeNodeBinary extends SyntaxTreeNode {
        final RuleType ruleType;
        final boolean headIsLeft;
        final SyntaxTreeNode leftChild;
        final SyntaxTreeNode rightChild;

        private SyntaxTreeNodeBinary(Category category, double probability, int hash, int totalDependencyLength, int headIndex,
                                     RuleType ruleType, boolean headIsLeft, SyntaxTreeNode leftChild, SyntaxTreeNode rightChild) {
            super(category, probability, hash, totalDependencyLength, headIndex);
            this.ruleType = ruleType;
            this.headIsLeft = headIsLeft;
            this.leftChild = leftChild;
            this.rightChild = rightChild;
        }

        @Override
        void accept(SyntaxTreeNodeVisitor v) {
            v.visit(this);
        }

        @Override
        public RuleType getRuleType() {
            return ruleType;
        }

    }

    public static class SyntaxTreeNodeLeaf extends SyntaxTreeNode {
        private SyntaxTreeNodeLeaf(
                String word, Category category, double probability, int hash, int totalDependencyLength, int headIndex
        ) {
            super(category, probability, hash, totalDependencyLength, headIndex);
            this.word = word;
        }

        private final String word;

        @Override
        void accept(SyntaxTreeNodeVisitor v) {
            v.visit(this);
        }

        @Override
        public RuleType getRuleType() {
            return RuleType.LEXICON;
        }

        public String getWord() {
            return word;
        }

    }

    static class SyntaxTreeNodeUnary extends SyntaxTreeNode {
        private SyntaxTreeNodeUnary(Category category, double probability, int hash, int totalDependencyLength, int headIndex, SyntaxTreeNode child) {
            super(category, probability, hash, totalDependencyLength, headIndex);

            this.child = child;
        }

        final SyntaxTreeNode child;

        @Override
        void accept(SyntaxTreeNodeVisitor v) {
            v.visit(this);
        }

        @Override
        public RuleType getRuleType() {
            return RuleType.UNARY;
        }

    }

    public String toString() {
        return CCGBankParsePrinterVisitor.tree2string(this);
    }

    public int getHeadIndex() {
        return headIndex;
    }

    @Override
    public int compareTo(SyntaxTreeNode o) {
        return Double.compare(o.probability, probability);
    }

    /**
     * Factory for SyntaxTreeNode. Using a factory so we can have different hashing/caching behaviour when N-best parsing.
     */
    public static class SyntaxTreeNodeFactory {
        private final int[][] categoryHash;
        private final int[][] dependencyHash;
        private final boolean hashWords;
        private final int maxSentenceLength;
        private final double unaryLogProb;
        private final double puncConjLogProb;

        /**
         * maxSentenceLength and numberOfLexicalCategories are needed so that it can pre-compute some hash values.
         */
        public SyntaxTreeNodeFactory(int maxSentenceLength,
                                     int numberOfLexicalCategories,
                                     double unaryLogProb,
                                     double puncConjLogProb) {
            this.unaryLogProb = unaryLogProb;
            this.puncConjLogProb = puncConjLogProb;
            this.maxSentenceLength = maxSentenceLength;
            hashWords = numberOfLexicalCategories > 0;
            categoryHash = makeRandomArray(maxSentenceLength, numberOfLexicalCategories + 1);
            dependencyHash = makeRandomArray(maxSentenceLength, maxSentenceLength);
        }

        private int[][] makeRandomArray(int x, int y) {
            Random random = new Random();
            int[][] result = new int[x][y];
            for (int i = 0; i < x; i++) {
                for (int j = 0; j < y; j++) {
                    result[i][j] = random.nextInt();
                }
            }

            return result;
        }

        public SyntaxTreeNodeLeaf makeTerminal(String word, Category category, double probability, int sentencePosition) {
            return new SyntaxTreeNodeLeaf(
                    word, category, probability,
                    hashWords ? categoryHash[sentencePosition % this.maxSentenceLength][category.getID()] : 0, 0, sentencePosition);
        }

        public SyntaxTreeNode makeUnary(Category category, SyntaxTreeNode child) {
            return new SyntaxTreeNodeUnary(category, child.probability + this.unaryLogProb, child.hash, child.totalDependencyLength, child.getHeadIndex(), child);
        }

        public SyntaxTreeNode makeBinary(Category category, SyntaxTreeNode left, SyntaxTreeNode right, RuleType ruleType, boolean headIsLeft, double spanScore) {

            int totalDependencyLength = (right.getHeadIndex() - left.getHeadIndex())
                    + left.totalDependencyLength + right.totalDependencyLength;

            int hash;
            if (right.getCategory().isPunctuation()) {
                // Ignore punctuation when calculating the hash, because we don't really care where a full-stop attaches.
                hash = left.hash;
            } else if (left.getCategory().isPunctuation()) {
                // Ignore punctuation when calculating the hash, because we don't really care where a full-stop attaches.
                hash = right.hash;
            } else {
                // Combine the hash codes in a commutive way, so that left and right branching derivations can still be equivalent.
                hash = left.hash ^ right.hash ^ dependencyHash[left.getHeadIndex()%this.maxSentenceLength][right.getHeadIndex()%this.maxSentenceLength];
            }


            double probability = left.probability + right.probability + spanScore;
            if(left.category.isPunctuation() && ruleType==RuleType.CONJ_BOTTOM)
                probability += this.puncConjLogProb;


            return new SyntaxTreeNodeBinary(
                    category,
                    probability, // log probabilities
                    hash,
                    totalDependencyLength,
                    headIsLeft ? left.getHeadIndex() : right.getHeadIndex(),
                    ruleType,
                    headIsLeft,
                    left,
                    right);
        }
    }

    abstract void accept(SyntaxTreeNodeVisitor v);

    public interface SyntaxTreeNodeVisitor {
        void visit(SyntaxTreeNodeBinary node);

        void visit(SyntaxTreeNodeUnary node);

        void visit(SyntaxTreeNodeLeaf node);
    }

    public abstract RuleType getRuleType();

    public Category getCategory() {
        return category;
    }
}