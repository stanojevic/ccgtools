package astar;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

import astar.Combinator.RuleProduction;
import astar.Combinator.RuleType;
import astar.SyntaxTreeNode.SyntaxTreeNodeFactory;
import astar.SyntaxTreeNode.SyntaxTreeNodeLeaf;

public class AStar {

    public AStar(List<String> validRootCategories,
                 Map<Category, List<Category>> unaryRules,
                 SyntaxTreeNodeFactory nodeFactory,
                 boolean useNormalForm) {
        this.useNormalForm = useNormalForm;
        this.nodeFactory = nodeFactory;
        this.unaryRules = unaryRules;

        List<Combinator> combinators = new ArrayList<>(Combinator.STANDARD_COMBINATORS);

        this.binaryRules = Collections.unmodifiableList(new ArrayList<>(combinators));

        List<Category> cats = new ArrayList<>();
        for (String cat : validRootCategories) {
            cats.add(Category.valueOf(cat));
        }
        possibleRootCategories = Collections.unmodifiableSet(new HashSet<>(cats));
    }

    private final Collection<Combinator> binaryRules;
    private final Map<Category, List<Category>> unaryRules;
    private final boolean useNormalForm;

    private final SyntaxTreeNodeFactory nodeFactory;

    private final Collection<Category> possibleRootCategories;

    // Included to try to help me suck less at log probabilities...
    private final static double CERTAIN = 0.0;
    private final static double IMPOSSIBLE = Double.NEGATIVE_INFINITY;

    static class AgendaItem implements Comparable<AgendaItem> {
        private final SyntaxTreeNode parse;

        AgendaItem(SyntaxTreeNode parse, double outsideProbabilityUpperBound, int startOfSpan, int spanLength) {
            this.parse = parse;
            this.startOfSpan = startOfSpan;
            this.spanLength = spanLength;
            this.cost = parse.probability + outsideProbabilityUpperBound;

        }

        private final int startOfSpan;
        private final int spanLength;
        private final double cost;

        /**
         * Comparison function used to order the agenda.
         */
        @Override
        public int compareTo(AgendaItem o) {
            int result = Double.compare(o.cost, cost);

            if (result != 0 && Math.abs(o.cost - cost) < 0.0000001) {
                // Allow some tolerance on comparisons of Doubles.
                result = 0;
            }

            if (result == 0) {
                // All other things being equal, it works best to prefer parser with longer dependencies (i.e. non-local attachment).
                return parse.totalDependencyLength - o.parse.totalDependencyLength;
            } else {
                return result;
            }
        }
    }

    /**
     * Takes supertagged input and returns a set of parses.
     * <p>
     * Returns null if the parse fails.
     */
    public List<SyntaxTreeNode> parseAstar(List<List<SyntaxTreeNodeLeaf>> supertags,
                                           double[][] spanScores,
                                           int maxSteps) {

        List<Category> empty = new ArrayList<>();

        final int sentenceLength = supertags.size();
        final PriorityQueue<AgendaItem> agenda = new PriorityQueue<>();
        final ChartCell[][] chart = new ChartCell[sentenceLength][sentenceLength];

        final double[][] outsideProbabilitiesUpperBound = computeOutsideProbabilities(supertags);


        for (int word = 0; word < sentenceLength; word++) {
            for (SyntaxTreeNode entry : supertags.get(word)) {
                agenda.add(new AgendaItem(entry, outsideProbabilitiesUpperBound[word][word + 1], word, 1));
            }
        }


        int step = 0;
        while (chart[0][sentenceLength - 1] == null) {
            // Add items from the agenda, until we have enough parses.

            final AgendaItem agendaItem = agenda.poll();
            if (agendaItem == null) {
                break;
            }

            // Try to put an entry in the chart.
            ChartCell cell = chart[agendaItem.startOfSpan][agendaItem.spanLength - 1];
            if (cell == null) {
                cell = new ChartCell();
                chart[agendaItem.startOfSpan][agendaItem.spanLength - 1] = cell;
            }

            if (cell.add(agendaItem.parse)) {
                // If a new entry was added, update the agenda.

                //See if any Unary Rules can be applied to the new entry.
                if(! (agendaItem.parse.getRuleType() == RuleType.UNARY &&
                     ((SyntaxTreeNode.SyntaxTreeNodeUnary) agendaItem.parse).child.getRuleType() == RuleType.UNARY)
                   && ! (useNormalForm && agendaItem.parse.getRuleType() == RuleType.RP)){  // prevent unaries over punc
                    // condition prevents triple unary sequence
                    for (Category unaryRuleProduction : unaryRules.getOrDefault(agendaItem.parse.getCategory(), empty)) {
                        if (agendaItem.spanLength == sentenceLength) {
                            break;
                        }
                        if(agendaItem.parse.getRuleType() == RuleType.UNARY &&
                           ((SyntaxTreeNode.SyntaxTreeNodeUnary) agendaItem.parse).child.getCategory().featurelessMathces(unaryRuleProduction)){
                            // skip non-sense unary seqences that just change features
                            continue;
                        }

                        agenda.add(new AgendaItem(nodeFactory.makeUnary(unaryRuleProduction, agendaItem.parse),
                                outsideProbabilitiesUpperBound[agendaItem.startOfSpan][agendaItem.startOfSpan + agendaItem.spanLength],
                                agendaItem.startOfSpan, agendaItem.spanLength));
                    }
                }

                // See if the new entry can be the left argument of any binary rules.
                for (int spanLength = agendaItem.spanLength + 1; spanLength < 1 + sentenceLength - agendaItem.startOfSpan; spanLength++) {
                    // agenda + chart
                    SyntaxTreeNode leftEntry = agendaItem.parse;

                    ChartCell rightCell = chart[agendaItem.startOfSpan + agendaItem.spanLength][spanLength - agendaItem.spanLength - 1];
                    if (rightCell == null) continue;

                    double spanScore = 0.;
                    if (spanScores != null) {
                        int resultStart = agendaItem.startOfSpan;
                        int resultEnd = agendaItem.startOfSpan+spanLength-1;
                        spanScore = spanScores[resultStart][resultEnd];
                    }
                    for (SyntaxTreeNode rightEntry : rightCell.getEntries()) {
                        updateAgenda(agenda, agendaItem.startOfSpan, spanLength, leftEntry, rightEntry, sentenceLength, outsideProbabilitiesUpperBound[agendaItem.startOfSpan][agendaItem.startOfSpan + spanLength], spanScore);
                    }
                }

                // See if the new entry can be the right argument of any binary rules.
                for (int startOfSpan = 0; startOfSpan < agendaItem.startOfSpan; startOfSpan++) {
                    // chart + agenda
                    int spanLength = agendaItem.startOfSpan + agendaItem.spanLength - startOfSpan;
                    SyntaxTreeNode rightEntry = agendaItem.parse;

                    ChartCell leftCell = chart[startOfSpan][spanLength - agendaItem.spanLength - 1];
                    if (leftCell == null) continue;

                    double spanScore = 0.;
                    if (spanScores != null) {
                        int resultStart = startOfSpan;
                        int resultEnd = startOfSpan+spanLength-1;
                        spanScore = spanScores[resultStart][resultEnd];
                    }
                    for (SyntaxTreeNode leftEntry : leftCell.getEntries()) {
                        updateAgenda(agenda, startOfSpan, spanLength, leftEntry, rightEntry, sentenceLength, outsideProbabilitiesUpperBound[startOfSpan][startOfSpan + spanLength], spanScore);
                    }
                }
            }
            step += 1;
            if(step > maxSteps)
                break;
        }

        if (chart[0][sentenceLength - 1] == null) {
            // Parse failure.
            SyntaxTreeNode left = null;
            int start = 0;
            while (start <= sentenceLength - 1) {
                int spanSize = sentenceLength - start - 1;
                while (spanSize >= 0 && chart[start][spanSize] == null)
                    spanSize--;

                SyntaxTreeNode right;
                if(spanSize < 0){
                    right = supertags.get(start).get(0);
                    spanSize = 0;
                }else{
                    right = Collections.min(chart[start][spanSize].getEntries());
                }

                if (left == null) {
                    left = right;
                } else {
                    left = nodeFactory.makeBinary(Category.GLUE, left, right, RuleType.NOISE, true, 0.);
                }
                start += spanSize + 1;
            }
            return Collections.singletonList(left);
        }

        // Read the parses out of the final cell.
        List<SyntaxTreeNode> parses = new ArrayList<>(chart[0][sentenceLength - 1].getEntries());

        // Sort the parses by probability.
        Collections.sort(parses);

        return parses;

    }

    /**
     * Computes an upper bound on the outside probabilities of a span, for use as a heuristic in A*.
     * The upper bound is simply the product of the probabilities for the most probable supertag for
     * each word outside the span.
     */
    private double[][] computeOutsideProbabilities(List<List<SyntaxTreeNodeLeaf>> supertags) {
        int sentenceLength = supertags.size();
        final double[][] outsideProbability = new double[sentenceLength + 1][sentenceLength + 1];

        final double[] fromLeft = new double[sentenceLength + 1];
        final double[] fromRight = new double[sentenceLength + 1];


        fromLeft[0] = CERTAIN;
        fromRight[sentenceLength] = CERTAIN;

        List<SyntaxTreeNode> bestTagsPerPos = new ArrayList<>();
        for(List<SyntaxTreeNodeLeaf> tags : supertags)
            bestTagsPerPos.add(Collections.min(tags));

        for (int i = 0; i < sentenceLength - 1; i++) {
            int j = sentenceLength - i;
            // The supertag list for words is sorted, so the most probably entry is at index 0.
            fromLeft[i + 1] = fromLeft[i] + bestTagsPerPos.get(i).probability;
            fromRight[j - 1] = fromRight[j] + bestTagsPerPos.get(j - 1).probability;
        }

        for (int i = 0; i < sentenceLength + 1; i++) {
            for (int j = i; j < sentenceLength + 1; j++) {
                outsideProbability[i][j] = fromLeft[i] + fromRight[j];
            }
        }

        return outsideProbability;
    }

    /**
     * Updates the agenda with the result of all combinators that can be applied to leftChild and rightChild.
     */
    private void updateAgenda(
            final PriorityQueue<AgendaItem> agenda,
            final int startOfSpan,
            final int spanLength,
            final SyntaxTreeNode leftChild,
            final SyntaxTreeNode rightChild,
            final int sentenceLength,
            double outsideProbabilityUpperBound,
            double spanScore) {

        for (RuleProduction production : getRules(leftChild.getCategory(), rightChild.getCategory())) {
            if (useNormalForm && (leftChild.getRuleType() == RuleType.FC || leftChild.getRuleType() == RuleType.GFC) &&
                    (production.ruleType == RuleType.FA || production.ruleType == RuleType.FC || production.ruleType == RuleType.GFC)) {
                // Eisner normal form constraint.
                continue;
            }
            if (useNormalForm && (rightChild.getRuleType() == RuleType.BX || leftChild.getRuleType() == RuleType.GBX) &&
                    (production.ruleType == RuleType.BA || production.ruleType == RuleType.BX || leftChild.getRuleType() == RuleType.GBX)) {
                // Eisner normal form constraint.
                continue;
            }
            if (useNormalForm && leftChild.getRuleType() == RuleType.UNARY &&
                    production.ruleType == RuleType.FA && leftChild.getCategory().isForwardTypeRaised()) {
                // Hockenmaier normal form constraint.
                continue;
            }
            if (useNormalForm && rightChild.getRuleType() == RuleType.UNARY &&
                    production.ruleType == RuleType.BA && rightChild.getCategory().isBackwardTypeRaised()) {
                // Hockenmaier normal form constraint.
                continue;
            }
            if (useNormalForm && spanLength == sentenceLength && !possibleRootCategories.contains(production.category)) {
                // Enforce that the root node must have one of a pre-specified list of categories.
                continue;
            }
            if (useNormalForm && rightChild.getRuleType() == RuleType.RP){
                // punctuation normal form to attach punctuation as high as possible
                continue;
            }
            SyntaxTreeNode newNode = nodeFactory.makeBinary(production.category, leftChild, rightChild, production.ruleType, production.headIsLeft, spanScore);
            agenda.add(new AgendaItem(newNode, outsideProbabilityUpperBound, startOfSpan, spanLength));
        }
    }

    private final ConcurrentHashMap<Category, ConcurrentHashMap<Category, Collection<RuleProduction>>> ruleCache = new ConcurrentHashMap<>();

    private Collection<RuleProduction> getRules(Category left, Category right) {
        return ruleCache.computeIfAbsent(left , x -> new ConcurrentHashMap<>())
                        .computeIfAbsent(right, x -> Combinator.getRules(left, right, binaryRules));
    }

    private static final class ChartCell {
        private double bestValue = IMPOSSIBLE;

        private final Map<Category, SyntaxTreeNode> keyToProbability = new HashMap<>();

        public ChartCell() {
        }

        /**
         * Possibly adds a @CellEntry to this chart cell. Returns true if the parse was added, and false if the cell was unchanged.
         */
        public boolean add(SyntaxTreeNode entry) {
            // See if the cell already has enough parses with this category.
            // All existing entries are guaranteed to have a higher probability
            if (isFull(entry.getCategory())) {
                return false;
            } else {
                addEntry(entry.getCategory(), entry);

                if (entry.probability > bestValue) {
                    bestValue = entry.probability;
                }

                return true;
            }
        }

        boolean isFull(Category category) {
            return keyToProbability.containsKey(category);
        }

        public Collection<SyntaxTreeNode> getEntries() {
            return keyToProbability.values();
        }

        void addEntry(Category category, SyntaxTreeNode newEntry) {
            keyToProbability.put(category, newEntry);
        }
    }

}