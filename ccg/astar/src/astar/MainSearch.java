package astar;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


public class MainSearch {

    private final Category[] categories;
    private final AStar astar;
    private final SyntaxTreeNode.SyntaxTreeNodeFactory factory;
    private final boolean useParallel;
    private final int maxSteps;
    private ExecutorService service = null;

    public static double[][][] emptySpans(int n){
        return new double[n][][];
    }

    private static Map<Category, List<Category>> constructUnaries(List<String> cats){
        Map<Category, List<Category>> unaryRules = new HashMap();
        for(int i=0 ; i<cats.size()/2; i++){
            Category child = Category.valueOf(cats.get(i));
            Category parent = Category.valueOf(cats.get(i+1));
            if(! unaryRules.containsKey(child)){
                unaryRules.put(child, new ArrayList<>());
            }
            unaryRules.get(child).add(parent);
        }
        return unaryRules;
    }

    public MainSearch(List<String> orderedCategories,
                      int maxSteps,
                      int numOfCpus,
                      double unaryProb,
                      double puncConjProb,
                      boolean useNormalForm) {
        this.maxSteps = maxSteps;
        this.useParallel = (numOfCpus != 1);
        if (numOfCpus > 1)
            System.setProperty("java.util.concurrent.ForkJoinPool.common.parallelism", Integer.toString(numOfCpus));
        this.categories = new Category[orderedCategories.size()];
        int index = 0;
        for (String cat_str : orderedCategories) {
            this.categories[index] = Category.valueOf(cat_str);
            index += 1;
        }

        Map<Category, List<Category>> unaryRules = constructUnaries(Arrays.asList(
                "NP", "S[X]/(S[X]/NP)",  // topicalization for English and Chinese
                "S[to]\\NP", "(S\\NP)\\(S\\NP)",  // added because it's frequent in English CCGbank ~ 1% of unaries
                "S[ng]\\NP", "(S\\NP)\\(S\\NP)",  // added because it's frequent in English CCGbank ~ 1% of unaries
                "N", "NP",
                "S[pss]\\NP", "NP\\NP",
                "S[ng]\\NP", "NP\\NP",
                "S[adj]\\NP", "NP\\NP",
                "S[to]\\NP", "NP\\NP",
                "S[to]\\NP", "N\\N",
                "S[dcl]/NP", "NP\\NP",
                "S[pss]\\NP", "S/S",
                "S[ng]\\NP", "S/S",
                "S[to]\\NP", "S/S",
                "S[ng]\\NP", "S\\S",
                "S[ng]\\NP", "NP",
                "NP", "S[X]/(S[X]\\NP)",
                "NP", "(S[X]\\NP)\\((S[X]\\NP)/NP)",
                "PP", "(S[X]\\NP)\\((S[X]\\NP)/PP)"
        ));

        this.factory = new SyntaxTreeNode.SyntaxTreeNodeFactory(
                100,
                Category.maxCatId(),
                Math.log(unaryProb),
                Math.log(puncConjProb)
                );
        this.astar = new AStar(
                Arrays.asList("S", "S[dcl]", "S[wq]", "S[q]", "S[qem]", "NP"),
                unaryRules,
                this.factory,
                useNormalForm);
    }

    public void shutdown(){
        if(this.service != null){
            this.service.shutdown();
            this.service = null;
        }
        // java.util.concurrent.ForkJoinPool.commonPool().shutdown();
    }

    public Future<List<String>> searchBatchFuture(List<List<String>> words,
                                                  long[][][] tagIndices,
                                                  double[][][] tagScores,
                                                  double[][] thresholds,
                                                  double[][][] spanScores) {
        if(this.service == null)
            this.service = Executors.newSingleThreadExecutor();
        return this.service.submit(() -> this.searchBatch(words, tagIndices, tagScores, thresholds, spanScores));
    }

    public List<String> searchBatch(List<List<String>> words,
                                    long[][][] tagIndices,
                                    double[][][] tagScores,
                                    double[][] thresholds,
                                    double[][][] spanScores) {
        int batchSize = words.size();
        IntStream stream = IntStream.range(0, batchSize);
        if (this.useParallel){
            stream = stream.parallel();
        }

        return  stream
                .mapToObj(i -> this.searchSingle(words.get(i), tagIndices[i],
                        tagScores[i], thresholds[i], spanScores[i]))
                .collect(Collectors.toList());
    }

    public String searchSingle(List<String> words,
                                long[][] tagIndices,
                                double[][] tagScores,
                                double[] thresholds,
                                double[][] spanScores
    ) {
        List<List<SyntaxTreeNode.SyntaxTreeNodeLeaf>> supertags = new ArrayList<>();
        for (int i = 0; i < words.size(); i++) {
            List<SyntaxTreeNode.SyntaxTreeNodeLeaf> currWordTags = new ArrayList<>();
            String word = words.get(i);
            long[] aTagIndices = tagIndices[i];
            double[] aTagScores = tagScores[i];
            double threshold = Double.NEGATIVE_INFINITY;
            if (thresholds != null)
                threshold = thresholds[i];
            for (int j = 0; j < aTagIndices.length; j++) {
                double score = aTagScores[j];
                int tagIndex = (int) aTagIndices[j];
                if(tagIndex >= this.categories.length){
                    throw new RuntimeException("\n\ntag index goes beyond available number of tags\n\n");
                }
                Category cat = this.categories[tagIndex];
                if (score >= threshold)
                    currWordTags.add(this.factory.makeTerminal(word, cat, score, i));
            }
            supertags.add(currWordTags);
        }
        List<SyntaxTreeNode> parses = this.astar.parseAstar(supertags, spanScores, this.maxSteps);
        return parses.get(0).toString();
    }

}


