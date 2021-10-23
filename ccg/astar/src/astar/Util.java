package astar;

public final class Util {
    public static int indexOfAny(String haystack, String needles) {
        for (int i = 0; i < haystack.length(); i++) {
            for (int j = 0; j < needles.length(); j++) {
                if (haystack.charAt(i) == needles.charAt(j)) {
                    return i;
                }
            }
        }

        return -1;
    }

    public static int findClosingBracket(String source) {
        return findClosingBracket(source, 0);
    }

    public static int findClosingBracket(String source, int startIndex) {
        int openBrackets = 0;
        for (int i = startIndex; i < source.length(); i++) {
            if (source.charAt(i) == '(') {
                openBrackets++;
            } else if (source.charAt(i) == ')') {
                openBrackets--;
            }

            if (openBrackets == 0) {
                return i;
            }
        }

        throw new Error("Mismatched brackets in string: " + source);
    }

    /**
     * Finds the first index of a needle character in the haystack, that is not nested in brackets.
     */
    public static int findNonNestedChar(String haystack, String needles) {
        int openBrackets = 0;

        for (int i = 0; i < haystack.length(); i++) {
            if (haystack.charAt(i) == '(') {
                openBrackets++;
            } else if (haystack.charAt(i) == ')') {
                openBrackets--;
            } else if (openBrackets == 0) {
                for (int j = 0; j < needles.length(); j++) {
                    if (haystack.charAt(i) == needles.charAt(j)) {
                        return i;
                    }
                }
            }
        }

        return -1;
    }

}
