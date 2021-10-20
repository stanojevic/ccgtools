package astar;

class CCGBankParsePrinterVisitor implements SyntaxTreeNode.SyntaxTreeNodeVisitor {

    public static String tree2string(SyntaxTreeNode entry) {
        StringBuilder result = new StringBuilder();
        if (entry == null) {
            return "FAIL";
        } else {
            entry.accept(new CCGBankParsePrinterVisitor(result));
        }
        return result.toString();
    }

    private final StringBuilder result;

    CCGBankParsePrinterVisitor(StringBuilder result) {
        this.result = result;
    }

    public void visit(SyntaxTreeNode.SyntaxTreeNodeBinary node) {
        result.append("(<T ")
              .append(node.getCategory().toString())
              .append(" ")
              .append(node.headIsLeft ? "0" : "1")
              .append(" 2> ");
        node.leftChild.accept(this);
        node.rightChild.accept(this);
        result.append(") ");
    }

    public void visit(SyntaxTreeNode.SyntaxTreeNodeUnary node) {
        result.append("(<T ");
        result.append(node.getCategory().toString());
        result.append(" 0 1> ");
        node.child.accept(this);
        result.append(") ");
    }

    public void visit(SyntaxTreeNode.SyntaxTreeNodeLeaf node) {
        result.append("(<L ");
        result.append(node.getCategory().toString());
        result.append(" X X ");
        result.append(normalize(node.getWord()));
        result.append(" ");
        result.append(node.getCategory().toString());
        result.append(">) ");
    }

    private static String normalize(String word) {
        if (word.length() > 1) {
            return word;
        } else if (word.equals("{")) {
            return "-LRB-";
        } else if (word.equals("}")) {
            return "-RRB-";
        } else if (word.equals("(")) {
            return "-LRB-";
        } else if (word.equals(")")) {
            return "-RRB-";
        }
        return word;
    }

}
