from .combinators import *
from .universal_tree_visualization import *
from .derivation import Node
from typing import List, Tuple
import os
import shutil as sh


def jupyter_display(tree, vtype=None, title=None, warning=False):
    from IPython.display import Image, display, HTML
    if vtype is None or vtype in [1, 'latex', 'proof']:
        try:
            tmp_fn = create_temp_file("", "png")
            save(tree, tmp_fn, vtype=1)
            _jupyter_display_object_with_title(Image(tmp_fn), title=title, warning=warning)
        except Exception as e:
            if vtype is None:
                jupyter_display(tree, vtype=2)
            else:
                raise e
    else:
        _jupyter_display_object_with_title(CCG_dot_Visualize._to_simple_node(tree),
                                           title=title, warning=warning)


def _jupyter_display_object_with_title(obj, title=None, warning=False):
    from IPython.display import display, HTML
    display(HTML("<br/>"))
    if title:
        display(HTML(f"<b><font size=\"3\" color=\"red\">{title}</font></b>"))
    if warning:
        display(HTML(f"<font color=\"red\">WARNING: {warning}</font>"))
    display(obj)


def jupyter_download(tree, vtype=None):
    return jupyter_download_link(lambda fn: save(tree=tree, fn=fn, vtype=vtype), name_prefix="tree")


def visualize(tree, graph_label="", vtype=None):
    fn = create_temp_file(graph_label, "pdf")
    save(tree, fn=fn, vtype=vtype)
    open_default(fn)


def save(tree, fn, vtype=None):
    if vtype is None:
        try:
            save(tree, fn, vtype=1)
        except Exception as e:
            save(tree, fn, vtype=2)
    elif vtype == 1 or vtype == 'latex' or vtype == 'proof':
        LaTeX.save_image(tree, fn)
    elif vtype == 2 or vtype == 'graphviz' or vtype == 'dot':
        CCG_dot_Visualize._to_simple_node(tree).save(fn)
    else:
        raise Exception(f"provided vtype {vtype} is not supported")


class CCG_dot_Visualize:

    @staticmethod
    def _to_simple_node(node) -> SimpleNode:
        if len(node.children) == 0:
            if not hasattr(node, 'semantics') or node.semantics is None:
                label = "%s\n%s\n%d" % (node.word, node.cat, node.span[0])
            else:
                label = "%s\n%s\n%s\n%d" % (node.word, node.cat, node.semantics, node.span[0])
            return SimpleNode(
                label=label,
                children=[],
                shape=Shape.RECTANGLE,
                color=Color.LIGHT_BLUE)
        elif len(node.children) == 1:
            if node.comb.is_unary_coord:
                shape = Shape.HEXAGON
                color = Color.BLUE
            else:
                shape = Shape.BOX
                color = Color.GREEN

            if not hasattr(node, 'semantics') or node.semantics is None:
                label="%s\n%s" % (node.comb, node.cat)
            else:
                label="%s\n%s\n%s" % (node.comb, node.cat, node.semantics)

            # mark head on the simple child
            return SimpleNode(
                label=label,
                children=[CCG_dot_Visualize._to_simple_node(child) for child in node.children],
                shape=shape, color=color)
        elif len(node.children) == 2:

            children = [CCG_dot_Visualize._to_simple_node(child) for child in node.children]

            comb = node.comb
            if isinstance(comb, B):
                if comb.is_forward:
                    color = Color.RED
                else:
                    color = Color.PURPLE
            elif isinstance(comb, Conj):
                color = Color.BLUE
                if comb.is_bottom:
                    children[0].arc_style = ArcStyle.BOLD
                    children[0].arc_color = Color.BLUE
                else:
                    children[1].arc_style = ArcStyle.BOLD
                    children[1].arc_color = Color.BLUE
            elif isinstance(comb, Punc):
                color = Color.LIGHT_BLUE
                if comb.punc_is_left:
                    children[0].arc_style = ArcStyle.DOTTED
                    children[0].arc_color = Color.BLACK
                else:
                    children[1].arc_style = ArcStyle.DOTTED
                    children[1].arc_color = Color.BLACK
            else:
                color = Color.BLACK

            shape = Shape.HEXAGON if isinstance(comb, Conj) else Shape.RECTANGLE

            if isinstance(comb, B):
                arrow = ">" if comb.is_forward else "<"
                cross = "" if comb.is_harmonic else "x"
                comb_str = arrow + "B" + str(comb.order) + cross
                # comb_str = str(comb)
            else:
                comb_str = str(comb)

            if not hasattr(node, 'semantics') or node.semantics is None:
                label="%s\n%s" % (comb_str, node.cat)
            else:
                label="%s\n%s\n%s" % (comb_str, node.cat, node.semantics)

            return SimpleNode(
                label=label,
                children=children,
                shape=shape, color=color)
        else:
            raise Exception("wrong number of children")

    @staticmethod
    def ipython(tree: Node):
        simple = CCG_dot_Visualize._to_simple_node(tree)
        return simple.ipython()

    @staticmethod
    def to_dot(tree: Node):
        simple = CCG_dot_Visualize._to_simple_node(tree)
        return simple.to_dot()


class CCGDepsDesc:

    def __init__(self, words, deps, leftmost_index=0):
        self.words = words
        self.deps = deps
        self.leftmost_index = leftmost_index

    def __iter__(self):
        for d in self.deps:
            yield d

    def add(self, dep):
        self.deps.add(dep)

    def _repr_html_(self):
        return self._construct_general_deps_desc()._repr_html_()

    def _construct_general_deps_desc(self):
        deps_list = [(d.head_pos, d.dep_pos, d.label, "dashed" if d.is_adj else "solid") for d in self.deps]
        return DepsDesc(words=self.words, deps=deps_list, starting_position=self.leftmost_index)

    def display(self, title: str = None, warning=False):
        _jupyter_display_object_with_title(self._construct_general_deps_desc(), title=title, warning=warning)

    def save(self, fn: str):
        self._construct_general_deps_desc().save(fn)

    def visualize(self, graph_label="ccg dependencies"):
        self._construct_general_deps_desc().visualize(graph_label)

    def download(self):
        from .visualization import jupyter_download
        return jupyter_download_link(self._construct_general_deps_desc().save, name_prefix="ccg_deps")


class DepsDesc:
    """This is a very general class for representing dependencies that are not necessarily coming out of CCG"""

    def __init__(self, words: List[str], deps: List[Tuple[int, int, str, str]], starting_position: int):
        self.starting_position = starting_position
        self.words = words
        self.deps = deps
        self.show_unconnected_words = False

    def _repr_html_(self):
        graph_label = "ipython_tree"
        file_type = "svg"
        file = create_temp_file(graph_label, file_type)
        self.save(file, include_disconnected_words=self.show_unconnected_words)
        with open(file) as fh:
            x = fh.read()
        x = re.sub(r"<svg width=\".*?\" height=\".*?\"", "<svg width=\"100%\" height=\"100%\"", x)
        return x

    def save_simple_deps_image(self, fn: str):
        run_latex(self.to_simple_deps_latex(), fn)

    def to_simple_deps_latex(self):
        res = ""
        res += "\\documentclass{standalone}\n"
        res += "\\usepackage{tikz-dependency}\n"
        res += "\\begin{document}\n"
        res += "\\begin{dependency}\n"

        res += "\\begin{deptext}\n"
        res += "\t" + " \\& ".join(escape_latex_text(w) for w in self.words) + "\\\\\n"
        res += "\\end{deptext}\n"

        for d in self.deps:
            res += "\t\\depedge{%d}{%d}{%s}\n" % (d[0]+1, d[1]+1, escape_latex_text(d[2]))

        res += "\\end{dependency}\n"
        res += "\\end{document}\n"
        return res

    def visualize(self,
                  graph_label: str = "dependencies",
                  file_type: str = "pdf",
                  renderer: str = "dot",
                  include_disconnected_words: bool = False,
                  include_word_positions: bool = False) -> None:
        file = create_temp_file(graph_label, file_type)
        self.save(file,
                  renderer=renderer,
                  include_disconnected_words=include_disconnected_words,
                  include_word_positions=include_word_positions)
        open_default(file)

    def save(self,
             fn: str,
             renderer: str = "dot",
             include_disconnected_words: bool = False,
             include_word_positions: bool = False) -> None:
        run_graphviz(dot_string=self.to_dot(include_disconnected_words=include_disconnected_words,
                                            include_word_positions=include_word_positions),
                     out_image_file=fn,
                     renderer=renderer)

    def to_dot(self, include_disconnected_words: bool = False, include_word_positions: bool = False):
        deps = self.deps
        words = enumerate(self.words, self.starting_position)
        if not include_disconnected_words:
            connected = {d[0] for d in deps} | {d[1] for d in deps}
            words = [w for w in words if w[0] in connected]

        if include_word_positions:
            words = [(i, "%s\n%d" % (w, i)) for i, w in words]
        node_strs = ["node%d[label=\"%s\" color=\"lightblue\" style=\"filled\"];" % (i, escape_for_dot(word)) for i, word in words]
        node_strs = "\n".join(node_strs)

        middle_strs = ["\nnode%d -> node%d [label=\"%s\",style=%s];" % d for d in deps]
        middle_strs = "\n".join(middle_strs)

        return "digraph G {\n%s\n%s\n}" % (node_strs, middle_strs)


class LaTeX:

    @staticmethod
    def visualize(tree: Node):
        file = create_temp_file("ccg", "pdf")
        LaTeX.save_image(tree, file)
        open_default(file)

    @staticmethod
    def save_image(tree: Node, fn: str):
        from os.path import realpath, dirname, join
        from os import getcwd
        import subprocess
        try:
            subprocess.check_output(['kpsewhich', 'standalone.sty'])  # check that standalone.sty is installed
        except FileNotFoundError:
            raise Exception("You need to have kpsewhich latex program installed")
        except subprocess.CalledProcessError:
            raise Exception("You need to have standalone.sty latex package installed")
        ccg_sty = join(realpath(join(getcwd(), dirname(__file__))), "ccg.sty")
        run_latex(LaTeX.to_latex(tree), out_file=fn, include_files=[ccg_sty])

    @staticmethod
    def to_latex(tree: Node):
        is_chinese = getattr(tree, 'language', 'English') == "Chinese"
        contains_semantics = False
        tree.assign_word_positions()
        tops = _node_topological_grouping(tree)
        terms = tops[0]
        ws = ["\\rm \\text{" + escape_latex_math(t.word) + "}" for t in terms]
        out = [" & ".join(ws)]
        for level in tops:
            curr_pos = 0
            line_comb  = []
            line_label = []
            line_sem   = []
            for node in level:
                skip = node.span[0] - curr_pos
                span_size = node.span[1] - node.span[0]
                if node.is_binary and node.comb.is_special_right_adj:
                    x = node.comb.span[0]-curr_pos
                    skip += x
                    span_size -= x
                line_comb  += [" "]*skip
                line_label += [" "]*skip
                line_sem   += [" "]*skip
                curr_pos = node.span[1]
                if node.is_term:
                    comb_cmd = "uuline"
                elif node.is_binary:
                    if node.comb.is_B_fwd or node.comb.is_B_bck:
                        f = "f" if node.comb.is_forward else "b"
                        x = "x" if node.comb.is_crossed else ""
                        s = ["apply", "comp", "comptwo", "compthree"][node.comb.order]
                        comb_cmd = f+x+s
                    elif node.comb.is_conj_top:
                        comb_cmd = "conjl"
                    elif node.comb.is_conj_bottom:
                        comb_cmd = "conjr"
                    elif node.comb.is_punc_left:
                        comb_cmd = "puncl"
                    elif node.comb.is_punc_right:
                        comb_cmd = "puncr"
                    elif node.comb.is_special_right_adj:
                        comb_cmd = "rightadj"
                    else:
                        comb_cmd = "tcb"
                elif node.is_unary:
                    if node.comb.is_type_raise:
                        comb_cmd = "ftype" if node.comb.is_forward else "btype"
                    else:
                        comb_cmd = "tc"
                else:
                    comb_cmd = "uuline"
                label = str(node.cat).replace("\\", "{\\bs}")
                line_comb.append("\\%s{%d}" % (comb_cmd, span_size))
                line_label.append("\\mc{%d}{\\it %s}" % (span_size, label))
                if hasattr(node, 'semantics') and node.semantics is not None:
                    line_sem.append("\\mc{%d}{\\textcolor{red}{%s}}" % (span_size, node.semantics.to_latex()))
                    contains_semantics = True
                else:
                    line_sem.append("\\mc{%d}{%s}" % (span_size, " "))
            skip = len(terms)-curr_pos
            line_comb  += [" "]*skip
            line_sem   += [" "]*skip
            line_label += [" "]*skip
            out.append(" & ".join(line_comb))
            out.append(" & ".join(line_label))
            if contains_semantics:
                out.append(" & ".join(line_sem))
            contains_semantics = False
        res = ""
        res += "\\documentclass{standalone}\n"
        if is_chinese:
            res += "\\usepackage[UTF8]{ctex}  % in case there are Chinese characters\n"
        res += "\\usepackage{xcolor}  % this is for coloring logical formulas\n"
        res += "\\usepackage{mathptmx}  % this is for nicer looking lambdas\n"
        res += "\\usepackage{amsmath}  % this is for \\operatorname\n"
        res += "\\usepackage{ccg}  % you can find the necessary style file ccg.sty in the repository of the parser\n"
        res += "% currently available at https://raw.githubusercontent.com/stanojevic/ccgtools/main/ccg/ccg.sty"
        res += "\\usepackage{amsmath}\n"
        res += "\\begin{document}\n"
        if is_chinese:
            res += "\\begin{CJK*}{UTF8}{gbsn}  % in case there are Chinese characters\n"
        res += "\\deriv{%d}{\n\t%s\n}\n" % (len(terms), "\\\\\n\t".join(out))
        if is_chinese:
            res += "\\end{CJK*}  % in case there are Chinese characters\n"
        res += "\\end{document}\n"
        return res


class ASCII_Art:

    @staticmethod
    def centerd_str(s: str, total: int) -> str:
        n = len(s)
        space = total-n
        lspace = rspace = space // 2
        if space % 2 == 1:
            rspace += 1
        return " "*lspace + s + " "*rspace

    @staticmethod
    def deriv2ascii(tree: Node) -> str:
        centerd_str = ASCII_Art.centerd_str
        tree.assign_word_positions()
        tops = _node_topological_grouping(tree)
        terms = tops.pop(0)
        lens = [max(len(t.word)+2, len(str(t.cat))+2, 5) for t in terms]
        out = ["".join([centerd_str(t.word, l)     for t, l in zip(terms, lens)]),
               "".join([centerd_str("-"*(l-2), l) for t, l in zip(terms, lens)]),
               "".join([centerd_str(str(t.cat), l) for t, l in zip(terms, lens)])]
        for level in tops:
            comb_line = ""
            cat_line = ""
            curr_word = 0
            for node in level:
                if node.is_binary and node.comb.is_special_right_adj:
                    span = (node.comb.span[0], node.span[1])
                    comb_str = "<R"
                else:
                    span = node.span
                    comb_str = str(node.comb)
                space_skip = " "*sum(lens[curr_word:span[0]])

                comb_line += space_skip
                cat_line += space_skip
                curr_word = span[1]

                node_len = sum(lens[span[0]:span[1]])
                comb_line += centerd_str("-"*(node_len-len(comb_str)-2)+comb_str, node_len)
                cat_line += centerd_str(str(node.cat), node_len)
            out.append(comb_line)
            out.append(cat_line)

        out_str = "\n".join(out)
        return out_str


def _node_topological_grouping(node):
    if node.is_term:
        return [[node]]
    elif node.is_unary:
        return _node_topological_grouping(node.child) + [[node]]
    elif node.is_binary:
        ltop = _node_topological_grouping(node.left)
        rtop = _node_topological_grouping(node.right)
        res = []
        for i in range(min(len(ltop), len(rtop))):
            res.append(ltop[i]+rtop[i])
        mtop = ltop if len(ltop)>len(rtop) else rtop
        res += mtop[len(res):]
        res += [[node]]
        return res
    else:
        raise Exception("I didn't expect this")
