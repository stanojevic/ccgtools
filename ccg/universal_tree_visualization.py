import os
from enum import Enum
from typing import Tuple, List
import atexit
import tempfile
import shutil as sh
import re


def escape_for_dot(s: str):
    return s.replace("\\", "\\\\").replace("\"", "\\\"")


def open_default(fn: str) -> None:
    import os
    from sys import platform
    if platform.startswith("win"):
        os.startfile(fn)
    elif platform.startswith("darwin"):
        os.system("open \"%s\"" % fn)
    elif platform.startswith("linux"):
        os.system("xdg-open \"%s\"" % fn)
    else:
        raise Exception("unknown system")


_tmp_list = []


def delete_at_exit(x):
    global _tmp_list
    _tmp_list.append(x)


def _deleting_at_exit():
    for x in _tmp_list:
        if os.path.isfile(x):
            os.unlink(x)
        elif os.path.isdir(x):
            sh.rmtree(x)


atexit.register(_deleting_at_exit)


def create_temp_file(prefix: str, extension: str) -> str:
    t = tempfile.NamedTemporaryFile(prefix=prefix + " ", suffix="." + extension).name
    # delete_at_exit(t)
    return t


def create_temp_dir(prefix: str, extension: str) -> str:
    x = create_temp_file(prefix, extension)
    if os.path.isfile(x):
        os.unlink(x)
    os.mkdir(x)
    return x


def run_graphviz(dot_string: str, out_image_file: str, renderer: str = "dot") -> None:
    """renderer determines the drawing algorithm. dot is good for DAG. circo for everything else"""
    import subprocess
    file_type = os.path.splitext(out_image_file)[1][1:]
    tmp_name = create_temp_file("dependencies_visualization", "dot")
    with open(tmp_name, "w") as fh:
        print(dot_string, file=fh)
    cmd = "%s -T%s \"%s\" -O" % (renderer, file_type, tmp_name)
    try:
        p = subprocess.Popen(cmd, shell=True)
        p.wait(1000)
        os.remove(tmp_name)
        sh.move(tmp_name + "." + file_type, out_image_file)
    except:
        raise Exception("graphviz dot failed")


def escape_latex_text(s: str) -> str:
    s = s.replace("\\", "\\textbackslash{}") \
        .replace("~", "\\textasciitilde{}")
    s = _escape_latex_general(s)
    return s


def escape_latex_math(s: str) -> str:
    s = s.replace("\\", "\\backslash{}") \
        .replace("~", "\\sim{}")
    s = _escape_latex_general(s)
    return s


def _escape_latex_general(s: str) -> str:
    return s.replace("%", "\\%") \
        .replace("$", "\\$") \
        .replace("&", "\\&") \
        .replace("{", "\\{") \
        .replace("}", "\\}") \
        .replace("_", "\\_") \
        .replace("-", "{-}") \
        .replace(" ", "\\ ")


def run_latex(tex: str, out_file: str, include_files: list = []) -> None:
    assert isinstance(include_files, list)
    from os.path import join
    import subprocess
    d = create_temp_dir("latex", "dir")
    for f in include_files:
        b = os.path.basename(f)
        sh.copyfile(f, join(d, b))
    with open(join(d, "main.tex"), "w") as fh:
        print(tex, file=fh)
    ret_code = subprocess.Popen("latex main 1> log.std 2> log.err", shell=True, cwd=d).wait(1000)
    if ret_code != 0:
        raise Exception("LaTeX failed")
    ext = out_file.split(".")[-1]
    if ext == "pdf":
        ret_code = subprocess.Popen("dvipdfmx main.dvi 1> log.std 2> log.err", shell=True, cwd=d).wait(1000)
        if ret_code != 0:
            raise Exception("dvipdfmx failed")
    elif ext == "png":
        ret_code = subprocess.Popen("dvipng -D 500 main.dvi -o main.png 1> log.std 2> log.err", shell=True, cwd=d).wait(1000)
        if ret_code != 0:
            raise Exception("dvipng failed")
    elif ext == "jpg" or ext == "jpeg":
        ret_code = subprocess.Popen("convert -density 500,500 main.dvi -strip main." + ext, shell=True, cwd=d).wait(1000)
        if ret_code != 0:
            raise Exception("ImageMagick failed")
    else:
        raise Exception("extension %s not supported" % ext)
    sh.copyfile(join(d, "main." + ext), out_file)
    sh.rmtree(d)


class ArcStyle(Enum):
    SOLID = "solid"
    DOTTED = "dotted"
    DASHED = "dashed"
    BOLD = "bold"
    INVISIBLE = "invis"


class Color(Enum):
    RED = "firebrick1"
    GREEN = "green3"
    BLUE = "blue"
    PURPLE = "purple"
    LIGHT_BLUE = "lightblue2"
    BLACK = "black"
    GRAY = "gray"
    YELLOW = "goldenrod2"


class Shape(Enum):
    TRAPEZIUM = "trapezium"
    HOUSE = "house"
    BOX = "box"
    HEXAGON = "hexagon"
    ELLIPSE = "ellipse"
    RECTANGLE = "rectangle"
    NOTHING = "plaintext"


class SimpleNode:

    def __init__(self,
                 label: str,
                 children,
                 color=None,
                 shape=None,
                 arc_color=None,
                 arc_style=None,
                 text_bold=False,
                 text_color=None,
                 text_size=-1,
                 position=0
                 ):
        self.label = label
        self.children = children
        self.color = color
        self.shape = shape
        self.arc_color = arc_color if arc_color else Color.BLACK
        self.arc_style = arc_style if arc_style else ArcStyle.SOLID
        self.text_bold = text_bold
        self.text_color = text_color if text_color else Color.BLACK
        self.text_size = text_size
        self.position = position

    def ipython(self):
        graph_label = "ipython_tree"
        file_type = "svg"
        file = create_temp_file(graph_label, file_type)
        self.save(file)
        with open(file, 'rb') as fh:
            x = fh.read()
        x = x.decode()
        x = re.sub(r"<svg width=\".*?\" height=\".*?\"", "<svg ", x)
        return x

    def visualize(self, graph_label: str = "CCG derivation", file_type: str = "pdf") -> None:
        file = create_temp_file(graph_label, file_type)
        self.save(file)
        open_default(file)

    def save(self, fn: str) -> None:
        run_graphviz(dot_string=self.to_dot(), out_image_file=fn)

    def _to_dot_rec(self, node_id: str) -> Tuple[str, List[str]]:
        out_str = ""
        terms = []
        out_str += node_id + "["
        if self.text_bold:
            out_str += "fontname=\"Times Bold\"; "
        out_str += "fontcolor=%s; " % self.text_color.value
        out_str += "label=\"" + escape_for_dot(self.label) + "\"; "
        shape = self.shape if self.shape else Shape.ELLIPSE
        out_str += "shape=%s; " % shape.value
        color = self.color if self.color else Color.BLACK
        out_str += "color=%s; " % color.value
        text_size = self.text_size if self.text_size > 0 else 10
        out_str += "fontsize=%d ; " % text_size
        out_str += "style=bold; "
        out_str += "];\n"

        for index, child in enumerate(self.children):
            child_name = "%s_%d" % (node_id, index)
            if child.children:
                s, t = child._to_dot_rec(child_name)
                out_str += s
                terms.extend(t)
            else:
                terms.append(child_name)
            out_str += "%s -- %s [color=%s, style=%s] ;\n" % (node_id,
                                                              child_name,
                                                              child.arc_color.value,
                                                              child.arc_style.value)

        return out_str, terms

    def all_terminals(self):
        if self.children:
            terminals = []
            for child in self.children:
                terminals.extend(child.all_terminals())
            return terminals
        else:
            return [self]

    def to_dot(self) -> str:
        out_str = "graph {\n"
        s, terms = self._to_dot_rec("node0")
        out_str += s
        out_str += "  subgraph {rank=same;rankdir=LR;\n"

        sorted_nodes_with_ids = sorted(zip(terms, self.all_terminals()), key=lambda x: x[1].position)

        for node_id, node in sorted_nodes_with_ids:
            color = node.color if node.color else Color.LIGHT_BLUE
            shape = node.shape if node.shape else Shape.NOTHING
            word = node.label
            out_str += "    " + node_id + "[ "
            if node.text_bold:
                out_str += "fontname=\"Times Bold\"; "
            text_color = node.text_color if node.text_color else Color.BLACK
            out_str += "fontcolor=%s; " % text_color.value
            out_str += "label=\"" + escape_for_dot(word) + "\" "
            out_str += "style=bold; "
            text_size = node.text_size if node.text_size > 0 else 10
            out_str += "fontsize=%d " % text_size
            out_str += "color=" + color.value + ";"
            out_str += "shape=" + shape.value + ";"
            out_str += "];\n"

        out_str += "    edge[style=\"invis\"];\n"

        if len(terms) > 1:
            out_str += "--".join([node_id for node_id, _ in sorted_nodes_with_ids])
            out_str += ";\n"

        out_str += "  }\n"

        out_str += "}\n"

        return out_str
