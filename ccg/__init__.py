from ccg.derivation import Node, DerivationLoader
from ccg.category import Category
from ccg import combinators as comb

load_file = DerivationLoader.iter_from_file
str2deriv = DerivationLoader.from_str
str2cat = Category.from_str

TypeChanging1 = comb.TypeChanging1  # class
TypeRaising = comb.TypeRaising  # class
tr_np_fwd = TypeRaising(cat_res=str2cat("S"), cat_arg=str2cat("NP"), is_forward=True)
tr_np_bck = TypeRaising(cat_res=str2cat("S\\NP"), cat_arg=str2cat("NP"), is_forward=False)

TypeChanging2 = comb.TypeChanging2  # class
RightAdjoin = comb.RightAdjoin  # class

Glue = comb.Glue  # class
glue = comb.Glue()

Conj = comb.Conj  # class
up_conj = comb.Conj(is_bottom=False)
bottom_conj = comb.Conj(is_bottom=True)

Punc = comb.Punc  # class
lpunc = comb.Punc(punc_is_left=True)
rpunc = comb.Punc(punc_is_left=False)

B = comb.B  # class
S = comb.S  # class

bx1f = B(is_forward=True, is_crossed=True, order=1)
bx1b = B(is_forward=False, is_crossed=True, order=1)
b1f = B(is_forward=True, is_crossed=False, order=1)
b0f = B(is_forward=True, is_crossed=False, order=0)
b0b = B(is_forward=False, is_crossed=False, order=0)

fapply = b0f
bapply = b0b
fcomp = b1f
fxcomp = bx1f

