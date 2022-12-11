
import sympy
#from sympy import sympify
#from IPython.display import Latex
from sympy.printing import latex as spl
def splitts(lhs, expr,terms_per_line=3):
    """
    This function is written to handle displaying long latex equations.
    """
    latex = r'\begin{aligned}' + '\n'
    terms = expr.as_ordered_terms()
    n_terms = len(terms)
    term_count = 1
    for i in range(n_terms):
        term = terms[i]
        term_start = r''
        term_end = r''
        sign = r'+'
        if term_count > terms_per_line:
            term_start = '&'
            term_count = 1
        if term_count == terms_per_line:
            term_end = r'\ldots \\' + '\n'
        if term.as_ordered_factors()[0]==-1:
            term = -1*term
            sign = r'-'
            
        if i == 0: 
            
            if sign == r'+': sign = r""
            latex += r'{:s} =& {:s} {:s} {:s} {:s}'.format(sympy.latex(lhs),
                                                     term_start,sign,sympy.latex(term),term_end)

        elif i == n_terms-1: # end
            latex += r'{:s} {:s} {:s} {:s}'.format(term_start,sign,sympy.latex(term),term_end)
        else: # middle
            latex += r'{:s} {:s} {:s} {:s}'.format(term_start,sign,sympy.latex(term),term_end)
        term_count += 1
    latex += r'\end{aligned}' + '\n'
    #return Latex(latex)
    #print(Latex(latex))
    #print(term)
    #print(latex)
    #print(spl(latex))
    return spl(latex)

