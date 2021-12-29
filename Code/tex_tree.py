import numpy as np
import os

def generate_tex_tree(M, history, path):

    alpha0 = M[0, 0]
    beta0  = M[0, 1]
    alpha1 = M[1, 0]
    beta1  = M[1, 1]

    with open(os.path.join(path, 'tex_tree.tex'), 'w') as f:

        f.write('\\begin{minipage}{\\textwidth}\n')
        f.write('\\begin{tikzpicture}\n') 
        f.write('\\tikzstyle{state} = [rectangle, text centered, minimum width=1.2cm, minimum height=1cm, draw=black, fill=orange!30]\n')
        f.write(r'\node[state] at (-10,0)  (h0){\shortstack{$\alpha_0=%u, \beta_0=%u$ \\ $\alpha_1=%u, \beta_1=%u$}};'%(alpha0, beta0, alpha1, beta1) + '\n')
        f.write(r'\node[state] at (-2, 5.7) (h1_0){\shortstack{$\alpha_0=%u, \beta_0=%u$ \\ $\alpha_1=%u, \beta_1=%u$}};'%(alpha0+1, beta0, alpha1, beta1) + '\n')
        f.write(r'\node[state] at (-2, 4.5) (h1_1){\shortstack{$\alpha_0=%u, \beta_0=%u$ \\ $\alpha_1=%u, \beta_1=%u$}};'%(alpha0, beta0+1, alpha1, beta1) + '\n')
        f.write(r'\node[state] at (-2, -4.5) (h1_2){\shortstack{$\alpha_0=%u, \beta_0=%u$ \\ $\alpha_1=%u, \beta_1=%u$}};'%(alpha0, beta0, alpha1+1, beta1) + '\n')
        f.write(r'\node[state] at (-2, -5.7) (h1_3){\shortstack{$\alpha_0=%u, \beta_0=%u$ \\ $\alpha_1=%u, \beta_1=%u$}};'%(alpha0, beta0, alpha1, beta1+1) + '\n')
        f.write(r'\node[state] at (6, 9) (h2_0){\shortstack{$\alpha_0=%u, \beta_0=%u$ \\ $\alpha_1=%u, \beta_1=%u$}};'%(alpha0+2, beta0, alpha1, beta1) + '\n')
        f.write(r'\node[state] at (6, 7.8) (h2_1){\shortstack{$\alpha_0=%u, \beta_0=%u$ \\ $\alpha_1=%u, \beta_1=%u$}};'%(alpha0+1, beta0+1, alpha1, beta1) + '\n')
        f.write(r'\node[state] at (6, 6.6) (h2_2){\shortstack{$\alpha_0=%u, \beta_0=%u$ \\ $\alpha_1=%u, \beta_1=%u$}};'%(alpha0+1, beta0, alpha1+1, beta1) + '\n')
        f.write(r'\node[state] at (6, 5.4) (h2_3){\shortstack{$\alpha_0=%u, \beta_0=%u$ \\ $\alpha_1=%u, \beta_1=%u$}};'%(alpha0+1, beta0, alpha1, beta1+1) + '\n')
        f.write(r'\node[state] at (6, 4.2) (h2_4){\shortstack{$\alpha_0=%u, \beta_0=%u$ \\ $\alpha_1=%u, \beta_1=%u$}};'%(alpha0+1, beta0+1, alpha1, beta1) + '\n')
        f.write(r'\node[state] at (6, 3.0) (h2_5){\shortstack{$\alpha_0=%u, \beta_0=%u$ \\ $\alpha_1=%u, \beta_1=%u$}};'%(alpha0, beta0+2, alpha1, beta1) + '\n')
        f.write(r'\node[state] at (6, 1.8) (h2_6){\shortstack{$\alpha_0=%u, \beta_0=%u$ \\ $\alpha_1=%u, \beta_1=%u$}};'%(alpha0, beta0+1, alpha1+1, beta1) + '\n')
        f.write(r'\node[state] at (6, 0.6) (h2_7){\shortstack{$\alpha_0=%u, \beta_0=%u$ \\ $\alpha_1=%u, \beta_1=%u$}};'%(alpha0, beta0+1, alpha1, beta1+1) + '\n')
        f.write(r'\node[state] at (6, -0.8) (h2_8){\shortstack{$\alpha_0=%u, \beta_0=%u$ \\ $\alpha_1=%u, \beta_1=%u$}};'%(alpha0+1, beta0, alpha1+1, beta1) + '\n')
        f.write(r'\node[state] at (6, -2) (h2_9){\shortstack{$\alpha_0=%u, \beta_0=%u$ \\ $\alpha_1=%u, \beta_1=%u$}};'%(alpha0, beta0+1, alpha1+1, beta1) + '\n')
        f.write(r'\node[state] at (6, -3.2) (h2_10){\shortstack{$\alpha_0=%u, \beta_0=%u$ \\ $\alpha_1=%u, \beta_1=%u$}};'%(alpha0, beta0, alpha1+2, beta1) + '\n')
        f.write(r'\node[state] at (6, -4.4) (h2_11){\shortstack{$\alpha_0=%u, \beta_0=%u$ \\ $\alpha_1=%u, \beta_1=%u$}};'%(alpha0, beta0, alpha1+1, beta1+1) + '\n')
        f.write(r'\node[state] at (6, -5.6) (h2_12){\shortstack{$\alpha_0=%u, \beta_0=%u$ \\ $\alpha_1=%u, \beta_1=%u$}};'%(alpha0+1, beta0, alpha1, beta1+1) + '\n')
        f.write(r'\node[state] at (6, -6.8) (h2_13){\shortstack{$\alpha_0=%u, \beta_0=%u$ \\ $\alpha_1=%u, \beta_1=%u$}};'%(alpha0, beta0+1, alpha1, beta1+1) + '\n')
        f.write(r'\node[state] at (6, -8) (h2_14){\shortstack{$\alpha_0=%u, \beta_0=%u$ \\ $\alpha_1=%u, \beta_1=%u$}};'%(alpha0, beta0, alpha1+1, beta1+1) + '\n')
        f.write(r'\node[state] at (6, -9.2) (h2_15){\shortstack{$\alpha_0=%u, \beta_0=%u$ \\ $\alpha_1=%u, \beta_1=%u$}};'%(alpha0, beta0, alpha1, beta1+2) + '\n')

        f.write('\\begin{scope}[cyan!40!black]\n')
        for idx, his in enumerate(history):
            if his[-2] == 0:
                if his[-1] == (0, 0):
                    colour = 'red'
                    break
                else:
                    colour = 'black'
        if colour == 'black':
            idx = ''
        f.write('\\draw[thick, ->, %s] (h0) -- (h1_0) node[sloped, pos=0.5, above] {Arm $0$} node[sloped, pos=0.5, below]{%s};\n'%(colour, idx))
        for idx, his in enumerate(history):
            if his[-2] == 0:
                if his[-1] == (1, 1):
                    colour = 'red'
                    break
                else:
                    colour = 'black'
        if colour == 'black':
            idx = ''
        f.write('\\draw[thick,->, %s] (h0) -- (h1_2) node[sloped, pos=0.5, above] {Arm $1$} node[sloped, pos=0.5, below]{%s};\n'%(colour, idx))
        f.write('\\draw[thick,->] (h1_0) -- (h2_0) node[sloped, pos=0.5, above] {Arm $0$};\n')
        f.write('\\draw[thick,->] (h1_0) -- (h2_2) node[sloped, pos=0.5, above] {Arm $1$};\n')
        f.write('\\draw[thick,->] (h1_1) -- (h2_4) node[sloped, pos=0.5, above] {Arm $0$};\n')
        f.write('\\draw[thick,->] (h1_1) -- (h2_6) node[sloped, pos=0.5, above] {Arm $1$};\n')
        f.write('\\draw[thick,->] (h1_2) -- (h2_8) node[sloped, pos=0.5, above] {Arm $0$};\n')
        f.write('\\draw[thick,->] (h1_2) -- (h2_10) node[sloped, pos=0.5, above] {Arm $1$};\n')
        f.write('\\draw[thick,->] (h1_3) -- (h2_12) node[sloped, pos=0.5, above] {Arm $0$};\n')
        f.write('\\draw[thick,->] (h1_3) -- (h2_14) node[sloped, pos=0.5, above] {Arm $1$};\n')
        f.write('\\end{scope}\n')
        f.write('\\end{tikzpicture}\n')
        f.write('\\end{minipage}\n')

    return None