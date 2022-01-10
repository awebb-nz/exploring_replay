import numpy as np

def generate_tex_tree(M, replays, save_path):

    # root prior values
    alpha0 = M[0, 0]
    beta0  = M[0, 1]
    alpha1 = M[1, 0]
    beta1  = M[1, 1]

    x0, y0 = -8, 0

    x1, y1 = -2, 5.6

    x2, y2 =  4, 9.2

    x1_0, y1_0 = x1,  y1
    # -------- #
    x1_2, y1_2 = x1, -y1

    x2_0, y2_0   = x2, y2
    x2_2, y2_2   = x2, y2-2.4
    x2_4, y2_4   = x2, y2-2.4*2
    x2_6, y2_6   = x2, y2-2.4*3
    # ---------- #
    x2_8, y2_8   = x2, y2-2.4*3 - 4
    x2_10, y2_10 = x2, y2-2.4*4 - 4
    x2_12, y2_12 = x2, y2-2.4*5 - 4
    x2_14, y2_14 = x2, y2-2.4*6 - 4

    with open(save_path, 'w') as f:

        f.write(r'\begin{minipage}{\textwidth}' + '\n')
        f.write(r'\begin{tikzpicture}' + '\n') 
        f.write(r'\tikzstyle{state}   = [rectangle, text centered, draw=black, minimum width=2.2cm, fill=orange!30]' + '\n')
        f.write(r'\tikzstyle{between} = [rectangle, draw=none, minimum width=3.1cm]' + '\n')

        # root
        f.write(r'\node[state] at (%.2f, %.2f)   '%(x0, y0)    + r'(h0){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0, beta0, alpha1, beta1) + '\n')
        
        # action 0 rew 1
        f.write(r'\node[state] at (%.2f, %.2f)  '%(x1_0, y1_0) + r'(h1_0){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0+1, beta0, alpha1, beta1) + '\n')
        f.write(r'\node[between] at (%.2f, %.2f) (h1_01){};'%(x1_0, y1_0-0.45) + '\n')
        # action 0 rew 0
        f.write(r'\node[state] at (%.2f, %.2f)'%(x1_0, y1_0-0.9)             + r'(h1_1){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0, beta0+1, alpha1, beta1) + '\n')

        # action 1 rew 1
        f.write(r'\node[state] at (%.2f, %.2f)  '%(x1_2, y1_2) + r'(h1_2){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0, beta0, alpha1+1, beta1) + '\n')
        f.write(r'\node[between] at (%.2f, %.2f) (h1_23){};'%(x1_2, y1_2-0.45) + '\n')
        # action 1 rew 0 
        f.write(r'\node[state] at (%.2f, %.2f)'%(x1_2, y1_2-0.9)             + r'(h1_3){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0, beta0, alpha1, beta1+1) + '\n')

        f.write(r'\node[state] at (%.2f, %.2f)  '%(x2_0, y2_0) + r'(h2_0){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0+2, beta0, alpha1, beta1) + '\n')
        f.write(r'\node[between] at (%.2f, %.2f) (h2_01){};'%(x2_0, y2_0-0.45) + '\n')
        f.write(r'\node[state] at (%.2f, %.2f)'%(x2_0, y2_0-0.9)             + r'(h2_1){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0+1, beta0+1, alpha1, beta1) + '\n')

        f.write(r'\node[state] at (%.2f, %.2f)  '%(x2_2, y2_2) + r'(h2_2){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0+1, beta0, alpha1+1, beta1) + '\n')
        f.write(r'\node[between] at (%.2f, %.2f) (h2_23){};'%(x2_2, y2_2-0.45) + '\n')
        f.write(r'\node[state] at (%.2f, %.2f)'%(x2_2, y2_2-0.9)             + r'(h2_3){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0+1, beta0, alpha1, beta1+1) + '\n')

        f.write(r'\node[state] at (%.2f, %.2f)  '%(x2_4, y2_4) + r'(h2_4){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0+1, beta0+1, alpha1, beta1) + '\n')
        f.write(r'\node[between] at (%.2f, %.2f) (h2_45){};'%(x2_4, y2_4-0.45) + '\n')
        f.write(r'\node[state] at (%.2f, %.2f)'%(x2_4, y2_4-0.9)             + r'(h2_5){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0, beta0+2, alpha1, beta1) + '\n')

        f.write(r'\node[state] at (%.2f, %.2f)  '%(x2_6, y2_6) + r'(h2_6){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0, beta0+1, alpha1+1, beta1) + '\n')
        f.write(r'\node[between] at (%.2f, %.2f) (h2_67){};'%(x2_6, y2_6-0.45) + '\n')
        f.write(r'\node[state] at (%.2f, %.2f)'%(x2_6, y2_6-0.9)             + r'(h2_7){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0, beta0+1, alpha1, beta1+1) + '\n')

        f.write(r'\node[state] at (%.2f, %.2f)  '%(x2_8, y2_8) + r'(h2_8){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0+1, beta0, alpha1+1, beta1) + '\n')
        f.write(r'\node[between] at (%.2f, %.2f) (h2_89){};'%(x2_8, y2_8-0.45) + '\n')
        f.write(r'\node[state] at (%.2f, %.2f)'%(x2_8, y2_8-0.9)             + r'(h2_9){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0, beta0+1, alpha1+1, beta1) + '\n')

        f.write(r'\node[state] at (%.2f, %.2f)'%(x2_10, y2_10) + r'(h2_10){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0, beta0, alpha1+2, beta1) + '\n')
        f.write(r'\node[between] at (%.2f, %.2f) (h2_1011){};'%(x2_10, y2_10-0.45) + '\n')
        f.write(r'\node[state] at (%.2f, %.2f)'%(x2_10, y2_10-0.9)             + r'(h2_11){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0, beta0, alpha1+1, beta1+1) + '\n')

        f.write(r'\node[state] at (%.2f, %.2f)'%(x2_12, y2_12) + r'(h2_12){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0+1, beta0, alpha1, beta1+1) + '\n')
        f.write(r'\node[between] at (%.2f, %.2f) (h2_1213){};'%(x2_12, y2_12-0.45) + '\n')
        f.write(r'\node[state] at (%.2f, %.2f)'%(x2_12, y2_12-0.9)             + r'(h2_13){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0, beta0+1, alpha1, beta1+1) + '\n')

        f.write(r'\node[state] at (%.2f, %.2f)'%(x2_14, y2_14) + r'(h2_14){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0, beta0, alpha1+1, beta1+1) + '\n')
        f.write(r'\node[between] at (%.2f, %.2f) (h2_1415){};'%(x2_14, y2_14-0.45) + '\n')
        f.write(r'\node[state] at (%.2f, %.2f)'%(x2_14, y2_14-0.9)             + r'(h2_15){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0, beta0, alpha1, beta1+2) + '\n')

        # horizon 0 action 0
        for idx, rep in enumerate(replays):
            if rep[1] == 0:
                if rep[-1] == (0, 0, 0, 0):
                    colour = 'red'
                    break
                else:
                    colour = 'black'
                    idx    = ''
            else:
                colour = 'black'
                idx    = ''
        f.write(r'\draw[->, thick, %s] ([yshift=5.5cm]h0)   |- (h1_01)  node[above, pos=0.8] {arm $0$} node[below, pos=0.8] {%s};'%(colour, idx) + '\n')
        
        # horizon 0 action 1
        for idx, rep in enumerate(replays):
            if rep[1] == 0:
                if rep[-1] == (0, 0, 0, 1):
                    colour = 'red'
                    break
                else:
                    colour = 'black'
                    idx    = ''
            else:
                colour = 'black'
                idx    = ''
        f.write(r'\draw[->, thick, %s] ([yshift=-5.5cm]h0)  |- (h1_23)  node[above, pos=0.8] {arm $1$} node[below, pos=0.8] {%s};'%(colour, idx) + '\n')
        
        # (horizon 0 action 0 rew 1) -> action 0
        for idx, rep in enumerate(replays):
            if rep[1] == 1:
                if rep[-1] == (0, 0, 0, 0):
                    colour = 'red'
                    break
                else:
                    colour = 'black'
                    idx    = ''
            else:
                colour = 'black'
                idx    = ''
        f.write(r'\draw[->, thick, %s] ([yshift=4.0cm]h1_0) |- (h2_01)  node[above, pos=0.8] {arm $0$} node[below, pos=0.8] {%s};'%(colour, idx) + '\n')
        
        # (horizon 0 action 0 rew 1) -> action 1
        for idx, rep in enumerate(replays):
            if rep[1] == 1:
                if rep[-1] == (0, 0, 0, 1):
                    colour = 'red'
                    break
                else:
                    colour = 'black'
                    idx    = ''
            else:
                colour = 'black'
                idx    = ''
        f.write(r'\draw[->, thick, %s] ([yshift=1.5cm]h1_0) |- (h2_23)  node[above, pos=0.8] {arm $1$} node[below, pos=0.8] {%s};'%(colour, idx) + '\n')
        
        # (horizon 0 action 0 rew 0) -> action 0
        for idx, rep in enumerate(replays):
            if rep[1] == 1:
                if rep[-1] == (0, 0, 1, 0):
                    colour = 'red'
                    break
                else:
                    colour = 'black'
                    idx    = ''
            else:
                colour = 'black'
                idx    = ''
        f.write(r'\draw[->, thick, %s] ([yshift=1.5cm]h1_1) |- (h2_45)  node[above, pos=0.8] {arm $0$} node[below, pos=0.8] {%s};'%(colour, idx) + '\n')
        
        # (horizon 0 action 0 rew 0) -> action 1
        for idx, rep in enumerate(replays):
            if rep[1] == 1:
                if rep[-1] == (0, 0, 1, 1):
                    colour = 'red'
                    break
                else:
                    colour = 'black'
                    idx    = ''
            else:
                colour = 'black'
                idx    = ''
        f.write(r'\draw[->, thick, %s] ([yshift=1.5cm]h1_1) |- (h2_67)  node[above, pos=0.8] {arm $1$} node[below, pos=0.8] {%s};'%(colour, idx) + '\n')
        
        # (horizon 0 action 1 rew 1) -> action 0
        for idx, rep in enumerate(replays):
            if rep[1] == 1:
                if rep[-1] == (1, 0, 2, 0):
                    colour = 'red'
                    break
                else:
                    colour = 'black'
                    idx    = ''
            else:
                colour = 'black'
                idx    = ''
        f.write(r'\draw[->, thick, %s] ([yshift=1.5cm]h1_2) |- (h2_89)  node[above, pos=0.8] {arm $0$} node[below, pos=0.8] {%s};'%(colour, idx) + '\n')
        
        # (horizon 0 action 1 rew 1) -> action 1
        for idx, rep in enumerate(replays):
            if rep[1] == 1:
                if rep[-1] == (1, 0, 2, 1):
                    colour = 'red'
                    break
                else:
                    colour = 'black'
                    idx    = ''
            else:
                colour = 'black'
                idx    = ''
        f.write(r'\draw[->, thick, %s] ([yshift=1.5cm]h1_2) |- (h2_1011) node[above, pos=0.8] {arm $1$} node[below, pos=0.8] {%s};'%(colour, idx) + '\n')
        
        # (horizon 0 action 1 rew 0) -> action 0
        for idx, rep in enumerate(replays):
            if rep[1] == 1:
                if rep[-1] == (1, 0, 3, 0):
                    colour = 'red'
                    break
                else:
                    colour = 'black'
                    idx    = ''
            else:
                colour = 'black'
                idx    = ''
        f.write(r'\draw[->, thick, %s] ([yshift=1.5cm]h1_3) |- (h2_1213) node[above, pos=0.8] {arm $0$} node[below, pos=0.8] {%s};'%(colour, idx) + '\n')
        
        # (horizon 0 action 1 rew 0) -> action 1
        for idx, rep in enumerate(replays):
            if rep[1] == 1:
                if rep[-1] == (1, 0, 3, 1):
                    colour = 'red'
                    break
                else:
                    colour = 'black'
                    idx    = ''
            else:
                colour = 'black'
                idx    = ''
        f.write(r'\draw[->, thick, %s] ([yshift=1.5cm]h1_3) |- (h2_1415) node[above, pos=0.8] {arm $1$} node[below, pos=0.8] {%s};'%(colour, idx) + '\n')

        f.write(r'\end{tikzpicture}' + '\n')
        f.write(r'\end{minipage}' + '\n')

    return None

def generate_big_tex_tree(h, save_path):

    y_max = 10
    x_max = 9

    with open(save_path, 'w') as f:

        f.write(r'\begin{minipage}{\textwidth}' + '\n')
        f.write(r'\begin{tikzpicture}' + '\n') 
        f.write(r'\tikzstyle{between} = [rectangle, draw=none]' + '\n')
        f.write(r'\tikzstyle{state}   = [rectangle, text centered, draw=black, minimum height=1mm, text width=3mm, inner sep=0pt, fill=orange!30]' + '\n')

        between_nodes = {hi:[] for hi in range(h)}

        for hi in range(1, h):

            num_nodes = 4**hi
            x_node    = -x_max + hi*2

            for idx, y_node in enumerate(reversed(np.linspace(-y_max*(hi+1)/h, y_max*(hi+1)/h, num_nodes))):
                node_name = str(hi) + '_b_' + str(idx)
                f.write(r'\node[between] at (%.2f, %.2f) (%s){};'%(x_node, y_node, node_name) + '\n')
                between_nodes[hi].append(node_name)

        state_nodes = {hi:[] for hi in range(h)}

        for hi in range(h):

            num_nodes = 4**hi
            x_node    = -x_max + hi*2

            if hi == 0:
                node_name = str(hi) + '_s_' + str(0)
                y_node    = 0
                f.write(r'\node[state] at (%.2f, %.2f) (%s){};'%(x_node, y_node, node_name) + '\n')
                state_nodes[hi].append(node_name)
            else:
                for idx, y_node in enumerate(reversed(np.linspace(-y_max*(hi+1)/h, y_max*(hi+1)/h, num_nodes))):
                    node_name = str(hi) + '_s_' + str(idx*2)
                    f.write(r'\node[state] at (%.2f, %.2f) (%s){};'%(x_node, y_node+0.08, node_name) + '\n')
                    state_nodes[hi].append(node_name)
                    node_name = str(hi) + '_s_' + str(idx*2+1)
                    f.write(r'\node[state] at (%.2f, %.2f) (%s){};'%(x_node, y_node-0.08, node_name) + '\n')
                    state_nodes[hi].append(node_name)

        for hi in range(h-1):
            for k in state_nodes[hi]:
                idx1 = int(k.split('_')[-1])
                for k1 in between_nodes[hi+1]:
                    idx2 = int(k1.split('_')[-1])
                    if hi != 0:
                        cond = (idx1*2 == idx2) or (idx1*2+1 == idx2)
                    else:
                        cond = (idx1*4 == idx2) or (idx1*4+1 == idx2) or (idx1*4+2 == idx2) or (idx1*4+3 == idx2)
                    if cond:
                        f.write(r'\draw[->, thick] (%s) -- (%s);'%(k, k1) + '\n')

        f.write(r'\end{tikzpicture}' + '\n')
        f.write(r'\end{minipage}' + '\n')

    return None