def generate_tex_tree(M, save_path):

    alpha0 = M[0, 0]
    beta0  = M[0, 1]
    alpha1 = M[1, 0]
    beta1  = M[1, 1]

    with open(save_path, 'w') as f:

        f.write(r'\begin{minipage}{\textwidth}' + '\n')
        f.write(r'\begin{tikzpicture}' + '\n') 
        f.write(r'\tikzstyle{state} = [rectangle, text centered, minimum width=1.2cm, minimum height=1cm, draw=black, fill=orange!30]' + '\n')
        f.write(r'\node[state] at (-8,0)       (h0){\begin{tabular}{c} $\begin{pmatrix} \alpha_0=%u & \beta_0=%u \\ \alpha_1=%u & \beta_1=%u \end{pmatrix}$ \end{tabular}};'%(alpha0, beta0, alpha1, beta1) + '\n')
        f.write(r'\node[state] at (-2, 5.6)  (h1_0){\begin{tabular}{c} $\begin{pmatrix} \alpha_0=%u & \beta_0=%u \\ \alpha_1=%u & \beta_1=%u \end{pmatrix}$ \\ \addlinespace[1ex]$\begin{pmatrix} \alpha_0=%u & \beta_0=%u \\ \alpha_1=%u & \beta_1=%u \end{pmatrix}$ \end{tabular}};'%(alpha0+1, beta0, alpha1, beta1, alpha0, beta0+1, alpha1, beta1) + '\n')
        f.write(r'\node[state] at (-2, -5.6) (h1_2){\begin{tabular}{c} $\begin{pmatrix} \alpha_0=%u & \beta_0=%u \\ \alpha_1=%u & \beta_1=%u \end{pmatrix}$ \\ \addlinespace[1ex]$\begin{pmatrix} \alpha_0=%u & \beta_0=%u \\ \alpha_1=%u & \beta_1=%u \end{pmatrix}$ \end{tabular}};'%(alpha0, beta0, alpha1+1, beta1, alpha0, beta0, alpha1, beta1+1) + '\n')
        f.write(r'\node[state] at (6, 9.2)   (h2_0){\begin{tabular}{c} $\begin{pmatrix} \alpha_0=%u & \beta_0=%u \\ \alpha_1=%u & \beta_1=%u \end{pmatrix}$ \\ \addlinespace[1ex]$\begin{pmatrix} \alpha_0=%u & \beta_0=%u \\ \alpha_1=%u & \beta_1=%u \end{pmatrix}$ \end{tabular}};'%(alpha0+2, beta0, alpha1, beta1, alpha0+1, beta0+1, alpha1, beta1) + '\n')
        f.write(r'\node[state] at (6, 6.8)   (h2_2){\begin{tabular}{c} $\begin{pmatrix} \alpha_0=%u & \beta_0=%u \\ \alpha_1=%u & \beta_1=%u \end{pmatrix}$ \\ \addlinespace[1ex]$\begin{pmatrix} \alpha_0=%u & \beta_0=%u \\ \alpha_1=%u & \beta_1=%u \end{pmatrix}$ \end{tabular}};'%(alpha0+1, beta0, alpha1+1, beta1, alpha0+1, beta0, alpha1, beta1+1) + '\n')
        f.write(r'\node[state] at (6, 4.4)   (h2_4){\begin{tabular}{c} $\begin{pmatrix} \alpha_0=%u & \beta_0=%u \\ \alpha_1=%u & \beta_1=%u \end{pmatrix}$ \\ \addlinespace[1ex]$\begin{pmatrix} \alpha_0=%u & \beta_0=%u \\ \alpha_1=%u & \beta_1=%u \end{pmatrix}$ \end{tabular}};'%(alpha0+1, beta0+1, alpha1, beta1, alpha0, beta0+2, alpha1, beta1) + '\n')
        f.write(r'\node[state] at (6, 2)     (h2_6){\begin{tabular}{c} $\begin{pmatrix} \alpha_0=%u & \beta_0=%u \\ \alpha_1=%u & \beta_1=%u \end{pmatrix}$ \\ \addlinespace[1ex]$\begin{pmatrix} \alpha_0=%u & \beta_0=%u \\ \alpha_1=%u & \beta_1=%u \end{pmatrix}$ \end{tabular}};'%(alpha0, beta0+1, alpha1+1, beta1, alpha0, beta0+1, alpha1, beta1+1) + '\n')
        f.write(r'\node[state] at (6, -2)    (h2_8){\begin{tabular}{c} $\begin{pmatrix} \alpha_0=%u & \beta_0=%u \\ \alpha_1=%u & \beta_1=%u \end{pmatrix}$ \\ \addlinespace[1ex]$\begin{pmatrix} \alpha_0=%u & \beta_0=%u \\ \alpha_1=%u & \beta_1=%u \end{pmatrix}$ \end{tabular}};'%(alpha0+1, beta0, alpha1+1, beta1, alpha0, beta0+1, alpha1+1, beta1) + '\n')
        f.write(r'\node[state] at (6, -4.4) (h2_10){\begin{tabular}{c} $\begin{pmatrix} \alpha_0=%u & \beta_0=%u \\ \alpha_1=%u & \beta_1=%u \end{pmatrix}$ \\ \addlinespace[1ex]$\begin{pmatrix} \alpha_0=%u & \beta_0=%u \\ \alpha_1=%u & \beta_1=%u \end{pmatrix}$ \end{tabular}};'%(alpha0, beta0, alpha1+2, beta1, alpha0, beta0, alpha1+1, beta1+1) + '\n')
        f.write(r'\node[state] at (6, -6.8) (h2_12){\begin{tabular}{c} $\begin{pmatrix} \alpha_0=%u & \beta_0=%u \\ \alpha_1=%u & \beta_1=%u \end{pmatrix}$ \\ \addlinespace[1ex]$\begin{pmatrix} \alpha_0=%u & \beta_0=%u \\ \alpha_1=%u & \beta_1=%u \end{pmatrix}$ \end{tabular}};'%(alpha0+1, beta0, alpha1, beta1+1, alpha0, beta0+1, alpha1, beta1+1) + '\n')
        f.write(r'\node[state] at (6, -9.2) (h2_14){\begin{tabular}{c} $\begin{pmatrix} \alpha_0=%u & \beta_0=%u \\ \alpha_1=%u & \beta_1=%u \end{pmatrix}$ \\ \addlinespace[1ex]$\begin{pmatrix} \alpha_0=%u & \beta_0=%u \\ \alpha_1=%u & \beta_1=%u \end{pmatrix}$ \end{tabular}};'%(alpha0, beta0, alpha1+1, beta1+1, alpha0, beta0, alpha1, beta1+2) + '\n')

        f.write(r'\draw[->, thick] ([yshift=5.5cm]h0)   |- (h1_0)  node[above, pos=0.7] {arm $0$};' + '\n')
        f.write(r'\draw[->, thick] ([yshift=-5.5cm]h0)  |- (h1_2)  node[above, pos=0.7] {arm $1$};' + '\n')
        f.write(r'\draw[->, thick] ([yshift=4.0cm]h1_0) |- (h2_0)  node[above, pos=0.7] {arm $0$};' + '\n')
        f.write(r'\draw[->, thick] ([yshift=1.5cm]h1_0) |- (h2_2)  node[above, pos=0.7] {arm $1$};' + '\n')
        f.write(r'\draw[->, thick] ([yshift=1.5cm]h1_0) |- (h2_4)  node[above, pos=0.7] {arm $0$};' + '\n')
        f.write(r'\draw[->, thick] ([yshift=1.5cm]h1_0) |- (h2_6)  node[above, pos=0.7] {arm $1$};' + '\n')
        f.write(r'\draw[->, thick] ([yshift=1.5cm]h1_2) |- (h2_8)  node[above, pos=0.7] {arm $0$};' + '\n')
        f.write(r'\draw[->, thick] ([yshift=1.5cm]h1_2) |- (h2_10) node[above, pos=0.7] {arm $1$};' + '\n')
        f.write(r'\draw[->, thick] ([yshift=1.5cm]h1_2) |- (h2_12) node[above, pos=0.7] {arm $0$};' + '\n')
        f.write(r'\draw[->, thick] ([yshift=1.5cm]h1_2) |- (h2_14) node[above, pos=0.7] {arm $1$};' + '\n')
        f.write(r'\end{tikzpicture}' + '\n')
        f.write(r'\end{minipage}' + '\n')

    return None