## Decomposition

Change in the value function due to learning about a particular $b'$:

$$
\begin{align*}
v(b')-v(b)=&\sum_a \pi(a\mid b')q(b', a)-\sum_{a'} \pi(a\mid b)q(b,a)\\
=&\sum_a\big[ ( \pi(a\mid b')-\pi(a\mid b) )q(b', a) \\
+& \pi(a\mid b)(q(b', a)-q(b, a))\big]
\end{align*}
$$

$$
\begin{align*}
q(b', a)-q(b, a)=&\sum_{b''}p(b''\mid b', a)\big[r(b',a) + \gamma v(b'')\big]\\
-&\sum_{b'}p(b'\mid b, a)\big[r(b,a) + \gamma v(b')\big]\\
=& r(b',a) + \gamma \sum_{b''}p(b''\mid b', a)v(b'') \\
-& r(b,a) + \gamma \sum_{b'}p(b'\mid b, a)v(b') \\
=& \underbrace{\vphantom{ \left(\frac{a^{\frac{0.3}{`}}}{b}\right)} r(b',a) - r(b,a)}_{\substack{\text{Difference in the expected} \\ \text{immediate return}}} + \underbrace{\gamma \big[ \sum_{b''}p(b''\mid b', a)v(b'') - \sum_{b'}p(b'\mid b, a)v(b') \big]}_{\substack{\text{Difference in the expected} \\ \text{future return}}}\\
\end{align*}
$$

Note that we can write

$$
\begin{align*}
cv(b'')-dv(b')=&cv(b'') - cv(b') + cv(b') - dv(b')\\
=&c(v(b'')-v(b')) + v(b')(c-d)
\end{align*}
$$

Therefore

$$
\begin{align*}
q(b', a) - q(b, a) =& r(b',a) - r(b,a)\\ 
+& \gamma \big[ \sum_{b''}p(b''\mid b', a)v(b'') - \sum_{b'}p(b'\mid b, a)v(b') \big]\\
=& r(b', a) - r(b, a)\\ 
+&\gamma \sum_{b''}p(b''\mid b', a)\big[ v(b'')-v(b')\big]\\ 
+& \gamma v(b')\big[ \sum_{b''}p(b''\mid b', a) - \sum_{b'}p(b'\mid b, a) \big]
\end{align*}
$$

Note that the last term goes to zero. Therefore, substituting in:

$$
\begin{align*}
v(b')-v(b)=&\sum_a \pi(a\mid b')q(b', a)-\sum_{a'} \pi(a\mid b)q(b,a)\\
=&\sum_a\big[ ( \pi(a\mid b')-\pi(a\mid b) )q(b', a) \\
+& \pi(a\mid b)(r(b', a) - r(b, a)\\ 
+&\gamma \sum_{b''}p(b''\mid b', a)\big[ v(b'')-v(b')\big]\\ 
=& \sum_a\big[ \pi(a\mid b')-\pi(a\mid b) \big]q(b', a) \\
+& \sum_{a}\pi(a\mid b)\big[ r(b', a)-r(b, a) \big]\\
+& \gamma\sum_{a}\pi(a\mid b)\big[ \sum_{b''}p(b''\mid b', a) (v(b'')-v(b')) \big]\\
\end{align*}
$$

Unrolling:

$$
\begin{align*}
v(b')-v(b)=&\sum_{i=0}^{\infty}\sum_{b'\in \mathcal{B}}\gamma^{i}P(b\rightarrow b', i, \pi(b)) \times\\
&\sum_a \Big(\big[ \pi(a\mid b')-\pi(a\mid b) \big]q(b', a) \\
&+\pi(a\mid b)\big[r(b', a)-r(b,a)\big]\Big)
\end{align*}
$$

