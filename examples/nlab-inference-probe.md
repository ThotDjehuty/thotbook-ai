# nLab Inference Probe — 10 questions to test ThotBook-AI

> **Purpose.** A copy-paste benchmark that probes whether the `thotbook-ai`
> engine (M1 corpus → M2 tokenizer → M3 QLoRA → M4 RAG → M5 serve) can answer
> open-ended research questions about the mathematics and physics catalogued on
> [ncatlab.org](https://ncatlab.org). Each probe is written in the style of the
> reference example **P01 · Functorial spectral geometry**: an open-ended
> question that demands a precise definition, the key theorems, the LaTeX
> equations, an illustration, and a list of the field's founding contributions.
>
> **How to run a probe.** Paste the `PROMPT` block verbatim into
> `thotbook/thotbook-ai`. Grade the raw answer against the golden-answer
> sketch and the evaluation notes. Every answer must ship LaTeX (`$$...$$`) and
> at least one ASCII/TikZ illustration — if it does not, the probe fails.

---

## P01 · Functorial spectral geometry

```
PROMPT
What is "functorial spectral geometry"? Give some basic theorems and a list
of Alain Connes' contributions to this field. Illustrate with equations in
LaTeX and at least one diagram.
```

**What it probes.** Recall of a niche nLab entry; the ability to state the
Connes distance formula and the reconstruction theorem precisely; knowledge of
Connes' bibliography (cyclic cohomology, Dixmier traces, the Standard Model,
Tomita–Takesaki); the ability to draw a functorial diagram.

**Golden-answer sketch.**

Spectral geometry describes a Riemannian manifold $M$ by spectral data — the
eigenvalue spectrum of its Dirac operator. A **spectral triple**
$(A, H, D)$ consists of an involutive algebra $A$ acting faithfully on a
Hilbert space $H$, and a self-adjoint operator $D$ with compact resolvent and
bounded commutators $[D, a]$, $a \in A$. *Functorial* spectral geometry
formulates the assignment

$$
M \;\longmapsto\; \big(C^\infty(M),\; L^2(S_M),\; \not D\big)
$$

as a functor from a category of (spin) manifolds to a category of spectral
triples whose morphisms are the **geometric (Morita) morphisms**, so that all
geometric invariants are natural transformations of this functor.

**Basic theorems.**

1. **Connes' reconstruction theorem.** Every commutative spectral triple
   $(A, H, D)$ with $A \simeq C^\infty(M)$ a smooth function algebra, satisfying
   finite-order smoothness, regularity, and orientability, recovers the smooth
   spin manifold $M$ and its Dirac operator.

2. **Connes distance formula.** For the commutative spectral triple of a
   Riemannian spin manifold,

$$
d_{\mathrm{geo}}(x, y) \;=\; \sup_{f \in C^\infty(M)} \Big\{ |f(x) - f(y)| \;:\; \big\| [\not D, f] \big\| \le 1 \Big\}.
$$

   This recovers exactly the geodesic distance, so the *spectral* data
   reconstruct the *metric* data.

3. **Local index formula (Connes–Moscovici).** The Chern character
   $\mathrm{ch}_\ast(D)$ of a (finitely summable, regular) spectral triple is a
   Hochschild/cyclic cocycle computed by residues of zeta functions of $D$,
   recovering the Atiyah–Singer index:

$$
\operatorname{Index}(\not D) \;=\; \int_M \widehat A(M) \wedge \mathrm{ch}(E).
$$

**Alain Connes' contributions (chronological list).**

| Year | Contribution |
|------|--------------|
| 1973–76 | Classification of type III factors; Tomita–Takesaki modular theory; $S$-matrix for flow of weights |
| 1980–85 | Cyclic cohomology/cyclic homology; Connes–Chern character; pairing with K-theory |
| 1985 | Noncommutative geometry as a field; **noncommutative torus** $A_\theta$ |
| 1986 | Spectral triple / noncommutative distance formula (Connes' distance) |
| 1989 | Noncommutative differential geometry; connections, curvature, Chern forms |
| 1990 | Connes–Lott model: Standard Model from an almost-commutative spectral triple |
| 1995 | Local index formula with Moscovici; Hopf cyclic cohomology |
| 1996 | Reconstruction of manifolds from spectral triples (axioms) |
| 2006+ | Spectral action $\mathrm{Tr}\big(f(D/\Lambda)\big)$; gravity from the Standard Model (Connes–Chamseddine) |
| 2011–18 | Quanta of spacetime; Heisenberg dynamics of the Dirac operator; modular Heisenberg algebras |

**Illustration.**

```
        spin manifold                     spectral triple
   ┌───────────────────┐          ┌──────────────────────────┐
   │  (M, g), spin      │   F      │  A = C∞(M)  acts on H    │
   │  Dirac operator D  │ ──────►  │  D = ⧸D  self-adjoint     │
   └───────────────────┘          │  [D,a] bounded ∀ a∈A      │
            │                     └──────────────────────────┘
            │ geodesic dist                        │
            │ d_geo(x,y)                           │ Connes distance
            ▼                                     ▼
     d_geo(x,y) = sup{|f(x)-f(y)| : ||[D,f]|| ≤ 1}   = d_Connes(x,y)
            │                     │
            └───────  natural in the functor M ↦ (A,H,D) ───────┘
```

**Evaluation notes.** Accept: definition of spectral triple verbatim; the
distance formula with the correct sup and the bound $\|[D,f]\| \le 1$; the
reconstruction theorem as the *commutative* case; at least 6 of the
contributions table entries; a diagram. Reject: claiming the distance formula
holds without the unitary/finite-sum condition; confusing cyclic cohomology
with de Rham cohomology.

---

## P02 · Spectral triples and the reconstruction of manifolds

```
PROMPT
State precisely the axioms of a (finite-summable, regular, orientable,
Poincaré-dual) spectral triple, and sketch how they imply that a commutative
spectral triple is a spin manifold. When does the Connes distance formula
reproduce the geodesic distance?
```

**What it probes.** Precise recall of the reconstruction axioms; the role of
the *commutativity* hypothesis; the degree-n summability and Hochschild cycle;
orientation and Poincaré duality; the smoothness/regularity condition.

**Golden-answer sketch.** A spectral triple $(A,H,D)$ is:

- **Finitely summable** of dimension $n$ if $|D|^{-1}$ is compact of Schatten
  class $\ell^{n+}$;
- **Regular** if $a$ and $[D,a]$ lie in $\bigcap_k \mathrm{dom}(\delta^k)$,
  $\delta(T) = [|D|, T]$;
- **Orientable** if there is a Hochschild $n$-cycle $\mathbf c$ with
  $\pi(\mathbf c) = \gamma$ (the $\mathbb{Z}_2$-grading);
- **Poincaré dual** in (even) K-homology: the Kasparov product with the
  fundamental class gives the Poincaré duality isomorphism.

Connes' theorem then states: a commutative spectral triple
$(C^\infty(M), L^2(S), \not D)$ satisfying these axioms is **exactly** the
data of a smooth spin manifold, and the Connes distance equals the geodesic
distance. The Hochschild character

$$
\mathrm{ch}(D) = \mathrm{Tr}_s\big(\gamma \cdot \mathbf c \cdot \exp(-D^2)\big)
$$

recovers the (twisted) fundamental class, and the spectral action
$\mathrm{Tr}(f(D/\Lambda))$ recovers the Einstein–Hilbert action plus matter.

**Illustration.**

```
  axioms                  objects recovered
  ──────                  ─────────────────
  A commutative   ───►    M smooth manifold, A ≅ C∞(M)
  D order 1       ───►    spin structure / Clifford bundle
  orientability   ───►    fundamental class [M] ∈ H_n
  Poincaré dual   ───►    KK-duality  K_*(A) ≅ K^*(A)
  finite summable ───►    dim M = n, spectral dimension
```

**Evaluation notes.** Reject answers that omit the commutative hypothesis or
claim noncommutative triples reconstruct manifolds. Accept the four axioms if
stated with the correct names and the Hochschild cycle condition.

---

## P03 · Cyclic cohomology and the Connes–Chern character

```
PROMPT
Define cyclic cohomology HC^n(A) of an algebra A, state the Connes exact
sequence linking Hochschild cohomology HH^n(A) to cyclic cohomology, and
explain how the Connes–Chern character ch: K_0(A) → HC^even(A) works. Why is
cyclic cohomology the right cohomological invariant for index theory on
foliations?
```

**What it probes.** Formal definition of cyclic cohomology (cyclic cochains,
the cyclic boundary $b$ and $B$); the Connes exact sequence; the pairing with
K-theory; motivation via the index theorem for foliations (Godbillon–Vey,
transversal index theorem).

**Golden-answer sketch.** A cyclic cochain is an $(n+1)$-linear functional
$\varphi$ on $A$ satisfying $\varphi(a_0 a_1, \dots, a_n) =
(-1)^n \varphi(a_n a_0, a_1, \dots)$; cyclic cohomology is the cohomology of
the $(b,B)$ bicomplex:

$$
HC^n(A) \;=\; H^n\big(\mathrm{Tot}(\mathrm{CC}^{\bullet,\bullet}(A)),\; b + B\big),
\qquad
HH^n(A) \;=\; H^n(A \otimes A^{\circ}, b).
$$

The **Connes exact sequence**

$$
0 \to \mathrm{HC}^n(A) \xrightarrow{S} \mathrm{HC}^{n+2}(A) \xrightarrow{B} \mathrm{HH}^{n+1}(A) \xrightarrow{I} \mathrm{HC}^{n+1}(A) \to 0
$$

relates them. For a Fredholm module / spectral triple, the Connes–Chern
character is the periodic cyclic cocycle

$$
\mathrm{ch}_n(a^0,\dots,a^n) = \mathrm{Tr}_s\big(\gamma\, a^0 [F, a^1]\cdots [F, a^n]\big),
$$

and the pairing $\langle \mathrm{ch}, [e]\rangle = \mathrm{Index}(D_e)$ computes
the index. On the leaf space of a foliation — whose "C*-algebra of holonomy"
$\mathrm{C}^\ast(\mathcal F)$ is highly noncommutative — de Rham cohomology
does not exist; cyclic cohomology does, and it yields the transversal index
theorem.

**Illustration.**

```
 K_0(A) × HC^{even}(A)  ──►  Z
    │ e                 ▲ ch      Index(D_e) = ⟨ch, e⟩
    │                   │
    ▼                   │
 K_*(C*(F))  ──┬──►  HC^*(C*(F))   ← transversal index theorem
    foliation C*-algebra (noncommutative!)
```

**Evaluation notes.** Accept: $(b,B)$ bicomplex and $S,B,I$ exact sequence;
the Fredholm-module character formula with the sign/$\mathrm{Tr}_s$; the
foliation motivation. Reject: stating cyclic = de Rham.

---

## P04 · The noncommutative torus and Morita equivalence

```
PROMPT
Describe the noncommutative torus A_θ, its smooth structure, its K-theory and
cyclic cohomology, and its irrational-rotation algebra universal property.
Explain what it means for A_θ to be Morita equivalent to A_{θ'}, and why this
matters for the Connes–Lott Standard-Model ansatz.
```

**What it probes.** Generators $U, V$ with $UV = e^{2\pi i\theta}VU$; the
smooth algebra $\mathcal{A}_\theta$; the Pimsner–Voiculescu exact sequence
(rank-2 K-groups, tracial state); Morita equivalence and the $SL(2,\mathbb Z)$
action; Rieffel projections; the role in the almost-commutative Standard Model.

**Golden-answer sketch.** For $\theta \in \mathbb R$, $A_\theta$ is generated
by unitaries $U, V$ with

$$
UV = e^{2\pi i\theta} VU,
$$

i.e. $A_\theta = C^\ast(U, V)$ — the universal C*-algebra of the
irrational-rotation action. The smooth subalgebra
$\mathcal A_\theta = \big\{ \sum a_{mn} U^m V^n : (a_{mn}) \text{ rapidly
decreasing}\big\}$ is a "noncommutative torus": its K-theory
$K_0(A_\theta) \cong K_1(A_\theta) \cong \mathbb Z^2$ (Pimsner–Voiculescu), with
the tracial state $\tau(a_{mn}) = a_{00}$ giving
$(\tau(\cdot), \frac{1}{2\pi i}\delta(\cdot)) : K_0 \to \mathbb R^2$.

Two irrational-rotation algebras are **Morita equivalent** iff $\theta' = \frac{a\theta + b}{c\theta + d}$, an $SL(2,\mathbb Z)$ action; Rieffel constructed the imprimitivity bimodules and projections of rank $\tau$.

**Illustration.**

```
   A_θ  ——UV = e^{2πiθ}VU──►  irrational rotation C*-algebra
     │                               │
   smooth subalgebra           Morita (SL(2,Z))  ⟺  θ'=(aθ+b)/(cθ+d)
     │                               │
   K_0 ≅ Z², τ(a)=a₀₀          Rieffel projection p, τ(p)=|p+qθ|
     │                               │
     └──────► almost-commutative triple  C∞(M) ⊗ A_θ  ──► Standard Model
```

**Evaluation notes.** Accept the commutation relation, Pimsner–Voiculescu
$\mathbb Z^2$ K-groups, and Morita criterion. Reject claims that $A_\theta$
has $K_0 \cong \mathbb Z$ (that is the gauge-equivariant or odd case).

---

## P05 · Kasparov KK-theory and the category of spectral triples

```
PROMPT
Define Kasparov's bivariant KK-theory, its product, and its functoriality.
Explain how spectral triples live in KK-theory and what Poincaré duality in KK
means for the Connes reconstruction program. Where does the Kasparov product
appear in physics (D-brane charge, T-duality)?
```

**What it probes.** The bivariant functor $KK(A,B)$; Kasparov cycles $(E, F)$;
the intersection/associative product; Poincaré duality $K_*(A) \cong K^*(A)$;
applications: D-brane charges as $K$-homology classes, T-duality and the
Bougourzi/Connes–Douglas–Schwarz constructions, string theory on the
noncommutative torus.

**Golden-answer sketch.** $KK(A,B)$ consists of (equivalence classes of)
Kasparov $A$–$B$-modules $(E, \phi, F)$: a graded Hilbert $B$-module with an
$A$-action $\phi$ and an odd, $B$-compact operator $F$ with
$a(F - F^\ast), a(F^2 - 1) \in K(E)$. The **Kasparov product**
$\circ : KK(A,B) \times KK(B,C) \to KK(A,C)$ is associative and makes
$KK$ a category. A spectral triple $(A,H,D)$ is a Kasparov cycle
$[A, H, F]$ with $F = D|D|^{-1}$, i.e. an element of $KK(A, \mathbb C)$ (an
extension of a Fredholm module). **Poincaré duality** is the requirement that
there exists a fundamental class $\Delta \in KK(A \otimes A, \mathbb C)$ whose
Kasparov product is the identity — giving the intersection-form
$K_*(A) \times K_*(A) \to \mathbb Z$.

**Illustration.**

```
  KK(A,B) × KK(B,C) ──∘──► KK(A,C)         [D] ∈ KK(A,C)  Fredholm module
       ▲                       │                    │
       │  spectral triple      │ KK with A, B = C   │
  (A,H,D) ↦ (H, F=D|D|⁻¹)      ▼                    ▼
   [A,H,D] ∈ KK(A,C)     Poincaré dual A ∘ A ──► KK(A⊗A, C)
       ▲                                             │
       │ D-brane charge [E] ∈ K^0                      │
   T-duality swaps KK(A,C) and KK(A^∨, C)  ◄──────────┘
```

**Evaluation notes.** Accept the Kasparov module definition (graded module, $F$
odd, $a(F-F^\ast), a(F^2-1) \in K(E)$), associativity of the product, and the
D-brane/T-duality connection. Reject conflating $KK$ with $K_0$ only.

---

## P06 · Differential cohomology, String structures and 2-groups

```
PROMPT
Explain what a String structure is, how it is a higher analog of a spin
structure, the role of the string 2-group and the Chern-Simons 2-gerbe, and
give the obstruction (w₂, p₁/2) picture. Show the relation to differential
cohomology on nLab.
```

**What it probes.** The Whitehead tower of $O(n)$; $B\mathrm{String}(n)$ as
3-connected cover of $B\mathrm{Spin}(n)$; the obstruction $\tfrac{1}{2}p_1$;
the String 2-group $\mathrm{String}(n)$ and its Lie 2-algebra $\mathfrak{string}$
(bundle gerbes, Baez–Crans, Schreiber); Chern–Simons 2-gerbes; differential
cohomology and the differential string structure; application: Green–Schwarz
anomaly cancellation.

**Golden-answer sketch.** The Whitehead tower of the orthogonal group is

$$
O(n) \;\to\; SO(n) \;\to\; \mathrm{Spin}(n) \;\to\; \mathrm{String}(n) \;\to\; \mathrm{Fivebrane}(n) \to \cdots
$$

Each step kills one homotopy group: $\pi_1(O) = \mathbb Z_2$ (orientation),
$\pi_2(SO) = \mathbb Z_2$ (spin), $\pi_3(\mathrm{Spin}) = \mathbb Z$
(string). A **String structure** on a spin manifold is a lift of the spin
frame bundle to a principal $\mathrm{String}(n)$-bundle; its obstruction is
the class

$$
\lambda = \tfrac{1}{2}p_1 \in H^4(B\mathrm{Spin}(n); \mathbb Z),
$$

i.e. $\lambda$ vanishes iff the bundle admits a String structure. In
higher/differential cohomology, the lift is refined to a
**Chern–Simons 2-gerbe** $\mathbf{CS} \in H^4_{\mathrm{diff}}$; the
differential string structure satisfies a 3-form equation
$d\, \mathbf{B} = \langle F_\omega \wedge F_\omega\rangle$, and
Green–Schwarz anomaly cancellation reads $\tfrac{1}{2}p_1 = [G]$ with
$G$ the field strength.

**Illustration.**

```
      Whitehead tower of O(n)
   ┌─────────────────────────────┐
   │  Fivebrane(n)  ── kills π₇   │
   │  String(n)     ── kills π₃   │   obstruction ½p₁ ∈ H⁴(Spin;Z)
   │  Spin(n)       ── kills π₂   │   String ⟺ lift of frame bundle
   │  SO(n)         ── kills π₁   │
   │  O(n)          base         │
   └─────────────────────────────┘
        Chern-Simons 2-gerbe CS ∈ H⁴_diff ,  d B = ⟨F,F⟩
```

**Evaluation notes.** Accept the tower, the $\tfrac12 p_1$ obstruction, the
2-gerbe/2-group vocabulary. Reject claims that String is a group in the naive
sense (it is a 2-group), or $\lambda = p_1$.

---

## P07 · Homotopy type theory and the univalence axiom

```
PROMPT
State the univalence axiom of homotopy type theory, explain why identity types
in a universe are equivalent to equivalence types, and relate univalence to
"structure, not identity". What does nLab list as the model-theoretic
realizations (simplicial sets, cubical sets, ∞-toposes)?
```

**What it probes.** The univalence axiom; identity types; the canonical
map $\mathrm{idtoeq}$; the equivalence
$(A = B) \simeq (A \simeq B)$; the higher-structure slogan; models: simplicial
sets (Voevodsky), cubical sets (CCHM), ∞-toposes (Lurie).

**Golden-answer sketch.** In a type theory with a universe $\mathcal U$,
for types $A, B : \mathcal U$ there is a canonical map

$$
\mathrm{idtoeq} : (A = B) \;\longrightarrow\; (A \simeq B),
$$

sending a path to the transport equivalence. The **univalence axiom**
asserts that this map is itself an equivalence:

$$
\mathrm{Univalence} : \qquad (A = B) \;\simeq\; (A \simeq B).
$$

Univalence says that equivalent types are *identical*: mathematical structure
is preserved under equivalence, i.e. mathematics is invariant under
weak equivalence — the "structure, not identity" principle. The axiom holds in
the simplicial set model (Voevodsky 2009), in cubical models (CCHM, and
computation), and more generally in every **∞-topos** (Lurie's theorem):
univalence is the internal statement that the universe classifier is
∞-categorical.

**Illustration.**

```
        identity type                  equivalence type
   ┌────────────────────┐         ┌────────────────────┐
   │  A = B  (paths)    │         │  A ≃ B (maps both  │
   │  p : A = B         │         │        ways)        │
   └────────────────────┘         └────────────────────┘
             │                          ▲
             └─────  idtoeq : = → ≃  ───┘
                      univalence: this is an equivalence
    slogan:  structure, not identity    ↔   equality = equivalence
```

**Evaluation notes.** Accept the canonical-map formulation and the equivalence
statement. Reject answers that state univalence as "all functions are
equivalences" or confuse it with extensionality of identity.

---

## P08 · Cohesive ∞-toposes and Schreiber's geometry of physics

```
PROMPT
Define a cohesive (∞,1)-topos: the four (co)reflective adjunctions (shape,
flat, sharp) and the cohesive axioms. Explain how Schreiber's "differential
cohomology in a cohesive ∞-topos" recovers de Rham theory, Chern–Simons
theory, and WZW models. Where does the "points-to-pieces" transform appear?
```

**What it probes.** The cohesive adjunction quadruple; the axioms (discrete,
codiscrete, cohesion, null cohesion); the shape modality and flat modality;
the points-to-pieces transform
$ʃX \to \flat X$; applications to differential cohomology
(bundles with connection), Chern–Simons/WZW, and the *gauge theory* slogan
"differential cohomology = higher gauge theory".

**Golden-answer sketch.** A cohesive $\infty$-topos
$\mathbf H$ over an $\infty$-base $\mathbf S$ has a string of adjoint
functor systems

$$
(\mathrm{Disc} \dashv \mathrm{Forget}) ,\quad
(\mathrm{Forget} \dashv \mathrm{Codisc}),\quad
(\Pi \dashv \mathrm{Disc}),\quad
(\mathrm{coDisc} \dashv \Gamma),
$$

assembled into the adjoint quadruple $\Pi \dashv \mathrm{Disc} \dashv
\Gamma \dashv \mathrm{coDisc}$, satisfying the cohesive axioms (e.g.
$\Pi$ preserves finite products, the "points-to-pieces" transform is
$\natural : \Pi X \to \Gamma X$, etc.). Modalities: **shape** $ʃ = \Pi \circ \mathrm{Disc}$, **flat** $\flat = \mathrm{Disc}\circ \Gamma$, **sharp** $\sharp = \mathrm{coDisc}\circ\Gamma$, with $ʃ \dashv \flat \dashv \sharp$.

For a cohesive ∞-topos, differential cohomology is organized by the
"cohesive structure": de Rham coefficients are $\flat_{\mathrm{dR}}$, and a
differential refinement of a characteristic class $c$ is a diagram

$$
\begin{array}{c}
  X \xrightarrow{\nabla} \mathbf{B}G_{\mathrm{conn}} \\
  \begin{array}{c} \flat_{\mathrm{dR}} \\[-2mm] \searrow \end{array}
  \!\! \to \!\! \mathbf{B}\mathbb U(1)_{\mathrm{conn}}
\end{array}
$$

For WZW models the relevant data is the Chern–Simons 2-gerbe
$\mathbf{CS}_G$ and the "higher pre-quantization" of the WZW functional; the
adjunction $\flat \dashv \sharp$ enforces the Bianchi identity
$d F = 0$ and gauge invariance.

**Illustration.**

```
          cohesive ∞-topos H  (smooth ∞-groupoids)
   ┌────────────────────────────────────────────┐
   │  ʃ ──shape──▶     points:  Γ ──flat──▶ ʃ    │
   │  ♭ ──flat────▶     discrete: Disc           │
   │  ♯ ──sharp───▶     codiscrete: coDisc       │
   └────────────────────────────────────────────┘
      "points-to-pieces":  natural ʃX → ♭X
      de Rham:  H^n_DR(X) = H^n(ʃX, ℝ)
      conn structure:  X → B G_conn  (higher bundles w/ connection)
```

**Evaluation notes.** Accept the adjoint quadruple, the modalities, the
points-to-pieces transform. Reject answers missing the difference between
shape and flat, or that confuse $\infty$-topos with 1-topos.

---

## P09 · The Atiyah–Singer index theorem and spectral flow

```
PROMPT
State the Atiyah–Singer index theorem, then explain the analytic index as the
index of a Fredholm operator D = d + d* on the spin Dirac operator, and the
local index formula. What is spectral flow, and how is it connected to
anomalies and to the Connes–Moscovici local index theorem?
```

**What it probes.** The AS theorem $\mathrm{Index}(D) = \int \mathrm{ch}(E) \wedge \mathrm{td}(M)$; the analytic/geometric (topological) index; heat-kernel proof; the local index formula (Connes–Moscovici); spectral flow as the winding of eigenvalues across a path of self-adjoint operators; the relation to anomalies (parity anomaly, gauge anomalies) and to the APS index / eta-invariant.

**Golden-answer sketch.** For an elliptic (e.g. Dirac) operator $D$ on a
closed manifold, the **Atiyah–Singer index theorem** is

$$
\operatorname{Index}(D) \;=\; \int_M \mathrm{ch}\big(\sigma(D)\big) \wedge \mathrm{td}(TM),
$$

where $\mathrm{ch}$ is the Chern character of the symbol and $\mathrm{td}$ the
Todd class. The **analytic index** is $\mathrm{Index}(D) = \dim\ker D - \dim \ker D^\ast$;
the theorem equates it with the topological index, computable from the symbol
class. For the spin Dirac operator this is
$\operatorname{Index}(\not D) = \int_M \widehat A(M)$. The
**local index formula** expresses this as a residue/spectral invariant:

$$
\operatorname{Index}(\not D) = \mathrm{Res}_{s=0}\, \mathrm{Tr}_s(\not D |\not D|^{-2s})
\qquad(\text{Wodzicki residue})
$$

**Spectral flow** of a 1-parameter path of self-adjoint Fredholm operators
$D_t$, $t\in[0,1]$, counts the net number of eigenvalues crossing zero:
$\mathrm{SF}(D_t) = \mathrm{Index}(D_1|_{\ker^\perp ...})$; it equals the
difference of the (APS) eta invariants, $\eta(D_1) - \eta(D_0)$, and is the 
analytic avatar of anomaly generation: the parity/gauge anomaly is
$\mathrm{SF}(\not D_t)$ over the interpolation, and the Connes–Moscovici local
index theorem computes such indices from the spectral triple data.

**Illustration.**

```
        Index(D) = dim ker D - dim ker D*          spectral flow
              │                                         ▲
   ┌──────────▼──────────┐                     eigenvalue t ────►
   │  Topological index  │                     ┌───►      ◄───┐
   │  = ∫ ch(E) ∧ td(M)  │                     │    0   crosses
   └──────────┬──────────┘                     │    0   zero 2×
              │                                └───►      ◄───┘
   heat kernel / zeta       SF(D_t) = η(D_1)-η(D_0)
   Residue_0 Tr_s(D|D|^-2s)   anomaly = SF over t∈[0,1]
```

**Evaluation notes.** Accept the equation with $\mathrm{ch}\wedge \mathrm{td}$, the
heat-kernel/local formula, and spectral-flow↔anomaly. Reject answers that give
the index as merely $\dim\ker D$ (missing the kernel of the adjoint), or
confuse the eta invariant with the spectral action.

---

## P10 · K-theory, D-brane charges and the Baum–Connes conjecture

```
PROMPT
Explain why D-brane charges are classified by K-theory rather than cohomology.
State the Baum–Connes conjecture and its role in the noncommutative geometry
of the space of Penrose tilings / irrational rotations. What is the role of
the associated "groupoid" C*-algebra?
```

**What it probes.** The Minasian–Moore / Witten K-theory classification of
RR-charges; K-theory vs cohomology (why the latter misses torsion); the
Baum–Connes isomorphism $K_*(G) \to K_*^{\mathrm{top}}(C^\ast_r(G))$; the
Connes–Haagerup picture of the noncommutative torus / irrational-rotation as
the C*-algebra of a groupoid; applications to the IQHE and to D-branes on
orbifolds.

**Golden-answer sketch.** RR-charge of a D-brane is the pushforward of a K-theory
class: $[E] \in K^0(M)$, and $\mathrm{Index}(\not D_E) = \int_M \mathrm{ch}([E])\wedge \widehat A$ shows that K-theory computes the same numbers cohomology does on torsion-free spaces — but K-theory sees **torsion** classes that cohomology misses, so D-brane charge is classified by $K^0(M)$, not $H^{\mathrm{even}}(M; \mathbb Z)$.

The **Baum–Connes conjecture** asserts that for a group $G$, the assembly map
$\mu : K^\mathrm{top}_\ast(G) \to K_\ast(C^\ast_r(G))$ is an isomorphism,
connecting the K-theory of the classifying space (topological) with the
K-theory of the reduced group C*-algebra. For the irrational-rotation
algebra $A_\theta$, realized as the groupoid C*-algebra of the
$\mathbb Z$-action $\mathbb T \curvearrowright \theta$, BC gives
$K_0(A_\theta) \cong \mathbb Z^2$, the Pimsner–Voiculescu exact sequence being
its concrete avatar. The same groupoid picture describes Penrose tilings
($C^\ast$ of the tiling groupoid) and the integer quantum Hall effect: the
Hall conductance is a $K_0$ element whose quantization is the integrality of
$K_0$.

**Illustration.**

```
  D-brane charge ∈ K^0(M)  (not H^even)  :  torsion visible only in K
        │
        ▼
  Index(⧸D_E) = ∫_M ch([E]) ∧ Â        RR-charge
        │
        ▼
  Baum-Connes:  K^top_*(G) ──μ──► K_*(C*_r(G))   (assembly map)
        │                          ▲
        │                    groupoid C*-algebra
        ▼                     C*(Z ⇉ T) ≅ A_θ
  Penrose tilings / IQHE  :  Hall σ ∈ K_0, quantization from integrality
```

**Evaluation notes.** Accept the K-theory-not-cohomology statement with the
torsion argument, the BC assembly-map formulation, and the groupoid
realization of $A_\theta$. Reject answers that say BC is proved (it is open in
general) or that drop the reduced/assembly qualification.

---

## Scoring rubric (per probe)

| Criterion | Weight | Pass |
|-----------|--------|------|
| Correct definition (nLab-level precision) | 30% | Key object + hypotheses verbatim |
| Theorems stated with hypotheses | 25% | At least one theorem + conditions |
| LaTeX equations correct | 20% | Every formula, correct signs/indices |
| Illustration present and informative | 10% | ≥ 1 ASCII/TikZ diagram |
| Contributions/context list | 15% | Named authors, chronological or structural |

**Baselines (M4 grading).** Compare model answers against: (1) the golden
sketch here; (2) the exact nLab page text (from the M1 mirror); (3) a
textbook treatment (e.g. Connes' *Noncommutative Geometry*, AC 1994); and
report ROC/AUC over ≥ 4 baselines with block-bootstrap CIs (ℓ≈21, B≥2000).

---

*Generated by the `thotbook-ai` pipeline (M1 nLab mirror → lakehouse →
tokenizer → model → serve). Companion dossier: `docs/source/showcase.rst`.*
