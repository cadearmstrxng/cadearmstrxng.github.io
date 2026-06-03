---
layout: default
title: "Inferring Discount Factors"
---

<div class="project-title">
  Project: Inferring Foresightedness in Dynamic, Noncooperative Games
</div>

<div class="proj-top">
  <div class="pub-note">
    <strong>Note:</strong> This work has been accepted for publication in 
    <em>IEEE Robotics and Automation Letters (RA-L), 2025</em>. View the publication <a href="https://ieeexplore.ieee.org/document/11217215" target="_blank" rel="noopener">here.
    </a>
  </div>

  <div class="proj-cta">
    <a class="proj-btn" href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11217215" target="_blank" rel="noopener">
      📄 Paper
    </a>
    <a class="proj-btn" href="https://github.com/cadearmstrxng/InverseGameDiscountFactor.jl" target="_blank" rel="noopener">
      💻 Code
    </a>
  </div>
</div>

<details class="section-drop">
<summary>Primary Motivation</summary>
<div class="section-drop__body" markdown="1">

Dynamic game theory is a increasingly popular tool for modeling multi-agent interactions. Game-theoretic models presume that each agent wishes to minimize their own private objective function that depends on other agents' actions. These games typically evolve over a fixed time horizon in which agents care equally about each moment in time regardless of how far into the future each agent plans. However, is this realistic? 

Take an example of drivers navigating a busy intersection. What if one (farsighted) driver slows down further in advance of the intersection than another (shortsighted) driver? A self-driving car that incorrectly predicts the behavior of both drivers may take prematurely evasive actions, resulting in uncessarily risky behavior.

Then, to interact with humans safely and efficiently, robots need to be able to account for decision-makers that vary in *foresightedness*, or how much they care about their current cost relative to their future costs. We can model this type of behavior being embedding a discount factor in agent objective functions. Then, via a gradient-based algorithm, we can invert the corresponding game and identify each agent's degree of foresightedness. Specifically, we contribute:
1. an explicitly foresighted game formulation
2. an efficient method to infer each agent's degree of foresightedness from data collected online

We demonstrate the benefits of modeling foresight through a series of simulated and real-world experiments. Before we get to that, let's discuss some of the background to this project and then get into the specifics of how it all works.

</div>
</details>

<details class="section-drop">
<summary>Background</summary>
<div class="section-drop__body" markdown="1">

<div class="content-block" markdown = "1">

**Notation Alert!**  
For notational convenience, I will use the following conventions:
- Superscript: player index
- Subscript: time index
- Lowercase bold: aggregations over time
- Omission of agent index: aggregation of all players

</div>

### Dynamic Games: An Introduction
A dynamic game is characterized by $N$ agents, with the $i^{th}$ agent's control input denoted $u^i_t \in \mathbb{R}^{m^i}$ for all discrete times $t \in [T] := (1, 2, ..., T)$ and a joint state variable $x_t \in \mathbb{R}^n$ following given dynamics $x_{t+1} = f_t(x_t, u^1_t, ..., u^n_t)$. Each agent has a cost function

$$
\begin{align}
J^i := \sum_{t=1}^T \Gamma^i(t;\gamma^i) C^i(x_t, u^i_t, u^{\neg i}_t;\theta^i)
\end{align}
$$

that depends on all player states and actions, as well as hidden parameters $\theta^i \in \mathbb{R}^k$ and *discount factor* $\gamma^i$. At each time $t$, agent $i$'s cost is comprised of parametrized function $C^i(\cdot)$ and scaled by parametrized *discounting function* $\Gamma^i(t;\gamma^i)$ which quantifies the importance of agent $i$'s cost at time $t$.

It should be noted that the above game formulation is general enough to handle arbitrary discounting functions, so long as they are twice differentiable in $\gamma^i$. Let's assume that the discounting function is an exponential, i.e., $\Gamma^i(t;\gamma^i) = (\gamma^i)^t$ where $\gamma^i \in (0, \infty)$.

<div class="quiz" data-answer="b">
  <div class="quiz__q"><strong>Quick check:</strong> What behavior is encoded when $\gamma^i > 1$ in our setting?</div>

  <div class="quiz__choices" role="group" aria-label="Quiz choices">
    <button class="quiz__choice" data-choice="a" type="button">A) Short-sighted behavior</button>
    <button class="quiz__choice" data-choice="b" type="button">B) Far-sighted (future-weighting) behavior</button>
  </div>

  <div class="quiz__feedback" aria-live="polite"></div>

  <div class="quiz__explanation">
    <strong>Explanation:</strong> In the above formulation, $\gamma^i$ scales future stage costs. Values of $\gamma^i > 1$ increase the relative importance
    of later costs, which corresponds to far-sighted behavior. If you are familiar with reinforcement-learning, this might appear odd as it doesn't follow the typical convention of $\gamma^i \in (0,1)$. Remember, we don't need to force a contraction in these settings like we do in RL.
  </div>
</div>

<div class="content-block" markdown = "1">

**Interesting Note**  

When $\gamma^i < 1$, observe that the discounting function $\Gamma^i(t;\gamma^i)$ rapidly approaches 0 as $t \to T$. Thus, the $i^{th}$ agent's actions do not depend upon costs incurred past some effective time horizon $\tilde{T}^i \in [T]$.

</div>

We refer to the sequence of game states as $\textbf{x} = (x^\top_1, x^\top_2, ..., x^\top_T)^\top$, agent $i$'s control sequence as $\textbf{u}^i = (u^{i, \top}_1, u^{i, \top}_2,...,u^{i, \top}_T)^\top$, the sequence of all agents' actions as $\textbf{u}^i = (\textbf{u}^{i, \top}, \textbf{u}^{i, \top},...,\textbf{u}^{i, \top})^\top$, and all agents' hidden parameters as $\theta = (\theta^{1,\top}, \theta^{2,\top}, ..., \theta^{N,\top})^\top$. Then, leveraging this notation, we can write each stage cost function $C^i(x_t, u^i_t, u^{\neg i}_t;\theta^i)$ as $C^i(x_t, u_t;\theta^i)$ and each total cost function as $J(\textbf{x},\textbf{u}; \gamma^i, \theta^i)$.

In general, we also can assign each agent with a set of inequality constraints $I^i(\textbf{x},\textbf{u}; \theta^i) \geq 0$ and a set of equality constraints $E^i(\textbf{x},\textbf{u}; \theta^i) = 0$. We can generally enforce the system dynamics $f_t(\cdot)$ via a group of equality constraints. Finally, we can define a dynamic game as a tuple:

$$
\begin{align}
\mathcal{G}(\theta, \gamma) = (\{J(\cdot; \gamma^i, \theta^i)\}_{i \in [N]}, \{I^i(\cdot; \theta^i)\}_{i \in [N]}, \{E^i(\cdot; \theta^i)\}_{i \in [N]}, x_1, T, N)
\end{align}
$$

### How to Solve Dynamic Games: Generalized Open-Loop Nash Equilibria (GOLNE)

For a given set of parameters $\theta$ and discount factors $\gamma$, a GOLNE of the game $\mathcal{G}(\theta, \gamma)$ is given by a point $(\textbf{x}^\star,\textbf{u}^\star)$ which jointly solves the following coupled optimization problems:

$$
\begin{aligned}
\forall i \in [N]
\begin{cases}
\underset{\textbf{x},\textbf{u}^i}{\min} &J^i(\textbf{x},\textbf{u}; \gamma^i, \theta^i) \\
\text{s.t.} ~~&E^i(\textbf{x},\textbf{u}; \theta^i) = 0 \\
&I^i(\textbf{x},\textbf{u}; \theta^i) \geq 0.
\end{cases}
\end{aligned}
$$

<div class="quiz" data-answer="a">
  <div class="quiz__q"><strong>Quick check:</strong> Assume all objective problems are globally minimized at the point $(\textbf{x}^\star,\textbf{u}^\star)$. At this point, can any agent unilaterally change their strategy (control sequence) to to further minimize their cost? In other words, does there exist a $\textbf{u}^i$ such that
  
  $$
    J^i(\textbf{x},\textbf{u}^{i},\textbf{u}^{\neg i\star}; \gamma^i, \theta^i) < J^i(\textbf{x},\textbf{u}^{i\star},\textbf{u}^{\neg i\star}; \gamma^i, \theta^i)
  $$

  </div>

  <div class="quiz__choices" role="group" aria-label="Quiz choices">
    <button class="quiz__choice" data-choice="a" type="button">A) No, the player strategies $\textbf{u}^\star$ optimize each agent's objective simultaneously, so no improvements can be made by an agent unilaterally changing their objective.</button>
    <button class="quiz__choice" data-choice="b" type="button">B) Yes, so long as every other player also gets to unilaterally change their strategy.</button>
    <button class="quiz__choice" data-choice="c" type="button">c) Yes, because a strategy that optimizes a single player's objective doesn't necessarily optimize other player's objectives.</button>
  </div>

  <div class="quiz__feedback" aria-live="polite"></div>

  <div class="quiz__explanation">
    <strong>Explanation:</strong> At a GOLNE, the strategies $\textbf{u}^\star$ have the property that
    
     $$
     \begin{align}
     J^i(\textbf{x},\textbf{u}^{i},\textbf{u}^{\neg i\star}; \gamma^i, \theta^i) \geq J^i(\textbf{x},\textbf{u}^{i\star},\textbf{u}^{\neg i\star}; \gamma^i, \theta^i)
     \end{align}
     $$

     In other words, no player can benefit by unilaterally deviating from their strategy. You can learn more about Nash equilibria <a href="https://en.wikipedia.org/wiki/Nash_equilibrium" target="_blank" rel="noopener noreferrer">here</a>!
  </div>
</div>

Like many other optimization problems, finding a global GOLNE is computationally intractable. Therefore, it is common to relax the Nash Equilibrium assumption to hold only in an open neighborhood of the point $(\textbf{x}^\star,\textbf{u}^\star)$. Such solutions are called local Nash equilibria. Practically, these points can be found by solving agents' first-order necessary conditions. 

Solving first order necessary conditions for the constrained, coupled optimization problems constitute a Mixed Complementarity Problem (MiCP), defined by decision variables $r \in \mathbb{\eta_r}$, $z \in \mathbb{\eta_z}$, as well as functions $c(r,z)$ and $h(r,z)$ such that:

$$
\begin{align}
c(r,z) &= 0\\
0 \leq z \perp h(&r,z) \geq 0
\end{align}
$$

In our implementation, we solved MiCPs efficiently via off-the-shelf solvers. Specifically, we used <a href="https://www.gams.com/latest/docs/S_PATH.html" target="_blank" rel="noopener noreferrer">PATH Solver</a>.

</div>
</details>

<details class="section-drop">
<summary>Problem Statement</summary>
<div class="section-drop__body" markdown="1">

Now that we understand what a game is, what a solution looks like, and how solutions can be found, we can introduce *inverse dynamic games*, which will enable us to use online trajectory data to solve for the discount factors and hidden parameters for each agent.

Let's presume that some observer has obtained noisy measurements of the game state over time, and denote these observations $\textbf{y} = [y_1(x_1),y_2(x_2),...,y_T(x_T)]$. For simplification purposes, assume observations are drawn independently from Gaussian models $y_t(x_t) \sim \mathcal{N}(h_t(x_t), \Sigma_t)$ with known covariance $\Sigma_t$ and where $h_t(x_t)$ describes the expected output of the sensor at time $t$. 

Given these observations, we want to find the $\theta, and \gamma$ that would most likely result in our observations if we played the game with them. 

<div class="content-block" markdown = "1">

**Example:**  

Say I observe an agent starting at an initial state $x_o$. Over $T$ time steps, I see the agent take actions that move it closer and closer to the origin, $(0,0)$. If I were to assume that the agent only cared about getting to some goal position $x^\star$, I would, based on my observations of the agent, guess that the goal position is $x^\star = (0,0)$. In our formulation, the goal position is a great example of what parameters we would want to store in $\theta$.

</div>

Equivalently, one can minimize covariance-weighted deviations from expected measurements. This results in the **Inverse Game Problem:**

**Problem 1.** Given a sequence of observations $\textbf{y}$, find parameters $\hat{\theta}$ which solve

$$
\begin{align}
\underset{\textbf{x},\textbf{u}, \gamma, \theta}{\min} ~\sum_{t=1}^T ~(h_t(x(t)) - y_t)^\top \Sigma_t^{-1} (h_t(x(t)) - y_t) \\
\text{s.t.} ~~(\textbf{x},\textbf{u}) \text{ is a GOLNE of }\mathcal{G}(\theta, \gamma)
\end{align}
$$

Problem 1 can be separated into an outer *inverse* problem and an inner *forward* problem. The inner GOLNE problem is parametrized by the game parameters $\theta$ and discount factors $\gamma$, which are the decision variables of the inverse problem. However, solving the outer inverse problem entails a multitude of computational challenges. Particularly, the first-order necessary conditions of the inner GOLNE problem are certainly nonlinear in $\gamma$, making the overall problem non-convex. In addition, since the inner problem may contain inequality constraints, its first-order necessary conditions involve complementarity conditions, making the problem non-smooth.

</div>
</details>

<details class="section-drop">
<summary>Solution Approach</summary>
<div class="section-drop__body" markdown="1">

Let's pause and recap. We want to solve an optimization problem that finds game parameters and discount factors that maximize the likelihood of producing some observations we have of the game. However, due to the game's structure, this optimization problem is both non-convex *and* non-smooth, making this a difficult problem to solve.

Our approach: a constrained gradient decent algorithm that requires us to take derivatives of game *solutions* $(\textbf{x},\textbf{u})$ with respect to parameters $(\gamma, \theta)$. First, we have to represent $\mathcal{G}(\theta, \gamma)$ as a workable form: let's convert the game's first-order necessary conditions into an MiCP!

### GOLNE as an MiCP

As one does when beginning a constrained optimization problem, we need begin by writing the agents' first order necessary conditions. First, we introduce Lagrange multipliers $\lambda^i$ and $\mu^i$ to represent agent $i$'s inequality and equality constraints, respectively. We can then write its Lagrangian as:

$$
\mathcal{L}^i (\textbf{x},\textbf{u},\lambda^i$, $\mu^i; \gamma, \theta) = C^i (\textbf{x},\textbf{u}; \gamma, \theta) - \lambda^{i,\top} I^i(\textbf{x},\textbf{u};\theta^i) -\mu^{i,\top} E^i(\textbf{x},\textbf{u};\theta^i).
$$

When the gradients of the constraints are linearly independent at a candidate solution point, we satisfy the linear independence constraint qualification. As such, the following Karush-Kuhn-Tucker (KKT) conditions must hold for each agent $i$:

$$
\begin{align}
\nabla_\textbf{x} \mathcal{L}^i = 0, \nabla_{\textbf{u}^i} \mathcal{L}^i = 0, E^i = 0 \\
0 \leq \lambda^i \perp I^i(\textbf{x},\textbf{u};\theta^i) \geq 0.
\end{align}
$$

These equations have the exact same form as the MiCP formulation we previously discussed! Let's make it even more explicit. 

The MiCP decision variable $r$ is now the aggregated states, actions, and equality Lagrangian multipliers for all agents, and the MiCP decision variable $z$ is now all agents' inequality Lagrange multipliers, i.e.,

$$
\begin{align}
r &= (\textbf{x}^\top, \textbf{u}^\top, \mu^{1,\top}, \mu^{2,\top},...,\mu^{N,\top})^\top\\
z &= (\lambda^{1,\top}, \lambda^{2,\top},...,\lambda^{N,\top})^\top.
\end{align}
$$

For brevity, let's define $v = [r^\top, z^\top]^\top$. Then, the parametrized MiCP for each agent can be written as:

$$
  \begin{align}
  c(v; \theta, \lambda) &= [(\nabla_\textbf{x}\mathcal{L}^i)^\top_{i\in[N]}, (\nabla_{\textbf{u}^i}\mathcal{L}^i)^\top_{i\in[N]}, (E^i)^\top_{i\in[N]}]^\top\\
  h(v; \theta, \lambda) &= [(I^i)^\top_{i\in[N]}]
  \end{align}
$$

For brevity, let's define a function $F$ such that $F = [c(\cdot)^\top,h(\cdot)^\top]^\top$.

### Optimizing Game Parameters with Gradient Descent

Now that we have constructed an MiCP that encodes our game's first order necessary conditions, we can do some interesting things with it. For example, applying the chain rule will allow us to achieve our goal of deriving game solutions with respect to the game's parameters.

Let's denote the objective of Problem 1 as $\mathcal{P}(\cdot)$ such that $\mathcal{P}(\textbf{x}(\theta,\gamma)) = \sum_{t=1}^T ~(h_t(x(t)) - y_t)^\top \Sigma_t^{-1} (h_t(x(t)) - y_t)$. We can then leverage the chain rule to compute its total derivative with respect to $(\theta,\gamma)$:

$$
\nabla_{(\theta, \lambda)} \mathcal{P}(\textbf{x}(\theta,\gamma)) = (\nabla_{(\theta, \lambda)} v)^\top (\nabla_v \textbf{x})^\top (\nabla_\textbf{x} \mathcal{P}),
$$

with which we can update $(\theta,\gamma)$ accordingly. The only term that is difficult to compute is $\nabla_{(\theta, \lambda)} v$. However, we can leverage <a href="https://en.wikipedia.org/wiki/Implicit_function_theorem" target="_blank" rel="noopener noreferrer">Implicit Function Theorem (IFT)</a> to derive exact values for $\nabla_{(\theta, \lambda)} v$ at a solution to the MiCP $v^\star$, where strict complementarity holds.

<details class="proof">
  <summary><strong>Derivation of $\nabla_{(\theta, \lambda)} v$</strong></summary>

  <div class="proof__body" markdown="1">
  As a reminder, we define an MiCP as

  $$
  \begin{align}
  c(r,z) &= 0\\
  0 \leq z \perp h(&r,z) \geq 0.
  \end{align}
  $$

  In the context of our game, the MiCP can store its first-order necessary conditions as:

  $$
  \begin{align}
  c(v; \theta, \lambda) &= [(\nabla_\textbf{x}\mathcal{L}^i)^\top_{i\in[N]}, (\nabla_{\textbf{u}^i}\mathcal{L}^i)^\top_{i\in[N]}, (E^i)^\top_{i\in[N]}]^\top\\
  h(v; \theta, \lambda) &= [(I^i)^\top_{i\in[N]}]
  \end{align}
  $$

  where $v = [r^\top, z^\top]^\top$ stores the game states, controls, and lagrange multipliers for all players $i \in [N]$ for all times $t \in [T]$. We define the function $F(v; \theta, \lambda) = [c(v; \theta, \lambda)^\top,h(v; \theta, \lambda)^\top]^\top$. Let's analyze this function at solution points $v^\star$.

  First, let's consider the complementarity constraints on $z$ and $h(\cdot)$ given by the second equation in the MiCP formulation. We can construct two index sets $\mathcal{I}$, which records all inactive inequality constraint dimensions of $h(\cdot)$, and $\mathcal{S}$, which records all other indices of $F$. Indexing $F$ at the elements of these sets results in vectors $F_\mathcal{I}$ and $F_\mathcal{S}$. 
  
  We know that the values of $F_\mathcal{I}$ are strictly positive, resulting in the Lagrange multipliers associated with these constraints being exactly $0$. As such, small changes in $(\theta, \lambda)$ preserve the positivity of the values of $F_\mathcal{I}$, and thus the corresponding Lagrange multipliers remain $0$. Thuse we find that $\nabla_{(\theta, \lambda)} v_\mathcal{I} = 0$, where $v_\mathcal{I}$ denotes the elements of $v$ that store the Lagrange multipliers corresponding to $F_\mathcal{I}$.

  Now, turning our attention to the remaining elements of $F$ stored in $F_\mathcal{S}$. At a solution to $F$, $F_\mathcal{S}$ must equal $0$. Therefore, if we perturb $(\theta, \lambda)$ on the solution manifold, then $F_\mathcal{S}$ still must equal $0$. We can then leverage <a href="https://en.wikipedia.org/wiki/Implicit_function_theorem" target="_blank" rel="noopener noreferrer">Implicit Function Theorem</a> and the stationarity of $F$ with respect to $v$ to write:

  $$
  \begin{align}
  0 &= D_{(\theta, \lambda)} F_\mathcal{S} = \nabla_{(\theta, \lambda)} F_\mathcal{S} + (\nabla_{v} F_\mathcal{S})(\nabla_{(\theta, \lambda)} v)\\
  &\implies (\nabla_{(\theta, \lambda)} v) = (\nabla_{v} F_\mathcal{S})^{-1} (\nabla_{(\theta, \lambda)} F_\mathcal{S})
  \end{align}
  $$
  
  where $D_{(\theta, \lambda)}$ denotes the total derivative with respect to $(\theta, \lambda)$. Then, when $\nabla_{v} F_\mathcal{S}$ is invertible, we can find exact values of $\nabla_{(\theta, \lambda)} v$!

  </div>
</details>

With these gradients taken, we can iteratively update $(\theta, \gamma)$ as needed to optimize $\mathcal{P}$ as shown in the algorithm below.

<div class="algo">
  <div class="algo__header">
    <span class="algo__name">Algorithm 1</span>
    <span class="algo__title">Gradient-Based Discount Factor Inference</span>
  </div>

  <div class="algo__meta">
    <div><span class="algo__key">Hyperparameters:</span> Learning rate $\alpha$</div>
    <div><span class="algo__key">Input:</span> Initial $\theta, \gamma$, observations $\textbf{y}$</div>
  </div>

  <ol class="algo__lines">
    <li>$\theta_o \gets \theta$</li>
    <li>$\gamma_o \gets \gamma$</li>
    <li>$k \gets 0$</li>
    <li>$\textbf{while } \textit{not converged } \textbf{do }$</li>
    <li>$~~~~~v_k \gets \text{solveInnerMiCP}(\theta_k, \gamma_k)$</li>
    <li>$~~~~~\nabla_{(\theta, \lambda)} \mathcal{P} \gets \text{calcGrad}(v_k, \theta_k, \gamma_k)$</li>
    <li>$~~~~~\theta_{k+1} \gets \theta_k - \alpha \nabla_{\theta} \mathcal{P}$</li>
    <li>$~~~~~\gamma_{k+1} \gets \max(0, \gamma_k - \alpha \nabla_{\gamma} \mathcal{P})$</li>
    <li>$~~~~~k \gets k + 1$</li>
    <li>$\textbf{end }$</li>
    <li>$\textbf{return } (\theta_k, \gamma_k, \textbf{x}, \textbf{u})$</li>
  </ol>
</div>

</div>
</details>

<details class="section-drop">
<summary>Experimental Results</summary>
<div class="section-drop__body" markdown="1">

Now that our method has been developed for inferring agent forsight and other game parameters has been developed, we can apply it to simulated and real-world data sets! We then compare the results of our experiments to baseline inverse game approaches, and show that our method outperforms them in both fully and partially observable settings. We have three experiments, shown below. Feel free to click through them and check them out!

{% include inf_disc_facts_experiments.html %}

</div>
</details>

<details class="section-drop">
<summary>Some Concluding Remarks</summary>
<div class="section-drop__body" markdown="1">

To summarize our work here, we formulate a noncooperative game that models potentially foresighted agents by associated each one with an unknown discount factor. We then solve for these
discount factors by rewriting the equilibrium conditions of the resulting game as a mixed complementarity problem to leverage the directional differentiability of its equilibrium solutions
with respect to unknown parameters. This enables us to solve an *inverse game* problem and identify discount factors and other game parameters via a gradient descent method. We then verify
and test our results in comparison to state-of-the-art baselines, in which we demonstrate superior performance.

Thank you for reading, and I hope you enjoyed! I'd like to acknowledge [David Fridovich-Keil](https://www.ae.utexas.edu/people/faculty/faculty-directory/fridovich-keil), [Kushagra Gupta](https://autonomy.oden.utexas.edu/profiles/kushagra-gupta), and [Xinjie Liu](https://autonomy.oden.utexas.edu/profiles/xinjie-liu) for their guidance and advising on this project. Their support was invaluable, especially given it was one of my first introductions to game theory. I'd also like to give special thanks to my co-author [Ryan J. Park](https://ryanjpark03.github.io/), who was a fantastic collaborator throughout.
</div>
</details>
