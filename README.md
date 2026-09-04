## Gaslike Social Motility (GSM) and Gaslike Social Motility Ranked (GSM-R) :

---

### **1. Core Concept & Inspiration**

The **Gaslike Social Motility (GSM)** algorithm is a novel swarm/population-based metaheuristic optimization framework.

* **Inspiration:** It is derived from a deterministic physical model of social motility using gaslike particles (originally developed by Parravano & Reyes, 2008).


* **Biological/Social Dynamics Modeled:**
1. Attraction between particles with similar characteristics/moods.


2. Formation of stable particle clusters.


3. Division of particle groups when a critical size is reached.


4. Dynamic spatial distribution changes caused by inter-group interactions.


5. Evolution of a particle's internal state ("mood") via local neighbor interactions.





---

### **2. Mathematical Formulation**

Each particle $i$ has an internal mood/state $x_t^i \in \mathbb{R}$ and a position $r_t^i \in \mathbb{R}^D$.

#### **A. Internal State (Mood) Update Rule**

A particle's mood at step $t+1$ is calculated based on its personal perception scaled by $(1-\epsilon)$ and the influence of neighbors within interaction radius $R$ scaled by $\epsilon$:

$$x_{t+1}^{i} = (1-\epsilon)f(r_{t}^{i}) + \frac{\epsilon}{\vert{}\eta_{t}^{i}\vert{}} \sum_{j \in \eta_{t}^{i}} f(r_{t}^{j})$$

* $f(\cdot)$: Objective function being minimized.


* $\eta_t^i$: Set of active neighbors where $\vert{}r_t^j - r_t^i\vert{} \le R$.


* $\vert{}\eta_t^i\vert{}$: Neighborhood size/cardinality.


* $\epsilon$: Coupling strength ($0 \le \epsilon \le 1$) governing neighbor influence.



#### **B. Spatial Position Update Rule**

The position for particle $i$ evolves as:

$$r_{t+1}^{i} = r_{\mathcal{N}} + \gamma \left( \sum_{j \in \eta_{t}^{i}} \frac{r_{B}^{j} - r_{t}^{i}}{\vert{}r_{B}^{j} - r_{t}^{i}\vert{}} \right) \left( x_{t+1}^{i} \sum_{j \in \eta_{t}^{i}} x_{t+1}^{j} \right)$$

* $r_{\mathcal{N}} \sim \mathcal{N}(r_{Best}, \sigma_{bs})$: A random vector sampled from a normal distribution centered on the global best position $r_{Best}$ with standard deviation $\sigma_{bs}$ (computed over $bs$ top particles). This term guides exploration toward the global best while introducing stochastic social variability.


* $r_B^j$: Best historical position visited by neighboring particle $j$.


* $\gamma$: Movement coupling factor.


* $x_{t+1}^{i} \sum_{j \in \eta_{t}^{i}} x_{t+1}^{j}$: Affinity factor determining direction and distance—good affinity attracts particles to dense optimal regions, whereas bad affinity repels them.



---

### **3. Key Hyperparameters**

| Parameter | Recommended Value | Role & Description |
| --- | --- | --- |
| **$R$** (Radius) | `0.1` | Defines neighborhood scope. Larger $R \to$ global exploration; smaller $R \to$ local exploitation.

 |
| **$\epsilon$** (Coupling) | `0.0001` | Controls internal state blending with neighbors.

 |
| **$\gamma$** (Step factor) | `0.001` | Controls motion scaling for cluster formation.

 |
| **$bs$** (Best subset) | `5` | Number of best-performing particles used to compute standard deviation $\sigma_{bs}$.

 |

---

### **4. Algorithm Steps (Pseudocode Overview)**

```text
Algorithm: Gaslike Social Motility (GSM)
--------------------------------------------------
1. Set parameters: R, epsilon, gamma, bs, Max_Generations G
2. Initialize random positions r^i in R^D and states x^i for i = 1..N
3. Set historical best positions r_B^i = r_t^i
4. For t = 1 to G:
   a. Evaluate fitness f(r_t^i) for all particles.
   b. Update local best r_B^i if current position improves f(r_t^i).
   c. Identify global best r_Best and best bs particles to compute sigma_bs.
   d. Determine neighbor set eta_t^i for each particle where |r_t^j - r_t^i| <= R.
   e. Compute internal state x_{t+1}^i using Eq (3).
   f. Generate r_N ~ N(r_Best, sigma_bs).
   g. Update position r_{t+1}^i using Eq (4).
5. Return overall global best position r_Best.
```[cite: 4]

---

### **5. Performance Benchmarking & Statistical Verification**

* **Tested Functions:** Evaluated on 22 standard benchmark functions ($F_1$ to $F_{22}$) comprising unimodal, multimodal, separable, non-separable, continuous, and scalable functions[cite: 4].
* **Competitor Algorithms:** Compared against **PSO, DE, BA, ABC, AHA, CBO, ECBO, AHA-AO, and SNS**[cite: 4].
* **Key Findings:**
  * Demonstrated superior convergence speed, precision, and robustness while requiring **significantly fewer iterations and smaller population sizes** (e.g., 50 particles, 50–100 iterations)[cite: 4].
  * Performs exceptionally well on unimodal non-separable functions ($F_1, F_5, F_6$) and non-separable multimodal functions ($F_9, F_{15}, F_{16}, F_{18}$)[cite: 4].
  * **Wilcoxon Signed-Rank Test ($\alpha = 0.05$):** Statistically rejected the null hypothesis ($H_0$) across all pairwise algorithm comparisons, confirming GSM's performance superiority[cite: 4].

---

### **6. Application Case: Minimum Cross-Entropy Image Segmentation**

GSM was applied to **Multilevel Thresholding (MTH)** image segmentation by optimizing Minimum Cross-Entropy Thresholding (MCET) objective functions[cite: 4]:

* **Objective Function:** Minimize cross-entropy $D(th)$ between the segmented image histogram and original image histogram to isolate homogenous image segments[cite: 4]:
  $$th_{opt} = \arg\min_{(th)} (D(th))$$
[cite: 4]
* **Validation Dataset:** Tested on standard USC-SIPI images (Boat, House, Airplane, Lake, Tank, Couple, Peppers, Truck, Hunter) across threshold levels $th \in \{2, 4, 8, 16\}$[cite: 4].
* **Evaluation Metrics:** Verified via PSNR, SSIM, FSIM, RMSE, QILV, UIQI, and HPSI[cite: 4]. GSM achieved high-quality segmentations with low computational overhead (only 50 particles, 100 iterations)[cite: 4].

```
# Installation Guide

Follow these steps to set up the environment and run the **Gaslike Social Motility (GSM)** optimization benchmark.

---

## Prerequisites

Ensure you have **Python 3.8+** installed on your system. You can verify your installation by running:

```bash
Here is a clean, GitHub-styled installation guide formatted in Markdown.

```markdown
# Installation Guide

Follow these steps to set up the environment and run the **Gaslike Social Motility (GSM)** optimization benchmark.

---

## Prerequisites

Ensure you have **Python 3.8+** installed on your system. You can verify your installation by running:

```bash
python --version

```

---

## 1. Clone the Repository

Clone this repository to your local machine and navigate into the project directory:

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

```

---

## 2. Create a Virtual Environment (Optional but Recommended)

It is recommended to use a virtual environment to manage dependencies:

### On macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate

```

### On Windows:

```cmd
python -m venv venv
venv\Scripts\activate

```

---

## 3. Install Dependencies

Install the required Python packages (`numpy` and `matplotlib`):

```bash
pip install numpy matplotlib

```

Or, if a `requirements.txt` file is present in the repository:

```bash
pip install -r requirements.txt

```

---

## 4. Run the Benchmark Script

Execute the comparative analysis script (`GSM` vs `GSM-R` on the Rastrigin function):

```bash
python comparison_GSM_KES26.py

```

---

## Expected Output

Upon running the script, the execution metrics will print to your terminal, and a convergence graph will be generated and saved:

```text
============================================================
GSM vs GSM-R  |  Rastrigin 50D
============================================================
Población : 50
Iteraciones : 1000
Semilla : 42

Ejecutando GSM Original...
Ejecutando GSM-R (Rank-Based)...

============================================================
RESULTADOS
============================================================
GSM   best final : ...
GSM-R best final : ...

Gráfica guardada: convergence_gsm_rastrigin.png

```

The output figure `convergence_gsm_rastrigin.png` will be saved in your project root folder.

```

```
