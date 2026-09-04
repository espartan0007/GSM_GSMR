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

# Installation Guide

Follow these steps to set up the environment and run the **Gaslike Social Motility (GSM)** optimization benchmark.

---

## Prerequisites

Ensure you have **Python 3.8+** installed on your system. You can verify your installation by running:


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
git clone https://github.com/espartan0007/GSM_GSMR/)

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

## Citation

If you use this algorithm or repository in your research, please cite the original publication:

```bibtex
@Article{algorithms18040199,
  author         = {Oscar D. Sanchez,  Reyes Luz M. Reyes, and Arturo Valdivia-G and Alanis, Alma Y. and Rangel-Heras, Eduardo},
  title          = {Gaslike Social Motility: Optimization Algorithm with Application in Image Thresholding Segmentation},
  journal        = {Algorithms},
  volume         = {18},
  year           = {2025},
  number         = {4},
  article-number = {199},
  doi            = {10.3390/algo18040199},
  publisher      = {MDPI}
}


