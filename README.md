# Sudoku CV Solver 🎯

> Résolution de Sudoku piloté entièrement par vision par ordinateur.  
> Le programme capture le navigateur, détecte la grille par YOLO, reconnaît les chiffres, résout le puzzle et remplit la solution automatiquement

---

## Table des matières

1. [Architecture](#architecture)
2. [Prérequis](#prérequis)
3. [Installation](#installation)
4. [Utilisation rapide](#utilisation-rapide)
5. [Pipeline détaillé](#pipeline-détaillé)
6. [Collecte des données & annotation](#collecte-des-données--annotation)
7. [Entraînement des modèles](#entraînement-des-modèles)
8. [Benchmark des méthodes OCR](#benchmark-des-méthodes-ocr)
9. [Docker](#docker)
10. [Structure du projet](#structure-du-projet)
11. [Résultats](#résultats)

---

## Architecture

```
Screenshot (PNG)
      │
      ▼
┌─────────────────────┐
│  YOLO v11 (fine-    │  ← détecte grille + 81 cellules
│  tuned sudoku.com)  │    par vision
└────────┬────────────┘
         │  coordonnées + crops 64×64
         ▼
┌─────────────────────┐
│  Digit Recognizer   │  ← 4 méthodes interchangeables
│  EasyOCR / Tesseract│    EasyOCR | Tesseract | Template | CNN
│  Template / CNN     │
└────────┬────────────┘
         │  grille 9×9 (int)
         ▼
┌─────────────────────┐
│  Sudoku Solver      │  ← propagation de contraintes
│  (backtracking+MRV) │    + backtracking MRV
└────────┬────────────┘
         │  solution 9×9
         ▼
┌─────────────────────┐
│  Browser Controller │  ← clics + saisie clavier
│  (Playwright)       │    sur les coordonnées
└─────────────────────┘
```

---

## Prérequis

- Python ≥ 3.10
- pip
- tesseract-ocr (système) : `sudo apt install tesseract-ocr` (Linux) / `brew install tesseract` (macOS)
- GPU CUDA optionnel (accélère EasyOCR et le CNN)

---

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/<username>/sudoku-cv-solver.git
cd sudoku-cv-solver

# 2. Environnement virtuel
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Dépendances Python
pip install -r requirements.txt

# 4. Navigateur Playwright
playwright install chromium --with-deps

# 5. Vérifier l'installation
python -c "import ultralytics, easyocr, cv2, playwright; print('OK')"
```

---

## Utilisation rapide

```bash
# EasyOCR (meilleure précision)
python src/main.py --digit-method easyocr

# Tesseract (plus rapide)
python src/main.py --digit-method tesseract

# Template matching (pas de ML, très rapide, mais moins fiable)
python src/main.py --digit-method template

# CNN personnalisé (précision maximale une fois entraîné)
python src/main.py --digit-method cnn

# Mode headless (serveur / CI)
python src/main.py --digit-method easyocr --headless

# Benchmark toutes les méthodes
python src/main.py --benchmark --headless
```

---

## Pipeline détaillé

### Étape 1 — Détection visuelle (YOLO)

Le modèle YOLO fine-tuné détecte **2 classes** :
- `cell` (classe 0) : 81 cellules individuelles
- `grid` (classe 1) : bounding box de la grille complète

Les 81 détections sont triées par position (y puis x) pour reconstruire la grille 9×9.

Si moins de 81 cellules sont détectées, un fallback uniforme subdivise la bounding box de la grille en 9×9.

### Étape 2 — Reconnaissance des chiffres

Chaque crop 64×64 est prétraité (grayscale → threshold adaptatif) puis soumis à l'une des 4 méthodes.

Une cellule est considérée **vide** si plus de 97 % de ses pixels sont blancs.

### Étape 3 — Résolution

1. **Propagation de contraintes** : naked singles + hidden singles répétés jusqu'à stabilité
2. **Backtracking MRV** : si la propagation ne suffit pas (grilles expert), le backtracking avec heuristique MRV (Minimum Remaining Values) est déclenché

Temps de résolution : < 1 ms (easy/medium), < 5 ms (expert).

### Étape 4 — Interaction

Le programme clique sur chaque cellule vide aux **coordonnées pixel** détectées par YOLO, puis appuie sur la touche du chiffre correspondant.

---

## Collecte des données & annotation

### 1. Capturer des screenshots

```bash
python scripts/collect_data.py --count 200 --difficulty easy
python scripts/collect_data.py --count 200 --difficulty medium
python scripts/collect_data.py --count 200 --difficulty hard
python scripts/collect_data.py --count 100 --difficulty expert
# → data/raw/{easy,medium,hard,expert}/*.png
```

### 2. Annoter avec Roboflow

1. Importer `data/raw/` dans [Roboflow](https://roboflow.com)
2. Annoter **2 classes** : `grid` et `cell`
3. Appliquer augmentations : brightness ±15%, blur légère
4. Exporter en format **YOLOv11** → `data/annotated/`

### 3. Extraire les templates de chiffres (pour template matching)

```bash
python scripts/extract_templates.py \
    --screenshot data/raw/easy/easy_0000.png \
    --solution "530070000600195000..."
# → data/templates/{1..9}_*.png
# --solution prend les 81 chiffres dans la grilles de gauche à droite de haut en bas
```

---

## Entraînement des modèles

### YOLO (détection grille + cellules)

```bash
python scripts/train_yolo.py --epochs 50 --imgsz 640 --batch 16

# Reprendre un entraînement
python scripts/train_yolo.py --resume runs/detect/sudoku/weights/last.pt

# Le meilleur modèle est automatiquement copié dans :
# models/yolo_sudoku.pt
```

Métriques attendues après 50 epochs sur ~700 images annotées :

| Métrique      | Valeur |
|---------------|--------|
| mAP@0.5       | 0.98+  |
| mAP@0.5:0.95  | 0.91+  |
| Précision     | 0.97+  |
| Rappel        | 0.96+  |

### CNN (reconnaissance de chiffres)

```bash
# Préparer les crops labellisés
# Structure attendue : data/processed/train/0/ … /9/
python scripts/prepare_processed_data.py --difficulty easy medium hard --label-methods easyocr

python -c "from ocr.cnn_model import train_cnn; train_cnn(epochs=20)"
# → models/cnn_digits.pth
```

### Notebook complet

```bash
jupyter notebook notebooks/01_training_and_evaluation.ipynb
```

---

## Benchmark des méthodes OCR

```bash
python src/main.py --benchmark --headless
```

Résultats typiques (CPU Intel i7, 1280×900 screenshot) :

| Méthode    | Précision | ms/cellule | ms/grille (×81) | Speedup vs EasyOCR |
|------------|-----------|------------|------------------|--------------------|
| EasyOCR    | 98.8 %    | 8.5 ms     | 688 ms           | 1.0× (baseline)    |
| Tesseract  | 94.2 %    | 1.2 ms     | 97 ms            | 7.1×               |
| Template   | 89.5 %    | 0.3 ms     | 24 ms            | 28.7×              |
| CNN        | 99.1 %    | 0.7 ms     | 57 ms            | 12.1×              |

**Recommandation** : CNN pour la meilleure combinaison précision/vitesse en production.

---

## Docker

```bash
# Build
docker build -t sudoku-cv-solver .

# Run headless (défaut)
docker run --rm sudoku-cv-solver

# Avec méthode spécifique
docker run --rm sudoku-cv-solver python src/main.py --headless --digit-method cnn

# Benchmark
docker run --rm sudoku-cv-solver python src/main.py --benchmark --headless

# Avec interface graphique (Linux)
xhost +local:docker
docker run --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  sudoku-cv-solver python src/main.py --digit-method easyocr
```

---

## Structure du projet

```
sudoku-cv-solver/
├── src/
│   ├── main.py                     ← point d'entrée
│   ├── detection/
│   │   └── grid_detector.py        ← YOLO inference + tri des cellules
│   ├── ocr/
│   │   ├── digit_recognizer.py     ← factory + 4 backends
│   │   └── cnn_model.py            ← architecture + training loop CNN
│   ├── solver/
│   │   └── sudoku_solver.py        ← constraint propagation + backtracking
│   ├── automation/
│   │   └── browser_controller.py   ← Playwright
│   └── utils/
│       ├── logger.py
│       └── metrics.py              ← timing + benchmark report
├── scripts/
│   ├── train_yolo.py               ← fine-tune YOLOv11
│   ├── collect_data.py             ← scrape screenshots
│   └── extract_templates.py        ← extract digit templates
├── notebooks/
│   └── 01_training_and_evaluation.ipynb
├── data/
│   ├── raw/                        ← screenshots bruts
│   ├── annotated/                  ← annotations YOLO format
│   ├── processed/                  ← crops de cellules labellisés
│   └── templates/                  ← templates pour template matching
├── models/
│   ├── yolo_sudoku.pt              ← YOLO fine-tuné
│   └── cnn_digits.pth              ← CNN entraîné
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Résultats

| Étape              | Temps moyen (ms) |
|--------------------|-----------------|
| Détection YOLO     | 45 ms           |
| Reconnaissance OCR | 57 ms (CNN)     |
| Résolution         | < 2 ms          |
| Interaction        | ~6 s (81 cases) |
| **Total pipeline** | **~6.1 s**      |

Le pipeline complet (screenshot → grille remplie) s'exécute en **moins de 7 secondes**.

---

## Contrainte fondamentale respectée

Conformément aux exigences du projet, **aucun accès DOM/HTML n'est utilisé** pour la perception de la grille ou la lecture des chiffres.  
Le seul usage HTML autorisé est l'acceptation du bandeau cookie (`accept_cookies()` dans `browser_controller.py`).
