<div align="center">

# CV Screening Tool

**Classez automatiquement des CV (FR/EN) par rapport à une offre d'emploi.**

Similarité sémantique, TF-IDF et correspondance des compétences, dans une interface Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-09A3D5?logo=spacy&logoColor=white)
![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-multilingue-7c3aed)
![License](https://img.shields.io/badge/License-MIT-green)

<!-- Ajoutez une capture d'écran : -->
<!-- ![Aperçu](docs/screenshot.png) -->

</div>

---

## Présentation

**CV Screening Tool** aide les recruteurs à présélectionner rapidement des candidats. On colle une offre d'emploi, on charge plusieurs CV, et l'outil renvoie un classement transparent : chaque score est décomposé et les compétences manquantes de chaque candidat sont listées.

## Fonctionnalités

- **Multi-format** : lecture des CV en PDF et DOCX
- **Bilingue** : CV et offres en français ou en anglais
- **Extraction d'informations** : compétences (spaCy `PhraseMatcher`), années d'expérience et niveau de formation
- **Score hybride** : sémantique, lexical et compétences combinés
- **Résultats détaillés** : classement, scores par critère, compétences trouvées et manquantes
- **Export CSV** : récupérez le classement pour le partager ou le retraiter
- **Interface claire** : tableau avec barres de progression, cartes de synthèse, onglets et graphique de comparaison

## Méthode de scoring

| Critère | Poids | Méthode |
|---|:---:|---|
| Similarité sémantique | **50 %** | Embeddings Sentence Transformers (modèle multilingue) + similarité cosinus |
| Similarité lexicale | **20 %** | Vectorisation TF-IDF + similarité cosinus |
| Compétences | **30 %** | Part des compétences de l'offre retrouvées dans le CV |

```
Score final = 0.5 × Sémantique + 0.2 × TF-IDF + 0.3 × Compétences
```

L'approche sémantique repère les profils pertinents même avec un vocabulaire différent. Le TF-IDF et le score de compétences ajoutent la précision sur les termes exacts.

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/iiyadh/CV-analyzer.git
cd CV-analyzer

# 2. (Recommandé) Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

> Le modèle Sentence Transformers est téléchargé automatiquement au premier lancement, ce qui peut prendre quelques instants.

## Utilisation

```bash
streamlit run app.py
```

L'application s'ouvre sur [http://localhost:8501](http://localhost:8501).

1. Collez la description du poste.
2. Chargez un ou plusieurs CV (PDF ou DOCX).
3. Cliquez sur **Classer les candidats**.
4. Consultez le classement, ouvrez la fiche d'un candidat ou exportez le CSV.

## Structure du projet

<!-- Adaptez selon votre arborescence -->
```
.
├── app.py              # Interface Streamlit
├── requirements.txt    # Dépendances Python
└── README.md
```

## Stack technique

- **Langage** : Python
- **NLP** : spaCy, Sentence Transformers, scikit-learn (TF-IDF, similarité cosinus)
- **Lecture de fichiers** : extraction de texte PDF et DOCX
- **Interface** : Streamlit

## Limites

- Les PDF scannés (images sans couche texte) ne sont pas lus sans OCR.
- La liste de compétences reconnues dépend du dictionnaire utilisé par le `PhraseMatcher`.
- Le score est une **aide à la décision** et ne remplace pas l'évaluation humaine. Il peut reproduire des biais présents dans les données ou les modèles.

## Pistes d'amélioration

- [ ] OCR pour les CV scannés
- [ ] Pondérations réglables depuis l'interface
- [ ] Enrichissement du dictionnaire de compétences
- [ ] Export PDF/Excel du rapport

## Licence

Distribué sous licence MIT. Voir le fichier [`LICENSE`](LICENSE).
