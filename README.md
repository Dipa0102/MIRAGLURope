# MIRA-GLURope

Modèle de langage causal **MiraGLURopeLM** (BPE) + tokenizer + checkpoint prêt à l’emploi.

## Contenu

- `model/mira_glu_rope.py` : définition du modèle `MiraGLURopeLM`
- `tokenizer/tokenizer_v3_fr12k.json` : tokenizer (inclut les tokens spéciaux `<|user|>`, `<|mira|>`, `<|endofturn|>`)
- `checkpoint/mira_glu_rope_v4_v3roles_ft_wiki02_ep3.pt` : poids du modèle
- `reports/latent_scan.md` et `reports/latent_scan.json` : analyse du latent
- `reports/latent_scan_figs/*.png` : graphiques générés (matplotlib)
- `analysis/plot_model.py` : script d’analyse (figures détaillées)
- `analysis/figures/*.png` : figures générées par `analysis/plot_model.py`
- `infer.py` : inférence (sampling) depuis le checkpoint

## Installation

Créer un environnement Python, puis installer les dépendances :

```bash
pip install -r requirements.txt
```

## Utilisation

Générer une réponse (format rôles v3) :

```bash
python infer.py --prompt "<|user|> Bonjour, comment vas-tu ? <|endofturn|>\n<|mira|>" --max-new 250
```

## Analyse (figures)

Générer les mêmes figures que dans `analysis/figures/` :

```bash
python analysis/plot_model.py --checkpoint checkpoint/mira_glu_rope_v4_v3roles_ft_wiki02_ep3.pt
```

Paramètres utiles :

- `--temperature` (défaut 0.85)
- `--top-k` (défaut 50)
- `--top-p` (défaut 0.92)
- `--rep-penalty` (défaut 1.25)
- `--device` (`cpu` ou `cuda`)

## Notes sur GitHub

Le fichier `.pt` est volumineux. Pour le publier sur GitHub, utilisez **Git LFS** (recommandé) :

```bash
git lfs install
git lfs track "*.pt"
```

