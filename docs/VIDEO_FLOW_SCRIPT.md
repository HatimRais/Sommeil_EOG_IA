# Script de la vidéo de démonstration — Sommeil_EOG_IA

Document prêt à l'emploi pour générer la vidéo avec **Google Flow** (moteur **Veo**).
Il contient : (1) la méthode d'assemblage, (2) le **style global** à réutiliser, (3) la **narration**
française minutée, (4) les **prompts exacts** (un par plan, à copier-coller dans Flow), (5) les réglages
qui évitent les erreurs et la redondance.

Durée cible : **~95 secondes** · Format : **16:9 · 1080p · 24 fps**.

---

## 0. Méthode (important — à lire avant de générer)

Flow / Veo génère des **clips de ~8 secondes** à partir d'un prompt texte. Pour une vidéo stable et
cohérente :

1. **Générez chaque plan séparément** (un prompt = un clip), dans l'ordre ci-dessous.
2. **Réutilisez la même phrase de style** (section 1) à la fin de chaque prompt → cohérence visuelle.
3. **Ne demandez PAS de texte à l'écran** (titres, chiffres, interface lisible) : Veo déforme le texte.
   Les titres, le logo « DeepSleep AI » et les vrais graphiques seront ajoutés **au montage** (CapCut,
   Premiere, Canva…) ou par **capture d'écran** réelle du dashboard et du notebook.
4. **Plan hybride recommandé** : les plans Flow servent d'**habillage cinématique** (intro, contexte,
   transitions, conclusion). Pour montrer le **vrai dashboard** et le **vrai hypnogramme**, intercalez
   un **screencast** (capture d'écran de `streamlit run app/dashboard.py` sur Patient_01).
5. Si un clip présente un artefact (mains/visages déformés, scintillement), **régénérez** avec la même
   seed/prompt plutôt que de modifier le texte.

---

## 1. Style global (À COLLER À LA FIN DE CHAQUE PROMPT)

```
Style: cinematic medical-technology aesthetic, deep navy and teal color palette (#0C4A6E, #1E5F8A, #5EB8A8), soft volumetric lighting, shallow depth of field, photorealistic, clean and calm, subtle camera motion, 16:9, 24fps, high detail, no on-screen text, no captions, no watermark, no logos.
```

**Négatif / à éviter (champ "negative prompt" si disponible) :**
```
text, captions, subtitles, watermark, logo, distorted hands, extra fingers, deformed faces, glitching letters, cluttered UI, oversaturated colors, cartoonish.
```

---

## 2. Narration française (voix off) — minutée

À enregistrer séparément (voix off humaine ou TTS) et à caler au montage sur les plans correspondants.

| Temps | Plan | Narration |
|------|------|-----------|
| 0:00–0:10 | 1 | « Chaque nuit, notre cerveau traverse plusieurs stades de sommeil. Les identifier est essentiel pour la santé. » |
| 0:10–0:22 | 2 | « Mais analyser une nuit entière, fenêtre par fenêtre, demande des heures de travail à un expert. » |
| 0:22–0:35 | 3 | « Notre projet part d'un seul signal : l'électro-oculogramme, qui capte les mouvements des yeux. » |
| 0:35–0:50 | 4 | « Un réseau de neurones convolutif apprend à reconnaître chaque stade à partir de ce signal. » |
| 0:50–1:02 | 5 | « Optimisé avec OpenVINO, le modèle s'exécute en temps réel sur le NPU, le GPU ou le simple processeur. » |
| 1:02–1:18 | 6 (screencast) | « Dans notre tableau de bord DeepSleep AI, il suffit de charger l'enregistrement : l'hypnogramme est reconstruit en quelques secondes. » |
| 1:18–1:30 | 7 (screencast) | « Sur des sujets jamais vus, l'IA atteint près de 86 % de concordance avec l'expert. » |
| 1:30–1:35 | 8 | « Sommeil EOG IA : analyser le sommeil, simplement. » |

---

## 3. Prompts exacts (un par plan)

> Copiez le bloc, **ajoutez la phrase de style (section 1)** à la fin, puis générez. Chaque plan ≈ 8 s.

### Plan 1 — Ouverture (0:00–0:10)
```
Aerial slow push-in over a quiet modern city at night under a deep blue sky, then a soft dissolve toward a calm hospital sleep-clinic window glowing warmly. Peaceful nocturnal mood, stars faintly twinkling.
```

### Plan 2 — Le problème (0:10–0:22)
```
Inside a dim sleep laboratory at night, a tired technologist in a white coat sits before multiple monitors showing abstract glowing biomedical waveforms (no readable text), rubbing their eyes. Stacks of paper charts beside them. Quiet, focused, slightly exhausting atmosphere, slow dolly-in.
```

### Plan 3 — Le signal EOG (0:22–0:35)
```
Extreme close-up of a sleeping person's closed eyes, small medical sensor electrodes gently placed near the temples, thin wires. Slow transition into an abstract glowing teal waveform that ripples horizontally as the eyes move beneath the eyelids. Macro, delicate, scientific beauty.
```

### Plan 4 — Le réseau de neurones (0:35–0:50)
```
Abstract 3D visualization of a deep neural network: a horizontal stream of luminous data points flowing left to right through stacked translucent convolutional layers, pulses of light propagating and converging into five glowing nodes. Dark navy background, teal and cyan light trails, elegant and futuristic, smooth camera orbit.
```

### Plan 5 — Accélération matérielle / NPU (0:50–1:02)
```
Macro shot of a sleek computer processor chip on a dark circuit board, glowing teal traces lighting up in sequence as energy flows toward the chip core. Tiny particles of light, depth of field, high-tech and clean. Slow rotating camera, sense of speed and efficiency.
```

### Plan 6 — Le médecin et le tableau de bord (1:02–1:18) — *cinématique de transition*
```
Over-the-shoulder shot of a clinician in a modern bright office looking at a large monitor displaying a colorful abstract sleep-stage chart with smooth steps in yellow, blue and red bands (no readable text). The clinician nods with satisfaction. Soft daylight, calm and professional, gentle push-in.
```
> **À intercaler ensuite avec le screencast réel** du dashboard (voir section 5).

### Plan 7 — Résultat / concordance (1:18–1:30) — *cinématique de transition*
```
Two abstract sleep-stage curves overlaying and gradually aligning into near-perfect synchronization on a dark screen, glowing teal and navy lines, a soft pulse of light when they match. Minimalist data-visualization aesthetic, satisfying convergence.
```
> **À intercaler avec la capture réelle** de l'hypnogramme IA vs expert et des métriques.

### Plan 8 — Clôture (1:30–1:35)
```
Slow zoom-out from a glowing crescent moon reflected on calm water at night, fading into a clean deep-navy gradient background with empty centered space reserved for a title. Serene, premium, minimalist.
```
> Ajoutez au montage le titre **« Sommeil_EOG_IA — DeepSleep AI »** sur l'espace réservé.

---

## 4. Réglages Flow recommandés

- **Aspect ratio** : 16:9 · **Résolution** : 1080p · **Modèle** : Veo (qualité la plus élevée disponible).
- **Outputs per prompt** : 2 à 4 variantes, gardez la meilleure.
- **Extend / Frames-to-Video** : pour enchaîner deux plans, utilisez la dernière image d'un clip comme
  image de départ du suivant → continuité fluide.
- **Cohérence** : ne changez jamais la phrase de style entre les plans.
- **Audio** : générez la vidéo **sans** dialogue parlé par Veo (la narration FR est ajoutée au montage).

---

## 5. Plan hybride : captures d'écran réelles (qualité maximale)

Pour les plans 6 et 7, rien ne vaut le **vrai produit**. Enregistrez votre écran (OBS Studio / Xbox Game Bar) :

1. Lancez le dashboard :
   ```bash
   streamlit run app/dashboard.py
   ```
2. Sélectionnez le périphérique (NPU / CPU), chargez **Patient_01** depuis `data/raw/`.
3. Filmez : l'inférence, l'hypnogramme IA, l'onglet « AI vs Expert », les métriques cliniques.
4. (Optionnel) Filmez le **notebook** `notebooks/Sommeil_EOG_IA_Demo.ipynb` qui défile avec ses graphiques.

Au montage : alternez les plans cinématiques de Flow (habillage) et les captures réelles (preuve),
en gardant la narration FR comme fil conducteur.

---

## 6. Ordre de montage final

```
[1 Flow] → [2 Flow] → [3 Flow] → [4 Flow] → [5 Flow]
   → [6 Flow court + SCREENCAST dashboard] 
   → [7 Flow court + SCREENCAST hypnogramme/métriques]
   → [8 Flow + titre incrusté]
```

Musique : nappe ambient calme, montée discrète vers la conclusion. Coupez chaque plan à ~6–8 s pour
garder un rythme dynamique et éviter toute répétition.
