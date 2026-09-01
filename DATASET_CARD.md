# Dataset Card

## Dataset Identity

- Dataset: **Frosch bottle 5**
- Version: **v6**
- Format: **COCO Segmentation**
- Source: **Roboflow export**
- Export date: **23 June 2026**
- Image size: **432 × 432**
- License: **CC BY 4.0**

## Dataset Size

| Split | Images | Annotations |
|---|---:|---:|
| Train | 739 | 2370 |
| Validation | 307 | 996 |
| Test | 0 | 0 |
| **Total** | **1046** | **3366** |

The actual available split is approximately:

- Train: 70.65%
- Validation: 29.35%
- Test: 0%

The supplied dataset therefore does not contain the handbook-requested 70/20/10 train/validation/test split.

## Collection Strategy

The project images were **provided by the project team lead**. They were not independently collected by the implementation author.

## Object Selection

The target object is the Frosch bottle.

Approximate physical bottle dimensions were **not known or recorded** during the project.

## Labelling

The dataset is a Roboflow COCO Segmentation export. The final artifacts do not preserve additional annotation-tool history beyond that export provenance.

## Class Distribution

| Class | Train | Validation | Total | Share of annotations |
|---|---:|---:|---:|---:|
| Frosch-bottle-UTNY-aUbJ-XBXs | 0 | 0 | 0 | 0.00% |
| bottle | 731 | 304 | 1035 | 30.75% |
| bump | 145 | 64 | 209 | 6.21% |
| capacity | 682 | 273 | 955 | 28.37% |
| damage | 48 | 24 | 72 | 2.14% |
| label | 727 | 303 | 1030 | 30.60% |
| scratch | 37 | 28 | 65 | 1.93% |

The first category exists in COCO metadata but has zero annotations.

## Preprocessing

The supplied dataset export uses auto-orientation/EXIF handling and resize to 432 × 432 using Stretch. No custom augmentation policy is claimed for the final training configuration.

## Dataset Limitations

- No separate test split.
- Physical bottle dimensions not recorded.
- Images provided by the team lead.
- Original collection protocol not retained in the final artifacts.
