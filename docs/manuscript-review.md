# Manuscript Review: Eelgrass Segmentation Paper

## 1. MDPI Remote Sensing Technical Note Format

### Key format issues with our current draft

| Requirement | Current draft | Action needed |
|-------------|--------------|---------------|
| Page limit | < 18 pages (not a word count) | We have room; typical is 14-18 pages, 4,000-6,000 words body |
| Abstract | ~300 words required (one paragraph) | Currently outlined at ~200; expand |
| Sections | Numbered IMRAD required | Need to add separate **Conclusions** section (MDPI requires Discussion and Conclusions to be separate) |
| Highlights | Obligatory for Remote Sensing | Missing entirely; need 3-5 bullet points |
| Back matter | Author Contributions, Funding, Data Availability, Acknowledgments, Conflicts of Interest | Only Data Availability and Acknowledgements outlined |
| References | Numbered [1], [2-4] in square brackets, ACS style | Currently author-year; needs reformatting |
| Maps | Must include scale bar, north arrow, coordinates | Study area map has coordinates and scale but verify north arrow |
| Figure text | Minimum 12 pt | Verify on existing plots |
| Keywords | 3-10 after abstract | Missing |

### Structural change: we have more room than expected

A Technical Note is < 18 pages, not ~3,000 words. Published examples run 4,000-6,000 words with 7-9 figures and 1-6 tables. We can afford a fuller treatment than initially planned.

---

## 2. Literature Review: Missing Citations

### Critical gaps (reviewers will expect these)

| Paper | Why critical |
|-------|-------------|
| **2025 SegFormer seagrass UAV paper** (Remote Sensing 17(14), 2518) "Eye in the Sky for Sub-Tidal Seagrass Mapping: Leveraging Unsupervised Domain Adaptation with SegFormer" | Uses SegFormer on UAV seagrass imagery at 3 cm. Most directly comparable published work. MUST cite. |
| **2025 DL seagrass survey** (Ocean Science Journal) "A Survey of Deep Learning Approaches for the Monitoring and Classification of Seagrass" | Latest comprehensive review. Shows awareness of full literature. |
| **Nahirnick et al. (2019)** RSEC "Mapping with confidence; delineating seagrass habitats using UAS" | Foundational BC UAS eelgrass paper. Reshitnyk is co-author. Essential for regional context. |
| **Roberts et al. (2017)** Ecography "Cross-validation strategies for data with temporal, spatial, hierarchical structure" | THE key reference for spatial CV. Reviewers will look for this. |
| **Ploton et al. (2020)** Nature Communications "Spatial validation reveals poor predictive performance of large-scale ecological mapping models" | High-impact demonstration of why spatial CV matters. |
| **Waycott et al. (2009)** PNAS "Accelerating loss of seagrasses across the globe" | Canonical global decline reference (29% lost, 7%/yr). |
| **Weidmann et al. (2019)** "A Closer Look at Seagrass Meadows: Semantic Segmentation" | Most-cited DL seagrass segmentation benchmark (mIoU 87.78%, underwater imagery). |

### Papers already cited that are confirmed correct
- Jeon et al. (2021) Ecological Informatics - confirmed
- Tallam et al. (2023) Remote Sensing 15(9), 2321 - confirmed (Precision 0.723, Recall 0.954, F1 0.809)
- Tahara et al. (2022) PeerJ 10, e14017 - confirmed (OA 0.818)
- Hobley et al. (2021) Remote Sensing 13(9), 1741 - confirmed (semi-supervised seagrass)
- Xie et al. (2021) NeurIPS - SegFormer, confirmed
- Zhou et al. (2018/2020) UNet++, confirmed
- Berman et al. (2018) CVPR - Lovasz loss, confirmed
- Kattenborn et al. (2021) ISPRS - DL for plant ecology, confirmed
- Osco et al. (2021) - DL for UAV RS review, confirmed

### Additional useful references
- Orth et al. (2006) BioScience - seagrass crisis review
- Nordlund et al. (2016) PLOS ONE - seagrass ecosystem services
- Duffy et al. (2018) - drone seagrass monitoring (non-DL)
- Ronneberger et al. (2015) - U-Net original
- Chen et al. (2018) - DeepLabv3+

### Note on Reshitnyk et al. kelp paper
No published peer-reviewed DL kelp paper by Reshitnyk was found. May be in preparation or review. Cite as "Reshitnyk et al., in preparation" or similar if not yet published. Cite Nahirnick et al. (2019) for the Hakai UAS seagrass work instead.

---

## 3. Redundancy and Content Assessment

### Recommendation: Move architecture/augmentation experiments to supplementary

Tables 2-3 (architecture comparison, augmentation tiers) document internal model development. They answer "how did we choose our config?" which is a Methods question, not a Results contribution. For a Technical Note, the contribution is the final model's performance and generalization, not the selection process.

**Proposed restructure:**
- In Methods 2.3, add 2-3 sentences summarizing selection: "We compared UNet++ and SegFormer at two tile sizes; SegFormer MiT-B2 at 1024 px yielded the highest validation IoU (Supplementary Table S1). We evaluated four augmentation tiers; domain-specific augmentations provided the best performance (Supplementary Table S2)."
- Remove Results sections 3.1 and 3.2
- Results becomes three clean sections: (1) Regional CV, (2) Final model pixel metrics, (3) Site-level area estimation

This saves ~300 words and two tables from main text, tightening the narrative.

**Counter-argument for keeping them:** The augmentation experiment (especially domain augmentation for water-column effects) is somewhat novel and practically useful. If kept, consider combining Tables 2-3 into one table.

### Current table count assessment

| Table | Keep/Move | Rationale |
|-------|-----------|-----------|
| Table 1 (dataset summary) | Keep | Essential context |
| Table 2 (architecture) | Move to supplementary | Model selection detail |
| Table 3 (augmentation) | Move to supplementary | Model selection detail |
| Table 4 (regional CV) | Keep | Core result |
| Table 5 (final model) | Keep | Core result |
| Table 6 (site-level area) | Keep | Applied contribution |

Main text: 4 tables. Supplementary: 2 tables. This is well within norms.

### Summary area metrics table has a formatting issue
Lines 191-199 have a duplicate table header. Fix when editing.

---

## 4. Anticipated Reviewer Issues

### HIGH PRIORITY (likely to trigger revision)

**A. Missing methodological details**
- Training hyperparameters: optimizer (AdamW), learning rate (1e-4), scheduler (OneCycleLR), batch size, epochs (200), early stopping criteria, loss function (Lovasz)
- Annotation process: who annotated, protocol (OBIA + manual), towed underwater video for subtidal edge validation
- Class imbalance: proportion of eelgrass vs background pixels
- Prediction threshold: state that 0.5 was used
- Inference pipeline: how chip predictions are stitched back to full orthomosaic
- SegFormer pretraining source: ImageNet (state explicitly)

**B. "N=12 is too few for R² to be meaningful"**
- Removing one point changes R² from 0.875 to 0.965
- Mitigate by: (1) reporting both with/without Beck (already done), (2) emphasizing scatter plot as primary evidence over the statistic, (3) not reporting R² to 3 decimal places (use 0.88 and 0.97), (4) framing Beck as a known edge case motivating multiclass models

**C. "IoU 0.72 is moderate"**
- Reviewers will compare to land-based vegetation segmentation (IoU > 0.8 common)
- Must proactively explain why eelgrass is harder: submerged, water column attenuation, variable density, ground truth informed by video (extends beyond what's visible in RGB)
- Compare to Tallam et al. (F1 0.809 at single site) and Jeon et al. to show our results are competitive at much larger scale

**D. North IoU 0.53 is concerning**
- Needs substantive explanation: what makes North different?
- From the metadata: North sites span only 2016-2018 (older sensors), diverse conditions (shadows, haze, bleaching, overcast), and eelgrass characteristics differ from Central/South
- Acknowledge this honestly as a limitation of geographic generalization

### MEDIUM PRIORITY (may trigger minor revision)

**E. "Why not compare to non-DL baselines?"**
- Reviewers often request comparison to NDVI threshold, random forest, or OBIA
- Mitigate: frame the paper as extending an existing DL pipeline (Reshitnyk et al. kelp model) to eelgrass, not as a novel DL vs traditional methods comparison
- Could note that the annotation workflow (OBIA) itself is the baseline being replaced

**F. "No per-site pixel metrics breakdown"**
- Add per-site IoU to supplementary (easy to compute from existing predictions)

**G. Regional CV naming and fold composition**
- Call it "leave-one-region-out cross-validation" not "3-fold CV"
- Report sites and chips per fold (already in docs/dataset-split-guidance.md)

**H. No confusion matrix**
- Include in supplementary for completeness
- Report both-class metrics (background F1 will be very high)

### LOW PRIORITY (unlikely to block acceptance)

- No confidence intervals on final model (single training run)
- No threshold sensitivity analysis
- No boundary metrics (boundary IoU)
- South region underrepresented (6 sites, all 2018)

---

## Action Plan for Revision

### Before writing final prose

| Priority | Action | Effort | Section affected |
|----------|--------|--------|-----------------|
| 1 | Add training details paragraph | Low | Methods 2.3 |
| 2 | Report chip counts and class balance | Low | Methods 2.2 |
| 3 | Document annotation process (OBIA + video) | Low | Methods 2.1 |
| 4 | State prediction threshold (0.5) | Low | Methods 2.4 |
| 5 | Describe inference/stitching for full orthomosaics | Low | Methods 2.4 |
| 6 | Move Tables 2-3 to supplementary | Medium | Methods 2.3, Results restructure |
| 7 | Rename to "leave-one-region-out CV" with fold composition | Low | Methods 2.3, Results 3.1 |
| 8 | Add Conclusions section (separate from Discussion) | Low | New section 5 |
| 9 | Add Highlights (3-5 bullets) | Low | Front matter |
| 10 | Add Keywords (3-10) | Low | After abstract |
| 11 | Add Author Contributions, Funding, Conflicts of Interest | Low | Back matter |
| 12 | Add missing citations (2025 SegFormer paper, Nahirnick, Roberts, Ploton, Waycott) | Medium | Throughout |
| 13 | Fix summary area metrics table formatting | Low | Results |
| 14 | Round R² to 2 decimal places (0.88, 0.97) | Low | Throughout |
| 15 | Add per-site IoU table to supplementary | Low | Supplementary |

### Key files
- Manuscript: `docs/manuscript.md`
- Results CSVs: `outputs/eval-metrics/*.csv`, `outputs/eval-area/ortho_areas.csv`
- Figures: `outputs/eval-plots/`
- Dataset metadata: `metadata_subset.csv`
- Companion paper: `/mnt/class_data/sdalgarno/lit/Reshitnyketal_KoM.pdf`
