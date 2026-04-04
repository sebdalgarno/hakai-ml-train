# Automated delineation of eelgrass extent from drone imagery using deep learning: a regional-scale evaluation across coastal British Columbia

**Target:** PeerJ (Research Article, no word limit, APC ~$1,700)

### TODO

See `docs/manuscript-review.md` for full review context.

**Writing (Seb):**
- [ ] Write Abstract (~250 words)
- [ ] Write Introduction (~500-700 words)
- [ ] Write Discussion (~600-800 words)
- [ ] Write Data and Code Availability statement
- [ ] Write Acknowledgements
- [ ] Compile reference list (author-year style, e.g., Jeon et al., 2021)
- [ ] Review Methods/Results drafted sections for accuracy and tone

**Methods gaps to fill:**
- [x] Add training details paragraph: optimizer, lr, scheduler, batch size, epochs, loss, pretraining
- [x] Report class imbalance (22.5% seagrass, 3.4:1 ratio)
- [x] State prediction threshold (0.5)
- [x] Describe inference pipeline (ONNX export, habitat-mapper tiling/stitching)
- [x] Explain 512 vs 1024 tile size comparison (random/centre crop from same 1024 chips)
- [x] Add spatial CV references (Roberts et al., 2017)
- [x] Expanded reference list with all critical citations
- [ ] Document annotation process: OBIA workflow, manual editing, towed underwater video (needs Seb input)

**Literature (missing critical citations):**
- [ ] 2025 SegFormer seagrass UAV paper (Remote Sensing 17(14), 2518)
- [ ] 2025 DL seagrass survey (Ocean Science Journal)
- [ ] Nahirnick et al. (2019) RSEC - foundational BC UAS eelgrass paper
- [ ] Roberts et al. (2017) Ecography - spatial CV
- [ ] Ploton et al. (2020) Nature Communications - spatial validation
- [ ] Waycott et al. (2009) PNAS - seagrass decline
- [ ] Confirm Reshitnyk et al. kelp paper publication status

**Figures:**
- [ ] Regenerate study area map at publication resolution (verify north arrow)
- [ ] Select 4 prediction panels from `outputs/visualize-pred-ortho/by_ortho.pdf`
- [ ] Export panels as high-resolution images
- [ ] Regenerate area scatter and error plots (notebooks reverted)

**Supplementary material:**
- [ ] Architecture comparison table (Table S1)
- [ ] Augmentation comparison table (Table S2)
- [ ] Per-site IoU breakdown for test sites (Table S3)
- [ ] Full survey metadata (Table S4)

**Information needed from Seb (not in repo):**

*Annotation and field methods:*
- [ ] Who annotated? One analyst or multiple? Inter-annotator check?
- [ ] Annotation criteria: what defined the eelgrass boundary (minimum density, percent cover)?
- [ ] Which sites had towed underwater video for subtidal edge? All Central Coast or a subset?
- [ ] Approximate time per orthomosaic for manual delineation (quantifies the bottleneck)
- [ ] Software used for annotation (QGIS, ArcGIS, etc.)?

*UAS surveys:*
- [ ] Flight altitude and overlap settings
- [ ] Tidal window targeted? What tide height range?
- [ ] GCPs or RTK/PPK for georeferencing?

*Regional differences (to explain North IoU 0.53):*
- [ ] What makes Gwaii Haanas ecologically different? Substrates, water clarity, eelgrass morphology?
- [ ] North only has 2016-2018 data with older sensors; is this a factor?
- [ ] Qualitative observations from North predictions?

*Beck site:*
- [ ] Confirm: ground truth includes ambiguous sparse intertidal areas?
- [ ] Is Beck representative or an unusual edge case?

*Practical context:*
- [ ] Manual processing time per site vs model time?
- [ ] Is anyone running habitat-mapper operationally on new surveys?
- [ ] Management decisions depending on these extent estimates?

*Admin:*
- [ ] Full author list, affiliations, ORCID IDs, corresponding author
- [ ] Funding sources and grant numbers
- [ ] Data sharing restrictions (Indigenous data sovereignty for Gwaii Haanas?)
- [ ] Reshitnyk et al. kelp paper: published, in review, or in preparation?

---

## Abstract (OUTLINE - ~250 words)

- Eelgrass (*Zostera marina*) monitoring increasingly relies on drone (UAS) surveys, but manual delineation of extent from orthomosaic imagery remains a bottleneck
- Submerged eelgrass presents detection challenges distinct from emergent species (e.g., kelp): signal is attenuated by the water column, density varies from sparse intertidal to dense subtidal, and the subtidal edge is a gradual depth-dependent transition
- We trained a SegFormer semantic segmentation model on 62 annotated drone orthomosaics from 30 sites across 3 regions of coastal British Columbia (South, Central, North)
- Site-level data splitting prevented spatial leakage; leave-one-region-out cross-validation assessed geographic generalization
- Final model achieved IoU 0.72, F1 0.84 on held-out test sites
- Site-level eelgrass area estimates correlated with ground truth (R² = 0.88, MAE = 3,363 m², mean % error = -8.9%)
- Leave-one-region-out CV demonstrated generalization across regions (IoU 0.53-0.69), with the range representing a realistic floor for application to new regions
- Key challenges were sparse eelgrass (where binary classification is inherently ambiguous) and subtidal edges (where ground truth informed by underwater video extends beyond what is visually discernible in the imagery)
- The model is deployed as an open-source tool (habitat-mapper) for use by the conservation and monitoring community

---

## 1. Introduction (OUTLINE - ~500-700 words)

**Seagrass importance:**
- Seagrass meadows provide critical ecosystem services: carbon sequestration, nursery habitat, sediment stabilization, water quality indicators (Waycott et al., 2009; Orth et al., 2006)
- Global declines documented (29% of known extent lost; Waycott et al., 2009); monitoring extent is essential for conservation and management
- *Zostera marina* (eelgrass) is the dominant seagrass species in the northeast Pacific

**Monitoring with drones:**
- UAS surveys increasingly used for local and regional seagrass monitoring (Duffy et al., 2018; Nahirnick et al., 2019)
- Manual delineation from orthomosaic imagery is the primary bottleneck (hours per site, subjective, does not scale)
- Same bottleneck identified for kelp mapping (Reshitnyk et al.)

**Challenges specific to eelgrass detection:**
- Eelgrass is submerged, so the signal is attenuated by the water column (unlike emergent kelp canopy visible at the surface)
- Density varies from sparse intertidal patches (difficult to distinguish from substrate) to dense subtidal meadows
- The subtidal edge is the hardest boundary to delineate: a gradual, depth-dependent transition where visibility degrades. Ground truth annotations at many sites were informed by concurrently collected towed underwater video, so the annotated extent can exceed what is visually discernible in the drone imagery alone
- Environmental variability across surveys (e.g., glint, shadows, fog, turbidity, tannins, overcast conditions) further complicates detection

**Prior work:**
- Deep learning for seagrass segmentation from drone imagery: Jeon et al. (2021), Tallam et al. (2023), Tahara et al. (2022), Hobley et al. (2021)
- These studies were limited to single study areas or small numbers of sites, with no evaluation of geographic generalization
- Importantly, all used random tile-level data splits, which inflate reported accuracy through spatial autocorrelation (Roberts et al., 2017; Ploton et al., 2020)
- Reshitnyk et al. demonstrated deep learning for kelp canopy mapping from UAS imagery using a CNN deployed via the open-source habitat-mapper tool; kelp is emergent (surface canopy), making it a comparatively simpler detection target

**Gap and contribution:**
- No study has evaluated deep learning for seagrass segmentation across multiple geographic regions with rigorous spatial cross-validation
- We present a regional-scale evaluation of a SegFormer model for eelgrass extent delineation, trained on 62 orthomosaics from 30 sites across 3 regions of coastal British Columbia
- We assess generalization through leave-one-region-out CV and evaluate site-level area estimation accuracy
- The model is deployed via habitat-mapper (alongside the kelp model) as a free, open-source tool

---

## 2. Materials and Methods (DRAFTED - ~1,000 words)

### 2.1 Study area and data

We compiled a dataset of 62 annotated UAS orthomosaics from 30 eelgrass monitoring sites across three regions of coastal British Columbia, Canada (Table 1; Fig. 1). The South region (6 sites, 6 orthomosaics) spans Pacific Rim National Park Reserve on the west coast of Vancouver Island. The Central region (9 sites, 34 orthomosaics) covers the Central Coast around Calvert Island, where the Hakai Institute conducts long-term ecological monitoring. The North region (15 sites, 22 orthomosaics) is located in Gwaii Haanas National Park Reserve on Haida Gwaii. Surveys were conducted between 2016 and 2024 during boreal summer months (May through August), timed for low tide conditions to maximize eelgrass visibility.

UAS platforms included the DJI Phantom Pro 3, DJI Phantom Pro 4, and Mavic 3E, producing orthomosaics at ground sampling distances of 1.6 to 5.6 cm. Orthomosaics were generated using structure-from-motion software (Agisoft Metashape Professional). Eelgrass extent was delineated by trained analysts using an object-based image analysis (OBIA) workflow, with manual editing where necessary. At several sites, the subtidal eelgrass edge was confirmed using concurrently collected towed underwater video. Delineation confidence was rated for each orthomosaic, with the majority of datasets rated as high confidence. Detailed survey metadata (i.e., date, sensor, resolution, tide height, delineation confidence, and image quality) are provided in Table S4.

**Table 1.** Dataset summary by region.

| Region | Sites | Orthomosaics | Date range | Resolution (cm) |
|--------|-------|-------------|------------|-----------------|
| South | 6 | 6 | 2018 | 2.4-5.2 |
| Central | 9 | 34 | 2017-2024 | 1.6-5.6 |
| North | 15 | 22 | 2016-2018 | 2.6-5.2 |
| **Total** | **30** | **62** | **2016-2024** | **1.6-5.6** |

### 2.2 Data preparation

Each orthomosaic and its corresponding label raster were tiled into 1024 x 1024 pixel chips. Labels were encoded as binary masks: background (0), eelgrass (1), and ignore/uncertain (-100). The ignore class was applied to pixels where the annotation was ambiguous (e.g., at the subtidal edge where visibility was poor), ensuring these pixels did not contribute to the training loss.

Data were split into training, validation, and test sets at the site level rather than the tile level. All tiles from a given site (including all repeat visits) were assigned to the same split, preventing spatial autocorrelation leakage between splits (Roberts et al., 2017). The 30 sites were divided into 18 training, 6 validation, and 6 test sites, stratified to maintain representation of all three geographic regions and a range of environmental conditions (e.g., lighting, water clarity, eelgrass density) in each split. The resulting dataset contained 10,245 training, 1,995 validation, and 1,761 test chips. Eelgrass pixels comprised 23.8% of valid pixels in the training set, 22.3% in validation, and 15.1% in test (background-to-eelgrass ratios of 3.2:1, 3.5:1, and 5.6:1, respectively). The lower eelgrass proportion in the test set reflects the inclusion of sites with sparser eelgrass coverage (e.g., Beck, Sedgwick).

### 2.3 Model selection and training

We selected the model architecture and augmentation strategy through experiments on a prototype dataset (a random subsample of chips from all sites, preserving site diversity for faster iteration). We compared UNet++ with a ResNet-34 backbone (Zhou et al., 2018) and SegFormer with a MiT-B2 backbone (Xie et al., 2021) at 512 and 1024 px tile sizes; SegFormer at 1024 px achieved the highest validation IoU (Table S1). For the tile size comparison, all models were trained on the same 1024 px chips; 512 px inputs were generated via random cropping in the dataloader (training) and centre cropping (validation), ensuring both tile sizes covered the same spatial regions. We then compared four augmentation tiers of increasing complexity, from geometric-only to domain-specific augmentations simulating water-column effects (i.e., colour shifts, turbidity simulation, refraction distortion); domain augmentation provided the best performance (Table S2).

The final model (SegFormer MiT-B2, 1024 px, domain augmentation) was implemented using segmentation-models-pytorch (Iakubovskii, 2019) and PyTorch Lightning (Falcon et al., 2019), with the encoder initialized from ImageNet-pretrained weights. We used Lovasz loss (Berman et al., 2018) to directly optimize IoU, which is well-suited for class-imbalanced segmentation tasks. The model was trained with a batch size of 6, a maximum learning rate of 1e-4 with the One Cycle learning rate scheduler (Smith & Topin, 2019), weight decay of 0.01, and bf16 mixed precision. Training ran for up to 1000 epochs with early stopping (patience of 15 epochs, monitoring validation IoU). Data augmentation during training included D4 geometric transforms (8-way rotation and flipping), random brightness and contrast adjustment, hue-saturation-value shifts, Gaussian and ISO noise, motion and Gaussian blur, affine transforms (rotation +/-5 degrees, scale 0.7-1.3), coarse dropout, grid distortion, elastic transforms, CLAHE, and colour jitter. Validation and test data received only ImageNet normalization. Tiles were normalized using channel-wise mean and standard deviation values from the ImageNet dataset. The model was trained on a single NVIDIA GPU. Training code is available at https://github.com/HakaiInstitute/hakai-ml-train.

### 2.4 Leave-one-region-out cross-validation

To evaluate geographic generalization, we conducted a leave-one-region-out cross-validation, holding out each region (North, Central, South) as the test set in turn. The remaining two regions were split into training (~82%) and validation (~18%) sets. This design tests how well the model performs when applied to an entirely unseen geographic region.

| Fold | Test region | Test sites | Test chips | Train sites | Train chips |
|------|-------------|-----------|------------|-------------|-------------|
| 1 | North | 15 | 23,771 | 9 | ~56,300 |
| 2 | Central | 9 | 55,828 | 15 | ~29,400 |
| 3 | South | 6 | 12,099 | 17 | ~64,500 |

### 2.5 Final model evaluation

The best model configuration was retrained on all training and validation data combined and evaluated on the held-out test sites.

We report four pixel-level metrics for the eelgrass class: intersection over union (IoU), precision, recall, and F1 score. Predictions were generated by applying a softmax activation to the model output and thresholding the eelgrass class probability at 0.5.

For inference on full orthomosaics, the trained model was exported to ONNX format and applied via the habitat-mapper tool (https://github.com/HakaiInstitute/habitat-mapper), which tiles the input orthomosaic into overlapping chips, runs inference on each chip, and merges predictions back into a single georeferenced raster.

To assess ecological utility, we compared predicted and ground truth eelgrass area at the orthomosaic level. For each test orthomosaic, eelgrass area (m²) was calculated by summing the predicted eelgrass pixels and multiplying by the pixel area derived from the orthomosaic ground sampling distance. We report the mean absolute error (MAE), root mean square error (RMSE), coefficient of determination (R²), and mean percentage error.

---

## 3. Results (DRAFTED - ~600 words)

### 3.1 Leave-one-region-out cross-validation

Model performance varied across regions when each was held out as the test set (Table 2). The South fold achieved the highest IoU (0.695), followed by Central (0.626) and North (0.529). Precision was consistently high across folds (0.70-0.97), while recall was more variable (0.68-0.71).

**Table 2.** Leave-one-region-out cross-validation results (test set for each fold).

| Test region | IoU | Precision | Recall | F1 |
|-------------|-----|-----------|--------|----|
| North | 0.529 | 0.702 | 0.681 | 0.692 |
| Central | 0.626 | 0.861 | 0.696 | 0.770 |
| South | 0.695 | 0.973 | 0.708 | 0.820 |

### 3.2 Final model: pixel metrics

The final model (trained on all training and validation data) achieved an IoU of 0.720 and F1 of 0.837 on the held-out test sites (Table 3). Precision (0.866) exceeded recall (0.810), indicating the model was conservative in its predictions.

**Table 3.** Final model performance.

| Split | IoU | Precision | Recall | F1 |
|-------|-----|-----------|--------|----|
| Validation | 0.674 | 0.825 | 0.786 | 0.805 |
| Test | 0.720 | 0.866 | 0.810 | 0.837 |

### 3.3 Site-level area estimation

Predicted eelgrass area was compared to ground truth across 12 test orthomosaics from 6 sites (Table 4; Fig. 3). Overall, predicted and ground truth areas were correlated (R² = 0.88, MAE = 3,363 m², mean percentage error = -8.9%). However, area metrics were heavily influenced by a single outlier: Beck, where the model underestimated area by 15,539 m² (-38.7%), accounting for 76% of the total squared error. On inspection, much of the disagreement at Beck occurred in areas of very sparse eelgrass where the ground truth delineation itself is debatable (i.e., eelgrass presence is ambiguous in the imagery). Excluding Beck, the model achieved R² = 0.97 and MAE = 2,256 m² across the remaining 11 orthomosaics.

The model performed well at sites with clear conditions and moderate to dense eelgrass (e.g., Bag Harbour, Triquet Bay), where area errors were within 10% of ground truth. At Triquet Bay, predicted area tracked ground truth consistently across three visits spanning 2018 to 2024, with errors of -5.2%, -1.9%, and -9.6%. At Sedgwick, results were inconsistent across visits: one orthomosaic (u0085) with cloud reflections was overestimated by 5,350 m² (+59.1%), while two others were underestimated by 23-37%.

**Table 4.** Site-level area comparison for test orthomosaics.

| Orthomosaic | Region | Resolution (cm) | GT area (m²) | Predicted area (m²) | Error (m²) | Error (%) |
|-------------|--------|----------------|--------------|--------------------|-----------| ---------|
| bag_harbour_u0490 | North | 2.7 | 51,436 | 52,267 | +832 | +1.6 |
| beck_u0409 | South | 5.0 | 40,122 | 24,583 | -15,539 | -38.7 |
| mcmullin_north_u0900 | Central | 2.3 | 13,416 | 14,089 | +673 | +5.0 |
| mcmullin_north_u1270 | Central | 2.5 | 14,099 | 12,306 | -1,792 | -12.7 |
| section_cove_u0249 | North | 4.4 | 11,030 | 9,163 | -1,868 | -16.9 |
| section_cove_u0487 | North | 4.0 | 10,455 | 7,511 | -2,944 | -28.2 |
| sedgwick_u0085 | North | 4.4 | 9,059 | 14,409 | +5,350 | +59.1 |
| sedgwick_u0260 | North | 4.1 | 9,291 | 7,155 | -2,136 | -23.0 |
| sedgwick_u0482 | North | 4.0 | 9,142 | 5,795 | -3,346 | -36.6 |
| triquet_bay_u0537 | Central | 4.0 | 32,983 | 31,266 | -1,717 | -5.2 |
| triquet_bay_u0709 | Central | 3.2 | 33,126 | 32,501 | -625 | -1.9 |
| triquet_bay_u1292 | Central | 2.4 | 36,798 | 33,264 | -3,534 | -9.6 |

**Summary area metrics:**

| Metric | All orthos (n=12) | Excluding Beck (n=11) |
|--------|-------------------|----------------------|
| MAE | 3,363 m² | 2,256 m² |
| RMSE | 5,149 m² | 2,641 m² |
| Mean bias | -2,221 m² | — |
| Mean % error | -8.9% | -6.2% |
| R² | 0.88 | 0.97 |

### Figures

**Figure 1.** Study area map showing 30 eelgrass monitoring sites across three regions of coastal British Columbia. Point size indicates the number of repeat visits per site. [outputs/eval-plots/study_area_map.png]

**Figure 2.** Example model predictions for one representative orthomosaic from each test site, ordered by decreasing per-site IoU. Each panel shows the RGB image, ground truth label, and model prediction (0.5 threshold). Top row: sites where the model performs well. Bottom row: sites where the model struggles. (a) Triquet Bay (IoU 0.83): dense meadow, clear water, accurate delineation. (b) Bag Harbour (IoU 0.76): good prediction under baseline conditions. (c) Section Cove (IoU 0.66): hazy lighting reduces recall. (d) McMullin North (IoU 0.58): clear conditions but model underperforms. (e) Beck (IoU 0.52): high precision (0.90) but low recall (0.55); model misses large areas of sparse eelgrass. (f) Sedgwick (IoU 0.49): errors in both directions; false positives from cloud reflections and false negatives in deeper water. [Select one representative ortho per site from outputs/visualize-pred-ortho/by_ortho.pdf]

**Figure 3.** Ground truth versus predicted eelgrass area (m²) for 12 test orthomosaics, coloured by region. The dashed line indicates 1:1 agreement. R² = 0.88. [outputs/eval-plots/area_scatter.pdf]

**Figure 4.** Area estimation error (m²) by orthomosaic, coloured by region. Negative values indicate underestimation. [outputs/eval-plots/area_error_by_ortho.pdf]

---

## 4. Discussion (OUTLINE - ~600-800 words)

**Performance and comparison:**
- Final model IoU (0.72) and F1 (0.84) are competitive with prior seagrass segmentation studies (Tallam et al., 2023 reported F1 0.81 at a single site; Jeon et al., 2021 reported higher IoU but used random tile splits which inflate metrics)
- Compared to the companion kelp model (Reshitnyk et al., IoU 0.82 for presence/absence), lower performance is expected given that eelgrass is submerged and signal is attenuated by the water column
- Critically, our metrics are evaluated under site-level splitting across 30 sites and 3 regions, whereas prior studies used random tile splits at 1-2 sites. Under comparable splitting, our results represent a more honest estimate of operational performance.

**Eelgrass density:**
- The model performs well on dense, continuous meadows (e.g., Bag Harbour, Triquet Bay) but struggles with sparse eelgrass where individual patches blend into substrate
- Beck was the worst-performing test site (-38.7% area underestimation), driven by large areas of very sparse eelgrass where the ground truth annotation itself is debatable. On inspection, the annotated extent at Beck includes regions where eelgrass presence is ambiguous in the imagery, suggesting the disagreement is partly a labelling issue rather than a pure model failure
- This highlights the limitations of binary (presence/absence) classification for eelgrass: sparse meadows occupy a continuum from clearly present to barely detectable, and a hard threshold between "eelgrass" and "not eelgrass" is inherently subjective at low densities
- A multiclass model distinguishing density classes (e.g., sparse, moderate, dense) would better represent this continuum and allow users to apply density thresholds appropriate to their monitoring objectives

**Depth and subtidal edge:**
- The underestimation bias (mean % error -8.9%) likely reflects conservative predictions at the subtidal edge, where eelgrass visibility decreases with depth and water column attenuation
- At many sites, the subtidal edge in the ground truth was confirmed using concurrently collected towed underwater video. The ground truth therefore extends beyond what is visually discernible in the RGB imagery alone, and the model is being evaluated against boundaries that are not fully recoverable from its input data
- This explains the consistent underestimation: the model clips predictions where the eelgrass signal fades in the imagery, even where the ground truth (informed by video) extends further into deeper water
- Incorporating bathymetric data (as an additional input channel or for post-hoc stratification of performance by depth) is a promising direction for improving subtidal edge predictions

**Regional generalization:**
- The leave-one-region-out CV addresses a practical question: what can users expect when applying the model to a new region?
- The IoU range of 0.53-0.69 across folds represents a realistic floor for out-of-sample performance at new regions
- North was most challenging (IoU 0.53): eelgrass characteristics in this region are least represented by the other two regions
- The gap between held-out test performance (IoU 0.72, where test sites share a region with training sites) and CV performance (IoU 0.53-0.69, where an entire region is unseen) confirms that within-region generalization is substantially easier than across-region generalization

**Area estimation for monitoring:**
- Despite pixel-level imprecision, the correlation between predicted and ground truth area (R² = 0.88, or 0.97 excluding Beck) suggests the model is useful for tracking site-level trends over time
- At Triquet Bay (3 visits, 2018-2024), predicted areas tracked ground truth consistently, supporting use in time-series monitoring
- However, reasonable area estimates do not necessarily imply good spatial predictions. At Sedgwick, per-site pixel metrics were poor (IoU 0.49, precision 0.67, recall 0.65), yet false positives and false negatives partially cancelled when summing pixels for area. For monitoring total eelgrass area this may be acceptable, but for mapping bed boundaries, such predictions would require substantial manual correction
- In current practice, practitioners use model predictions as a starting point and manually modify the predicted extent in GIS to produce final delineations. This hybrid workflow substantially reduces processing time compared to fully manual delineation, even when manual correction is required
- More broadly, the model can be viewed as a measurement tool rather than an end-to-end solution (Freeland-Haynes, 2026). Like any imperfect measurement instrument, its outputs contain systematic and site-specific biases that can be corrected through calibration. Where site-level area is the primary output (e.g., for monitoring temporal trends) and specific eelgrass polygons are not required, a calibration approach could produce bias-corrected area estimates with associated uncertainty. For example, classifying a stratified random sample of points across the predicted raster at each new site (following standard accuracy assessment protocols; e.g., Olofsson et al., 2014) would allow estimation of site-specific precision and false negative rates, which could then be used to correct the predicted area and provide confidence intervals. This approach accounts for distribution shift across sites (i.e., the fact that model performance varies with local conditions) and would be substantially less effort than full manual delineation while providing statistically valid area estimates. Additionally, sites that undergo manual correction of predictions (the current operational workflow) automatically produce calibration data that could be accumulated over time

**Limitations:**
- Some conditions (e.g., tannins, turbidity, glint) are only represented in training data and not directly evaluated in the test set
- Binary classification does not distinguish eelgrass density classes or intertidal vs subtidal extent
- No depth information is incorporated; performance likely degrades with increasing water depth
- South region is underrepresented (6 sites, all from 2018)
- Area estimation is based on 12 test orthomosaics; the Beck outlier demonstrates sensitivity to individual sites at this sample size

**Future directions:**
- Evaluate performance stratified by depth using available bathymetric data
- Develop multiclass models to distinguish density classes (sparse, moderate, dense) and submerged vs unsubmerged eelgrass
- Explore statistical calibration of site-level area estimates as an alternative to manual correction in GIS
- Systematic assessment of annotation uncertainty, particularly at subtidal edges

---

## 5. Data and Code Availability (OUTLINE)

- The eelgrass segmentation model is freely available via the habitat-mapper command-line tool: https://github.com/HakaiInstitute/habitat-mapper
- Training code: https://github.com/HakaiInstitute/hakai-ml-train
- The companion kelp canopy model (Reshitnyk et al.) is also available via habitat-mapper
- [TODO: dataset availability; discuss with Hakai/Parks Canada]

---

## Acknowledgements (OUTLINE)

- Luba Reshitnyk: eelgrass annotations, data gathering
- Taylor Denouden: developed the hakai-ml-train codebase and habitat-mapper deployment tool
- Parks Canada (Gwaii Haanas, Pacific Rim) and Hakai Institute for data collection and sharing
- CV4E 2026 workshop
- [TODO: funding sources]

---

## References (TO BE COMPILED)

**Seagrass ecology:**
- Waycott, M. et al. (2009). Accelerating loss of seagrasses across the globe threatens coastal ecosystems. *PNAS*, 106(30), 12377-12381.
- Orth, R.J. et al. (2006). A Global Crisis for Seagrass Ecosystems. *BioScience*, 56(12), 987-996.

**UAS seagrass monitoring:**
- Nahirnick, N.K. et al. (2019). Mapping with confidence; delineating seagrass habitats using Unoccupied Aerial Systems (UAS). *Remote Sensing in Ecology and Conservation*, 5(2), 121-135.
- Duffy, J.P. et al. (2018). Spatial assessment of intertidal seagrass meadows using optical imaging systems and a lightweight drone. *Estuarine, Coastal and Shelf Science*, 200, 169-180.

**Seagrass DL segmentation:**
- Jeon, B.-K. et al. (2021). Semantic segmentation of seagrass habitat from drone imagery based on deep learning: A comparative study. *Ecological Informatics*, 66, 101430.
- Tallam, K. et al. (2023). Application of Deep Learning for Classification of Intertidal Eelgrass from Drone-Acquired Imagery. *Remote Sensing*, 15(9), 2321.
- Tahara, S. et al. (2022). Species level mapping of a seagrass bed using an unmanned aerial vehicle and deep learning technique. *PeerJ*, 10, e14017.
- Hobley, B. et al. (2021). Semi-Supervised Segmentation for Coastal Monitoring Seagrass Using RPA Imagery. *Remote Sensing*, 13(9), 1741.
- [2025 SegFormer seagrass UAV paper] (Remote Sensing, 17(14), 2518) - Eye in the Sky for Sub-Tidal Seagrass Mapping. [TODO: confirm full citation]
- [2025 DL seagrass survey] (Ocean Science Journal) - A Survey of Deep Learning Approaches for the Monitoring and Classification of Seagrass. [TODO: confirm full citation]

**Kelp/marine habitat DL:**
- Reshitnyk, L.Y. et al. (in preparation). Deep learning CNNs automate canopy detection of habitat-forming kelps from high resolution RGB imagery. [TODO: confirm publication status]

**DL architectures and methods:**
- Xie, E. et al. (2021). SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. *NeurIPS*, 34.
- Zhou, Z. et al. (2018). UNet++: A Nested U-Net Architecture for Medical Image Segmentation. *DLMIA/MICCAI Workshop*, LNCS 11045, 3-11.
- Berman, M. et al. (2018). The Lovasz-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks. *CVPR*, 4413-4421.
- Smith, L.N. & Topin, N. (2019). Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates. *arXiv:1708.07120*.

**DL for remote sensing (reviews):**
- Kattenborn, T. et al. (2021). Review on Convolutional Neural Networks (CNN) in vegetation remote sensing. *ISPRS J. Photogrammetry and Remote Sensing*, 173, 24-49.
- Osco, L.P. et al. (2021). A review on deep learning in UAV remote sensing. *Int. J. Applied Earth Observation and Geoinformation*, 102, 102456.

**Spatial cross-validation:**
- Roberts, D.R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913-929.
- Ploton, P. et al. (2020). Spatial validation reveals poor predictive performance of large-scale ecological mapping models. *Nature Communications*, 11, 4540.

**Area estimation and accuracy assessment:**
- Olofsson, P. et al. (2014). Good practices for estimating area and assessing accuracy of land change. *Remote Sensing of Environment*, 148, 42-57.
- Freeland-Haynes, L. (2026). What do I do with my (imperfect) model? CV4Ecology Workshop lecture.

**Software:**
- Iakubovskii, P. (2019). Segmentation Models Pytorch. https://github.com/qubvel/segmentation_models.pytorch
- Falcon, W. et al. (2019). PyTorch Lightning. https://github.com/Lightning-AI/pytorch-lightning

---

## Supplementary Material

**Table S1.** Architecture experiment results (validation set, prototype data).

| Model | Tile size | IoU | Precision | Recall | F1 |
|-------|-----------|-----|-----------|--------|----|
| UNet++ (ResNet-34) | 512 | 0.508 | 0.794 | 0.585 | 0.674 |
| UNet++ (ResNet-34) | 1024 | 0.541 | 0.854 | 0.596 | 0.702 |
| SegFormer (MiT-B2) | 512 | 0.586 | 0.846 | 0.656 | 0.739 |
| SegFormer (MiT-B2) | 1024 | 0.620 | 0.849 | 0.697 | 0.765 |

**Table S2.** Augmentation experiment results (validation set, prototype data, SegFormer MiT-B2 1024).

| Augmentation tier | Key additions | IoU | Precision | Recall | F1 |
|-------------------|--------------|-----|-----------|--------|----|
| Baseline | Geometric only | 0.606 | 0.875 | 0.664 | 0.755 |
| Default | Contrast, brightness, noise, blur | 0.638 | 0.867 | 0.707 | 0.779 |
| Scale | Resolution variation | 0.651 | 0.864 | 0.725 | 0.789 |
| Domain | Water-column effects (colour, turbidity, refraction) | 0.668 | 0.868 | 0.744 | 0.801 |

**Table S3.** Per-site pixel metrics for test sites (final model).

| Site | Region | Chips | IoU | Precision | Recall | F1 |
|------|--------|-------|-----|-----------|--------|----|
| Triquet Bay | Central | 468 | 0.834 | 0.942 | 0.879 | 0.909 |
| Bag Harbour | North | 395 | 0.761 | 0.857 | 0.871 | 0.864 |
| Section Cove | North | 196 | 0.659 | 0.913 | 0.703 | 0.795 |
| McMullin North | Central | 496 | 0.581 | 0.747 | 0.722 | 0.735 |
| Beck | South | 74 | 0.518 | 0.898 | 0.550 | 0.682 |
| Sedgwick | North | 132 | 0.494 | 0.674 | 0.649 | 0.661 |

Note: Beck has high precision (0.90) but low recall (0.55), confirming the model misses sparse eelgrass rather than hallucinating it. Sedgwick has the lowest precision (0.67), consistent with false positives from cloud reflections.

**Table S4.** Full survey metadata for 62 orthomosaics. [TODO: export from metadata_subset.csv]
