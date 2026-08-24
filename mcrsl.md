Data Descriptor

## MCR-SL: A Multimodal, Context-Rich Skin Lesion Dataset for Skin Cancer Diagnosis

Maria Castro-Fernandez 1,* , Thomas Roger Schopf 2, Irene Castaño-Gonzalez 3 , Belinda Roque-Quintana 3 , [URL 🔗](https://orcid.org/0000-0002-8188-7066)

Herbert Kirchesch 4, Samuel Ortega 1,5,6 , Himar Fabelo 1,7,8 , Fred Godtliebsen 6 and Gustavo M. Callico 1,* [URL 🔗](https://orcid.org/0000-0002-9794-490X)

- 1 Research Institute for Applied Microelectronics (IUMA), Universidad de Las Palmas de Gran Canaria, 35001 Las Palmas de Gran Canaria, Spain

- 2 Norwegian Center for E-Health Research, University Hospital of North-Norway, 9038 Tromsø, Norway

- 3 Department of Dermatology, Hospital Universitario de Gran Canaria Dr. Negrín, Barranco de la Ballena s/n, 35010 Las Palmas de Gran Canaria, Spain

- 4 Dermatology Private Office, 51147 Cologne, Germany

- 5 Norwegian Institute of Food, Fisheries and Aquaculture Research (Nofima), 9291 Tromsø, Norway

- 6 Department of Mathematics and Statistics, UiT The Arctic University of Norway, 9037 Tromsø, Norway

- 7 Fundación Canaria Instituto de Investigación Sanitaria de Canarias (FIISC), 35019 Las Palmas de Gran Canaria, Spain

- 8 Research Unit, Hospital Universitario de Gran Canaria Dr. Negrín, 35019 Las Palmas de Gran Canaria, Spain

- \* Correspondence: mcastro@iuma.ulpgc.es (M.C.-F.); gustavo@iuma.ulpgc.es (G.M.C.)

, Conceição Granja [URL 🔗](https://orcid.org/0000-0001-7896-8634)

2

## Abstract

Well-annotated datasets are fundamental for developing robust artificial intelligence mod- els, particularly in medical fields. Many existing skin lesion datasets have limitations in image diversity (including only clinical or dermoscopic images) or metadata, which hinder their utility for mimicking real-world clinical practice. The purpose of the MCR-SL dataset is to introduce a new, meticulously curated dataset that addresses these limita- tions. The MCR-SL dataset was collected from 60 subjects at the University Hospital of North Norway and comprises 779 clinical images and 1352 dermoscopic images of 240 unique lesions. The lesion types included are nevus, seborrheic keratosis, basal cell carcinoma, actinic keratosis, atypical nevus, melanoma, squamous cell carcinoma, angioma, and dermatofibroma. Labels were established by combining the consensus of a panel of four dermatologists with histopathology reports for the 29 excised lesions, with the latter serving as the gold standard. The resulting dataset provides a comprehensive resource with clinical and dermoscopic images and rich clinical context, ensuring a high level of clinical relevance, surpassing many existing resources in that matter. The MCR-SL dataset provides a holistic and reliable foundation for validating artificial intelligence models, enabling a more nuanced and clinically relevant approach to automated skin lesion diagnosis that mirrors real-world clinical practice.

Dataset: The data presented in this study are openly available in Zenodo at https://zenodo. [URL 🔗](https://zenodo.org/records/17306338)

[org/records/17306338](https://zenodo.org/records/17306338)

Dataset License: CC-BY

Keywords: skin cancer; multimodal; clinical data; dermoscopy; machine learning

Academic Editor: Kesheng (John)Wu

Received: 4 September 2025

Revised: 10 October 2025

Accepted: 15 October 2025

Published: 18 October 2025

Citation:

Castro-Fernandez, M.;

Schopf, T.R.; Castaño-Gonzalez, I.; Roque-Quintana, B.; Kirchesch, H.; Ortega, S.; Fabelo, H.; Godtliebsen, F.; Granja, C.; Callico, G.M. MCR-SL:A Multimodal, Context-Rich Skin Lesion Dataset for Skin Cancer Diagnosis. Data 2025, 10, 166. https://doi.org/ 10.3390/data10100166 [URL 🔗](https://doi.org/10.3390/data10100166)

(accessed on 16 October 2025).

Copyright: © 2025 by the authors. Licensee MDPI, Basel, Switzerland. This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license (https://creativecommons.org/ licenses/by/4.0/). [URL 🔗](https://creativecommons.org/licenses/by/4.0/)


## 1. Summary

Skin cancer is one of the most prevalent types of cancer worldwide, and it is expected

to continue growing in the fair-skinned population in 2050 [1]. It is diagnosed by dermatol- ogists, who evaluate the look of the lesion and other factors such as the subject’s risk factors (e.g., family history of skin cancer) or any associated symptoms (e.g., itching, bleeding, or pain) of the lesion. There are two image modalities typically used to depict skin lesions for diagnosis: clinical and dermoscopic images. The first shows the lesion as it appears to the naked eye. At the same time, the latter is captured using a dermoscope, which illuminates the skin with polarized or non-polarized light, removes surface reflections, and magnifies the lesion. An example of each modality is shown in Figure 1. [URL 🔗](#page-0)

*Figure 1. (a) A clinical and (b) a dermoscopic image of the same lesion (L0098).*

Artificial Intelligence (AI) models have been applied to skin lesion classification

for some years. The initial breakthrough came with Convolutional Neural Networks (CNNs), which demonstrated the ability to classify skin lesions just as well or better than human dermatologists on carefully curated datasets [2–5]. Commonly used architectures included Inception (v3 and v4), ResNet50, EfficientNet, and ensembles based on these models. However, when these unimodal models were tested with real-world, lower-quality images (such as those collected with a smartphone), their performance often dropped significantly [6]. To tackle these robustness and generalizability challenges, two areas of research emerged: (1) Developing more ‘dermatologist-like’ and multimodal approaches, which involved using attention mechanisms for explainability and combining image data with crucial clinical metadata about the patient and lesion, mirroring a human’s holistic assessment [7–9]. (2) Applying entirely new, more powerful architectures like the Vision Transformer (ViT), which leverage global context and attention mechanisms to process images, thereby promising greater robustness and accuracy than traditional CNNs [10,11]. [URL 🔗](#page-0)

The development of those models has been greatly promoted by the existence of

public datasets, such as the HAM10000, BCN20000, PH2, or PAD-UFES-20 dataset [12–15]. However, the ability of these models to accurately classify skin lesions and assist in clinical diagnosis is directly dependent on the richness and quality of the data used for their training and validation. While several datasets of skin lesions are publicly available, many present limitations in terms of image diversity, detailed metadata, or the methodology for establishing ground truth labels [15,16]. [URL 🔗](#page-0)


To address these limitations, this work introduces a new multimodal dataset of skin

lesions, collected and curated to provide a comprehensive resource for the scientific com- munity. The dataset includes clinical and dermoscopic images, as well as tabular metadata about the subjects, lesions, and diagnoses, covering clinical data, skin cancer risk factors, lesion characteristics (e.g., lesion diameter or body location), and diagnostic information. The Multimodal, Context-Rich Skin Lesion dataset (MCR-SL) is the result of a data acquisi- tion campaign carried out at the University Hospital of North Norway (UNN) and was initially created to serve as a controlled test dataset for the AI models developed within the European project WARIFA (Watching the Risk Factors, Grant Agreement: 101017385) [17]. The project targeted automatic skin cancer prevention and detection based on smartphone applications, which necessitated a dataset that reflects the challenging, non-ideal conditions inherent in data captured by the general public. While even curated datasets show sig- nificant variability in lighting and focus [18], real-life collected data presents even deeper variability in lighting, motion blur, and lack of focus. Therefore, the MCR-SL dataset was specifically curated to capture this diversity, making it an ideal resource for testing robust models intended to be used in challenging scenarios. [URL 🔗](#page-0)

The MCR-SL dataset comprises 2131 images documenting 240 skin lesions from

60 subjects. It includes a combination of 779 clinical images and 1352 dermoscopic images, covering diagnostic categories, including: nevus (NEV), seborrheic keratosis (SK), basal cell carcinoma (BCC), actinic keratosis (AK), atypical nevus (ATY), melanoma (MEL), squamous cell carcinoma (SCC), angioma (ANG), dermatofibroma (DF), and unknown (UNK). A central feature of this dataset is its approach to ground truth labeling. The diagnosis for each lesion was established in two ways: first, a panel of four dermatologists diagnosed every lesion; then, for those lesions that had been excised, the histopathology results served as the gold standard. A unified diagnosis combines both to serve as the gold standard. In addition to the images, the dataset includes extensive anonymized metadata: 9 attributes for lesions, 22 for subjects, and 16 attributes in total for diagnoses (encompassing dermatological, histopathological, and unified diagnoses). All data underwent a thorough curation process to ensure integrity and consistency, which included image standardization and the removal of all subject-identifying information to maintain privacy.

The MCR-SL dataset distinguishes itself from existing resources by combining key

strengths often found in isolation in other datasets, as shown in Table 1 (Note that in this table and the following ones “#” stands for “number of”). For instance, while datasets like PAD-UFES-20 [15] include detailed subject metadata, they consist solely of clinical images. Conversely, popular dermoscopic-only datasets such as PH2 [14] offer detailed dermoscopic criteria of the images, but lack the crucial context provided by clinical images and extensive metadata. The well-known HAM10000 dataset [12] provides a large number of images and robust ground truth, but its metadata is often limited to basic subject demographics, and its image modalities can vary across different challenges. The MCR-SL dataset combines both clinical and dermoscopic images with extensive subject and lesion metadata, which provides the critical context typically available to a clinician. This holistic structure aims to mirror real-world clinical practice, enabling a more nuanced approach to lesion diagnosis where experts consider a subject’s complete history, individual characteristics, and the nuanced details of the lesion itself. Furthermore, for those lesions that were excised (around 12% of the lesions in the dataset), our ground truth labeling combines the consensus of an expert panel of dermatologists for all the lesions with histopathology reports (when available), providing a robust and reliable label for each lesion. [URL 🔗](#page-0)


*Table 1. Comparison of characteristics for some public skin lesion datasets and the MCR-SL dataset.*

| Dataset | # | Classes | Image | Gold | Fields | Subject’s | Lesion | Diagnosis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Name | Images | Included | Modality | Standard | with IDs | Data | Data | Variables |
|   |   |   |   | Mixed |   |   |   |   |
| PH2 | 200 | NEV, MEL, | Dermoscopic | (Histology, | Image | - | Dermoscopic | - |
|   |   | ATY |   | Expert |   |   | criteria |   |
|   |   |   |   | Consensus) |   |   |   |   |
|   |   |   |   | Mixed |   |   |   |   |
|   |   | NEV, MEL, |   | (Histology, |   |   |   |   |
| BCN20000 | 10,015 | BCC, SK, AK, | Dermoscopic | Follow-up, | Lesion, Image | Age, sex | Body location | Verification |
|   |   | ANG, DF |   | Confocal, |   |   |   | Type (dx_type) |
|   |   |   |   | Expert |   |   |   |   |
|   |   |   |   | Consensus) |   |   |   |   |
|   |   | NEV, MEL, |   | Mixed |   |   |   |   |
| HAM10000 | 19,424 | BCC, SCC, SK, | Dermoscopic | (Histology, | Subject, Lesion, | Age, sex | Body location | - |
|   |   | AK, ANG, |   | Expert | Image |   |   |   |
|   |   | DF, other |   | Consensus) |   |   |   |   |
|   |   |   |   | Mixed (100% |   |   | Body location, |   |
|   |   | NEV, MEL, |   | Biopsy for | Subject, Lesion, | Age, sex, skin | lesion |   |
| PAD-UFES-20 | 2298 | BCC, SCC, | Clinical | cancers; Expert | Image | cancer risk | diameter, | - |
|   |   | SK, AK |   | Consensus for |   | factors, others | others |   |
|   |   |   |   | others) |   |   |   |   |
|   |   | NEV, SK, BCC, |   | Mixed |   | Age, sex, skin | Body location, | Certainty, |
| MCR-SL | 779; | AK,ATY, | Clinical, | (Histology, | Subject, Lesion, | cancer risk | lesion | image quality, |
|   | 1352 | MEL, ANG, | Dermoscopic | Expert | Image | factors, others | diameter, | time |
|   |   | DF, UNK |   | Consensus) |   |   | others |   |

The single-center, Scandinavian origin of the dataset is a limitation to generalizability,

as well as its limited size. However, its design incorporates several features to facilitate future expansion through multi-center or international collaboration. The relational struc- ture was chosen to ensure that new data entries can be seamlessly integrated. By using unique identifiers and standardized tables, the addition of new subjects, lesions, and im- ages is straightforward. Furthermore, the modular design accommodates the inclusion of new data collection points without requiring a redesign of the core database schema. For instance, a new table for data collection locations could be added to account for other clinics or hospitals. This forward-looking approach allows the dataset to grow over time and provides a flexible framework for potential collaborative, multi-center studies.

Nowadays, the field is heading towards the usage of multimodal models, like the one

proposed by Yan et al. [19], but the modalities under investigation are expanding rapidly. A variety of advanced non-invasive imaging and spectral modalities are currently being researched to improve skin cancer detection, sometimes in combination. Some of these techniques include Confocal Reflectance Microscopy (RCM), Optical Coherence Tomog- raphy (OCT), Laser Speckle Contrast Imaging (LSCI), Photoacoustic Imaging (PAI), and Raman Spectroscopy (RS). Among these, the development and application of Multi and Hyperspectral Imaging (MHSI) is a deeply researched area, with a strong body of work focusing on its unique ability to capture rich spectral signatures (providing molecular and chemical information) over a wide spatial area without the high cost or complexity associ- ated with modalities like OCT [20–23]. The growing trend emphasizes the complementary nature of these data sources, as evidenced by studies combining dermoscopic images with RCM [24], a “four-modal device” comprising OCT, photoacoustic tomography, ultrasound, and Raman spectroscopy developed for in vivo skin lesion assessment [25], or the use of LSCI, hyperspectral, and photoacoustic imaging for functional and molecular 3D mapping of tumors [26]. Furthermore, the integration of structural and spectral data, such as OCT with Raman Spectroscopy [27], highlights the shift toward models that can interpret a deep, feature-rich portrait of the lesion. This trajectory suggests that future multimodal models will in- creasingly incorporate rich spectral data, such asMHSI, alongside traditional clinical, dermoscopic, and patient metadata, to offer a truly holistic and non-invasive diagnostic assessment. [URL 🔗](#page-0)


## 2. Data Description

The MCR-SL dataset documents 240 unique skin lesions collected from 60 subjects.

It consists of 779 clinical images, 1352 dermoscopic images. Each lesion has a diagnosis from a panel of dermatologists, and for the 29 lesions that were excised, a histopathological diagnosis is also included. The dataset encompasses various types of skin lesions, including NEV, SK, BCC, AK, ATY, MEL, SCC, ANG, DF, and UNK.

Table 2 summarizes the distribution of lesion types, detailing the number of lesions [URL 🔗](#page-0)

and subjects associated with each specific type. The percentages for subjects and lesions are calculated with respect to the total number of subjects (60) and lesions (240) in the dataset, respectively. Note that the percentages for lesions sum to 100%, but the percentages for subjects do not, as some subjects may present with multiple types of skin lesion types. For example, a subject with both nevi and seborrheic keratoses is counted in both categories, which is why the sum of subjects across categories can exceed the total number of unique subjects (60). How the lesions were diagnosed is explained in Section 3.4. [URL 🔗](#page-0)

*Table 2. Lesion distribution by unified and histopathological diagnoses. The numbers in parentheses represent the proportion of subjects and lesions out of the total subjects and lesions in the dataset.*

|   |   | Diagnosed by | Diagnosed by |
| --- | --- | --- | --- |
| Lesion Type | Malignancy | Histopathology | Dermatologists |
|   |   | Subjects Lesions | Subjects Lesions |
| BCC |   | Malignant 18 (30.0%) 20 (8.3%) 18 (30.0%) | 26 (10.8%) |
| MEL | Malignant | 3 (5.0%) | 3 (1.3%) 7 (11.7%) 8 (3.3%) |
| SCC | Malignant | 0 (0.0%) 0 (0.0%) | 5 (8.3%) 5 (2.1%) |
| NEV | Non-Malignant | 3 (5.0%) | 3 (1.3%) 37 (61.7%) 85 (35.4%) |
| SK | Non-Malignant | 1 (1.7%) | 1 (0.4%) 34 (56.6%) 84 (35.0%) |
| AK | Non-Malignant | 0 (0.0%) | 0 (0.0%) 10 (16.7%) 12 (5.0%) |
| ATY | Non-Malignant | 2 (3.3%) | 2 (0.8%) 6 (10.0%) 7 (2.9%) |
| ANG | Non-Malignant | 0 (0.0%) 0 (0.0%) | 2 (3.3%) 4 (1.7%) |
| DF | Non-Malignant | 0 (0.0%) 0 (0.0%) | 2 (3.3%) 2 (0.8%) |
| UNK | - | 0 (0.0%) | 0 (0.0%) 6 (10.0%) 7 (2.9%) |
| Total |   | 27 (45.0%) 29 (12.1%) 60 (100.0%) 240 (100.0%) |   |

Beyond the formal lesion and histopathological diagnoses, the cohort’s phenotypical

characteristics were recorded. Notably, the subject’s Fitzpatrick types were not formally recorded. Regarding the Fitzpatrick Skin Type (FST), we acknowledge that this information was not formally assessed during the consultation. However, the metadata collected already includes the “subject’s skin reaction to sun exposure.” As confirmed by our clinical experts, the levels recorded in this variable (specifically those describing the reaction of ‘red and pain,’ ‘red,’ and ‘tanning without reddening’) align directly with the core criteria used to determine FST I, FST II, and FST III, respectively. Considering the Scandinavian setting where FST I–III are most common, this field can be accepted as a clinically acceptable surrogate for FST.

To better illustrate them, the association between subjects’ characteristics and lesion’s

characteristics with the lesion malignancy is explored in Tables 3 and 4, respectively. In both tables, the missing values have been managed to allow the analysis of these relationships. In the case of categorical data, they have been treated as “unknown”, while in the case of numerical data, they have been input as the mean of the attribute. Then, Chi-squared test p-values are included to indicate the strength of association between each attribute and lesion malignancy. In these tables, NM and M stand for Non-malignant and Malignant, respectively. [URL 🔗](#page-0)


*Table 3. Distribution of demographic and clinical attributes in the dataset (related to Lesions), including counts and percentages for each category, stratified by lesion malignancy.*

| Attribute |   | Values # % # NM %NM # M %M p-Value |   |   |   |
| --- | --- | --- | --- | --- | --- |
| [No Missing Values/Total] |   |   |   |   |   |
| 1.235–12.083 | 199 83% | 23 12% | 5 | 3% | 0.0030 |
| 12.083–22.867 Diameter | 34 14% | 17 50% | 0 | 0% | 0.0030 |
| 22.867–33.65 [238/240] | 2 1% | 0 0% | 0 | 0% | 0.0030 |
| 33.65–44.433 | 1 0% | 0 0% | 0 | 0% | 0.0030 |
| 55.217–66.0 | 1 0% | 1 100% | 0 | 0% | 0.0030 |
| Back | 99 41% | 11 11% | 3 | 3% | 0.0005 |
|   | Arms 45 19% | 2 4% | 1 | 2% | 0.0005 |
| Face Location group | 41 17% | 16 39% | 1 | 2% | 0.0005 |
| Torso [232/240] | 31 13% | 10 32% | 0 | 0% | 0.0005 |
| Legs | 12 5% | 2 17% | 1 | 8% | 0.0005 |
| Head | 4 2% | 1 25% | 0 | 0% | 0.0005 |
| unknown | 8 3% | 0 0% | 0 | 0% | 0.0005 |
| Lesion Lesion status when captured | 235 98% | 38 16% | 6 | 3% | 0.0047 |
| Biopsied lesion [240/240] | 5 2% | 4 80% | 0 | 0% | 0.0047 |
|   | Voluntary sample 197 82% | 16 8% | 6 | 3% | 0.0000 |
|   | BCC 25 10% | 22 100% | 0 | 0% | 0.0000 |
| SK Referral diagnosis | 7 3% | 0 0% | 0 | 0% | 0.0000 |
| [240/240] | MEL 5 2% | 3 60% | 0 | 0% | 0.0000 |
|   | NEV 5 2% | 0 0% | 0 | 0% | 0.0000 |
| Morbus Bowen | 1 0% | 1 100% | 0 | 0% | 0.0000 |
| carcinoma |   |   |   |   |   |
|   | Non-malignant 192 80% |   |   |   |   |
| Malignancy Malignant | 42 18% |   |   |   |   |
| unknown | 6 2% |   |   |   |   |

*Table 4. Distribution of demographic and clinical attributes in the dataset (related to Subjects), including counts and percentages for each category, stratified by lesion malignancy.*

| Attribute | Values |   |   | # % # NM %NM # M %M p-Value |   |
| --- | --- | --- | --- | --- | --- |
| [No Missing Values/Total] |   |   |   |   |   |
| Age | 14.9–40.7 | 8 13% | 0 | 0% | 8 100% |
| [59/60] | 40.7–66.3 | 23 38% |   | 13 57% | 10 43% 0.582 |
|   | 66.3–92.0 | 29 48% |   | 19 66% | 10 34% |
| Sex | Female | 33 55% |   | 12 36% | 21 64% 0.008 |
| [60/60] | Male | 27 45% |   | 20 74% | 7 26% |
| Height (cm) | 145.9–162.3 | 14 23% | 4 29% |   | 10 71% |
| [59/60] | 162.3–178.7 | 27 45% |   | 17 63% | 10 37% 0.053 |
|   | 178.7–195.0 | 19 32% |   | 11 58% | 8 42% |
| Weight (kg) | 38.9–66.0 | 19 32% | 6 32% |   | 13 68% |
| [59/60] | 66.0–93.0 | 32 53% |   | 20 62% | 13 41% 0.496 |
|   | 93.0–120.0 | 9 15% | 6 67% |   | 4 44% |
|   | Brown | 25 42% |   | 12 48% | 13 52% |
|   | Fair blonde | 19 32% |   | 10 53% | 9 47% |
| Natural hair color | Dark |   |   |   | 0.382 |
| (≤18 years old) | brown, | 12 20% |   | 9 75% | 3 25% |
| [60/60] | black |   |   |   |   |
|   | Red or | 3 5% |   | 1 33% | 2 67% |
|   | auburn |   |   |   |   |
|   | Blonde | 1 2% | 0 | 0% | 1 100% |


*Table 4. Cont.*

| Attribute Values | # % # NM %NM # M %M p-Value |   |   |
| --- | --- | --- | --- |
| [No Missing Values/Total] |   |   |   |
| Red | 29 48% | 16 55% | 13 45% |
| Brown |   |   | 0.844 |
| Skin reaction to without 1st |   |   |   |
| sun exposure becoming | 22 37% | 12 55% | 10 45% |
| [60/60] red |   |   |   |
| Red with | 9 15% | 4 44% | 5 56% |
| pain |   |   |   |
| Few Number of moles | 21 35% | 14 67% | 7 33% |
| Some (≤18 years old) | 18 30% | 5 28% | 13 72% 0.065 |
| Many [53/60] | 14 23% | 8 57% | 6 43% |
| Unknown | 7 12% | 5 71% | 2 29% |
| Yes Moles > 5 mm | 30 50% | 14 47% | 16 53% |
| No [55/60] | 25 42% | 16 64% | 9 36% 0.361 |
| Unknown | 5 8% | 2 40% | 3 60% |
| Moles > 20 cm No | 60 100% | 32 53% | 28 47% 1.000 |
| [60/60] |   |   |   |
| Some | 24 40% | 9 38% | 15 62% |
| Few Number of moles (now) | 22 37% | 15 68% | 7 32% 0.133 |
| Many [53/60] | 7 12% | 3 43% | 4 57% |
| Unknown | 7 12% | 5 71% | 2 29% |
| 0 | 28 47% | 14 50% | 14 50% |
| 1–2 Number of severe | 13 22% | 7 54% | 6 46% |
| 3–5 sunburns | 8 13% | 3 38% | 5 62% 0.617 |
| >5 [52/60] | 3 5% | 2 67% | 1 33% |
| Unknown | 8 13% | 6 75% | 2 25% |
| No Sunbed use | 54 90% | 29 54% | 25 46% |
| Yes [58/60] | 4 7% | 1 25% | 3 75% 0.218 |
| Unknown | 2 3% | 2 100% | 0 0% |
| No History of cancer | 39 65% | 17 44% | 22 56% 0.073 |
| Yes [60/60] | 21 35% | 15 71% | 6 29% |
| No History of skin cancer | 41 68% | 19 46% | 22 54% |
| Yes [56/60] | 15 25% | 9 60% | 6 40% 0.102 |
| Unknown | 4 7% | 4 100% | 0 0% |
| History of skin cancer No | 50 83% | 25 50% | 25 50% |
| (close relatives) |   |   | 0.418 |
| Yes [60/60] | 10 17% | 7 70% | 3 30% |
| No Organ transplant | 57 95% | 30 53% | 27 47% |
| Yes [59/60] | 2 3% | 2 100% | 0 0% 0.234 |
| Unknown | 1 2% | 0 0% | 1 100% |
| No Immunosuppression | 54 90% | 30 56% | 24 44% |
| Yes [59/60] | 5 8% | 2 40% | 3 60% 0.448 |
| Unknown | 1 2% | 0 0% | 1 100% |
| Plastic | 35 58% | 20 57% | 15 43% |
| surgery Patients derived from |   |   | 0.040 |
| Dermatology [60/60] | 17 28% | 11 65% | 6 35% |
| Volunteer | 8 13% | 1 12% | 7 88% |
| Subjects with known yes | 32 53% |   |   |
| malignant lesions no | 28 47% |   |   |

Regarding the images, they are stored in separate folders for clinical and dermoscopic

image modalities. Both types of images were cropped to standardized sizes, which are


detailed in the Methods section. It is important to note that many images of the same lesion are near duplicates, captured with slight variations in lighting, focus, or rotation.

The images are provided in PNG (.png) format and utilize the sRGB color space.

Each image is accompanied by extensive metadata detailing the lesions and subjects’ characteristics. The metadata is organized into multiple tables (provided as spreadsheets) designed to function as a relational database. The attributes and structure of these tables are further explained in Section 2.1. [URL 🔗](#page-0)

## 2.1. Dataset Structure

The dataset is composed of both images and contextual data tables, which together

provide a comprehensive record of skin lesions. The images are organized into two separate folders based on modality: dermoscopic and clinical. The contextual data is stored in several tables that contain clinical information about each lesion and the subjects. Each of these elements is an entity in our dataset, with the Lesion entity serving as the central element that connects all other data. This structure is further detailed in the Entity-Relationship Diagram shown in Figure 2. [URL 🔗](#page-0)

*Figure 2. Entity-Relationship Diagram: This diagram visually represents the relationships between the dataset’s core entities. Attributes are not included in the figure for the sake of simplicity.*

In it, each rectangle represents an entity of our database, and the diamonds represent

the relationship between the two entities connected to them. The numbers indicate the cardinality of the relationship, which specifies how many instances of one entity can be associated with instances of another. For instance, the relationship “subject-lesion (1,1):(1,M)” in the diagram shows that a single Subject can have multiple Lesions (1 to M), but each Lesion belongs to a single Subject. In contrast, the relationship “lesion-unified (1,1):(1,1)” in the diagram indicates that each Lesion is linked to a single Unified diagnosis, and each Unified diagnosis corresponds to a single Lesion. Also, the relationship “histopath- unified (0,1):(1,1)” between Histopathology diagnosis and Unified diagnosis entities shows that each Histopathology diagnosis is associated with a single Unified diagnosis; however, a Unified diagnosis may not be linked to any Histopathology diagnosis. Each entity and its relationships are further explained below.

## 2.1.1. Lesion Entity

The Lesion entity serves as the central entity of the dataset. Each entry is uniquely

identified by a lesion_id, which is tied to a specific subject. This entity contains a unified diagnosis for each lesion and is associated with multiple images (both clinical and der- moscopic). Additional attributes, such as the lesion’s diameter (diameter) or the referring physician’s diagnosis (referral_diagnosis), are also included. All attributes of the Lesion entity are described in detail in Table 5. [URL 🔗](#page-0)


*Table 5. Attributes of the Lesion entity.*

| Attribute | Data Type | Description |
| --- | --- | --- |
| lesion_id | string | A unique identifier for the lesion. |
| referral_diagnosis | text | The initial diagnosis provided |
|   |   | during the subject’s referral. |
| lesion_status_when_captured | categorical | The status of the lesion at the time |
|   |   | of imaging. |
| location | categorical | The anatomical location of the |
|   |   | lesion on the subject’s body. |
| location_group | categorical | A broader classification of the |
|   |   | lesion’s location. |
| diameter | numerical | The measured diameter of the |
|   |   | lesion in millimeters. |
| malignancy | categorical | The malignancy status of the lesion |
|   |   | (i.e., malignant, non-malignant). |
| lesion_diagnosis | text | The unified diagnosis assigned to |
|   |   | the lesion. |
|   |   | The unique identifier of the specific |
| diagnosis_image_id | string | image used by the dermatologists |
|   |   | to make their diagnoses. |

Note that a few lesions were included even if they had only one image modality, as

they were accompanied by other lesions from the same subject for which bothmodalities were available. This allowed for a more complete dataset and a holistic analysis of each subject.

## 2.1.2. Subject Entity

The Subjects table contains extensive, anonymized clinical data and risk factors, ob-

tained through a questionnaire filled in by the subject. Each entry is uniquely identified by its subject_id. The specific attributes of this table are detailed in Table 6. [URL 🔗](#page-0)

*Table 6. Attributes of the Subject entity.*

| Attribute | Data Type | Description |
| --- | --- | --- |
| subject_id | string | A unique identifier for the subject. |
| derived_from | categorical | The hospital’s department that derived the subject. |
| year_of_birth | integer | The subject’s year of birth. |
| age | integer | The subject’s age. |
| sex | categorical | The subject’s sex. |
| height | numerical | Subject height in centimeters. |
| weight | numerical | Subject weight in kilograms. |
| natural_hair_color | categorical | The subject’s natural hair color at 18 years old. |
| skin_reaction_to_sun | categorical | How the subject’s skin reacts to sun exposure without sun protection. |
| number_of_moles | integer | The total number of moles on the subject at 18 years old. |
| moles_bigger_5mm | integer | Current number of moles larger than 5mm. |
| moles_bigger_20cm | integer | Current number of moles larger than 20cm. |
| moles_body | integer | Current number of moles on the body. |
| sunburn_number | integer | The number of severe sunburns the subject has experienced. |
| sunburn_age | text | The age at which the subject experienced severe sunburns. |
| sunburn_number_group | categorical | A categorized group for the number of sunburns. |
| sunbed | boolean | Whether the subject has used a sunbed. |
| h_cancer | boolean | History of hereditary cancer. |
| h_skin_cancer | boolean | History of hereditary skin cancer. |
| h_skin_cancer_relatives | boolean | History of skin cancer in close relatives. |
| organ_transplant | boolean | Whether the subject has had an organ transplant. |
| immunosuppresion | boolean | Whether the subject is on immunosuppressive medication. |


## 2.1.3. Image Entities

The Images entity stores metadata for all the acquired images. Each entry is uniquely

identified by its image_id and is linked to a specific lesion via the lesion_id. It also includes the modality attribute, which specifies the type of image (e.g., clinical or dermoscopic). The attributes of the Images entity are detailed in Table 7, which serves as the key link connecting the image files to the rich contextual information of lesions and subjects stored in the dataset. [URL 🔗](#page-0)

*Table 7. Attributes of the Image entity.*

| Attribute Data Type Description |   |   |
| --- | --- | --- |
| image_id | string | A unique identifier for each image. |
| lesion_id | string | A unique identifier for the lesion depicted in the image. |
| modality | categorical The modality of the image (clinical or dermoscopic). |   |

- 2.1.4. Diagnostic Entities: Dermatology, Histopathology, and Unified Diagnosis The dataset contains three distinct types of diagnosis:

- (1) Dermatology Diagnosis: A diagnosis provided by a panel of dermatologists assigned to each lesion.

- (2) Histopathology Diagnosis: A diagnosis derived from histopathology reports, available for a subset of 29 excised lesions (out of 240). This report also contains tumor thickness information when applicable.

- (3) Unified Diagnosis: The definitive label for this dataset, derived by synthesizing the dermatology and histopathology diagnoses. The methodology for generating this label is detailed in the Methods section.

The attributes of the Dermatology, Histopathology, and Unified diagnosis entities are

detailed in Table 8, Table 9, and Table 10, respectively. The Dermatology entity contains individual diagnoses from each expert, whereas the Unified entity holds the definitive final diagnosis for each lesion. With both expert and histopathology diagnoses available, two analyses can be performed: first, to calculate the diagnostic accuracy of the dermatologists for the 29 excised lesions; and second, to analyze the interobserver variability among the experts. The diagnostic accuracy derived from this subset of histologically proven lesions will be used to extrapolate the dermatologists’ expected performance on the larger set of non-confirmed lesions. [URL 🔗](#page-0)

*Table 8. Attributes of the Dermatology diagnosis entity.*

| Attribute Data Type Description |   |   |
| --- | --- | --- |
| diagnosis_id | string | A unique identifier for each diagnosis. |
| lesion_id | string | The identifier of the lesion the diagnosis refers to. |
| image_id | string | The identifier of the image that was diagnosed. |
| expert_id | string | The identifier of the dermatologist who provided the diagnosis. |
| diagnosis | string | The primary diagnosis provided by the expert (e.g., NEV, MEL). |
| 2nd_option | string | An optional second choice or differential diagnosis. |
| certainty | categorical | A numerical rating of the expert’s confidence in their diagnosis. |
|   |   | Potential values are 0%, 25%, 50%, 75%, and 100%. |
| image_rating | integer | The expert’s rating of the image quality, ranging from 1 to 10. |
| time | datetime The time taken by the expert to provide the diagnosis. |   |


*Table 9. Attributes of the Histopathology diagnosis entity.*

| Attribute | Data Type | Description |
| --- | --- | --- |
| diagnosis_id | string | A unique identifier for each histopathology diagnosis. |
| lesion_id | string | The identifier of the lesion the diagnosis refers to. |
| procedure | string | The type of procedure described in the report (e.g., biopsy, excision). |
| tumor_thickness | float | The Breslow thickness of the tumor, if applicable. |
| diagnosis | string | The final diagnosis from the histopathology report (e.g., NEV, MEL). |

*Table 10. Attributes of the Unified diagnosis entity.*

| Attribute | Data Type | Description |
| --- | --- | --- |
| diagnosis_id | string | A unique identifier for the unified diagnosis. |
| lesion_id | string | The identifier of the lesion the diagnosis refers to. |
| dermatology_diagnosis | string | The final diagnosis selected by the dermatology experts. |
| histopathology_diagnosis | string | The diagnosis from the histopathology report, used as the ground |
|   |   | truth when available. |
| diagnosis_id_histopath | string | The unique identifier of the histopathological diagnosis of the lesion. |
| unified_diagnosis | string | The final ground truth diagnosis for the lesion. |

## 3. Methods

## 3.1. Ethics Declaration

This study was conducted in accordance with the Declaration of Helsinki. The dataset

was obtained in partnership with the dermatology and plastic surgery departments at UNN. The data collection campaign received approval from the Regional Committee for Medical and Health Research Ethics (North) (Ref.: 392439).

## 3.2. Participants and Selection Criteria

Eligibility criteria include subjects with skin lesions belonging to one of the following

types (previously introduced in Table 2): NEV, ATY, SK, AK, ANG, DF, BCC, SCC, or MEL. These skin lesion types have been selected to aid in the development of AI-based algorithms to learn the key differences among benign, malignant, and premalignant lesions. In order to make comparisons between lesions, at least two lesions (of any type) per subject were captured. This protocol allows future users to explore and account for intra-subject variability whenever two lesions of the same type are available for a given subject. The importance of capturing this variability is underscored by studies such as Rotemberg et al. [28], whose work demonstrated that models evaluated accounting for this variability improve their performance. The MCR-SL dataset’s design, which provides the necessary Subject IDs and Lesion IDs for creating subject-disjoint validation splits, directly enables realistic model evaluation without introducing patient-level data leakage. Given resources and time constraints during the collection phase, this strategy was also an efficient way to expand the dataset size while maintaining high data quality. [URL 🔗](#page-0)

Note that originally, only melanoma, BCC, or SCC were considered as eligible skin

cancer lesions, but the inclusion criteria were extended during the data collection to decrease the probability of missing a relevant lesion for the study. The methodology for consolidating these diagnoses and establishing ground truth is detailed in Section 3.4. [URL 🔗](#page-0)

Participants include patients and volunteers from the dermatology and plastic surgery

departments at UNN. In this context, patients are individuals who sought medical care, while volunteers are individuals recruited specifically for the study who did not seek treatment. Both departments were asked to collaborate to increase the potential number of


subjects and lesions in the final dataset, finding patients with at least one lesion fitting the eligibility criteria.

The recruitment process differed between the departments. In the case of dermatology,

dermatologists derived every subject after screening them. For plastic surgery, patients were scheduled to participate 45 min before their surgery. This approach prioritized subject convenience, as other methods would have required additional hospital appointments.

Additionally, a few subjects volunteered to participate to increase the number of benign

lesions collected during the campaign. Images of as many lesions as possible were taken from all participants, including those the subjects were referred to, and any incidental findings.

## 3.3. Data Acquisition Workflow

Data collection and diagnosis were performed as three distinct steps, separately. First,

a questionnaire is used to gather information about the subject’s demographic profile (e.g., age, sex) and skin cancer risk factors, while images of their lesions are acquired. Then, the histopathology reports for the excised lesions were collected through the plastic surgery department personnel. Finally, once the image collection ended, a panel of dermatologists was asked to diagnose one image per lesion. The image provided for each lesion was a randomly selected dermoscopic image, except for a few cases where none was available, in which case a clinical image was used instead. The workflow is illustrated in Figure 3. The consolidation of the diagnoses from dermatology and histopathology is explained in Section 3.4. [URL 🔗](#page-0)

*Figure 3. Workflow of the data acquisition process. (a) The subject signs the informed consent form; (b) Collecting clinical data via a questionnaire completed by the subject; (c) Clinical and dermoscopic images are acquired using a smartphone and a portable dermoscope; (d) Diameter of the skin lesion is measured with a caliper; (e) After the subject interview, data is verified and transferred from the smartphone to be stored in a secure, encrypted system; (f) Histopathology diagnosis are obtained for the excised lesions; (g) Dermatology diagnoses are collected from a panel of four expert dermatologists (E1–E4). Note that the dataset labels (final diagnoses) are derived from the combination of both diagnoses (f,g).*

Image and subject metadata were collected on-site at UNN. Data collection was carried

out by a researcher familiar with the appearance of clinical and dermoscopic images, though without formal training in their acquisition. Room illumination corresponded to a standard clinical setup. Clinical images were generally collected as close-up views of the lesions, although in a few cases focusing issues prevented the acquisition of optimal images. The equipment used for image collection consisted of a Xiaomi Redmi 9A smartphone (Xiaomi


Corp., Beijing, China), equipped with a 13 MP rear camera (f/2.2 aperture, 28 mm focal length), and a DermLite HÜD 2 portable dermatoscope (3Gen Inc., San Juan Capistrano, CA, USA) providing × 10 magnification under polarized light. The dermoscope, with an outer diameter of approximately 59.2 mm and a lens diameter of 12.5 mm, was attached directly to the smartphone for dermoscopic acquisitions. Clinical images were captured at close range using ambient room illumination without flash.

The equipment was selected based on the specific goals of the WARIFA project. This

decision was driven by two primary rationales: (1) alignment with deployment reality and (2) robustness testing. We determined that a mid-class smartphone reflects the typical consumer device and image quality that the deployed AI models will encounter. We acknowledge that higher-end imaging setups could have provided better image quality; however, our choice prioritized accessibility and realism over optimal resolution. The variability, resolution limits, and subtle artifacts introduced by a consumer-grade device (unlike those from a highly controlled clinical camera) create a realistic stress test necessary to evaluate the robustness and generalizability of AI models.

Assuming two lesions per subject, the total time required for carrying out the whole

process was around 15–20 min. The steps followed are outlined below:

- 1. Informed consent: When the subject arrives in the room, they are informed about the study. Then, the subject is given the informed consent form to read and sign if they are willing to participate in the study (Figure 3a). Estimated time: 5 min.

- 2. Clinical data collection: If the informed consent form is signed, the subject is asked to fill out a questionnaire in situ, so the data collector can clarify any questions the subject may have if needed (Figure 3b). Estimated time: 10 min.

- 3. Clinical and dermoscopic image acquisition: A smartphone-based digital camera is used by the data collector for capturing the images with and without the dermoscope attached to the device (Figure 3c). Estimated time: 30 s per lesion.

- 4. Diameter measurement of the skin lesion: The lesion is measured by the data collector with a caliper gauge (Figure 3d). Estimated time: 20 s per lesion.

- 5. Data Storage: All acquired data are verified and stored in a secure, encrypted storage system (Figure 3e). Estimated time: 5 min per lesion.

Our strategy involved capturing multiple images of the same lesion (near duplicates).

Even in curated datasets, there is significant variability in lightning and focus [18]. Our images are meant to reflect the variability caused by real-world data imperfections, which we assume have deeper variability in lighting, motion blur, and lack of focus. To capture their differences, a mixed approach was utilized: [URL 🔗](#page-0)

- 1. Real-World Baseline: Images were first acquired using the default automatic settings for all parameters. This captures the natural, heterogeneous noise expected from the average user of the WARIFA application.

- 2. User Manipulation Scenario: Subsequent images of the same lesion were taken by deliberately adjusting settings such as brightness/exposure and focus. Crucially, this adjustment was performed using the typical user interface (e.g., tap-to-focus or brightness sliders) without setting specific technical values for ISO or exposure time.

This combined approach allows the MCR-SL dataset to simultaneously model two

critical challenges: the unpredictable variability of automatic settings and the effects of non-expert user adjustments. Rather than introducing an undesirable bias, this variability is a key feature that allows researchers to test and evaluate whether machine learning models are truly robust against changes in a user’s environment, device settings, and active manipulation. This strengthens model generalizability for clinical application, which is the primary goal of this dataset.


Regarding the subjects, note that they were guided through the questionnaire, but

their answers were entirely their own. Participation was encouraged, but not mandatory, which resulted in missing values for some cases. The questionnaire was written in Nor- wegian to make it easier for subjects to understand; an English version is available in the Supplementary Material.

Special consideration was given to two subjects who were underage at the time of

data collection. In these cases, the consent form was signed by their legal guardians. Furthermore, questions asking about characteristics at 18 years old were answered with their current status. Consequently, the answers for the current number of moles and the number of moles at 18 years old are the same for these subjects.

As mentioned, the diagnoses of the excised lesions were obtained from their

histopathology reports (Figure 3f). To do that, all the recent reports for a given patient were collected, sorted, and translated into English to extract the relevant information: procedure, diagnosis, and tumor thickness (when applicable). [URL 🔗](#page-0)

Once the data collection was completed, the dermatology diagnoses of the acquired

images were performed remotely by a panel of four dermatologists (Figure 3g) who had high or very high levels of expertise (self-reported). They used a customized software interface developed for this purpose (Figure 4). [URL 🔗](#page-0)

*Figure 4. View of the software interface showing a list of potential diagnoses and certain fields for quantifying the certainty level of the diagnosis, the image quality, and for adding a comment about the image or the lesion.*

Using this software interface, the dermatologists could examine the image shown

and provide a diagnosis for the lesion, alongside other variables. For each image, the dermatologists were asked to specify their certainty of diagnosis on a scale of 0%, 25%, 50%, 75%, and 100% (A 100% rating meant they were completely sure about their answer, while a 0% rating meant the opposite). They also provided a rating of image quality on a scale from 1 to 10, with 1 being the lowest and 10 the highest quality, and the software


automatically recorded the time spent on each image. The dermatologists were encouraged to add comments about the lesion and were given the option to specify a lesion type if it was not available in the predefined class options. Finally, if they were unsure about their primary diagnosis, they were asked to provide a second diagnostic option.

## 3.4. Diagnosis Consolidation and Ground Truth Determination

The lesions were diagnosed in two ways: by dermatologists and, when possible, by

histopathology.

First, the software interface and corresponding images were sent to a panel of four

expert dermatologists, who independently provided a diagnosis, among other variables. Using only one image per lesion to diagnose, the dermatology diagnosis was determined by taking the most frequent diagnosis from the four experts (i.e., majority voting). If no experts provided the same diagnosis, the diagnosis for that lesion was labeled as UNK. This happened with six lesions (around 2.5% of the total).

A tie-breaker criterion was implemented for the cases where two labels were proposed

(Figure 5), based on the collective accuracy of the experts who proposed each label. This occurred for 26 out of 240 lesions (around 10.8% of the total). The experts were ranked according to their individual accuracy against the histopathology diagnoses available within the same dataset. For each tied diagnosis, the average accuracy of the proposing experts was calculated, and the diagnosis associated with the pair of experts with the highest average accuracy was selected as the final dermatology diagnosis. This approach ensured that the decision reflected the combined expertise of the agreeing parties rather than relying on a single individual. Specific accuracy values are reported in Figure 5, and additional details can be found in the dataset. [URL 🔗](#page-0)

*Figure 5. Tie-breaker criterion for the dermatology diagnosis, explained with an example.*

Histopathology is widely considered the definitive diagnosis because it provides a

final, conclusive diagnosis based on the microscopic examination of tissue, rather than on visual patterns alone. However, it is well-documented that it is susceptible to interobserver variability [29]. Similarly, clinical and dermoscopic diagnoses are also subject to variability, as shown by studies on expert consensus [4], but a diagnosis reached by a consensus of expert dermatologists can still serve as a highly reliable benchmark [30]. [URL 🔗](#page-0)

The reality of public machine learning datasets is that the primary diagnostic label

often comes from different sources (histopathology, single image consensus, follow-up, or confirmation by in vivo confocal microscopy). That is the case for both BCN20000 and HAM10000, the latter with a 50% of confirmed by histopathology lesions [12,13]. The PAD-UFES-20 dataset relies primarily on histopathology. However, in the case of [URL 🔗](#page-0)


the ISIC2020 [28], widely known and used as well, the percentage lowers to 14% for the training set and 7% for the test set [31]. [URL 🔗](#page-0)

In our case, we established a single, definitive ground truth for each lesion by combin-

ing the diagnosis from histopathology and dermatology (multi-expert panel consensus), which minimizes the risk of individual error. To create the unified diagnosis for our dataset, we adopted a hierarchical approach to diagnosis. For all lesions that were excised, the histopathology diagnosis was prioritized and used as the definitive gold standard (Figure 6). For the remaining lesions, which were not excised, the consensus diagnosis from the expert panel served as the ground truth. This systematic integration of expert consensus and histopathology provides a more robust and less ambiguous label for each image in the dataset than a diagnosis from a single expert. [URL 🔗](#page-0)

*Figure 6. Unified diagnosis: if the lesion has a diagnosis from histopathology, that is the gold standard. If not, then it is the diagnosis given by the panel of dermatologists.*

Even though our diagnosis label approach is consistent with community standards,

we acknowledge that one limitation of the MCR-SL dataset is the low number of confirmed cases by histopathology (29 out of 240, around 12%), which means that the majority of the labels rely on the panel consensus accuracy. Older studies have demonstrated that single dermatologists typically achieve high accuracy (e.g., >70% nearly two decades ago [32]), and modern research confirms that a majority vote consensus achieves a balanced improvement in both sensitivity and specificity [33], compared to individual experts’ performance. For these reasons, we believe that the multi-expert consensus, documented with the diagnosis-related variables, provides a reliable clinical ground truth relevant to testing diagnostic models. [URL 🔗](#page-0)

## 3.5. Data Curation and Validation

To ensure the integrity and consistency of the dataset, several steps were taken during

data collection and post-processing (Figure 7). [URL 🔗](#page-0)

First, to protect subject privacy and comply with ethical guidelines, all identifying

information, such as names, birth dates, and exact data collection dates were removed. A relational-like database was established, allowing unique identifiers to be assigned to each subject, lesion, and image immediately after collection. These codes serve to identify and link data records while ensuring anonymity. Any free-text fields containing potentially identifying information were also removed to complete the anonymization process.

In the following subsections, the data curation process for the images, metadata, and,

specifically, the experts’ feedback is presented.


*Figure 7. Data curation and validation workflow. The flowchart details the four main stages of data preparation. Data collection involves acquiring images and associated clinical metadata. The data then undergoes Anonymization, where identifying information is removed (✕) and replaced with unique subject and lesion IDs (✓). The subsequent stage is Data Curation, which comprises two parallel processes: Image Curation & Standardization and Metadata Validation. The former involves rejecting unsuitable images (thumbs down symbol) and standardizing accepted images by cropping them (indicated by the red frame) to isolate the lesion. Simultaneously, metadata validation focuses on consistency checks, standardizing categorical data, and handling missing values in the clinical data. Finally, all refined data feeds into Diagnosis Consolidation for the final expert assessment.*

## 3.5.1. Image Standardization and Curation

To homogenize the image data, most camera settings were standardized. While

automatic white balancing and exposure were used by default to capture the first image of all the lesions, another image was also captured with manually adjusted brightness for comparison (a near duplicate of the first one). This process ensured consistent image capture across the dataset.

Following data collection, a visual inspection and curation process was performed.

Images of poor, unrecoverable quality (e.g., highly unfocused or improperly framed images) were removed. However, we intentionally retained images of moderate to lower quality (e.g., slight blurriness or sub-optimal lighting) to allow for comparisons between higher and lower quality images and to study their effect on other variables, such as diagnosis accuracy.

Note that since a formal, automated quality analysis was performed, we do not provide

a precise image count for a predefined “low quality” threshold, nor its exact numerical distribution across lesion types. Defining that threshold could be interesting work to do in the future. However, we also believe that the most scientifically robust measure is the one provided by clinical experts themselves. We have provided the necessary data for researchers to conduct an inter-observer analysis on diagnosis-related variables, including the dermatologists’ image quality ratings and diagnostic certainty scores. This complete quality metadata also enables users to conduct the requested quantitative analysis, which remains a potential research line for users of the MCR-SL dataset.

After the visual inspection, the class imbalance is significant and represents an ir-

refutable limitation for standard skin lesion image classification tasks. However, we believe the decision to retain all images, even those with limited representation, is essential because


the dataset’s value is derived from its multimodality and rich contextual metadata, not solely from the size of the lesion classes. This is particularly true for rare, high-priority lesions like melanoma. The low count for melanoma directly reflects its low natural preva- lence compared to benign lesions (e.g., nevi); retaining these highly valuable, confirmed cases is paramount, and they should not be discarded by any means.

For all images, even those belonging to underrepresented types, they provide crucial

knowledge connected to the other variables. These images carry detailed information related to the experts’ diagnoses (including image rating, diagnosis certainty, and time spent diagnosing that particular image). Thus, even though we considered discarding these images, we concluded that the nuances and knowledge to be extracted from the interconnection between the various variables justify their inclusion in the dataset.

Ultimately, we believe providing the complete, unaltered dataset empowers future

users. Researchers can choose to ignore or exclude these minority classes to suit their specific benchmarking needs. Conversely, removing the images would permanently eliminate the valuable multimodal context associated with them, hindering broader research applications.

The remaining images were then manually cropped to contain only the region of inter-

est, excluding artifacts such as the frame of the dermoscopic lens or stray hairs (Figure 8). While this cropping does not necessarily reflect routine clinical practice, it is a standard step performed to ensure a clear judgment and help the AI model focus on the lesion instead of irrelevant artifacts [29,34]. To further standardize the image data, dermoscopic images were consistently cropped to a size of 1750 × 1750 pixels. This dimension was chosen as the maximum squared size that could be obtained while eliminating the dermoscope’s frame. Clinical images were cropped to 512 × 512, 1024 × 1024, or 1750 × 1750 pixels, depending on the lesion size. [URL 🔗](#page-0)

*Figure 8. An example of before (a) and after (b) cropping one of the collected images. The red frame indicates the cropping area.*

## 3.5.2. Metadata Validation and Consolidation

In parallel with image curation, the metadata was reviewed for consistency and

completeness. Missing values were not imputed or removed from the dataset, leaving the choice of how to handle them to end users of the dataset. All categorical data, such as lesion locations and diagnoses, were standardized to a predefined list. For instance, we collapsed the original body locations into a more compact list of categories but maintained both fields to provide flexibility for future researchers. Data types were also validated to ensure, for example, that age was stored as a numerical value. Beyond these standard measures, the expert ratings on image quality, collected for a fraction of the dataset (29 out of 240 lesions), serve as a valuable, human-centric validation of the dataset’s visual integrity.

## 3.5.3. Experts’ Feedback

The MCR-SL dataset offers comprehensive raw and joined diagnostic data for every

lesion, which is essential for evaluating model robustness against uncertain, realistic clinical inputs. This extensive metadata includes the individual diagnoses proposed by experts, their certainty levels, and the time spent diagnosing. By providing the raw individual


diagnoses of all dermatologists who labeled the images, the dataset enables researchers to explore diagnostic uncertainty via expert votes and certainty scores, or to apply alternative consensus criteria by leveraging the raw voting data in place of the established tiebreaker. Furthermore, the dataset ensures complete transparency regarding the ground truth by clearly distinguishing between lesions confirmed by histopathology (the highest confidence level) and those confirmed by the dermatology diagnosis (multi-expert consensus). This transparent stratification allows researchers to utilize the data selectively, for example, by using only the cases confirmed by histopathology for critical validation tasks

However, all image quality ratings from expert E002 were lost due to an unrecoverable

technical error. Given the nature of this complete data loss, imputation was not feasible and was thus not attempted. This loss primarily serves as a limitation for future analyses of inter-observer variability for image quality ratings. Future users planning this specific analysis must note that the comparison pool for image quality is limited to the three remaining experts. Crucially, all other diagnostic metadata (e.g., final diagnosis, certainty score) provided by expert E002, and all the data related to the other experts, were unaffected by this loss and are fully retained in the dataset for all other forms of analysis.

Notably, the remaining experts’ responses varied concerning image quality: those

who generally gave lower ratings to the images also tended to achieve lower accuracy in their diagnosis. This suggests that some experts may be less comfortable diagnosing when image quality is limited. However, further studies are needed to fully analyze this inter-observer variability and the impact of image quality on diagnostic confidence.

Supplementary Materials: The questionnaire used for the data collection can be downloaded at: https://www.mdpi.com/article/10.3390/data10100166/s1. [URL 🔗](https://www.mdpi.com/article/10.3390/data10100166/s1)

Author Contributions: Conceptualization, M.C.-F.; Methodology, M.C.-F., T.R.S., H.K., B.R.-Q. and I.C.-G.; Software, M.C.-F. and S.O.; Validation, M.C.-F., T.R.S., H.K., B.R.-Q. and I.C.-G.; Formal Analysis, M.C.-F.; Investigation, M.C.-F., T.R.S., H.K., B.R.-Q. and I.C.-G.; Resources, H.F. and G.M.C.; Data Curation, M.C.-F.; Writing—Original Draft Preparation, M.C.-F.; Writing—Review and Editing, T.R.S., H.K., B.R.-Q., I.C.-G., H.F., S.O., F.G. and G.M.C.; Visualization, M.C.-F.; Supervision, H.F. and G.M.C.; Project Administration, M.C.-F. and C.G.; Funding Acquisition, F.G., G.M.C., T.R.S. and C.G. All authors have read and agreed to the published version of the manuscript.

Funding: This work was completed while Maria Castro-Fernandez was a beneficiary of a predoctoral fellowship from the 2022 Ph.D. Training Program for Research Staff of the University of Las Palmas de Gran Canaria (ULPGC). The data collection was performed as part of the tasks in the Watching the Risk Factors (WARIFA) project. This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 101017385. The labeling software used is an adaptation of the original version created during a research project supported by the IKT+ initiative, funded by the Research Council of Norway (grant no. 332901).

Institutional Review Board Statement: The study was conducted in accordance with the Declaration of Helsinki and approved by the Regional Committee for Medical and Health Research Ethics (North) (Ref.: 392439) at UNN Hospital.

Informed Consent Statement: Informed consent was obtained from all subjects involved in the study.

Data Availability Statement: The data presented in this study are publicly available at https://doi. org/10.5281/zenodo.17306338 (Uploaded on 10 October 2025). [URL 🔗](https://doi.org/10.5281/zenodo.17306338)

Acknowledgments: The authors would like to thank the Dermatology and Plastic Surgery Depart- ments at UNN Hospital for their invaluable assistance in facilitating subject recruitment and data collection. Authors also thank Pablo Hernández Morera for his valuable comments, which have helped to improve some of the descriptions presented in this manuscript. During the preparation of this manuscript, the authors used Gemini 2.5 Flash to polish a human-written text. The authors have reviewed and edited the output and take full responsibility for the content of this publication.


Conflicts of Interest: The authors declare no conflicts of interest. The funders had no role in the design of the study; in the collection, analyses, or interpretation of data; in the writing of the manuscript; or in the decision to publish the results.

## Abbreviations

The following abbreviations are used in this manuscript:

AI

CNN Convolutional Neural Network

ViT

MCR-SL Multimodal, Context-Rich Skin Lesion

UNN University Hospital of North Norway

WARIFA Watching the Risk Factors

NEV Nevus

SK

BCC Basal Cell Carcinoma

AK

ATY

MEL Melanoma

SCC

ANG Angioma

DF

UNK Unknown

NM Non-malignant

M Malignant

Artificial Intelligence

Vision Transformer

Seborrheic Keratosis

Actinic Keratosis

Atypical nevus

Squamous Cell Carcinoma

Dermatofibroma

## References

- 1. Wang, R.; Chen, Y.; Shao, X.; Chen, T.; Zhong, J.; Ou, Y.; Chen, J. Burden of Skin Cancer in Older Adults from 1990 to 2021 and Modelled Projection to 2050. JAMA Dermatol. 2025, 161, 715. [CrossRef]

- 2. Esteva, A.; Kuprel, B.; Novoa, R.A.; Ko, J.; Swetter, S.M.; Blau, H.M.; Thrun, S. Dermatologist-Level Classification of Skin Cancer with Deep Neural Networks. Nature 2017, 542, 115–118. Erratum in Nature 2017, 546, 686. https://doi.org/10.1038/nature22985. [CrossRef]

- 3. Brinker, T.J.; Hekler, A.; Enk, A.H.; Klode, J.; Hauschild, A.; Berking, C.; Schilling, B.; Haferkamp, S.; Schadendorf, D.; Holland- Letz, T.; et al. Deep Learning Outperformed 136 of 157 Dermatologists in a Head-to-Head Dermoscopic Melanoma Image Classification Task. Eur. J. Cancer 2019, 113, 47–54. [CrossRef] [PubMed]

- 4. Haenssle, H.A.; Fink, C.; Schneiderbauer, R.; Toberer, F.; Buhl, T.; Blum, A.; Kalloo, A.; Ben Hadj Hassen, A.; Thomas, L.; Enk, A.; et al. Man against Machine: Diagnostic Performance of a Deep Learning Convolutional Neural Network for Dermoscopic Melanoma Recognition in Comparison to 58 Dermatologists. Ann. Oncol. 2018, 29, 1836–1842. [CrossRef] [PubMed]

- 5. Ha, Q.; Liu, B.; Liu, F. Identifying Melanoma Images Using EfficientNet Ensemble: Winning Solution to the SIIM-ISIC Melanoma Classification Challenge. arXiv 2020, arXiv:2010.05351.

- 6. Dascalu, A.; Walker, B.N.; Oron, Y.; David, E.O. Non-Melanoma Skin Cancer Diagnosis: A Comparison between Dermoscopic and Smartphone Images by Unified Visual and Sonification Deep Learning Algorithms. J. Cancer Res. Clin. Oncol. 2021, 148, 2497–2505. [CrossRef] [PubMed]

- 7. Pacheco, A.G.C.; Krohling, R.A. The Impact of Patient Clinical Information on Automated Skin Cancer Detection. Comput. Biol. Med. 2020, 116, 103545. [CrossRef] [PubMed]

- 8. Pacheco, A.G.C.; Krohling, R.A. An Attention-Based Mechanism to Combine Images and Metadata in Deep Learning Models Applied to Skin Cancer Classification. IEEE J. Biomed. Health Inf. 2021, 25, 3554–3563. [CrossRef]

- 9. Castro-Fernandez, M.; Hernandez, A.; Fabelo, H.; Balea-Fernandez, F.J.; Ortega, S.; Callico, G.M. Towards Skin Cancer Self- Monitoring through an Optimized MobileNet with Coordinate Attention. In Proceedings of the 2022 25th Euromicro Conference on Digital System Design (DSD), Maspalomas, Spain, 31 August–2 September 2022; IEEE: New York, NY, USA, 2022; pp. 607–614.

- 10. Nie, Y.; Sommella, P.; Carratù, M.; O’Nils, M.; Lundgren, J. A Deep CNN Transformer Hybrid Model for Skin Lesion Classification of Dermoscopic Images Using Focal Loss. Diagnostics 2022, 13, 72. [CrossRef] [PubMed]

- 11. Gallazzi, M.; Biavaschi, S.; Bulgheroni, A.; Gatti, T.M.; Corchs, S.; Gallo, I. A Large Dataset to Enhance Skin Cancer Classification with Transformer-Based Deep Neural Networks. IEEE Access 2024, 12, 109544–109559. [CrossRef]


- 12. Tschandl, P.; Rosendahl, C.; Kittler, H. The HAM10000 Dataset, a Large Collection of Multi-Source Dermatoscopic Images of Common Pigmented Skin Lesions. Sci. Data 2018, 5, 180161. [CrossRef]

- 13. Combalia, M.; Codella, N.C.F.; Rotemberg, V.; Helba, B.; Vilaplana, V.; Reiter, O.; Carrera, C.; Barreiro, A.; Halpern, A.C.; Puig, S.; et al. BCN20000: Dermoscopic Lesions in the Wild. arXiv 2019, arXiv:1908.02288. [CrossRef]

- 14. Mendonca, T.; Ferreira, P.M.; Marques, J.S.; Marcal, A.R.S.; Rozeira, J. PH2—A Dermoscopic Image Database for Research and Benchmarking. In Proceedings of the 2013 35th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), Osaka, Japan, 3–7 July 2013; IEEE: Osaka, Japan, 2013; pp. 5437–5440.

- 15. Pacheco, A.G.C.; Lima, G.R.; Salomão, A.S.; Krohling, B.; Biral, I.P.; de Angelo, G.G.; Alves, F.C.R., Jr.; Esgario, J.G.M.; Simora, A.C.; Castro, P.B.C.; et al. PAD-UFES-20: A Skin Lesion Dataset Composed of Patient Data and Clinical Images Collected from Smartphones. Data Brief 2020, 32, 106221. [CrossRef]

- 16. Codella, N.; Rotemberg, V.; Tschandl, P.; Celebi, M.E.; Dusza, S.; Gutman, D.; Helba, B.; Kalloo, A.; Liopyris, K.; Marchetti, M.; et al. (ISIC). Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration arXiv 2019, arXiv:1902.03368. [CrossRef]

- 17. Watching the Risk Factors: Artificial Intelligence and the Prevention of Chronic Conditions|WARIFA Project|Fact Sheet|H2020|CORDIS|European Commission. Available online: https://cordis.europa.eu/project/id/101017385/es (accessed on 27 October 2021).

- 18. Petrie, T.C.; Larson, C.; Heath, M.; Samatham, R.; Davis, A.; Berry, E.G.; Leachman, S.A. Quantifying Acceptable Artefact Ranges for Dermatologic Classification Algorithms. Ski. Health Dis. 2021, 1, e19. [CrossRef]

- 19. Yan, S.; Yu, Z.; Primiero, C.; Vico-Alonso, C.; Wang, Z.; Yang, L.; Tschandl, P.; Hu, M.; Ju, L.; Tan, G.; et al. A Multimodal Vision Foundation Model for Clinical Dermatology. Nat. Med. 2025, 31, 2691–2702. [CrossRef]

- 20. Johansen, T.H.; Møllersen, K.; Ortega, S.; Fabelo, H.; Garcia, A.; Callico, G.M.; Godtliebsen, F. Recent Advances in Hyperspectral Imaging for Melanoma Detection. WIREs Comput. Stat. 2020, 12, e1465. [CrossRef]

- 21. Leon, R.; Martinez-Vega, B.; Fabelo, H.; Ortega, S.; Melian, V.; Castaño, I.; Carretero, G.; Almeida, P.; Garcia, A.; Quevedo, E.; et al. Non-Invasive Skin Cancer Diagnosis Using Hyperspectral Imaging for In-Situ Clinical Support. J. Clin. Med. 2020, 9, 1662. [CrossRef]

- 22. Aloupogianni, E.; Ishikawa, M.; Ichimura, T.; Sasaki, A.; Kobayashi, N.; Obi, T. Design of a Hyper-Spectral Imaging System for Gross Pathology of Pigmented Skin Lesions. In Proceedings of the 2021 43rd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), Guadalajara, Mexico, 1–5 November 2021; IEEE: New York, NY, USA, 2021; pp. 3605–3608.

- 23. Hetz, M.J.; Garcia, C.N.; Haggenmüller, S.; Brinker, T.J. Advancing Dermatological Diagnosis: Development of a Hyperspectral Dermatoscope for Enhanced Skin Imaging. arXiv 2024, arXiv:2403.00612. [CrossRef]

- 24. De Pascalis, A.; Perrot, J.L.; Tognetti, L.; Rubegni, P.; Cinotti, E. Review of Dermoscopy and Reflectance Confocal Microscopy Features of the Mucosal Melanoma. Diagnostics 2021, 11, 91. [CrossRef]

- 25. Roth, B.; Kukk, A.F.; Wu, D.; Panzer, R.; Emmert, S. Four-Modal Device Comprising Optical Coherence Tomography, Photoacoustic Tomography, Ultrasound, and Raman Spectroscopy Developed for in Vivo Skin Lesion Assessment. Biomed. Opt. Express 2025, 16, 1792–1806. [CrossRef]

- 26. Stridh, M.; Dahlstrand, U.; Naumovska, M.; Engelsberg, K.; Gesslein, B.; Sheikh, R.; Merdasa, A.; Malmsjö, M. Functional and Molecular 3D Mapping of Angiosarcoma Tumor Using Non-Invasive Laser Speckle, Hyperspectral, and Photoacoustic Imaging. Orbit 2024, 43, 453–463. [CrossRef]

- 27. Wu, D.; Fedorov Kukk, A.; Panzer, R.; Emmert, S.; Roth, B. In Vivo Differentiation of Cutaneous Melanoma from Benign Nevi with Dual–Modal System of Optical Coherence Tomography and Raman Spectroscopy. J. Biophotonics 2025, 18, e70040. [CrossRef] [PubMed]

- 28. Rotemberg, V.; Kurtansky, N.; Betz-Stablein, B.; Caffery, L.; Chousakos, E.; Codella, N.; Combalia, M.; Dusza, S.; Guitera, P.; Gutman, D.; et al. A Patient-Centric Dataset of Images and Metadata for Identifying Melanomas Using Clinical Context. Sci. Data 2021, 8, 34. [CrossRef] [PubMed]

- 29. Daneshjou, R.; Barata, C.; Betz-Stablein, B.; Celebi, M.E.; Codella, N.; Combalia, M.; Guitera, P.; Gutman, D.; Halpern, A.; Helba, B.; et al. Checklist for Evaluation of Image-Based Artificial Intelligence Reports in Dermatology: CLEAR DermConsensus Guidelines from the International Skin Imaging Collaboration Artificial IntelligenceWorking Group. JAMADermatol. 2022, 158, 90–96. [CrossRef] [PubMed]

- 30. Bourkas, A.N.; Barone, N.; Bourkas, M.E.C.; Mannarino, M.; Fraser, R.D.J.; Lorincz, A.; Wang, S.C.; Ramirez-Garcialuna, J.L. Diagnostic Reliability in Teledermatology: A Systematic Review and a Meta-Analysis. [PubMed] BMJ Open 2023, 13, e068207. [CrossRef]

- 31. ISIC Archive. ISIC 2020: Training Data. Available online: https://gallery.isic-archive.com/#!/topWithHeader/onlyHeaderTop/ gallery?filter=%5B%22collections%7C70%22%5D (accessed on 2 October 2025). [URL 🔗](https://gallery.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery?filter=[%22collections%7C70%22])


- 32. Tran, H.; Chen, K.; Lim, A.C.; Jabbour, J.; Shumack, S. Assessing Diagnostic Skill in Dermatology: A Comparison between General Practitioners and Dermatologists. Australas. J. Dermatol. 2005, 46, 230–234. [CrossRef] [PubMed]

- 33. Nervil, G.G.; Ternov, N.K.; Lorentzen, H.; Kromann, C.; Ingvar, Å.; Nielsen, K.; Tolsgaard, M.; Vestergaard, T.; Hölmich, L.R. Teledermoscopic Triage of Melanoma-Suspicious Skin Lesions Is Safe: A Retrospective Comparative Diagnostic Accuracy Study with Multiple Assessors. J. Telemed. Telecare 2025, 31, 1296–1307. [CrossRef]

- 34. Barata, C.; Celebi, M.E.; Marques, J.S. A Survey of Feature Extraction in Dermoscopy Image Analysis of Skin Cancer. IEEE J. Biomed. Health Inf. 2019, 23, 1096–1109. [CrossRef]

Disclaimer/Publisher’s Note: The statements, opinions and data contained in all publications are solely those of the individual author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to people or property resulting from any ideas, methods, instructions or products referred to in the content.
