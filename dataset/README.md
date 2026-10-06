# FACE Dataset

This folder contains the synthetic portrait images and image-pair definitions used in the **FACE benchmark**.

The dataset is organized into two image folders and three JSON files containing the controlled image pairs used during VLM evaluation.

## Dataset Structure

```text
dataset/
├── base_images/
├── variations/
├── base_image_pairs.json
├── base_in_counterfactuals_image_pairs.json
└── counterfactual_image_pairs.json
```

## `base_images/`

Contains the synthetic base portraits used to construct the FACE dataset.

Each image represents an identity defined by demographic attributes such as **race, gender, and age**.

Example filenames:

```text
White_Male_Young.png
White_Female_MiddleAged.png
MiddleEastern_Male_Old.png
SoutheastAsian_Male_Young.png
```

The filename structure is:

```text
<Race>_<Gender>_<Age>.png
```

These images provide the starting identities from which the controlled appearance variations are generated.

## `variations/`

Contains the counterfactual appearance variants generated from the base portraits.

Each image modifies a particular appearance attribute while attempting to preserve the underlying identity and other portrait characteristics.

The manipulated appearance attributes include:

- Skin tone
- Facial expression
- Hairstyle
- Cultural markers
- Facial hair
- Tattoos
- Piercings

Example filenames:

```text
White_Male_Young_skin_fair.png
White_Male_Young_skin_dark.png
White_Male_Young_expr_sad.png
White_Male_Young_hair_dreadlocks.png
White_Male_Young_tat_necktattoo.png
White_Male_Young_pierc_earandnosepiercings.png
```

The filenames encode the base identity, manipulated appearance attribute, and corresponding attribute value.

## `base_image_pairs.json`

Contains controlled pairs constructed from the portraits in `base_images/`.

The paired identities differ in exactly **one demographic attribute**:

- Race
- Gender
- Age

This pairing strategy enables demographic comparisons while limiting other demographic differences between the two presented identities.

## `base_in_counterfactuals_image_pairs.json`

Contains cross-identity pairs constructed from the counterfactual portraits in `variations/`.

Images are paired while keeping the **same appearance variation fixed**. The underlying identities differ in exactly one demographic attribute: **race, gender, or age**.

These pairs are used to examine demographic associations while controlling for the applied appearance variation.

## `counterfactual_image_pairs.json`

Contains controlled pairs constructed from the counterfactual portraits in `variations/`.

Images within a pair share the same underlying identity and appearance-attribute category but differ in the specific value of that attribute.

For example, a pair may compare two skin-tone variants or two hairstyle variants of the same underlying identity.

These pairs form the controlled comparisons used for analyzing appearance-conditioned associations.

## Dataset Relationships

The relationship between the image folders and pair-definition files can be summarized as:

```text
base_images/
     │
     ├──> base_image_pairs.json
     │
     └──> Counterfactual Appearance Generation
                     │
                     ▼
                variations/
                     │
                     ├──> base_in_counterfactuals_image_pairs.json
                     │
                     └──> counterfactual_image_pairs.json
```

The JSON files contain the image-pair definitions used by the VLM inference scripts, while the corresponding portrait images are stored in `base_images/` and `variations/`.

## Note

The FACE dataset consists of synthetically generated portraits and controlled appearance modifications created for auditing VLM behavior. The demographic and appearance labels represent experimental generation conditions and should not be interpreted as predictions or assessments of real individuals.
