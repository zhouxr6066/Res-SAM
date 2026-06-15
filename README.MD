# Res-SAM Framework for GPR Underground Hazard Detection

Res-SAM adopts a **two-stage processing workflow** to efficiently detect underground hazards and structures. First, the **Segment Anything Model (SAM)** preprocesses the GPR images, rapidly marking potential anomaly candidate regions without additional training. Then, **Reservoir Computing (RC)** is used to calculate the anomaly score heatmap. By integrating **Dual-Directional Echo State Networks (2D-ESN) and image patching techniques**, the software refines the anomalies within the candidate regions with high precision.

This software can detect **underground risks** such as **cavities, cracks, and loose regions**, as well as **man-made structures** like **pipes and manholes** (which are not considered risks but still valuable for detection). The anomaly score heatmap highlights high-risk areas for further analysis.


# Res-SAM Framework Overview

Detecting underground anomalies using GPR faces three significant challenges. First, anomaly involves unknowns, thus, there is a notable shortage of labeled anomaly data. Acquiring anomaly samples is costly, resulting in highly imbalanced datasets dominated by normal (non-target) samples. This imbalance severely restricts the training process of data-driven approaches. Second, the high variability of subsurface environments, including diverse soils and structures, leads to significant differences in EM wave characteristics. Such variability impedes the generalization capability, causing approaches trained on one scenario to likely perform poorly in others. Third, underground anomalies typically present as gradual changes in EM wave reflections rather than sharp, distinct boundaries. This gradual transition complicates precise anomaly localization, making conventional methods designed for visual images inadequate for wholly and accurately delineating anomaly boundaries in real-world applications.


![framework_overview](images/framework_overview.png)

Res-SAM leverages minimal non-target training data, readily obtainable from the detection area, and avoids computationally intensive training, enabling rapid deployment in unexplored environments. It synergizes visual discernibility with local dynamic variations in GPR data caused by underground anomalies. Recognizing that practical subsurface anomaly detection necessitates clearly delineating entire affected regions, Res-SAM provides rectangular outlines of anomalies along with their categorization. As depicted in the figure, Res-SAM operates in two main phases: "Feature Collection" and "Anomaly Detection". 

In the Feature Collection phase, local patches are extracted from non-target GPR data and fitted individually using the Dual-Directional Echo State Network (2D-ESN). Acknowledging the vertical continuity of EM waves and horizontal correlations due to the consistency of the subsurface medium, effective changing information exists both vertically and horizontally within GPR data. 2D-ESN, a Reservoir Computing Network that incorporates two reservoirs in its hidden layer, connects each data point within the patch to its neighbors in both vertical and horizontal directions. Fitting each patch with 2D-ESN adequately captures the local data-inherent dual-directional changing information, resulting in a compact fitted readout model serving as "dual-directional dynamic features"
of the original patch. These features, derived from patches on non-target data and denoted as "normal features", collectively represent local changing information in non-target data and are stored in a "Feature Bank".

In the Anomaly Detection phase, SAM is first introduced to identify initial candidate anomaly regions, either guided by simple click prompts (typically single-digit number left- and right-mouse clicks) in the **click-guided** mode, or by segmenting the entire B-scan into coarse regions in the **fully automatic** mode, where each region is fitted via 2D-ESN with extracted dynamic feature compared against the feature bank to discard normal parts before further refinement. Anomalies in GPR data often have indistinct boundaries due to EM wave reflection, diffraction, and attenuation, necessitating further refinement. To this end, we focus on local changing information within GPR data caused by underground anomalies, aiming to precisely adjust the candidate region. Specifically, for each point within the candidate region, we extract a local patch centered on this point and fit this patch using 2D-ESN. The obtained dynamic features are compared with the normal features in the feature bank. Patches exhibiting significantly different dynamics (high anomaly likelihood) indicate anomalies; overlapping anomaly patches are then merged to precisely delineate anomaly regions. Owing to the similar changing information in GPR data, similar subsurface structures result in comparable dynamic features, while those from different structures exhibit significant variances, reflecting their distinct changing information along and among EM waves. Therefore, the confirmed anomaly regions undergo another round of 2D-ESN fitting to derive category-specific dynamic features, enabling anomaly categorization.

The two modes (click-guided & fully automatic) are automatically switched by default in our software. If you click the target area on the B-Scan, you will enter the click-guided mode. If you directly click the "Output Predicted Area" button, you will enter the fully automatic mode.


# Pseudo-code of Res-SAM

```pseudocode
Algorithm: Res-SAM Framework
Input:
  Test image X
  Positive clicks C⁺
  Negative clicks C⁻
  Normal set 𝔇
  Patch size r
  Stride s
  Mask threshold τ_mask
  Anomaly threshold τ_anom
Output:
  Bounding boxes ℛ aligned to X

# Step 1: Feature collection from normal data
B ← ∅
for each I ∈ 𝔇 do
    P ← UNFOLD(I, k = r, s = r)
    for each p ∈ P do
        f ← FIT_2DESN(p, r)
        append f to B
    end for
end for

# Step 2: SAM-based candidate region identification:
M_norm ← SAM_INFER(X, C⁺, C⁻)          # normalized mask in [0,1]
R_M    ← BOUNDING_BOX(M_norm, τ_mask)
x₀, y₀ ← top-left corner of R_M
X_M    ← CROP(X, R_M)
H_M, W_M ← SIZE(X_M)

# Step 3: Patch fitting and heatmap generating
P_M ← UNFOLD(X_M, k = r, s = s)
G_M ← CENTER_COORDS on X_M for patches in P_M
A_M ← ZEROS(H_M, W_M)
for i = 1 to |P_M| do
    f ← FIT_2DESN(P_M[i], r)
    d ← min_j DISTANCE(f, B[j])
    c ← G_M[i]                          # center c_x, c_y in X_M
    A_M[c_y - r/2 : c_y + r/2 , c_x - r/2 : c_x + r/2] ←
        A_M[c_y - r/2 : c_y + r/2 , c_x - r/2 : c_x + r/2] + d
end for
A_M ← NORMALIZE(A_M)				# anomaly heatmap in [0,1]

# Step 4: Outlining the anomaly region
ℛ_M ← BOUNDING_BOX(A_M, τ_anom)
if ℛ_M ≠ ∅ then
    x₁′, y₁′, x₂′, y₂′ ← ℛ_M
    ℛ ← [x₁′ + x₀, y₁′ + y₀, x₂′ + x₀, y₂′ + y₀]
else
    ℛ ← ∅
end if

return ℛ
```

# Guidance Video for Res-SAM

https://github.com/user-attachments/assets/5fa823db-3749-4fcd-b17a-d27495e8bdf6

# System Requirements

To ensure smooth operation, the following system specifications are recommended:

- **Operating System**: Linux (Ubuntu 20.04+), Windows 10/11
- **Python Version**: 3.8 or higher
- **GPU Support**: NVIDIA GPU, CUDA 12.4+ recommended for acceleration
- **Memory Requirements**: 16GB RAM or more recommended

# Installation Guide

Following the steps below, the installation typically takes about half an hour under normal conditions.

1. Clone the project repository:

```bash
git clone https://github.com/your-repo/GPR-Subsurface-Anomaly-Detection.git
cd GPR-Subsurface-Anomaly-Detection
```

2. Create a conda environment and install the dependencies based on whether you are using a GPU or CPU:

```bash
conda env create -f environment-cpu.yml # CPU
conda env create -f environment-gpu.yml # GPU, you may change the CUDA version according to your device.
conda activate res-sam
```

3. Download Segment Anything Model pre-trained weights:

- [Segment Anything](https://github.com/facebookresearch/segment-anything)

The package is included in the environment files already. Additionally, Segment Anything requires downloading the pre-trained model weights: [ViT-L SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth),
which should be placed in the `sam` directory of the project.

# Usage

## Running the Software

Simply run the following command:

```bash
python main.py
```

To switch to the Chinese version, modify the `os.environ['LANGUAGE']` variable in `main.py`:

```python
os.environ['LANGUAGE'] = 'zh-CN'
```


Then, execute the command again:

```bash
python main.py
```


## Software Interface & Operations

![image-20250326095209567](images/image-20250326095209567.png)

1. Select an Image
   Click the "Choose an Image" button to load an image from your local storage.

![image-20250326095332711](images/image-20250326095332711.png)

2. Mark Areas of Interest (Optional)
   - Left-click to mark green points, indicating areas of interest (potential anomaly regions).
   - Right-click to mark red points, indicating non-anomalous areas.
   - The SAM model will use these marked points to draw a blue bounding box around the suspected anomaly.
   - The software will then generate an anomaly score heatmap within this area to visualize risk levels.
   - This step can be repeated multiple times for adjustment.

![image-20250326095522051](images/image-20250326095522051.png)

3. Output Predicted Area

Click "Output Predicted Area" button, and the system will further refine the suspicious regions based on the heatmap:

- If Step 2 included marked points: a red bounding box is generated around those areas to precisely delineate the anomaly.

- If no points were marked (since Step 2 is optional): the system automatically identifies an area of interest, runs prediction, and outputs a red bounding box covering that region.

The output takes about 5 seconds under normal conditions.

![image-20250326095916035](images/image-20250326095916035.png)

4. Save Processed Image

Once satisfied with the adjustments, click "Save the Image" to save the processed image locally.

![image-20250326100049407](images/image-20250326100049407.png)

![image-20250326100203941](images/image-20250326100203941.png)

5. Output Image

The saved image includes the processed visualization.


# Data Input & Output

- Input: The software supports PNG and JPEG formats for GPR radar images.

- Output: The processed results can be saved in PNG and JPEG formats.

# Core Modules & API

- `sam/sam.py`: Integrates operations related to the Segment Anything Model (SAM).
  - `predict`: Generates the highest-confidence object mask.
  - `find_box`: Computes the bounding box corresponding to the mask.

- `PatchRes` Module: The core functionality module of the software
  - Computes and visualizes anomaly score heatmaps.
  - Optimizes bounding boxes for detected anomalies.

- `ui` Module: Implements the Graphical User Interface (GUI) to enhance usability.

- `i18n` Module: Provides multi-language translation support.

# 📄 License

This project is licensed under the **GNU General Public License v3.0** - see the [LICENSE](LICENSE) file for details.
