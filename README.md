# DIAMOND Challenge: Predicting ci-DME Development

Welcome to the DIAMOND Challenge! This challenge focuses on developing predictive models for the development of clinically significant Diabetic Macular Edema (ci-DME) within a year using Ultra-Wide Field Color Fundus Photography (UWF-CFP) images. Our goal is to encourage innovative approaches that achieve high sensitivity, specificity, and precision on both in-domain and out-of-domain data.

<p float="left">
  <img src="images/miccai2024.png" height="150" />
  <img src="images/diamond.png" height="150" />
</p>

## Overview

For this challenge, participants are required to submit their code as a **Zip file** using a standardized Singularity image provided by the organizers. This approach ensures consistency in the execution environment across all submissions. The challenge is structured to facilitate a blind evaluation of submissions, with feedback on performance provided directly to participants daily.

## Data
### Synthetic Dataset
To assist in the development and local testing of your algorithms, we are providing a synthetic dataset generated using images from the Deep Diabetic Retinopathy Image Dataset (DeepDRiD) -  [DeepDRiD GitHub](https://github.com/deepdrdoc/DeepDRiD). This dataset is available under the Creative Commons Attribution Share Alike 4.0 International license. For the purpose of this challenge, ci-DME labels have been synthetically generated based on the following rule: `A ci-DME event is predicted to occur in the next visit if the DR label is >= moderate non-proliferative DR; otherwise, it is considered non-ci-DME`.
> Download the `ultra-widefield_images` folder from [DeepDRiD GitHub](https://github.com/deepdrdoc/DeepDRiD) and use the `training_set.csv` and `validation_set.csv` files provided in this repository to test your algorithms.

### One Patient Dataset
In addition to the synthetic dataset, we are also providing real image data from two patients. This dataset is intended to give participants a more accurate feel for the kind of data they will be working with and to aid in the fine-tuning of their predictive models.
Participants will have access to sample data and an artificially generated dataset for model development. The full challenge dataset remains hidden and will be used by the organizers for the final evaluation of submissions.



## Submission Method

- **Code Submission**: Submit your code as a **Zip file** containing all necessary scripts and documentation.
- **Singularity Image**: Use the Singularity image provided by the organizers for development and testing. This ensures compatibility and fairness in the evaluation process. The image can be downloaded using this [link](https://drive.google.com/drive/folders/1A4nw-upR_TP19InQJpcX-UZ9xtuhetrg). The list of libraries included in this image can be found in the file `list_lib_diamond.txt`.
- **Local Testing**: Participants must test their solutions locally using the provided Singularity image before submission. To run your code using the singularity image locally: `singularity exec --nv /path/to/diamond.sif python main.py`
- **Library Requests**: If a specific library is needed that is not included in the Singularity image, participants can request its inclusion via [this form](https://docs.google.com/forms/d/e/1FAIpQLScHLU8zwy0qNVFs_A8XY8SsVtDETB3hBP2olY8dCdnOhgqZuw/viewform).

**IMPORTANT:** The function `generate_val_csv` saves a CSV file of the predictions. THIS FUNCTION IS CRITICAL FOR EVALUATING THE MODEL'S PERFORMANCE ON THE VALIDATION SET. PARTICIPANTS SHOULD NOT CHANGE THIS FUNCTION IN ANY WAY TO ENSURE THE INTEGRITY OF THE EVALUATION PROCESS. THE CODE OF THE FINALISTS WILL BE REVIEWED. IN CASE OF MODIFICATION OF THIS FUNCTION, THE TEAM WILL BE DISQUALIFIED.
    

### Using the `train.sh` Script for Submission

For efficient management and execution of your model/models with different parameters or configurations, include a `train.sh` shell script file in your submission. Here’s a template for what your `train.sh` might look like:

```bash

# First run with a specific parameter set and a 3-day limit. Convert days to hours: 3 days * 24 hours/day = 72 hours.
timeout 72h python main.py --learning_rate 1e-4 --batch_size 16 --backbone resnet50 --submission_name $1

# Second run with a different parameter set and a 2-day limit. Convert days to hours: 2 days * 24 hours/day = 48 hours. 
timeout 48h python main.py --learning_rate 1e-3 --batch_size 8 --backbone efficientnet_b0 --submission_name $1

# Third run with a 2-day limit. Convert days to hours: 2 days * 24 hours/day = 48 hours. 
timeout 48h python main.py --learning_rate 1e-3 --batch_size 8 --backbone resnet101 --submission_name $1
```
We will call this bash script with your submission's name as an argument ($1). Submission names are used to save logs, checkpoints, result files, and allows you to access saved checkpoints from previous submissions.

**Customization Instructions:**

- **Parameter Settings**: Modify the `--learning_rate`,  `--batch_size` and `--backbone` parameters as necessary to fit your model’s requirements. This is just an example, you can add all elements of your argparse. 

- **Running time setup:** The `train.sh` script uses `timeout` to manage the execution time for each run, allowing participants to strategically distribute the total time allocation (7 days) across different runs, for example, allocate 3 days for the first run, 2 days for the second and 2 days for the third. You can add a line for each run. 


## Schedule

- **Registration Period**: April 1 - July 31, 2024
- **Sample Data and Artificial Dataset Release**: April 1, 2024
- **Code Submission Period**: April 1 - July 31, 2024
- **Finalists Announcement**: August 1, 2024
- **Final Submission Period (Code + Report)**: August 1 -  September 15, 2024
- **Results Announcement**: Challenge Day, October 10, 2024

## Getting Started

- **Starting Code**: Refer to this repository for the PyTorch-based starter code.
- **Singularity Setup**: Follow the [Singularity installation guide](https://sylabs.io/guides/latest/user-guide/) to set up your local environment for testing.
- **Submission Guide**: Detailed instructions for preparing and submitting your code can be found in the `Submission` section.
  
**Note on Hyperparameters and GPU Specifications:**
The hyperparameters provided within the starter code, including learning rate, epochs, etc., serve merely as examples. Participants are encouraged to adjust these values according to their specific needs to optimize model performance.
Moreover, the GPU model used for the reference training runs is a **Tesla V100s**, equipped with **32 GB of memory**. The choice of batch size is crucial and directly influenced by the combination of the model architecture and the image size being processed. As a guideline:
- For **Resnet50** or **EfficientNet_b0** with images of size **448x448**, a maximum **batch size of 80 is recommended**.
- For **Resnet101** with images of size **448x448**, **the batch size should not exceed 48**.
- For **EfficientNet_b7** with images of size **448x448**, a maximum **batch size of 8 is advised**.
  
These batch size recommendations are based on ensuring optimal memory utilization and may need adjustment depending on the specific model configuration and available GPU resources. Participants must consider these factors when configuring their training setups to achieve the best balance between training speed and model accuracy.

## Pre-evaluation

Participants are encouraged to thoroughly test their models locally using the provided Singularity image. Each team is allowed three submission attempts, with each attempt providing an opportunity to refine their models based on performance feedback.


## Assessment Aims

1. Predict the development of ci-DME within a year using in-domain UWF-CFP data from France.
2. Predict the development of ci-DME within a year using out-of-domain UWF-CFP data from Algeria.

## Leaderboard

Below is the current leaderboard showcasing the top-performing participants and their teams, based on various evaluation metrics:

| Participant |  Team    | Score  |  AUC   | F1 Score |   ECE  | Name          |
|-------------|----------|--------|--------|----------|--------|---------------|
| agaldran    |    -     | 1.2184 | 0.7203 | 0.0454   | 0.0492 | Submission_3  |
| agaldran    |    -     | 1.1765 | 0.6880 | 0.0000   | 0.0229 | Submission 1  |
| agaldran    |    -     | 1.1096 | 0.6150 | 0.0000   | 0.0107 | Submission 2  |
| qqqqqpy     |    -     | 1.0898 | 0.5732 | 0.0526   | 0.0194 | TL_den_optos1 |
| pzhang1     | DF41     | 1.0486 | 0.5719 | 0.0000   | 0.0465 | Exp_1         |
| yyydido     | AIFUTURE | 0.9521 | 0.6269 | 0.1458   | 0.4920 | submit0703    |
| wsy66       |   -      | 0.9317 | 0.4697 | 0.0000   | 0.0759 | Ensemble3     |
|             |          |        |        |          |        |


## Submission

Submit your Zip file containing the code via [this form](https://docs.google.com/forms/d/e/1FAIpQLSfIDaXXr35gYxEfbPAZF2NXaRFx6lKSP8pB05XhaqHSSVZMOg/viewform?usp=pp_url). 

## Requesting Additional Libraries

If your solution requires libraries not present in the provided Singularity image, please request their inclusion via [this form](https://docs.google.com/forms/d/e/1FAIpQLScHLU8zwy0qNVFs_A8XY8SsVtDETB3hBP2olY8dCdnOhgqZuw/viewform).

## Contact

For inquiries regarding the challenge rules, submission process, or for technical support, please contact [gwenole.quellec@inserm.fr](mailto:gwenole.quellec@inserm.fr) and [mostafa.elhabibdaho@univ-brest.fr](mailto:mostafa.elhabibdaho@univ-brest.fr).

We wish all participants the best of luck and look forward to innovative solutions that advance our understanding and predictive capabilities in the realm of ci-DME development.
