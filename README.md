# Image Classification for a City Dog Show

A Python-based image classification pipeline built to automate contestant registration for a citywide dog show. This project uses pre-trained Convolutional Neural Networks (CNNs) to verify if submitted registration images depict actual dogs and to accurately identify their specific breeds. 

Developed as part of the AI Programming with Python curriculum, this project evaluates three different CNN architectures (AlexNet, VGG, and ResNet) to determine the optimal balance between classification accuracy and computational runtime.

## Project Objectives

* **Verification:** Determine which image classification algorithm works best at distinguishing images of "dogs" from "not dogs" to filter out invalid show registrations.
* **Breed Identification:** Evaluate the accuracy of the best algorithm in correctly identifying specific dog breeds.
* **Performance Profiling:** Measure and compare the runtime of each algorithm to analyze the trade-off between computational efficiency and accuracy.
* **Edge Case Handling:** Analyze the model's performance on morphologically similar breeds (e.g., Great Pyrenees vs. Kuvasz, Beagle vs. Walker Hound).

## Technical Stack & Skills

* **Language:** Python 3
* **Core Libraries:** `argparse`, `time`, `os`
* **Deep Learning Framework:** PyTorch (Torchvision for pre-trained models)
* **Concepts:** Command Line Interfaces (CLI), Data Structures (Dictionaries/Lists for result mapping), File I/O, Modular Programming.

## Architecture & Logic

The project is heavily modularized to parse inputs, generate labels, and compute statistics efficiently. The main execution script, `check_images.py`, ties together several custom modules:
* `get_input_args.py`: Parses command line arguments for the image directory, architecture choice, and dog names reference file.
* `get_pet_labels.py`: Extracts the true ground-truth labels directly from the image filenames.
* `classify_images.py`: Feeds the images through the pre-trained ImageNet CNN to generate predicted labels.
* `adjust_results4_isadog.py`: Compares predicted labels and ground-truth labels against a master text file of valid dog names.
* `calculates_results_stats.py`: Computes absolute counts and percentage accuracies for breed matches and dog/not-dog classifications.

## Usage

To run the classification pipeline, execute `check_images.py` from the terminal. The script utilizes `argparse` to accept three optional arguments.

**Arguments:**
* `--dir`: Path to the folder of pet images (default: `pet_images/`)
* `--arch`: CNN model architecture to use (`vgg`, `resnet`, or `alexnet` - default: `vgg`)
* `--dogfile`: Text file containing the list of valid dog names (default: `dognames.txt`)
* ## Results & Evaluation

The classification pipeline was evaluated using three different pre-trained CNN architectures (VGG, ResNet, and AlexNet) across a dataset of 40 images (30 dog images, 10 non-dog images). The objective was to determine the "best" model based on two primary metrics:
1. **Verification Accuracy:** Correctly distinguishing dogs from non-dogs.
2. **Breed Identification Accuracy:** Correctly classifying the specific breed of the dog.

### Architecture Performance Comparison

While running the batch evaluation across all three models, distinct trade-offs between architectural complexity, classification accuracy, and basic verification emerged:

| Metric | VGG (Selected Model) | ResNet | AlexNet |
| :--- | :--- | :--- | :--- |
| **% Correct Dogs** | **100.0%** | < 100.0% | **100.0%** |
| **% Correct Not-Dogs** | **100.0%** | < 100.0% | **100.0%** |
| **% Correct Breed** | **93.3%** | Moderate | Lowest |

### Key Findings

* **The Optimal Model (VGG):** The **VGG** architecture proved to be the most effective model for this application. It successfully filtered out all 10 non-dog images (100% accuracy) and identified general dog images perfectly (100% accuracy). Furthermore, it achieved the highest breed classification accuracy at **93.3%** (with a measured runtime of ~2.3 seconds).
* **Algorithmic Trade-offs:** While AlexNet matched VGG's perfect 100% accuracy in basic dog vs. non-dog verification, it struggled with granular breed identification. Conversely, ResNet performed better than AlexNet at identifying specific breeds but failed to achieve the perfect 100% baseline for basic verification. 
* **Edge Case Limitations:** The 93.3% breed accuracy in the VGG model highlights a known limitation in computer vision: distinguishing between morphologically similar classes. The model's only errors were misclassifications of closely related breeds, specifically confusing a **Great Pyrenees** for a **Kuvasz**, and a **Beagle** for a **Walker Hound**. Despite these edge cases, the system demonstrated exceptional reliability for the primary goal of the registration system.

**Example Command:**
```bash
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
