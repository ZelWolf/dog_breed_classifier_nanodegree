Image Classification for a City Dog Show
A Python-based image classification pipeline built to automate contestant registration for a citywide dog show. This project uses pre-trained Convolutional Neural Networks (CNNs) to verify if submitted registration images depict actual dogs and to accurately identify their specific breeds.

Developed as part of the AI Programming with Python curriculum, this project evaluates three different CNN architectures (AlexNet, VGG, and ResNet) to determine the optimal balance between classification accuracy and computational runtime.

Project Objectives
Verification: Determine which image classification algorithm works best at distinguishing images of "dogs" from "not dogs" to filter out invalid show registrations.

Breed Identification: Evaluate the accuracy of the best algorithm in correctly identifying specific dog breeds.

Performance Profiling: Measure and compare the runtime of each algorithm to analyze the trade-off between computational efficiency and accuracy.

Edge Case Handling: Analyze the model's performance on morphologically similar breeds (e.g., Great Pyrenees vs. Kuvasz, Beagle vs. Walker Hound).

Technical Stack & Skills
Language: Python 3

Core Libraries: argparse, time, os

Deep Learning Framework: PyTorch (Torchvision for pre-trained models)

Concepts: Command Line Interfaces (CLI), Data Structures (Dictionaries/Lists for result mapping), File I/O, Modular Programming.

Architecture & Logic
The project is heavily modularized to parse inputs, generate labels, and compute statistics efficiently. The main execution script, check_images.py, ties together several custom modules:

get_input_args.py: Parses command line arguments for the image directory, architecture choice, and dog names reference file.

get_pet_labels.py: Extracts the true ground-truth labels directly from the image filenames.

classify_images.py: Feeds the images through the pre-trained ImageNet CNN to generate predicted labels.

adjust_results4_isadog.py: Compares predicted labels and ground-truth labels against a master text file of valid dog names.

calculates_results_stats.py: Computes absolute counts and percentage accuracies for breed matches and dog/not-dog classifications.

Usage
To run the classification pipeline, execute check_images.py from the terminal. The script utilizes argparse to accept three optional arguments.

Arguments:

--dir: Path to the folder of pet images (default: pet_images/)

--arch: CNN model architecture to use (vgg, resnet, or alexnet - default: vgg)

--dogfile: Text file containing the list of valid dog names (default: dognames.txt)

Example Command:

Bash
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
Results & Evaluation
The pipeline was tested against a dataset of 40 images (30 dog images, 10 non-dog images). Below are the performance metrics utilizing the VGG architecture.

Summary Statistics:

Total Images Processed: 40

Dog Detection Accuracy: 100.0%

Non-Dog Detection Accuracy: 100.0%

Breed Identification Accuracy: 93.3%

Total Elapsed Runtime: 23 seconds

Key Insights:
The VGG model proved highly capable, successfully filtering out all 10 non-dog images (including cats, geckos, polar bears, and squirrels) with 100% accuracy.

The breed classification accuracy of 93.3% highlights a known limitation with CNNs: distinguishing between closely related breeds. The model encountered two misclassifications due to visual similarities:

A Great Pyrenees was misclassified as a Kuvasz.

A Beagle was misclassified as a Walker Hound.

Despite these edge cases, VGG demonstrated exceptional reliability for the primary goal of the dog show registration system.
