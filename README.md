# MINAS-BR

This project implements the MINAS-BR algorithm and related experiments for multi-label data stream classification.

## Prerequisites

- **Java JDK 8 or higher**  
  Make sure `java` and `javac` are available in your system PATH.

- **Maven**  
  Used for dependency management and building the project.  
  Download and install from [https://maven.apache.org/download.cgi](https://maven.apache.org/download.cgi).

- **Git**  
  To clone the repository.

## Step-by-Step Guide

### 1. Clone the Repository

```sh
git clone https://github.com/CostaJoel/MINAS-BR-v2.git
cd MINAS-BR-v2
```

### 2. Install Dependencies and Build the Project

Use Maven to download dependencies and build the project.  
This will also generate an executable JAR file.

```sh
mvn clean package
```

After building, the JAR file will be located in the `target/` directory.  
If you configured the Maven Assembly Plugin, the file will be named like:

```
target/MINAS-BR-v2-jar-with-dependencies.jar
```

### 3. Prepare the Dataset

Place your `.arff` dataset files in the `resources/datasets/` directory.  
Example files are already provided in this folder.

### 4. Run the Application

Use the following command to run the JAR file, replacing the parameters as needed:

```sh
java -jar target/MINAS-BR-v2-jar-with-dependencies.jar <tipoAlgoritmo> <dataSetName> <trainPath> <testPath> <outputDirectory> <k_ini> <omega> <L>
```

#### Parameters:

- `<tipoAlgoritmo>`: Algorithm type (`minas-br`, `upperbounds`, or `lowerbounds`)
- `<dataSetName>`: Name of the dataset (e.g., `MOA-5C-7C-2D`)
- `<trainPath>`: Path to the training dataset (e.g., `resources/datasets/MOA-5C-7C-2D-train.arff`)
- `<testPath>`: Path to the test dataset (e.g., `resources/datasets/MOA-5C-7C-2D-test.arff`)
- `<outputDirectory>`: Directory for output results (e.g., `resources/results_output`)
- `<k_ini>`: Initial micro-cluster percentage (e.g., `0.05`)
- `<omega>`: Window size (e.g., `2000`)
- `<L>`: Number of classes (e.g., `7`)

#### Example:

```sh
java -jar target/MINAS-BR-v2-jar-with-dependencies.jar minas-br MOA-5C-7C-2D resources/datasets/MOA-5C-7C-2D-train.arff resources/datasets/MOA-5C-7C-2D-test.arff resources/results_output 0.05 2000 7
```

### 5. Output

Results and logs will be saved in the specified output directory.

---

## Troubleshooting

- **ClassNotFoundException**:  
  Make sure you are using the `-jar-with-dependencies.jar` file, which includes all required libraries.

- **File Not Found**:  
  Ensure the dataset paths are correct and files exist.

- **Permission Issues**:  
  Run the terminal as administrator if you encounter permission errors.

---

## Project Structure

- `src/`: Java source code
- `resources/datasets/`: Example datasets
- `target/`: Compiled classes and generated JARs (after build)
- `libs/`: External JARs (if any)

---

## License

MIT License (or specify your license here)
