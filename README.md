# 1. Environment
## 1.1 Python dependencies

This project is developed and tested with **Python 3.8**, main dependencies of this project are listed in `requirements.txt`. and can be installed with:

```pip install -r requirements.txt```


## 1.2 PyTorch and CUDA

PyTorch-related packages are not included in requirements.txt, as the appropriate PyTorch and CUDA configuration depends on the user's local hardware, operating system, GPU driver, and CUDA environment.

Please install a compatible version of PyTorch separately according to your local environment. Refer to the official PyTorch installation guide for the recommended installation command.

For reference, the project was developed and tested on an NVIDIA RTX 40-series GPU using Python 3.8, PyTorch 2.4.1, and CUDA 12.1. The corresponding installation command is:

python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

This configuration is provided as a reference only; please select the appropriate PyTorch build based on your local situation including hardware and operating system.

## 1.3 C++ compiler

A working C++ compiler is also required to build native extensions used by the project. Supported compilers include:

* GCC / G++ on Linux or Windows
* Clang / Clang++ on macOS

The compiler is not installed automatically by this project and should be configured according to your operating system. Please refer to the relevant compiler documentation or the PyTorch C++ extension documentation for further details.


This project includes an automatic C++ compilation step.  
When running the main script, it will attempt to automatically compile the required C++ source files using `g++` or `clang++`.

However, **automatic compilation may fail** under some circumstances, such as:
- No available C++ compiler is installed
- Compiler is not in the system PATH
- Platform-specific issues 

If automatic compilation fails, please compile the C++ program **manually** as follows:

#### For AMP mode:
```bash
g++ -std=gnu++17 -O3 -Wall -Wextra -Wno-unused-parameter -pthread \
    src/gen_bin_amp_cpp.cpp -o build/processor_amp
```
#### For DEL mode:
```bash
g++ -std=gnu++17 -O3 -Wall -Wextra -Wno-unused-parameter -pthread \
    src/gen_bin_del_cpp.cpp -o build/processor_del
```

# 2.Input File Format
The input data of both following format can be provided across multiple files. The program will automatically merge **all input files within the target directory** and perform downstream analysis on the combined dataset.

## 2.1 Integer copy number format
The input file should be a tab-separated values (TSV) file with the following columns:

```tsv
GDC_Aliquot	Chromosome	Start	End	Copy_Number	Major_Copy_Number	Minor_Copy_Number
0aeed974-9ad8-4208-9c09-43418386665f	chr1	61735	3754463	4	3	1
0aeed974-9ad8-4208-9c09-43418386665f	chr1	3760034	86902042	3	2	1
0aeed974-9ad8-4208-9c09-43418386665f	chr1	86902552	93174578	5	4	1
0aeed974-9ad8-4208-9c09-43418386665f	chr1	93175263	248930189	4	3	1
```
>The example input shown above is taken from the first few lines of the TCGA file  
`TCGA-BLCA.0aeed974-9ad8-4208-9c09-43418386665f.ascat3.allelic_specific.seg.txt`.

Make sure there are columns named `Chromosome`, `Start`, `End`, `Copy_Number`,The first column can have other name, as long as it is used to mark different case.

### 2.2 Log2ratio format copy number
```tsv
Sample	Chromosome	Start	End	Num_Probes	Segment_Mean
S0001	1	3218610	5838773	289	0.408563
S0001	1	5844802	9012691	360	0.845405
S0001	1	9012737	28712545	1834	0.838700
S0001	1	28716679	33008483	523	2.682358
S0001	1	33019130	33032412	3	1.244822
S0001	1	33045735	34651385	191	2.682358
S0001	1	34653822	40301092	657	3.974571
```
In this format, `Segment_Mean` represent copy number value, with its meaning of Segment_Mean = log2(Tumor signal / Normal reference). CRESCENT can also handle this format, make sure the column name is exactly `Segment_Mean`, `Num_Probes`in the distance is not necessary.



# 3.Conducting Analysis

1. Create one folder per analysis under `Main_pack/Data/input`, for example `Main_pack/Data/input/EXAMPLE`.
2. Edit the control panel in `Main_pack/run/run.py`:
   - `PROJECT_NAMES`: project folder names, such as `["EXAMPLE"]` or `["PROJECT_A", "PROJECT_B"]`.
   - `MUTATION_TYPE`: `"amp"` or `"del"`.
   - `CHROMOSOME_NUMBERS`: leave empty for all chromosomes, or use a subset such as `[1, 8, 12]`.
   - `ONLY_USE_FOCAL`: whether to remove arm-level events.
   - `CLASSIFICATION_THRESHOLD`: probability threshold for the final segments.
   - `USE_CHUNKING`: enable bounded temporary storage; when disabled, low disk space produces a warning but does not stop execution.
   - `SAMPLES_PER_CHUNK`: maximum temporary samples stored at once; reduce it when disk space is limited.
3. Run the control panel from any working directory:

```bash
python /path/to/CRESCENT/Main_pack/run/run.py
```

Results are written under `Main_pack/run/result/<project>/<mutation_type>`.

The implementation is in `Main_pack/run/pipeline.py`; normal users only need to edit the control panel.

4. Visualization through /CRESCENT/visualization/dataset_check.py


