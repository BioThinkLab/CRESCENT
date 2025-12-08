## Environment
This project is developed and tested with **Python 3.8**, main dependencies of this project are listed in `requirements.txt`.

>Please note that the configurations of `torch`, `CUDA`, and other related environments may vary depending on your hardware and system.

We recommend creating a virtual environment using conda:
```bash
conda create -n your_env_name python=3.8
conda activate your_env_name
pip install -r requirements.txt
```
or using pip 
```bash
pip install -r requirements.txt
```


## Input File Format
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

The input data can be provided across multiple files. The program will automatically merge **all input files within the target directory** and perform downstream analysis on the combined dataset.

## Conducting Analysis
* Create a new directory (with any name you like, suppose it's "EXAMPLE") under `Data/input` to store the files used for a single analysis. 
* change parameters in the head of `run.py` , major parameters are in the following.
  * `project_name`: same as the name of the directory you put input file.
  * `mutation_type`: the mutation type you want to analyse, can only be `"amp"` or `"del"`, which means amplification and deletion.
  * `classification_threshold`:The decision threshold used to convert predicted probabilities into binary class labels, default=0.5.

* after changing parameter above, run `run.py` and it will go automatically. The final result will be in `/Main_pack/run/result`

 
## Output File

| Chromosome | Start    | End      |
|------------|----------|----------|
| chr1       | 1222897  | 1252207  |
| chr1       | 1398056  | 1430190  |
| chr1       | 39729231 | 39759523 |
| chr1       | 39759570 | 39831595 |

The final output is a tsv file, each row is a potential CNV segment, the 3 columns contains basic information of each segment, including the chromosome it belongs to and the start, end point of it.

If you need more detailed result for further manual analysis, the confidence data are saved in `/Main_pack/run/bin_with_case_amp/<name_of_input_directory>`or`/Main_pack/run/bin_with_case_del/<name_of_input_directory>` depending on mutation type, the confidence of each bin is saved in the last column "prob", which means the confidence of this bin being contained in a CNV segments. You can visualization the confidence data and the heatmap of copy number across a single chromosome using `/Main_pack/visualization/dataset_check.py`.


 ## C++ Compilation (Auto & Manual)

This project includes an automatic C++ compilation step.  
When running the main script, it will attempt to automatically compile the required C++ source files using `g++` or `clang++`.

However, **automatic compilation may fail** under some circumstances, such as:
- No available C++ compiler is installed
- Compiler is not in the system PATH
- Incompatible compiler version
- Platform-specific issues (e.g., Windows environment)

If automatic compilation fails, please compile the C++ program manually, the source code is in `/Main_pack/run/src`, please make sure the excutable file is in `/Main_pack/run/build` and named `processor_amp` and `processor_del` example compile command are as follows:

#### For AMP:
```bash
g++ -std=gnu++17 -O3 -Wall -Wextra -Wno-unused-parameter -pthread \
    src/gen_bin_amp_cpp.cpp -o build/processor_amp
```
#### For DEL:
```bash
g++ -std=gnu++17 -O3 -Wall -Wextra -Wno-unused-parameter -pthread \
    src/gen_bin_del_cpp.cpp -o build/processor_del
```
