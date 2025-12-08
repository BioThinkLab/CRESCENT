import os
import glob
import pandas as pd


#    input_dir = sorted/{cancer_type}"  # 存储各染色体数据文件的目录
#    output_dir = bin_with_case/{cancer_type}"  # 处理结果输出目录


class Processor:
    def __init__(self, input_dir, output_dir):
        """
        初始化时设置输入与输出目录，便于后续修改。
        """
        self.input_dir = input_dir
        self.output_dir = output_dir

    def process_file(self, file_path):
        """
        处理单个 TSV 文件：
         1. 读取数据并过滤 is_arm_level 为 True 的行；
         2. 取出 Start 与 End 坐标生成 bin；
         3. 对每个 bin，遍历所有分段：如果该分段覆盖了当前 bin，
            则将该行的 Copy_Number 累加到其 GDC_Aliquot 对应的列上；
         4. 计算每个 bin 的总和，并根据全局各 GDC_Aliquot 总和对列排序；
         5. 返回处理后的 DataFrame。
        """
        try:
            # 读取 TSV 文件（制表符分隔）
            df = pd.read_csv(file_path, sep='\t')
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

        # 检查必须的列是否存在
        required_columns = ['GDC_Aliquot', 'Chromosome', 'Start', 'End', 'Copy_Number', 'is_arm_level']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error in file {file_path}: 缺少必要的列: {', '.join(missing_columns)}")
            return None

        try:
            # 删除 is_arm_level 为 True 的行
            df = df[df['is_arm_level'] != True]
        except Exception as e:
            print(f"Error filtering is_arm_level in file {file_path}: {e}")
            return None

        try:
            # 取出所有 Start 与 End 坐标，去重后排序
            points = pd.unique(df[['Start', 'End']].values.ravel())
            points = sorted(points)
        except Exception as e:
            print(f"Error processing Start/End columns in file {file_path}: {e}")
            return None

        if len(points) < 2:
            print(f"Warning: 文件 {file_path} 中的 Start 和 End 坐标不足以生成 bin")
            return None

        try:
            # 生成相邻两个点构成的 bin 区间
            bins = [(points[i], points[i+1]) for i in range(len(points) - 1)]
        except Exception as e:
            print(f"Error generating bins for file {file_path}: {e}")
            return None

        results = []
        # 用于统计当前文件内各个 GDC_Aliquot 的全局累计 Copy_Number
        global_cn_totals = {}

        for bin_start, bin_end in bins:
            # 初始化当前 bin 的结果字典
            bin_dict = {'start': bin_start, 'end': bin_end}
            # 如果文件中存在 Chromosome 列，则取第一行的值作为当前 bin 的染色体标识
            if 'Chromosome' in df.columns:
                bin_dict['chr'] = df.iloc[0]['Chromosome']

            # 遍历该文件内每个分段（行）
            for index, row in df.iterrows():
                try:
                    seg_start = row['Start']
                    seg_end = row['End']
                    # 若当前分段覆盖该 bin（包含左右端点）
                    if seg_start <= bin_start and seg_end >= bin_end:
                        aliquot = row['GDC_Aliquot']
                        n = row['Copy_Number']
                        bin_dict[aliquot] = bin_dict.get(aliquot, 0) + n
                        global_cn_totals[aliquot] = global_cn_totals.get(aliquot, 0) + n
                except Exception as e:
                    print(f"Error processing row {index} in file {file_path}: {e}")
                    continue

            try:
                # 计算当前 bin 中所有 GDC_Aliquot 的累加和
                bin_dict['sum'] = sum(value for key, value in bin_dict.items() if key not in ['start', 'end', 'chr'])
            except Exception as e:
                print(f"Error calculating sum for bin {bin_start}-{bin_end} in file {file_path}: {e}")
                bin_dict['sum'] = None
            results.append(bin_dict)

        try:
            result_df = pd.DataFrame(results)
        except Exception as e:
            print(f"Error creating DataFrame for file {file_path}: {e}")
            return None

        # 获取所有出现过的 GDC_Aliquot 列（不包括 start, end, chr, sum）
        aliquot_ids = list(global_cn_totals.keys())
        # 根据全局累计总和降序排列（总和大的放在左侧）
        aliquot_ids.sort(key=lambda x: global_cn_totals[x], reverse=True)

        try:
            # 补全缺失的 GDC_Aliquot 列（若某个 bin 没有该值，则填 0）
            for aliquot in aliquot_ids:
                if aliquot not in result_df.columns:
                    result_df[aliquot] = 0
                else:
                    result_df[aliquot] = result_df[aliquot].fillna(0)

            # 确定最终输出的列顺序：start, end, (chr), 各个 GDC_Aliquot 列，再 sum 列
            cols = ['start', 'end']
            if 'chr' in result_df.columns:
                cols.append('chr')
            cols = cols + aliquot_ids + ['sum']
            result_df = result_df[cols]

            # 按 bin 起点排序
            result_df = result_df.sort_values(by='start')
        except Exception as e:
            print(f"Error organizing DataFrame columns for file {file_path}: {e}")
            return None

        return result_df

    def process_all_files(self):
        """
        自动遍历输入目录下所有 TSV 文件，对每个文件调用 process_file 进行处理，
        并将结果输出到指定的输出目录中（文件名保持不变）。
        """
        if not os.path.isdir(self.input_dir):
            print(f"Error: 输入目录 {self.input_dir} 不存在或不是一个目录")
            return

        os.makedirs(self.output_dir, exist_ok=True)

        file_list = glob.glob(os.path.join(self.input_dir, "*"))
        if not file_list:
            print(f"Warning: 在输入目录 {self.input_dir} 中没有找到 TSV 文件")
            return

        for file_path in file_list:
            print(f"Processing file: {file_path}")
            result_df = self.process_file(file_path)
            if result_df is None:
                print(f"Skipping file {file_path} due to errors.")
                continue
            try:
                base_name = os.path.basename(file_path)
                output_file = os.path.join(self.output_dir, base_name)
                result_df.to_csv(output_file, sep='\t', index=False)
                print(f"Processed {file_path} -> {output_file}")
            except Exception as e:
                print(f"Error writing output for file {file_path}: {e}")


if __name__ == '__main__':
    cancer_type = "BRCA"
    # 在这里设置输入和输出目录，方便日后修改
    input_dir = f"/Users/sanjati/jangoTemp/temp2/pycharmD/preprocess/Version0209/output/sorted/{cancer_type}"  # 存储各染色体数据文件的目录
    output_dir = f"/Users/sanjati/jangoTemp/temp2/pycharmD/preprocess/Version0209/output/bin_with_case/{cancer_type}"  # 处理结果输出目录

    processor = Processor(input_dir, output_dir)
    processor.process_all_files()
