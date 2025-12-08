#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <filesystem>
#include <set>

namespace fs = std::filesystem;

// ===== Data structures =====
struct Record {
    std::string GDC_Aliquot;
    std::string Chromosome;
    long long Start;
    long long End;
    double Copy_Number;
    bool is_arm_level = false;   // default false, in case the column is missing
};

struct BinResult {
    long long start;
    long long end;
    std::string chr;
    // 这里存的是“当前 n 值”，即从 2 开始减去缺失
    std::unordered_map<std::string, double> aliquotValues;
};

class Processor {
public:
    Processor(const std::string& inputDir,
              const std::string& outputDir,
              bool useArmFilter)
        : input_dir(inputDir),
          output_dir(outputDir),
          use_arm_filter(useArmFilter) {}

    // =========================
    // Read one file and append records
    // =========================
    bool read_file_append(const fs::path& file_path,
                          std::vector<std::string> required,
                          std::vector<Record>& all_records,
                          std::unordered_map<std::string, int>& colIndex_ref,
                          bool& header_initialized)
    {
        std::ifstream infile(file_path);
        if (!infile.is_open()) {
            std::cerr << "Error: cannot open file " << file_path << std::endl;
            return false;
        }

        std::string headerLine;
        if (!std::getline(infile, headerLine)) {
            std::cerr << "Error: cannot read header from " << file_path << std::endl;
            return false;
        }
        std::vector<std::string> headers = split(headerLine, '\t');

        // If arm-level filtering is disabled, "is_arm_level" is no longer required
        if (!use_arm_filter) {
            required.erase(
                std::remove(required.begin(), required.end(), "is_arm_level"),
                required.end()
            );
        }

        // Initialize header mapping (only once, assume all files share the same header)
        if (!header_initialized) {
            for (size_t i = 0; i < headers.size(); ++i) {
                colIndex_ref[headers[i]] = static_cast<int>(i);
            }

            for (const auto& col : required) {
                if (colIndex_ref.find(col) == colIndex_ref.end()) {
                    std::cerr << "Error in file " << file_path
                              << ": missing required column: " << col << std::endl;
                    return false;
                }
            }
            header_initialized = true;
        }

        // Read data lines
        std::string line;
        while (std::getline(infile, line)) {
            if (line.empty()) continue;
            auto tokens = split(line, '\t');
            if (tokens.size() < colIndex_ref.size()) continue;

            Record rec;
            rec.GDC_Aliquot = tokens[colIndex_ref["GDC_Aliquot"]];
            rec.Chromosome  = tokens[colIndex_ref["Chromosome"]];

            try {
                rec.Start       = std::stoll(tokens[colIndex_ref["Start"]]);
                rec.End         = std::stoll(tokens[colIndex_ref["End"]]);
                rec.Copy_Number = std::stod(tokens[colIndex_ref["Copy_Number"]]);
            } catch (...) {
                std::cerr << "Warning: numeric conversion error in file "
                          << file_path << ", line skipped." << std::endl;
                continue;
            }

            // Only read "is_arm_level" when filtering is enabled and column exists
            if (use_arm_filter && colIndex_ref.count("is_arm_level")) {
                const std::string& isArmStr = tokens[colIndex_ref["is_arm_level"]];
                rec.is_arm_level = (isArmStr == "True" || isArmStr == "true");
            }

            all_records.push_back(std::move(rec));
        }

        return true;
    }

    // =========================
    // Core logic for a single chromosome (DEL 专用):
    // 1. 只用 Copy_Number < 2 的片段生成 bin 边界
    // 2. 所有样本初始 n = 2.0
    // 3. 若某 DEL 片段完全覆盖 bin (Start <= bin.start && End > bin.end)
    //    则 n -= (2 - Copy_Number)
    // =========================
    std::vector<BinResult> process_merged_records_del(
        const std::vector<Record>& records_for_chr,
        std::vector<std::string>& out_sorted_aliquotIds
    ) {
        if (records_for_chr.empty()) {
            return {};
        }

        // 1. arm-level 过滤
        std::vector<Record> filtered;
        filtered.reserve(records_for_chr.size());
        for (const auto& rec : records_for_chr) {
            if (!use_arm_filter || !rec.is_arm_level) {
                filtered.push_back(rec);
            }
        }

        if (filtered.empty()) {
            std::cerr << "Warning: no records remain after filtering for chromosome "
                      << records_for_chr.front().Chromosome
                      << ". (Possibly all segments are arm-level.)" << std::endl;
            return {};
        }

        // 2. 收集样本列表（这里用“所有读到的样本”，包括可能只有 arm-level 的，
        //    这样他们在所有 bin 中 n 一直是 2）
        std::set<std::string> aliquot_set;
        for (const auto& rec : records_for_chr) {
            aliquot_set.insert(rec.GDC_Aliquot);
        }
        std::vector<std::string> aliquot_list(aliquot_set.begin(), aliquot_set.end());

        if (aliquot_list.empty()) {
            std::cerr << "Warning: no aliquots found for chromosome "
                      << records_for_chr.front().Chromosome << std::endl;
            return {};
        }

        // 3. 用 Copy_Number < 2 的片段生成边界点
        std::set<long long> pointSet;
        for (const auto& rec : filtered) {
            if (rec.Copy_Number < 2.0) {
                pointSet.insert(rec.Start);
                pointSet.insert(rec.End);
            }
        }

        if (pointSet.size() < 2) {
            std::cerr << "Warning: not enough DEL breakpoints to build bins for chromosome "
                      << filtered.front().Chromosome
                      << " (need at least 2 distinct coordinates from CN<2 segments)." << std::endl;
            return {};
        }

        std::vector<long long> points(pointSet.begin(), pointSet.end());
        std::sort(points.begin(), points.end());

        // 4. 构建 bins
        std::vector<std::pair<long long,long long>> bins;
        bins.reserve(points.size() - 1);
        for (size_t i = 0; i + 1 < points.size(); ++i) {
            bins.emplace_back(points[i], points[i+1]);
        }

        std::string chrValue = filtered.front().Chromosome;
        std::vector<BinResult> results;
        results.reserve(bins.size());

        // 用于按照“总缺失量”排序样本（非必须，只是为了美观）
        std::unordered_map<std::string, double> total_loss;  // 越大表示缺失越多

        // 5. 对每个 bin：初始 n=2，再累减
        for (const auto& b : bins) {
            BinResult binRes;
            binRes.start = b.first;
            binRes.end   = b.second;
            binRes.chr   = chrValue;

            // 初始：所有样本 n=2.0
            for (const auto& id : aliquot_list) {
                binRes.aliquotValues[id] = 2.0;
            }

            // DEL 段累减：Copy_Number<2 && 完全覆盖 bin
            for (const auto& rec : filtered) {
                if (rec.Copy_Number < 2.0 &&
                    rec.Start <= b.first &&
                    rec.End   >  b.second)   // 注意这里是 ">"
                {
                    double diff = 2.0 - rec.Copy_Number;  // 可以是 1, 2, 或小数
                    auto& n_ref = binRes.aliquotValues[rec.GDC_Aliquot];
                    n_ref -= diff;

                    // 统计总缺失量，便于后续排序
                    total_loss[rec.GDC_Aliquot] += diff;
                }
            }

            results.push_back(std::move(binRes));
        }

        // 6. 排序样本顺序（按总缺失量从大到小；无缺失的样本排在后面）
        std::vector<std::pair<std::string,double>> vec;
        vec.reserve(aliquot_list.size());
        for (const auto& id : aliquot_list) {
            double loss = 0.0;
            auto it = total_loss.find(id);
            if (it != total_loss.end()) loss = it->second;
            vec.emplace_back(id, loss);
        }

        std::sort(vec.begin(), vec.end(),
                  [](const auto& a, const auto& b){
                      if (a.second != b.second)
                          return a.second > b.second; // 缺失多的在前
                      return a.first < b.first;       // 再按名字排
                  });

        out_sorted_aliquotIds.clear();
        out_sorted_aliquotIds.reserve(vec.size());
        for (auto& p : vec) {
            out_sorted_aliquotIds.push_back(p.first);
        }

        return results;
    }

    // =========================
    // Process directory and write per-chromosome merged files
    // =========================
    void process_directory_merged_del() {
        fs::create_directories(output_dir);

        std::vector<std::string> required = {
            "GDC_Aliquot", "Chromosome", "Start", "End", "Copy_Number", "is_arm_level"
        };

        std::vector<Record> all_records;
        std::unordered_map<std::string,int> colIndex;
        bool header_initialized = false;

        size_t file_count = 0;
        for (const auto& entry : fs::directory_iterator(input_dir)) {
            if (!entry.is_regular_file()) continue;
            auto path = entry.path();
            if (path.extension() != ".tsv" && path.extension() != ".txt") continue;

            std::cout << "Reading file: " << path << std::endl;
            if (!read_file_append(path, required, all_records, colIndex, header_initialized)) {
                std::cerr << "Warning: file skipped due to errors: " << path << std::endl;
                continue;
            }
            ++file_count;
        }

        if (file_count == 0) {
            std::cerr << "Error: no valid TSV/TXT files found in input directory: "
                      << input_dir << std::endl;
            return;
        }

        if (all_records.empty()) {
            std::cerr << "Warning: no valid records were read from input directory: "
                      << input_dir << std::endl;
            return;
        }

        // === 按 Chromosome 分组，并跳过性染色体 ===
        std::map<std::string, std::vector<Record>> chr_groups;
        for (const auto& rec : all_records) {
            std::string chr = rec.Chromosome;
            std::string chr_lower = chr;
            std::transform(chr_lower.begin(), chr_lower.end(), chr_lower.begin(), ::tolower);

            // 跳过性染色体：chrX / chrY
            if (chr_lower == "chrx" || chr_lower == "chry") {
                continue;
            }

            chr_groups[rec.Chromosome].push_back(rec);
        }

        if (chr_groups.empty()) {
            std::cerr << "Warning: no chromosome groups found after reading all records."
                      << std::endl;
            return;
        }

        // 对每条染色体单独做 DEL 逻辑，并各自产生一个输出文件
        for (const auto& kv : chr_groups) {
            const std::string& chr = kv.first;
            const std::vector<Record>& records_for_chr = kv.second;

            std::cout << "Processing chromosome: " << chr
                      << " with " << records_for_chr.size() << " records." << std::endl;

            std::vector<std::string> sorted_aliquotIds_chr;
            auto results = process_merged_records_del(records_for_chr, sorted_aliquotIds_chr);
            if (results.empty()) {
                std::cerr << "Warning: no bin results generated for chromosome "
                          << chr << ". Nothing will be written for this chromosome."
                          << std::endl;
                continue;
            }

            // 输出文件名：cnv_<Chromosome>.tsv
            fs::path out = fs::path(output_dir) / ("cnv_" + chr + ".tsv");
            std::ofstream ofs(out);
            if (!ofs.is_open()) {
                std::cerr << "Error: cannot open output file for writing: " << out << std::endl;
                continue;
            }

            // Header：沿用第一份代码风格
            ofs << "Chromosome\tStart\tEnd";
            for (auto& id : sorted_aliquotIds_chr) ofs << "\t" << id;
            ofs << "\n";

            // Body：每个 bin 的 n 值
            for (auto& r : results) {
                ofs << r.chr << "\t" << r.start << "\t" << r.end;
                for (auto& id : sorted_aliquotIds_chr) {
                    double v = r.aliquotValues.at(id);
                    ofs << "\t" << v;
                }
                ofs << "\n";
            }

            ofs.close();
            std::cout << "Success. Output written to: " << out << std::endl;
        }
    }

private:
    std::string input_dir;
    std::string output_dir;
    bool use_arm_filter;

    static std::vector<std::string> split(const std::string& s, char delim) {
        std::vector<std::string> elems;
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, delim)) elems.push_back(item);
        return elems;
    }
};

// =========================
// main
// =========================
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_dir> <output_dir> <use_arm_filter: 0|1>\n";
        std::cerr << "  Example: " << argv[0]
                  << " ../Data/input/merged_BLCA ./output/bin_with_case_del 1\n";
        return 1;
    }

    std::string input_dir  = argv[1];
    std::string output_dir = argv[2];
    bool use_arm_filter    = std::stoi(argv[3]) != 0;

    std::cout << "Input directory : " << input_dir  << std::endl;
    std::cout << "Output directory: " << output_dir << std::endl;
    std::cout << "Arm-level filtering: "
              << (use_arm_filter ? "ENABLED" : "DISABLED") << std::endl;

    Processor processor(input_dir, output_dir, use_arm_filter);
    processor.process_directory_merged_del();
    return 0;
}
