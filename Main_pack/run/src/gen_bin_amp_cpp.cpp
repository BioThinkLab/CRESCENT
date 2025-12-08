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
    std::unordered_map<std::string, double> aliquotValues;
    double sum;
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
    // Core logic for a single chromosome:
    // given records from one chromosome, do binning & aggregation
    // (logic same as old process_merged_records, but restricted to one chr)
    // =========================
    std::vector<BinResult> process_merged_records(
        const std::vector<Record>& all_records_for_chr,
        std::vector<std::string>& out_sorted_aliquotIds
    ) {
        std::vector<Record> filtered;
        filtered.reserve(all_records_for_chr.size());

        // Control whether to apply is_arm_level filtering
        for (const auto& rec : all_records_for_chr) {
            if (!use_arm_filter || !rec.is_arm_level) {
                filtered.push_back(rec);
            }
        }

        if (filtered.empty()) {
            std::cerr << "Warning: no records remain after filtering for chromosome "
                      << (all_records_for_chr.empty() ? "UNKNOWN" : all_records_for_chr.front().Chromosome)
                      << ". (Possibly all segments are arm-level.)" << std::endl;
            return {};
        }

        // Collect breakpoints for this chromosome
        std::set<long long> pointSet;
        for (const auto& rec : filtered) {
            pointSet.insert(rec.Start);
            pointSet.insert(rec.End);
        }

        if (pointSet.size() < 2) {
            std::cerr << "Warning: not enough distinct coordinates to build bins for chromosome "
                      << filtered.front().Chromosome << "." << std::endl;
            return {};
        }

        std::vector<long long> points(pointSet.begin(), pointSet.end());
        std::sort(points.begin(), points.end());

        // Build bins
        std::vector<std::pair<long long,long long>> bins;
        bins.reserve(points.size() - 1);
        for (size_t i = 0; i + 1 < points.size(); ++i) {
            bins.emplace_back(points[i], points[i+1]);
        }

        // All records here should be from the same chromosome
        std::string chrValue = filtered.front().Chromosome;
        std::vector<BinResult> results;
        results.reserve(bins.size());
        std::unordered_map<std::string, double> global_cn;

        for (const auto& b : bins) {
            BinResult binRes;
            binRes.start = b.first;
            binRes.end   = b.second;
            binRes.chr   = chrValue;
            binRes.sum   = 0.0;

            for (const auto& rec : filtered) {
                if (rec.Start <= b.first && rec.End >= b.second) {
                    binRes.aliquotValues[rec.GDC_Aliquot] += rec.Copy_Number;
                    global_cn[rec.GDC_Aliquot]            += rec.Copy_Number;
                }
            }

            for (const auto& p : binRes.aliquotValues)
                binRes.sum += p.second;

            results.push_back(std::move(binRes));
        }

        if (global_cn.empty()) {
            std::cerr << "Warning: no aliquot values accumulated for chromosome "
                      << chrValue << ". Bin results will be empty." << std::endl;
            return {};
        }

        // Sort aliquots by their global accumulated CN (within this chromosome)
        std::vector<std::pair<std::string,double>> vec(global_cn.begin(), global_cn.end());
        std::sort(vec.begin(), vec.end(),
                  [](auto& a, auto& b){ return a.second > b.second; });

        out_sorted_aliquotIds.clear();
        out_sorted_aliquotIds.reserve(vec.size());
        for (auto& p : vec) out_sorted_aliquotIds.push_back(p.first);

        // Fill missing aliquots with 0
        for (auto& binRes : results) {
            for (const auto& id : out_sorted_aliquotIds) {
                if (!binRes.aliquotValues.count(id))
                    binRes.aliquotValues[id] = 0.0;
            }
        }

        return results;
    }

    // =========================
    // Process directory and write per-chromosome merged files
    // =========================
    void process_directory_merged() {
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

        // === 新增：按 Chromosome 分组 ===
        // === 按 Chromosome 分组，并跳过性染色体 ===
        std::map<std::string, std::vector<Record>> chr_groups;
        for (const auto& rec : all_records) {

            // 统一转小写，防止 "chrX" / "chrx" 混乱
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

        // 对每条染色体单独做旧版逻辑，并各自产生一个输出文件
        for (const auto& kv : chr_groups) {
            const std::string& chr = kv.first;
            const std::vector<Record>& records_for_chr = kv.second;

            std::cout << "Processing chromosome: " << chr
                      << " with " << records_for_chr.size() << " records." << std::endl;

            std::vector<std::string> sorted_aliquotIds_chr;
            auto results = process_merged_records(records_for_chr, sorted_aliquotIds_chr);
            if (results.empty()) {
                std::cerr << "Warning: no bin results generated for chromosome "
                          << chr << ". Nothing will be written for this chromosome."
                          << std::endl;
                continue;
            }

            // 输出文件名：merged_<Chromosome>.tsv
            fs::path out = fs::path(output_dir) / ("cnv_" + chr + ".tsv");
            std::ofstream ofs(out);
            if (!ofs.is_open()) {
                std::cerr << "Error: cannot open output file for writing: " << out << std::endl;
                continue;
            }

            // Header
            ofs << "start\tend\tchr";
            for (auto& id : sorted_aliquotIds_chr) ofs << "\t" << id;
            ofs << "\tsum\n";

            // Body; replace 0 with 2 as in original logic
            for (auto& r : results) {
                ofs << r.start << "\t" << r.end << "\t" << r.chr;
                for (auto& id : sorted_aliquotIds_chr) {
                    double v = r.aliquotValues[id];
                    ofs << "\t" << (v == 0.0 ? 2.0 : v);
                }
                ofs << "\t" << (r.sum == 0.0 ? 2.0 : r.sum) << "\n";
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
    processor.process_directory_merged();
    return 0;
}
