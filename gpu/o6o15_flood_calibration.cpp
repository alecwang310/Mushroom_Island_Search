#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../engine/continentalness.h"

struct IslandRecord {
    int64_t seed;
    int area;
    int cx;
    int cz;
};

static int64_t read_integer(const std::string& line, const char* key) {
    const std::string token = std::string("\"") + key + "\":";
    size_t start = line.find(token);
    if (start == std::string::npos)
        throw std::runtime_error(std::string("missing field ") + key);
    start += token.size();
    while (start < line.size() && line[start] == ' ') start++;
    size_t end = start;
    if (end < line.size() && (line[end] == '-' || line[end] == '+')) end++;
    while (end < line.size() && line[end] >= '0' && line[end] <= '9') end++;
    return std::stoll(line.substr(start, end - start));
}

static std::vector<IslandRecord> load_records(const char* path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error(std::string("cannot open ") + path);
    std::vector<IslandRecord> records;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        records.push_back({
            read_integer(line, "seed"),
            static_cast<int>(read_integer(line, "area")),
            static_cast<int>(read_integer(line, "cx")),
            static_cast<int>(read_integer(line, "cz")),
        });
    }
    return records;
}

static void append_even_sample(std::vector<IslandRecord>& output,
                               const std::vector<IslandRecord>& source,
                               size_t count) {
    if (source.size() <= count) {
        output.insert(output.end(), source.begin(), source.end());
        return;
    }
    for (size_t index = 0; index < count; index++) {
        size_t source_index = index * source.size() / count;
        output.push_back(source[source_index]);
    }
}

static std::vector<IslandRecord> make_sample(
        const std::vector<IslandRecord>& records, size_t per_lower_band) {
    std::vector<IslandRecord> band_4m;
    std::vector<IslandRecord> band_45m;
    std::vector<IslandRecord> band_5m;
    std::vector<IslandRecord> band_55m;
    std::vector<IslandRecord> band_6m;
    for (const auto& record : records) {
        if (record.area >= 6'000'000) band_6m.push_back(record);
        else if (record.area >= 5'500'000) band_55m.push_back(record);
        else if (record.area >= 5'000'000) band_5m.push_back(record);
        else if (record.area >= 4'500'000) band_45m.push_back(record);
        else band_4m.push_back(record);
    }
    std::vector<IslandRecord> sample;
    append_even_sample(sample, band_4m, per_lower_band);
    append_even_sample(sample, band_45m, per_lower_band);
    append_even_sample(sample, band_5m, per_lower_band);
    append_even_sample(sample, band_55m, per_lower_band);
    sample.insert(sample.end(), band_6m.begin(), band_6m.end());
    return sample;
}

static std::vector<double> thresholds() {
    return {-0.95, -0.925, -0.91, -0.90, -0.89,
            -0.88, -0.875, -0.87, -0.86, -0.85};
}

static double pearson(const std::vector<IslandRecord>& records,
                      const std::vector<int64_t>& estimates,
                      size_t threshold_index) {
    size_t count = records.size();
    size_t offset = threshold_index * count;
    long double sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0;
    for (size_t index = 0; index < count; index++) {
        long double x = records[index].area;
        long double y = estimates[offset + index];
        sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
    }
    long double numerator = count * sxy - sx * sy;
    long double left = count * sxx - sx * sx;
    long double right = count * syy - sy * sy;
    if (left <= 0.0 || right <= 0.0) return 0.0;
    return static_cast<double>(numerator / std::sqrt(left * right));
}

static int64_t median(std::vector<int64_t> values) {
    if (values.empty()) return 0;
    size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    return values[middle];
}

int main(int argc, char** argv) {
    const char* input_path = argc > 1 ? argv[1] : "gpu/islands_4m.jsonl";
    const char* output_path = argc > 2
        ? argv[2] : "gpu/o6o15_flood_calibration.csv";
    const int workers = argc > 3 ? std::stoi(argv[3]) : 28;
    const size_t per_lower_band = argc > 4
        ? static_cast<size_t>(std::stoul(argv[4])) : 250;
    const int max_cells = argc > 5 ? std::stoi(argv[5]) : 1'000'000;
    if (workers <= 0 || per_lower_band == 0 || max_cells <= 0)
        return 2;

    try {
        const auto all_records = load_records(input_path);
        const auto records = make_sample(all_records, per_lower_band);
        const auto levels = thresholds();
        const size_t record_count = records.size();
        std::vector<int64_t> estimates(record_count * levels.size(), 0);
        std::atomic<size_t> next_job{0};
        const size_t job_count = record_count * levels.size();

        auto worker = [&]() {
            while (true) {
                size_t job = next_job.fetch_add(1);
                if (job >= job_count) break;
                size_t threshold_index = job / record_count;
                size_t record_index = job % record_count;
                const auto& record = records[record_index];
                estimates[job] = cont_flood_fill_2oct(
                    static_cast<uint64_t>(record.seed),
                    record.cx, record.cz, levels[threshold_index], max_cells);
            }
        };

        std::vector<std::thread> threads;
        for (int index = 0; index < workers; index++)
            threads.emplace_back(worker);
        for (auto& thread : threads) thread.join();

        std::ofstream output(output_path);
        if (!output)
            throw std::runtime_error(std::string("cannot write ") + output_path);
        output << "threshold,actual_min,actual_count,estimate_passes,"
                  "true_positives,retention_pct,below_passes,below_pass_pct,"
                  "precision_pct,median_estimate,median_ratio,pearson_r,"
                  "capped_count\n";
        output << std::fixed << std::setprecision(6);
        const int area_minima[] = {
            4'000'000, 4'500'000, 5'000'000, 5'500'000, 6'000'000,
        };

        for (size_t threshold_index = 0;
             threshold_index < levels.size(); threshold_index++) {
            size_t offset = threshold_index * record_count;
            double correlation = pearson(records, estimates, threshold_index);
            int capped_count = 0;
            for (size_t index = 0; index < record_count; index++)
                capped_count += estimates[offset + index] >= max_cells * 16LL;
            for (int area_minimum : area_minima) {
                int actual_count = 0;
                int estimate_passes = 0;
                int true_positives = 0;
                int below_passes = 0;
                std::vector<int64_t> matching_estimates;
                std::vector<int64_t> ratios;
                for (size_t index = 0; index < record_count; index++) {
                    bool actual = records[index].area >= area_minimum;
                    bool estimated = estimates[offset + index] >= area_minimum;
                    actual_count += actual;
                    estimate_passes += estimated;
                    true_positives += actual && estimated;
                    below_passes += !actual && estimated;
                    if (actual) {
                        matching_estimates.push_back(estimates[offset + index]);
                        ratios.push_back(estimates[offset + index] * 1'000'000LL
                            / records[index].area);
                    }
                }
                int below_count = static_cast<int>(record_count) - actual_count;
                double retention = actual_count > 0
                    ? 100.0 * true_positives / actual_count : 0.0;
                double below_rate = below_count > 0
                    ? 100.0 * below_passes / below_count : 0.0;
                double precision = estimate_passes > 0
                    ? 100.0 * true_positives / estimate_passes : 0.0;
                output << levels[threshold_index] << ',' << area_minimum << ','
                       << actual_count << ',' << estimate_passes << ','
                       << true_positives << ',' << retention << ','
                       << below_passes << ',' << below_rate << ','
                       << precision << ',' << median(matching_estimates) << ','
                       << median(ratios) / 1'000'000.0 << ','
                       << correlation << ',' << capped_count << '\n';
            }
        }

        std::cout << "all_records=" << all_records.size()
                  << " sampled_records=" << record_count
                  << " thresholds=" << levels.size()
                  << " workers=" << workers
                  << " max_cells=" << max_cells << '\n';
        std::cout << "csv=" << output_path << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
