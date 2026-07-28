#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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

static int hex_distance(int x, int z) {
    return std::max({std::abs(x), std::abs(z), std::abs(-x - z)});
}

static int row_parity(int row) {
    int parity = row % 2;
    return parity < 0 ? parity + 2 : parity;
}

static std::vector<double> make_thresholds() {
    std::vector<double> thresholds;
    for (int index = 0; index <= 80; index++)
        thresholds.push_back(-1.20 + index * 0.01);
    return thresholds;
}

static int connected_cells(const std::vector<double>& values,
                           int radius, double threshold,
                           bool* clipped) {
    const int side = radius * 2 + 1;
    const int cells = side * side;
    std::vector<uint8_t> visited(cells, 0);
    std::vector<int> queue(cells);
    int queue_head = 0;
    int queue_tail = 0;

    for (int x = -1; x <= 1; x++) {
        for (int z = -1; z <= 1; z++) {
            if (hex_distance(x, z) > 1) continue;
            int index = (x + radius) * side + z + radius;
            if (values[index] < threshold && !visited[index]) {
                visited[index] = 1;
                queue[queue_tail++] = index;
            }
        }
    }

    const int neighbor_x[6] = {-1, 1, 0, 0, -1, 1};
    const int neighbor_z[6] = {0, 0, -1, 1, 1, -1};
    int count = 0;
    *clipped = false;
    while (queue_head < queue_tail) {
        int index = queue[queue_head++];
        int x = index / side - radius;
        int z = index % side - radius;
        count++;
        if (hex_distance(x, z) == radius)
            *clipped = true;
        for (int direction = 0; direction < 6; direction++) {
            int next_x = x + neighbor_x[direction];
            int next_z = z + neighbor_z[direction];
            if (hex_distance(next_x, next_z) > radius) continue;
            int next_index = (next_x + radius) * side + next_z + radius;
            if (!visited[next_index] && values[next_index] < threshold) {
                visited[next_index] = 1;
                queue[queue_tail++] = next_index;
            }
        }
    }
    return count;
}

static double pearson(const std::vector<IslandRecord>& records,
                      const std::vector<int64_t>& estimates,
                      size_t threshold_index) {
    const size_t count = records.size();
    const size_t offset = threshold_index * count;
    long double sum_x = 0.0;
    long double sum_y = 0.0;
    long double sum_xx = 0.0;
    long double sum_yy = 0.0;
    long double sum_xy = 0.0;
    for (size_t index = 0; index < count; index++) {
        long double x = records[index].area;
        long double y = estimates[offset + index];
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_yy += y * y;
        sum_xy += x * y;
    }
    long double numerator = count * sum_xy - sum_x * sum_y;
    long double left = count * sum_xx - sum_x * sum_x;
    long double right = count * sum_yy - sum_y * sum_y;
    if (left <= 0.0 || right <= 0.0) return 0.0;
    return static_cast<double>(numerator / std::sqrt(left * right));
}

int main(int argc, char** argv) {
    const char* input_path = argc > 1 ? argv[1] : "gpu/islands_4m.jsonl";
    const char* output_path = argc > 2
        ? argv[2] : "gpu/o6o15_area_calibration.csv";
    const int workers = argc > 3 ? std::stoi(argv[3]) : 28;
    const int step = argc > 4 ? std::stoi(argv[4]) : 250;
    const int radius = argc > 5 ? std::stoi(argv[5]) : 4;
    if (workers <= 0 || step <= 0 || radius < 1)
        return 2;

    try {
        const std::vector<IslandRecord> records = load_records(input_path);
        const std::vector<double> thresholds = make_thresholds();
        const size_t record_count = records.size();
        const size_t threshold_count = thresholds.size();
        std::vector<int64_t> estimates(record_count * threshold_count, 0);
        std::vector<uint8_t> clipped(record_count * threshold_count, 0);
        std::atomic<size_t> next_record{0};
        const int side = radius * 2 + 1;
        const double spacing_z = step * 0.86602540378443864676;
        const double cell_area = static_cast<double>(step) * step
            * 0.86602540378443864676 * 16.0;

        auto worker = [&]() {
            std::vector<double> values(side * side,
                std::numeric_limits<double>::infinity());
            while (true) {
                size_t record_index = next_record.fetch_add(1);
                if (record_index >= record_count) break;
                const IslandRecord& record = records[record_index];
                ContEngine engine;
                cont_engine_init(&engine, static_cast<uint64_t>(record.seed), 0);
                cont_engine_disable_shift(&engine);
                engine.cont_octA_count = 1;
                engine.cont_octB_count = 1;

                std::fill(values.begin(), values.end(),
                          std::numeric_limits<double>::infinity());
                for (int x = -radius; x <= radius; x++) {
                    for (int z = -radius; z <= radius; z++) {
                        if (hex_distance(x, z) > radius) continue;
                        int sample_x = record.cx + x * step
                            + row_parity(z) * (step / 2);
                        int sample_z = record.cz
                            + static_cast<int>(std::lround(z * spacing_z));
                        int index = (x + radius) * side + z + radius;
                        values[index] = cont_sample(&engine, sample_x, sample_z);
                    }
                }

                for (size_t threshold_index = 0;
                     threshold_index < threshold_count; threshold_index++) {
                    bool touches_boundary = false;
                    int cells = connected_cells(
                        values, radius, thresholds[threshold_index],
                        &touches_boundary);
                    size_t output_index = threshold_index * record_count
                        + record_index;
                    estimates[output_index] = static_cast<int64_t>(
                        cells * cell_area);
                    clipped[output_index] = touches_boundary ? 1 : 0;
                }
            }
        };

        std::vector<std::thread> threads;
        for (int worker_index = 0; worker_index < workers; worker_index++)
            threads.emplace_back(worker);
        for (auto& thread : threads) thread.join();

        const int area_minima[] = {
            4'000'000, 4'500'000, 5'000'000, 5'500'000,
            6'000'000, 6'500'000, 7'000'000,
        };
        std::ofstream output(output_path);
        if (!output)
            throw std::runtime_error(std::string("cannot write ") + output_path);
        output << "threshold,actual_min,actual_count,estimate_passes,"
                  "true_positives,retention_pct,below_count,below_passes,"
                  "below_pass_pct,precision_pct,clipped_pass_pct,pearson_r\n";
        output << std::fixed << std::setprecision(6);

        double best_score = -1.0;
        double best_threshold = 0.0;
        int best_true_positive = 0;
        int best_false_positive = 0;
        for (size_t threshold_index = 0;
             threshold_index < threshold_count; threshold_index++) {
            double correlation = pearson(records, estimates, threshold_index);
            size_t offset = threshold_index * record_count;
            for (int area_minimum : area_minima) {
                int actual_count = 0;
                int estimate_passes = 0;
                int true_positives = 0;
                int below_count = 0;
                int below_passes = 0;
                int clipped_passes = 0;
                for (size_t index = 0; index < record_count; index++) {
                    bool actual = records[index].area >= area_minimum;
                    bool estimated = estimates[offset + index] >= area_minimum;
                    actual_count += actual;
                    estimate_passes += estimated;
                    true_positives += actual && estimated;
                    below_count += !actual;
                    below_passes += !actual && estimated;
                    clipped_passes += estimated && clipped[offset + index];
                }
                double retention = actual_count > 0
                    ? 100.0 * true_positives / actual_count : 0.0;
                double below_rate = below_count > 0
                    ? 100.0 * below_passes / below_count : 0.0;
                double precision = estimate_passes > 0
                    ? 100.0 * true_positives / estimate_passes : 0.0;
                double clipped_rate = estimate_passes > 0
                    ? 100.0 * clipped_passes / estimate_passes : 0.0;
                output << thresholds[threshold_index] << ','
                       << area_minimum << ',' << actual_count << ','
                       << estimate_passes << ',' << true_positives << ','
                       << retention << ',' << below_count << ','
                       << below_passes << ',' << below_rate << ','
                       << precision << ',' << clipped_rate << ','
                       << correlation << '\n';

                if (area_minimum == 6'000'000) {
                    double score = retention - below_rate;
                    if (score > best_score) {
                        best_score = score;
                        best_threshold = thresholds[threshold_index];
                        best_true_positive = true_positives;
                        best_false_positive = below_passes;
                    }
                }
            }
        }

        std::cout << "records=" << record_count
                  << " thresholds=" << threshold_count
                  << " step=" << step << " radius=" << radius
                  << " workers=" << workers << '\n';
        std::cout << "csv=" << output_path << '\n';
        std::cout << std::fixed << std::setprecision(2)
                  << "best_6m_threshold=" << best_threshold
                  << " score=" << best_score
                  << " true_positives=" << best_true_positive
                  << "/21 below_6m_passes=" << best_false_positive << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
