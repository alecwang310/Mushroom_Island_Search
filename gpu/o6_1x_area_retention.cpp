#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
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

struct GridEstimate {
    int cells = 0;
    bool clipped = false;
};

static inline float fade_value(float value) {
    return value * value * value
        * (value * (value * 6.0f - 15.0f) + 10.0f);
}

static inline float gradient_dot(uint32_t hash, float x, float y, float z) {
    const uint32_t gradient = hash & 15u;
    float first = gradient < 8u ? x : y;
    float second = gradient < 4u
        ? y : ((gradient == 12u || gradient == 14u) ? x : z);
    if (gradient & 1u)
        first = -first;
    if (gradient & 2u)
        second = -second;
    return first + second;
}

static inline uint32_t permutation_pair(
        const ContTieredParams& params, uint32_t index) {
    const uint32_t wrapped = index & 0xFFu;
    const uint32_t next = (wrapped + 1u) & 0xFFu;
    return static_cast<uint32_t>(params.perm[0][wrapped])
        | (static_cast<uint32_t>(params.perm[0][next]) << 8);
}

static float sample_o6(const ContTieredParams& params, float x, float z) {
    const float offset_x = params.offset_a[0];
    const float offset_z = params.offset_c[0];
    const float cached_d2 = params.cached_d2[0];
    const float cached_t2 = params.cached_t2[0];
    const float shifted_x = x + offset_x;
    const float shifted_z = z + offset_z;
    const int cell_x = static_cast<int>(std::floor(shifted_x));
    const int cell_z = static_cast<int>(std::floor(shifted_z));
    const float local_x = shifted_x - static_cast<float>(cell_x);
    const float local_z = shifted_z - static_cast<float>(cell_z);
    const uint32_t hash_x = static_cast<uint32_t>(cell_x) & 0xFFu;
    const uint32_t hash_z = static_cast<uint32_t>(cell_z) & 0xFFu;
    const float fade_x = fade_value(local_x);
    const float fade_z = fade_value(local_z);

    uint32_t pair = permutation_pair(params, hash_x);
    const uint32_t first_a = (pair & 0xFFu) + params.cached_h2[0];
    const uint32_t first_b = (pair >> 8) + params.cached_h2[0];
    pair = permutation_pair(params, first_a);
    const uint32_t second_a = (pair & 0xFFu) + hash_z;
    const uint32_t second_b = (pair >> 8) + hash_z;
    pair = permutation_pair(params, first_b);
    const uint32_t third_a = (pair & 0xFFu) + hash_z;
    const uint32_t third_b = (pair >> 8) + hash_z;

    pair = permutation_pair(params, second_a);
    const uint32_t fourth_a = pair & 0xFFu;
    const uint32_t fourth_b = pair >> 8;
    pair = permutation_pair(params, second_b);
    const uint32_t fifth_a = pair & 0xFFu;
    const uint32_t fifth_b = pair >> 8;
    pair = permutation_pair(params, third_a);
    const uint32_t sixth_a = pair & 0xFFu;
    const uint32_t sixth_b = pair >> 8;
    pair = permutation_pair(params, third_b);
    const uint32_t seventh_a = pair & 0xFFu;
    const uint32_t seventh_b = pair >> 8;

    float lower_left = gradient_dot(fourth_a, local_x, cached_d2, local_z);
    float lower_right = gradient_dot(
        sixth_a, local_x - 1.0f, cached_d2, local_z);
    float upper_left = gradient_dot(
        fifth_a, local_x, cached_d2 - 1.0f, local_z);
    float upper_right = gradient_dot(
        seventh_a, local_x - 1.0f, cached_d2 - 1.0f, local_z);
    float lower_left_back = gradient_dot(
        fourth_b, local_x, cached_d2, local_z - 1.0f);
    float lower_right_back = gradient_dot(
        sixth_b, local_x - 1.0f, cached_d2, local_z - 1.0f);
    float upper_left_back = gradient_dot(
        fifth_b, local_x, cached_d2 - 1.0f, local_z - 1.0f);
    float upper_right_back = gradient_dot(
        seventh_b, local_x - 1.0f, cached_d2 - 1.0f, local_z - 1.0f);

    lower_left = std::fmaf(
        fade_x, lower_right - lower_left, lower_left);
    upper_left = std::fmaf(
        fade_x, upper_right - upper_left, upper_left);
    lower_left_back = std::fmaf(
        fade_x, lower_right_back - lower_left_back, lower_left_back);
    upper_left_back = std::fmaf(
        fade_x, upper_right_back - upper_left_back, upper_left_back);
    lower_left = std::fmaf(
        cached_t2, upper_left - lower_left, lower_left);
    lower_left_back = std::fmaf(
        cached_t2, upper_left_back - lower_left_back, lower_left_back);
    return std::fmaf(
        fade_z, lower_left_back - lower_left, lower_left)
        * params.amplitude[0] * params.cont_dbl_amp;
}

static constexpr double HEX_AREA_SCALE =
    0.86602540378443864676 * 16.0;
static constexpr int DEFAULT_STEP = 250;
static constexpr int DEFAULT_RADIUS = 16;
static constexpr int DEFAULT_ESTIMATE_MIN_AREA = 6'000'000;
static constexpr double MIN_THRESHOLD = -1.05;
static constexpr double MAX_THRESHOLD = 0.00;
static constexpr double THRESHOLD_STEP = 0.01;

static int64_t read_integer(const std::string& line, const char* key) {
    std::string token = std::string("\"") + key + "\":";
    size_t start = line.find(token);
    if (start == std::string::npos)
        return 0;
    start += token.size();
    while (start < line.size() && line[start] == ' ')
        ++start;
    size_t end = start;
    if (end < line.size() && (line[end] == '-' || line[end] == '+'))
        ++end;
    while (end < line.size() && line[end] >= '0' && line[end] <= '9')
        ++end;
    return std::stoll(line.substr(start, end - start));
}

static std::vector<IslandRecord> load_records(const char* path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error(std::string("cannot open ") + path);

    std::vector<IslandRecord> records;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty())
            continue;
        records.push_back({
            read_integer(line, "seed"),
            static_cast<int>(read_integer(line, "area")),
            static_cast<int>(read_integer(line, "cx")),
            static_cast<int>(read_integer(line, "cz")),
        });
    }
    return records;
}

static std::vector<double> make_thresholds() {
    std::vector<double> thresholds;
    for (int index = 0;; ++index) {
        double threshold = MIN_THRESHOLD + index * THRESHOLD_STEP;
        if (threshold > MAX_THRESHOLD + 1e-9)
            break;
        thresholds.push_back(threshold);
    }
    return thresholds;
}

static int grid_index(int col, int row, int radius) {
    const int width = radius * 2 + 1;
    return (row + radius) * width + col + radius;
}

static GridEstimate largest_component(const std::vector<double>& values,
                                      int radius, double threshold) {
    const int width = radius * 2 + 1;
    const int cell_count = width * width;
    const int directions[6][2] = {
        {-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, 1}, {1, -1},
    };
    std::vector<uint8_t> visited(cell_count, 0);
    std::vector<int> queue(cell_count);
    GridEstimate best;

    for (int row = -radius; row <= radius; ++row) {
        for (int col = -radius; col <= radius; ++col) {
            const int start = grid_index(col, row, radius);
            if (visited[start] || values[start] >= threshold)
                continue;

            int head = 0;
            int tail = 0;
            int cells = 0;
            bool clipped = false;
            visited[start] = 1;
            queue[tail++] = start;

            while (head < tail) {
                const int current = queue[head++];
                const int current_row = current / width - radius;
                const int current_col = current % width - radius;
                ++cells;
                if (current_row == -radius || current_row == radius
                        || current_col == -radius || current_col == radius)
                    clipped = true;

                for (const auto& direction : directions) {
                    const int next_col = current_col + direction[0];
                    const int next_row = current_row + direction[1];
                    if (next_col < -radius || next_col > radius
                            || next_row < -radius || next_row > radius)
                        continue;
                    const int next = grid_index(next_col, next_row, radius);
                    if (!visited[next] && values[next] < threshold) {
                        visited[next] = 1;
                        queue[tail++] = next;
                    }
                }
            }

            if (cells > best.cells
                    || (cells == best.cells && clipped && !best.clipped)) {
                best.cells = cells;
                best.clipped = clipped;
            }
        }
    }
    return best;
}

static std::vector<double> sample_o6_grid(const IslandRecord& record,
                                          int step, int radius) {
    const int width = radius * 2 + 1;
    const double spacing_z = step * 0.86602540378443864676;
    const int half_step = step / 2;
    std::vector<double> values(width * width);
    const uint64_t seed = static_cast<uint64_t>(record.seed);
    ContTieredParams params;
    cont_batch_init_tiered(&seed, 1, 0, &params);

    for (int row = -radius; row <= radius; ++row) {
        const int parity = row & 1;
        const float sample_z = static_cast<float>(record.cz)
            + static_cast<float>(row) * static_cast<float>(spacing_z);
        for (int col = -radius; col <= radius; ++col) {
            const float sample_x = static_cast<float>(record.cx)
                + static_cast<float>(col * step + parity * half_step);
            values[grid_index(col, row, radius)] = sample_o6(
                params,
                sample_x * params.lacunarity[0],
                sample_z * params.lacunarity[0]);
        }
    }
    return values;
}

static int64_t estimate_area(const GridEstimate& estimate, int step) {
    const double area = estimate.cells * step * step * HEX_AREA_SCALE;
    return static_cast<int64_t>(std::llround(area));
}

static std::vector<int> make_area_minima(const std::vector<IslandRecord>& records) {
    int maximum = 0;
    for (const IslandRecord& record : records)
        maximum = std::max(maximum, record.area);

    std::vector<int> minima;
    for (int area = 6'000'000; area <= maximum; area += 1'000'000)
        minima.push_back(area);
    if (minima.empty())
        minima.push_back(6'000'000);
    return minima;
}

int main(int argc, char** argv) {
    const char* input_path = argc > 1 ? argv[1] : "gpu/islands_6m.jsonl";
    const char* csv_path = argc > 2
        ? argv[2] : "gpu/o6_1x_area_retention.csv";
    const char* records_path = argc > 3
        ? argv[3] : "gpu/o6_1x_area_estimates.csv";
    const int workers = argc > 4 ? std::stoi(argv[4]) : 28;
    const int step = argc > 5 ? std::stoi(argv[5]) : DEFAULT_STEP;
    const int radius = argc > 6 ? std::stoi(argv[6]) : DEFAULT_RADIUS;
    const int estimate_min_area = argc > 7
        ? std::stoi(argv[7]) : DEFAULT_ESTIMATE_MIN_AREA;

    if (workers <= 0 || step <= 0 || radius < 1) {
        std::cerr << "invalid workers, step, or radius\n";
        return 2;
    }

    const std::vector<IslandRecord> records = load_records(input_path);
    const std::vector<double> thresholds = make_thresholds();
    const std::vector<int> area_minima = make_area_minima(records);
    const size_t result_count = records.size() * thresholds.size();
    std::vector<GridEstimate> estimates(result_count);
    std::atomic<size_t> next_record{0};

    auto worker = [&]() {
        while (true) {
            const size_t index = next_record.fetch_add(1);
            if (index >= records.size())
                return;
            const std::vector<double> values = sample_o6_grid(
                records[index], step, radius);
            for (size_t threshold_index = 0;
                 threshold_index < thresholds.size(); ++threshold_index) {
                estimates[index * thresholds.size() + threshold_index] =
                    largest_component(values, radius,
                                      thresholds[threshold_index]);
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (int index = 0; index < workers; ++index)
        threads.emplace_back(worker);
    for (std::thread& thread : threads)
        thread.join();

    std::ofstream output(csv_path);
    if (!output)
        throw std::runtime_error(std::string("cannot open ") + csv_path);
    output << "threshold,actual_min,total,strict_kept,"
              "strict_retention_pct,clipped_kept,clipped_retention_pct\n";
    output << std::fixed << std::setprecision(2);

    for (size_t threshold_index = 0;
         threshold_index < thresholds.size(); ++threshold_index) {
        for (int actual_min : area_minima) {
            int total = 0;
            int strict_kept = 0;
            int clipped_kept = 0;
            for (size_t record_index = 0;
                 record_index < records.size(); ++record_index) {
                if (records[record_index].area < actual_min)
                    continue;
                ++total;
                const GridEstimate& estimate = estimates[
                    record_index * thresholds.size() + threshold_index];
                const int64_t area = estimate_area(estimate, step);
                if (area >= estimate_min_area)
                    ++strict_kept;
                if (area >= estimate_min_area || estimate.clipped)
                    ++clipped_kept;
            }
            const double strict_retention = total > 0
                ? 100.0 * strict_kept / total : 0.0;
            const double clipped_retention = total > 0
                ? 100.0 * clipped_kept / total : 0.0;
            output << thresholds[threshold_index] << ',' << actual_min << ','
                   << total << ',' << strict_kept << ','
                   << strict_retention << ',' << clipped_kept << ','
                   << clipped_retention << '\n';
        }
    }

    std::ofstream record_output(records_path);
    if (!record_output)
        throw std::runtime_error(
            std::string("cannot open ") + records_path);
    record_output << "seed,actual_area,cx,cz,threshold,estimated_area,"
                     "estimated_cells,clipped\n";
    record_output << std::fixed << std::setprecision(2);
    for (size_t record_index = 0;
         record_index < records.size(); ++record_index) {
        for (size_t threshold_index = 0;
             threshold_index < thresholds.size(); ++threshold_index) {
            const GridEstimate& estimate = estimates[
                record_index * thresholds.size() + threshold_index];
            record_output << records[record_index].seed << ','
                          << records[record_index].area << ','
                          << records[record_index].cx << ','
                          << records[record_index].cz << ','
                          << thresholds[threshold_index] << ','
                          << estimate_area(estimate, step) << ','
                          << estimate.cells << ','
                          << (estimate.clipped ? 1 : 0) << '\n';
        }
    }

    std::cout << "records=" << records.size()
              << " thresholds=" << thresholds.size()
              << " step=" << step
              << " radius=" << radius
              << " estimate_min_area=" << estimate_min_area << '\n';
    std::cout << "retention_csv=" << csv_path << '\n';
    std::cout << "records_csv=" << records_path << '\n';
    return 0;
}
