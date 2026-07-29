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

static constexpr double HEX_AREA_SCALE =
    0.86602540378443864676 * 16.0;
static constexpr int DEFAULT_STEP = 250;
static constexpr int DEFAULT_RADIUS = 16;
static constexpr int DEFAULT_ESTIMATE_MIN_AREA = 6'000'000;
static constexpr double MIN_THRESHOLD = -1.05;
static constexpr double MAX_THRESHOLD = -0.40;
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
    ContEngine engine;
    cont_engine_init_6oct(
        &engine, static_cast<uint64_t>(record.seed), 0);

    for (int row = -radius; row <= radius; ++row) {
        const int parity = row & 1;
        const int z = record.cz + static_cast<int>(row * spacing_z);
        for (int col = -radius; col <= radius; ++col) {
            const int x = record.cx + col * step + parity * half_step;
            values[grid_index(col, row, radius)] = cont_sample(
                &engine, x, z);
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
