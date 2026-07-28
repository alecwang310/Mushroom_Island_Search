#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "../engine/continentalness.h"

struct IslandRecord {
    int64_t seed;
    int cx;
    int cz;
};

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
            static_cast<int>(read_integer(line, "cx")),
            static_cast<int>(read_integer(line, "cz")),
        });
    }
    return records;
}

static bool has_triple(const ContEngine& engine, int cx, int cz,
                       int step, int radius, double threshold) {
    constexpr int grid_size = 512;
    const double spacing_z = step * 0.86602540378443864676;
    const double half_spacing_x = step * 0.5;
    const int origin_x = -(grid_size / 2) * step;
    const int origin_z = static_cast<int>(-(grid_size / 2) * spacing_z);

    const int center_row = static_cast<int>(
        std::floor((cz - origin_z) / spacing_z + 0.5));
    const int center_parity = center_row & 1;
    const int center_col = static_cast<int>(std::floor(
        (cx - origin_x - center_parity * half_spacing_x) / step + 0.5));
    const int width = radius * 2 + 1;
    std::vector<double> values(width * width);
    std::vector<uint8_t> low(width * width);

    for (int row = -radius; row <= radius; ++row) {
        int global_row = center_row + row;
        int parity = global_row & 1;
        int z = static_cast<int>(std::lround(
            origin_z + global_row * spacing_z));
        for (int col = -radius; col <= radius; ++col) {
            int global_col = center_col + col;
            int x = origin_x + global_col * step
                + static_cast<int>(parity * half_spacing_x);
            int index = (row + radius) * width + col + radius;
            values[index] = cont_sample(&engine, x, z);
            low[index] = values[index] < threshold;
        }
    }

    for (int row = -radius + 1; row < radius; ++row) {
        int parity = (center_row + row) & 1;
        for (int col = -radius + 1; col < radius; ++col) {
            int index = (row + radius) * width + col + radius;
            if (!low[index])
                continue;

            int neighbor_count = 0;
            const int directions[6][2] = {
                {0, -1}, {0, 1}, {-1, 0}, {-1, parity ? 1 : -1},
                {1, 0}, {1, parity ? 1 : -1},
            };
            for (const auto& direction : directions) {
                int neighbor_row = row + direction[0];
                int neighbor_col = col + direction[1];
                int neighbor_index = (neighbor_row + radius) * width
                    + neighbor_col + radius;
                neighbor_count += low[neighbor_index] != 0;
            }
            if (neighbor_count >= 2)
                return true;
        }
    }
    return false;
}

int main(int argc, char** argv) {
    const char* input_path = argc > 1 ? argv[1] : "gpu/islands_4m.jsonl";
    const int radius = argc > 2 ? std::stoi(argv[2]) : 6;
    const std::vector<int> steps = {500, 410, 320};
    const std::vector<double> thresholds = {
        -0.55, -0.50, -0.45, -0.40, -0.35, -0.30, -0.28,
        -0.26, -0.24, -0.22, -0.20, -0.15, -0.10,
    };

    try {
        const auto records = load_records(input_path);
        std::cout << "records=" << records.size()
                  << " radius_cells=" << radius << "\n";

        for (int step : steps) {
            std::vector<std::vector<int>> kept(
                thresholds.size(), std::vector<int>(1, 0));
            for (const auto& record : records) {
                ContEngine engine;
                cont_engine_init(&engine,
                                 static_cast<uint64_t>(record.seed), 0);
                cont_engine_disable_shift(&engine);
                engine.cont_octA_count = 1;
                engine.cont_octB_count = 0;

                for (size_t threshold_index = 0;
                     threshold_index < thresholds.size();
                     ++threshold_index) {
                    if (has_triple(engine, record.cx, record.cz, step,
                                   radius, thresholds[threshold_index]))
                        ++kept[threshold_index][0];
                }
            }

            std::cout << "step=" << step << "\n";
            for (size_t i = 0; i < thresholds.size(); ++i) {
                int count = kept[i][0];
                double percentage = records.empty()
                    ? 0.0 : 100.0 * count / records.size();
                std::cout << "  threshold=" << thresholds[i]
                          << " kept=" << count << "/" << records.size()
                          << " (" << percentage << "%)\n";
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
    return 0;
}
