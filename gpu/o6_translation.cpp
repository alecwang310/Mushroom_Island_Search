#include <atomic>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
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

struct LargerResult {
    int64_t seed;
    int original_area;
    int translated_area;
    int original_cx;
    int original_cz;
    int translated_cx;
    int translated_cz;
    int dx_index;
    int dz_index;
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
            static_cast<int>(read_integer(line, "area")),
            static_cast<int>(read_integer(line, "cx")),
            static_cast<int>(read_integer(line, "cz")),
        });
    }
    return records;
}

static int64_t floor_division(int64_t numerator, int64_t denominator) {
    int64_t quotient = numerator / denominator;
    int64_t remainder = numerator % denominator;
    if (remainder != 0 && ((remainder < 0) != (denominator < 0)))
        --quotient;
    return quotient;
}

static int64_t ceiling_division(int64_t numerator, int64_t denominator) {
    return -floor_division(-numerator, denominator);
}

static int lower_translation(int coordinate, int64_t period, int border) {
    return static_cast<int>(ceiling_division(
        -static_cast<int64_t>(border) - coordinate, period));
}

static int upper_translation(int coordinate, int64_t period, int border) {
    return static_cast<int>(floor_division(
        static_cast<int64_t>(border) - coordinate, period));
}

int main(int argc, char** argv) {
    const char* input_path = argc > 1 ? argv[1] : "gpu/islands_4m.jsonl";
    const char* output_path = argc > 2
        ? argv[2] : "gpu/o6_translation_larger.csv";
    const int worker_count = argc > 3 ? std::stoi(argv[3]) : 28;
    const double threshold = argc > 4 ? std::stod(argv[4]) : -0.95;
    constexpr int64_t o6_period = 512LL * 256LL;
    constexpr int world_border = 7'500'000;
    constexpr int coarse_gate = 3'000'000;
    constexpr int max_cells = 1'000'000;

    try {
        const auto records = load_records(input_path);
        std::atomic<size_t> next_record{0};
        std::atomic<size_t> completed{0};
        std::atomic<uint64_t> positions_tested{0};
        std::atomic<uint64_t> o6o15_candidates{0};
        std::atomic<uint64_t> coarse_nonzero{0};
        std::atomic<uint64_t> coarse_passed{0};
        std::atomic<uint64_t> full_floods{0};
        std::mutex result_mutex;
        std::mutex progress_mutex;
        std::vector<LargerResult> larger;

        auto worker = [&]() {
            while (true) {
                size_t record_index = next_record.fetch_add(1);
                if (record_index >= records.size())
                    return;
                const IslandRecord& record = records[record_index];
                int min_dx = lower_translation(
                    record.cx, o6_period, world_border);
                int max_dx = upper_translation(
                    record.cx, o6_period, world_border);
                int min_dz = lower_translation(
                    record.cz, o6_period, world_border);
                int max_dz = upper_translation(
                    record.cz, o6_period, world_border);

                ContEngine engine;
                cont_engine_init(&engine,
                                 static_cast<uint64_t>(record.seed), 0);
                cont_engine_disable_shift(&engine);
                engine.cont_octA_count = 1;
                engine.cont_octB_count = 1;

                uint64_t local_positions = 0;
                uint64_t local_candidates = 0;
                uint64_t local_coarse_nonzero = 0;
                uint64_t local_coarse_passed = 0;
                uint64_t local_full_floods = 0;
                std::vector<LargerResult> local_larger;

                for (int dx_index = min_dx; dx_index <= max_dx; ++dx_index) {
                    for (int dz_index = min_dz; dz_index <= max_dz;
                         ++dz_index) {
                        if (dx_index == 0 && dz_index == 0)
                            continue;
                        ++local_positions;
                        int translated_cx = static_cast<int>(
                            record.cx + dx_index * o6_period);
                        int translated_cz = static_cast<int>(
                            record.cz + dz_index * o6_period);
                        if (cont_sample(&engine, translated_cx,
                                        translated_cz) >= threshold)
                            continue;
                        ++local_candidates;

                        int64_t coarse_area = cont_flood_fill_6oct(
                            static_cast<uint64_t>(record.seed),
                            translated_cx, translated_cz, max_cells);
                        if (coarse_area <= 0)
                            continue;
                        ++local_coarse_nonzero;
                        if (coarse_area < coarse_gate)
                            continue;
                        ++local_coarse_passed;

                        ++local_full_floods;
                        int64_t full_area = cont_flood_fill(
                            static_cast<uint64_t>(record.seed),
                            translated_cx, translated_cz, max_cells);
                        if (full_area > record.area) {
                            local_larger.push_back({
                                record.seed, record.area,
                                static_cast<int>(full_area),
                                record.cx, record.cz,
                                translated_cx, translated_cz,
                                dx_index, dz_index,
                            });
                        }
                    }
                }

                positions_tested.fetch_add(local_positions);
                o6o15_candidates.fetch_add(local_candidates);
                coarse_nonzero.fetch_add(local_coarse_nonzero);
                coarse_passed.fetch_add(local_coarse_passed);
                full_floods.fetch_add(local_full_floods);
                if (!local_larger.empty()) {
                    std::lock_guard<std::mutex> lock(result_mutex);
                    larger.insert(larger.end(), local_larger.begin(),
                                  local_larger.end());
                }

                size_t done = completed.fetch_add(1) + 1;
                if (done % 100 == 0 || done == records.size()) {
                    std::lock_guard<std::mutex> lock(progress_mutex);
                    std::cout << "processed=" << done << "/"
                              << records.size() << " candidates="
                              << o6o15_candidates.load() << " floods="
                              << full_floods.load() << "\n";
                }
            }
        };

        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (int i = 0; i < worker_count; ++i)
            workers.emplace_back(worker);
        for (auto& thread : workers)
            thread.join();

        std::sort(larger.begin(), larger.end(),
                  [](const LargerResult& left, const LargerResult& right) {
                      return left.translated_area > right.translated_area;
                  });
        std::ofstream output(output_path);
        if (!output)
            throw std::runtime_error(std::string("cannot write ")
                                     + output_path);
        output << "seed,original_area,translated_area,original_cx,original_cz,"
                  "translated_cx,translated_cz,dx_index,dz_index\n";
        for (const auto& result : larger) {
            output << result.seed << ',' << result.original_area << ','
                   << result.translated_area << ',' << result.original_cx << ','
                   << result.original_cz << ',' << result.translated_cx << ','
                   << result.translated_cz << ',' << result.dx_index << ','
                   << result.dz_index << '\n';
        }

        std::cout << "records=" << records.size()
                  << " workers=" << worker_count
                  << " o6_period_pipeline_coords=" << o6_period
                  << " threshold=" << threshold << "\n";
        std::cout << "positions_tested=" << positions_tested.load()
                  << " o6o15_candidates=" << o6o15_candidates.load()
                  << " coarse_nonzero=" << coarse_nonzero.load()
                  << " coarse_passed=" << coarse_passed.load()
                  << " full_floods=" << full_floods.load()
                  << " larger_results=" << larger.size() << "\n";
        if (!larger.empty()) {
            std::cout << "largest_translated_area="
                      << larger.front().translated_area << " seed="
                      << larger.front().seed << " translated_cx="
                      << larger.front().translated_cx << " translated_cz="
                      << larger.front().translated_cz << "\n";
        }
        std::cout << "csv=" << output_path << "\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
    return 0;
}
