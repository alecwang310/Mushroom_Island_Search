#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../engine/continentalness.h"

struct IslandRecord {
    int64_t seed;
    int area;
    int cx;
    int cz;
};

struct StepResult {
    int step;
    std::vector<double> required;
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

static double required_threshold(const ContEngine& engine, int cx, int cz,
                                 int step, int radius) {
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
        }
    }

    double best = std::numeric_limits<double>::infinity();
    for (int row = -radius + 1; row < radius; ++row) {
        int parity = (center_row + row) & 1;
        for (int col = -radius + 1; col < radius; ++col) {
            int index = (row + radius) * width + col + radius;
            double lowest = std::numeric_limits<double>::infinity();
            double second_lowest = std::numeric_limits<double>::infinity();
            const int directions[6][2] = {
                {0, -1}, {0, 1}, {-1, 0}, {-1, parity ? 1 : -1},
                {1, 0}, {1, parity ? 1 : -1},
            };
            for (const auto& direction : directions) {
                int neighbor_row = row + direction[0];
                int neighbor_col = col + direction[1];
                int neighbor_index = (neighbor_row + radius) * width
                    + neighbor_col + radius;
                double value = values[neighbor_index];
                if (value < lowest) {
                    second_lowest = lowest;
                    lowest = value;
                } else if (value < second_lowest) {
                    second_lowest = value;
                }
            }
            double candidate = std::max(
                values[index], std::max(lowest, second_lowest));
            best = std::min(best, candidate);
        }
    }
    return best;
}

static std::vector<double> make_thresholds() {
    std::vector<double> thresholds;
    for (int i = 0; i <= 90; ++i)
        thresholds.push_back(-1.0 + i * 0.01);
    return thresholds;
}

static std::vector<int> make_area_minima() {
    return {4'000'000, 4'500'000, 5'000'000, 5'500'000,
            6'000'000, 6'500'000, 7'000'000, 7'500'000};
}

static int count_area(const std::vector<IslandRecord>& records, int area_min) {
    int count = 0;
    for (const auto& record : records)
        count += record.area >= area_min;
    return count;
}

static int count_kept(const std::vector<IslandRecord>& records,
                      const std::vector<double>& required,
                      int area_min, double threshold) {
    int count = 0;
    for (size_t i = 0; i < records.size(); ++i)
        count += records[i].area >= area_min && required[i] < threshold;
    return count;
}

static double retention(const std::vector<IslandRecord>& records,
                        const std::vector<double>& required,
                        int area_min, double threshold) {
    int total = count_area(records, area_min);
    if (total == 0)
        return 0.0;
    return 100.0 * count_kept(records, required, area_min, threshold)
        / total;
}

static std::string format_area(int area) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(1)
           << static_cast<double>(area) / 1'000'000.0 << "M+";
    return output.str();
}

static void write_csv(const char* path, const std::vector<IslandRecord>& records,
                      const std::vector<StepResult>& results,
                      const std::vector<double>& thresholds,
                      const std::vector<int>& area_minima) {
    std::ofstream output(path);
    if (!output)
        throw std::runtime_error(std::string("cannot write ") + path);
    output << "step,threshold,area_min,total,kept,retention_percent\n";
    output << std::fixed << std::setprecision(6);
    for (const auto& result : results) {
        for (double threshold : thresholds) {
            for (int area_min : area_minima) {
                int total = count_area(records, area_min);
                int kept = count_kept(records, result.required,
                                      area_min, threshold);
                double percent = total == 0 ? 0.0
                    : 100.0 * kept / total;
                output << result.step << ',' << threshold << ',' << area_min
                       << ',' << total << ',' << kept << ',' << percent
                       << '\n';
            }
        }
    }
}

static void write_svg(const char* path, const std::vector<IslandRecord>& records,
                      const std::vector<StepResult>& results,
                      const std::vector<double>& thresholds,
                      const std::vector<int>& area_minima) {
    constexpr int panel_width = 380;
    constexpr int panel_height = 280;
    constexpr int columns = 2;
    const int rows = static_cast<int>((area_minima.size() + columns - 1)
                                      / columns);
    const int width = columns * panel_width;
    const int height = rows * panel_height + 35;
    const int plot_x = 48;
    const int plot_y = 35;
    const int plot_width = 300;
    const int plot_height = 190;
    const char* colors[] = {"#2563eb", "#16a34a", "#dc2626"};

    std::ofstream output(path);
    if (!output)
        throw std::runtime_error(std::string("cannot write ") + path);
    output << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\""
           << width << "\" height=\"" << height
           << "\" viewBox=\"0 0 " << width << ' ' << height << "\">\n";
    output << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    output << "<text x=\"20\" y=\"22\" font-family=\"sans-serif\" "
              "font-size=\"16\" font-weight=\"bold\">O6 triple retention "
              "by island area</text>\n";

    for (size_t area_index = 0; area_index < area_minima.size(); ++area_index) {
        int panel_origin_x = static_cast<int>(area_index % columns)
            * panel_width;
        int panel_origin_y = static_cast<int>(area_index / columns)
            * panel_height + 35;
        int total = count_area(records, area_minima[area_index]);
        output << "<text x=\"" << panel_origin_x + plot_x
               << "\" y=\"" << panel_origin_y + 20
               << "\" font-family=\"sans-serif\" font-size=\"13\">"
               << format_area(area_minima[area_index]) << " (n=" << total
               << ")</text>\n";
        output << "<rect x=\"" << panel_origin_x + plot_x
               << "\" y=\"" << panel_origin_y + plot_y
               << "\" width=\"" << plot_width << "\" height=\""
               << plot_height << "\" fill=\"none\" stroke=\"#888\"/>\n";
        output << "<text x=\"" << panel_origin_x + plot_x - 30
               << "\" y=\"" << panel_origin_y + plot_y + 5
               << "\" font-family=\"sans-serif\" font-size=\"10\">100</text>\n";
        output << "<text x=\"" << panel_origin_x + plot_x - 20
               << "\" y=\"" << panel_origin_y + plot_y + plot_height / 2 + 4
               << "\" font-family=\"sans-serif\" font-size=\"10\">50</text>\n";
        output << "<text x=\"" << panel_origin_x + plot_x - 14
               << "\" y=\"" << panel_origin_y + plot_y + plot_height + 4
               << "\" font-family=\"sans-serif\" font-size=\"10\">0</text>\n";

        for (size_t result_index = 0; result_index < results.size();
             ++result_index) {
            const auto& result = results[result_index];
            output << "<polyline fill=\"none\" stroke=\""
                   << colors[result_index] << "\" stroke-width=\"2\" points=\"";
            for (double threshold : thresholds) {
                double x_fraction = (threshold + 1.0) / 0.9;
                double y_fraction = retention(
                    records, result.required, area_minima[area_index],
                    threshold) / 100.0;
                double x = panel_origin_x + plot_x + x_fraction * plot_width;
                double y = panel_origin_y + plot_y
                    + (1.0 - y_fraction) * plot_height;
                output << x << ',' << y << ' ';
            }
            output << "\"/>\n";
        }
    }

    output << "<text x=\"20\" y=\"" << height - 8
           << "\" font-family=\"sans-serif\" font-size=\"11\">"
              "Threshold (O6 &lt; x): -1.0 to -0.1</text>\n";
    output << "<text x=\"760\" y=\"22\" font-family=\"sans-serif\" "
              "font-size=\"11\" fill=\"#2563eb\">step 500</text>\n";
    output << "<text x=\"835\" y=\"22\" font-family=\"sans-serif\" "
              "font-size=\"11\" fill=\"#16a34a\">step 410</text>\n";
    output << "<text x=\"910\" y=\"22\" font-family=\"sans-serif\" "
              "font-size=\"11\" fill=\"#dc2626\">step 320</text>\n";
    output << "</svg>\n";
}

int main(int argc, char** argv) {
    const char* input_path = argc > 1 ? argv[1] : "gpu/islands_4m.jsonl";
    const int radius = argc > 2 ? std::stoi(argv[2]) : 6;
    const char* csv_path = argc > 3 ? argv[3] : "gpu/o6_retention.csv";
    const char* svg_path = argc > 4 ? argv[4] : "gpu/o6_retention.svg";
    const std::vector<int> steps = {500, 410, 320};
    const std::vector<double> thresholds = make_thresholds();
    const std::vector<int> area_minima = make_area_minima();

    try {
        const auto records = load_records(input_path);
        std::vector<StepResult> results;
        results.reserve(steps.size());
        for (int step : steps) {
            StepResult result{step, {}};
            result.required.reserve(records.size());
            for (const auto& record : records) {
                ContEngine engine;
                cont_engine_init(&engine,
                                 static_cast<uint64_t>(record.seed), 0);
                cont_engine_disable_shift(&engine);
                engine.cont_octA_count = 1;
                engine.cont_octB_count = 0;
                result.required.push_back(required_threshold(
                    engine, record.cx, record.cz, step, radius));
            }
            results.push_back(std::move(result));
        }

        write_csv(csv_path, records, results, thresholds, area_minima);
        write_svg(svg_path, records, results, thresholds, area_minima);
        std::cout << "records=" << records.size()
                  << " radius_cells=" << radius << "\n";
        std::cout << "csv=" << csv_path << "\nsvg=" << svg_path << "\n";
        for (const auto& result : results) {
            std::cout << "step=" << result.step << "\n";
            for (double threshold : {-0.60, -0.50, -0.40, -0.30,
                                     -0.24, -0.20}) {
                std::cout << "  threshold=" << threshold;
                for (int area_min : area_minima) {
                    std::cout << " " << format_area(area_min) << "="
                              << retention(records, result.required,
                                           area_min, threshold) << "%";
                }
                std::cout << "\n";
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
    return 0;
}
