#include "cpp_engine/data_engine.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <numeric>

namespace sweep {

static std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool in_quotes = false;
    for (char c : line) {
        if (c == '"') in_quotes = !in_quotes;
        else if (c == ',' && !in_quotes) { fields.push_back(field); field.clear(); }
        else field += c;
    }
    fields.push_back(field);
    return fields;
}

static Value parse_value(const std::string& s) {
    if (s.empty() || s == "null" || s == "NULL" || s == "NA" || s == "NaN") return std::monostate{};
    try { size_t pos; double v = std::stod(s, &pos); if (pos == s.length()) return v; } catch (...) {}
    if (s == "true" || s == "True" || s == "TRUE") return true;
    if (s == "false" || s == "False" || s == "FALSE") return false;
    return s;
}

DataFrame parse_csv(const std::string& csv_text, bool has_header) {
    DataFrame df;
    std::istringstream ss(csv_text);
    std::string line;
    bool first = true;
    while (std::getline(ss, line)) {
        if (line.empty()) continue;
        auto fields = split_csv_line(line);
        if (first && has_header) { df.columns = fields; first = false; continue; }
        Row row;
        for (const auto& f : fields) row.push_back(parse_value(f));
        df.rows.push_back(row);
        first = false;
    }
    if (!df.columns.empty() && !df.rows.empty()) {
        for (size_t col = 0; col < df.columns.size(); col++) {
            Column c;
            for (const auto& row : df.rows) c.push_back(col < row.size() ? row[col] : std::monostate{});
            df.column_data[df.columns[col]] = c;
        }
    }
    return df;
}

std::string dataframe_to_csv(const DataFrame& df) {
    std::string csv;
    for (size_t i = 0; i < df.columns.size(); i++) { if (i > 0) csv += ","; csv += df.columns[i]; }
    csv += "\n";
    for (const auto& row : df.rows) {
        for (size_t i = 0; i < row.size(); i++) {
            if (i > 0) csv += ",";
            std::visit([&](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, std::monostate>) csv += "";
                else if constexpr (std::is_same_v<T, bool>) csv += arg ? "true" : "false";
                else if constexpr (std::is_same_v<T, std::string>) csv += arg;
                else if constexpr (std::is_arithmetic_v<T>) csv += std::to_string(arg);
            }, row[i]);
        }
        csv += "\n";
    }
    return csv;
}

std::vector<double> DataFrame::get_numeric(const std::string& col) const {
    std::vector<double> result;
    auto it = column_data.find(col);
    if (it == column_data.end()) return result;
    for (const auto& v : it->second) if (std::holds_alternative<double>(v)) result.push_back(std::get<double>(v));
    return result;
}

std::vector<std::string> DataFrame::get_strings(const std::string& col) const {
    std::vector<std::string> result;
    auto it = column_data.find(col);
    if (it == column_data.end()) return result;
    for (const auto& v : it->second) {
        if (std::holds_alternative<std::string>(v)) result.push_back(std::get<std::string>(v));
        else if (std::holds_alternative<double>(v)) result.push_back(std::to_string(std::get<double>(v)));
    }
    return result;
}

DataFrame df_head(const DataFrame& df, int n) { DataFrame result; result.columns = df.columns; int count = std::min(n, static_cast<int>(df.rows.size())); result.rows.assign(df.rows.begin(), df.rows.begin() + count); return result; }
DataFrame df_tail(const DataFrame& df, int n) { DataFrame result; result.columns = df.columns; int start = std::max(0, static_cast<int>(df.rows.size()) - n); result.rows.assign(df.rows.begin() + start, df.rows.end()); return result; }

DataFrame df_select(const DataFrame& df, const std::vector<std::string>& columns) {
    DataFrame result; result.columns = columns;
    for (const auto& row : df.rows) {
        Row new_row;
        for (const auto& col : columns) {
            auto it = std::find(df.columns.begin(), df.columns.end(), col);
            if (it != df.columns.end()) { size_t idx = it - df.columns.begin(); new_row.push_back(idx < row.size() ? row[idx] : std::monostate{}); }
        }
        result.rows.push_back(new_row);
    }
    return result;
}

DataFrame df_filter(const DataFrame& df, const std::string& column, const std::string& op, double value) {
    DataFrame result; result.columns = df.columns;
    auto it = std::find(df.columns.begin(), df.columns.end(), column); if (it == df.columns.end()) return result;
    size_t col_idx = it - df.columns.begin();
    for (const auto& row : df.rows) {
        if (col_idx >= row.size() || !std::holds_alternative<double>(row[col_idx])) continue;
        double v = std::get<double>(row[col_idx]); bool keep = false;
        if (op == ">" || op == "gt") keep = v > value; else if (op == "<" || op == "lt") keep = v < value;
        else if (op == ">=" || op == "gte") keep = v >= value; else if (op == "<=" || op == "lte") keep = v <= value;
        else if (op == "==" || op == "eq") keep = v == value; else if (op == "!=" || op == "ne") keep = v != value;
        if (keep) result.rows.push_back(row);
    }
    return result;
}

DataFrame df_sort(const DataFrame& df, const std::string& column, bool ascending) {
    DataFrame result = df; auto it = std::find(result.columns.begin(), result.columns.end(), column); if (it == result.columns.end()) return result;
    size_t col_idx = it - result.columns.begin();
    std::sort(result.rows.begin(), result.rows.end(), [&](const Row& a, const Row& b) {
        if (col_idx >= a.size() || col_idx >= b.size()) return false;
        if (std::holds_alternative<double>(a[col_idx]) && std::holds_alternative<double>(b[col_idx])) { double va = std::get<double>(a[col_idx]), vb = std::get<double>(b[col_idx]); return ascending ? va < vb : va > vb; }
        if (std::holds_alternative<std::string>(a[col_idx]) && std::holds_alternative<std::string>(b[col_idx])) return ascending ? std::get<std::string>(a[col_idx]) < std::get<std::string>(b[col_idx]) : std::get<std::string>(a[col_idx]) > std::get<std::string>(b[col_idx]);
        return false;
    });
    return result;
}

double column_sum(const Column& col) { double sum = 0; for (const auto& v : col) if (std::holds_alternative<double>(v)) sum += std::get<double>(v); return sum; }
double column_mean(const Column& col) { int count = 0; double sum = 0; for (const auto& v : col) if (std::holds_alternative<double>(v)) { sum += std::get<double>(v); count++; } return count > 0 ? sum / count : 0; }
double column_min(const Column& col) { double min_val = INFINITY; for (const auto& v : col) if (std::holds_alternative<double>(v)) { double d = std::get<double>(v); if (d < min_val) min_val = d; } return min_val; }
double column_max(const Column& col) { double max_val = -INFINITY; for (const auto& v : col) if (std::holds_alternative<double>(v)) { double d = std::get<double>(v); if (d > max_val) max_val = d; } return max_val; }
double column_std(const Column& col) { double mean = column_mean(col), sq_sum = 0; int count = 0; for (const auto& v : col) if (std::holds_alternative<double>(v)) { double d = std::get<double>(v) - mean; sq_sum += d * d; count++; } return count > 1 ? std::sqrt(sq_sum / (count - 1)) : 0; }

std::vector<ColumnStats> df_describe(const DataFrame& df) {
    std::vector<ColumnStats> stats;
    for (const auto& col_name : df.columns) {
        ColumnStats s; s.name = col_name; auto it = df.column_data.find(col_name); if (it == df.column_data.end()) { stats.push_back(s); continue; }
        s.count = it->second.size(); s.null_count = 0; for (const auto& v : it->second) if (std::holds_alternative<std::monostate>(v)) s.null_count++;
        s.mean = column_mean(it->second); s.std_dev = column_std(it->second); s.min_val = column_min(it->second); s.max_val = column_max(it->second);
        std::vector<double> nums; for (const auto& v : it->second) if (std::holds_alternative<double>(v)) nums.push_back(std::get<double>(v));
        std::sort(nums.begin(), nums.end()); if (!nums.empty()) { size_t mid = nums.size() / 2; s.median = nums.size() % 2 == 0 ? (nums[mid - 1] + nums[mid]) / 2 : nums[mid]; }
        stats.push_back(s);
    }
    return stats;
}

std::vector<std::vector<double>> df_correlation(const DataFrame& df) {
    std::vector<std::vector<double>> corr, numeric_cols; for (const auto& col_name : df.columns) numeric_cols.push_back(df.get_numeric(col_name));
    for (size_t i = 0; i < numeric_cols.size(); i++) { std::vector<double> row; for (size_t j = 0; j < numeric_cols.size(); j++) {
        if (i == j) { row.push_back(1.0); continue; }
        if (numeric_cols[i].size() != numeric_cols[j].size() || numeric_cols[i].empty()) { row.push_back(0); continue; }
        double mx = std::accumulate(numeric_cols[i].begin(), numeric_cols[i].end(), 0.0) / numeric_cols[i].size(), my = std::accumulate(numeric_cols[j].begin(), numeric_cols[j].end(), 0.0) / numeric_cols[j].size(), num = 0, dx = 0, dy = 0;
        for (size_t k = 0; k < numeric_cols[i].size(); k++) { double a = numeric_cols[i][k] - mx, b = numeric_cols[j][k] - my; num += a * b; dx += a * a; dy += b * b; }
        double denom = std::sqrt(dx * dy); row.push_back(denom < 1e-10 ? 0 : num / denom);
    } corr.push_back(row); }
    return corr;
}

DataFrame df_fill_na(const DataFrame& df, const std::string& column, const Value& fill_value) { DataFrame result = df; auto it = std::find(result.columns.begin(), result.columns.end(), column); if (it == result.columns.end()) return result; size_t col_idx = it - result.columns.begin(); for (auto& row : result.rows) if (col_idx < row.size() && std::holds_alternative<std::monostate>(row[col_idx])) row[col_idx] = fill_value; return result; }
DataFrame df_drop_na(const DataFrame& df) { DataFrame result; result.columns = df.columns; for (const auto& row : df.rows) { bool has_na = false; for (const auto& v : row) if (std::holds_alternative<std::monostate>(v)) { has_na = true; break; } if (!has_na) result.rows.push_back(row); } return result; }

DataFrame df_drop_duplicates(const DataFrame& df) {
    DataFrame result; result.columns = df.columns; std::vector<std::string> seen;
    for (const auto& row : df.rows) {
        std::string key;
        for (const auto& v : row) std::visit([&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::monostate>) key += "|";
            else if constexpr (std::is_same_v<T, bool>) key += arg ? "T|" : "F|";
            else if constexpr (std::is_same_v<T, std::string>) key += arg + "|";
            else if constexpr (std::is_arithmetic_v<T>) key += std::to_string(arg) + "|";
        }, v);
        if (std::find(seen.begin(), seen.end(), key) == seen.end()) { seen.push_back(key); result.rows.push_back(row); }
    }
    return result;
}

DataFrame df_rename(const DataFrame& df, const std::string& old_name, const std::string& new_name) { DataFrame result = df; for (auto& col : result.columns) if (col == old_name) col = new_name; return result; }

DataFrame df_group_by(const DataFrame& df, const std::string& column, const std::string& agg_func) {
    DataFrame result; result.columns.push_back(column); result.columns.push_back(agg_func); auto it = std::find(df.columns.begin(), df.columns.end(), column); if (it == df.columns.end()) return result; size_t col_idx = it - df.columns.begin(); std::unordered_map<std::string, std::vector<double>> groups;
    for (const auto& row : df.rows) {
        if (col_idx >= row.size()) continue; std::string key;
        std::visit([&](auto&& arg) { using T = std::decay_t<decltype(arg)>; if constexpr (std::is_same_v<T, std::monostate>) key = ""; else if constexpr (std::is_same_v<T, bool>) key = arg ? "true" : "false"; else if constexpr (std::is_same_v<T, std::string>) key = arg; else if constexpr (std::is_arithmetic_v<T>) key = std::to_string(arg); }, row[col_idx]);
        for (size_t c = 0; c < row.size(); c++) if (c != col_idx && std::holds_alternative<double>(row[c])) groups[key].push_back(std::get<double>(row[c]));
    }
    for (const auto& [key, values] : groups) {
        Row row; row.push_back(key);
        if (agg_func == "sum" || agg_func == "count") { row.push_back(static_cast<double>(agg_func == "count" ? values.size() : 0)); if (agg_func == "sum") row.back() = std::accumulate(values.begin(), values.end(), 0.0); }
        else if (agg_func == "mean") { double sum = std::accumulate(values.begin(), values.end(), 0.0); row.push_back(values.empty() ? 0 : sum / values.size()); }
        else if (agg_func == "min") row.push_back(values.empty() ? 0 : *std::min_element(values.begin(), values.end()));
        else if (agg_func == "max") row.push_back(values.empty() ? 0 : *std::max_element(values.begin(), values.end()));
        else row.push_back(std::accumulate(values.begin(), values.end(), 0.0));
        result.rows.push_back(row);
    }
    return result;
}

}  // namespace sweep
