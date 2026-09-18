#include "cpp_engine/search_ranker.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <sstream>

namespace sweep {

double tfidf_score(const std::string& query, const std::string& text) {
    auto query_tokens = tokenize(query);
    auto text_tokens = tokenize(text);
    if (query_tokens.empty() || text_tokens.empty()) return 0.0;
    std::unordered_map<std::string, int> tf;
    for (const auto& t : text_tokens) tf[t]++;
    double score = 0.0; int matched = 0;
    for (const auto& qt : query_tokens) {
        auto it = tf.find(qt);
        if (it != tf.end()) {
            double tf_val = static_cast<double>(it->second) / text_tokens.size();
            std::string lower_text;
            lower_text.reserve(std::min(text.size(), static_cast<size_t>(200)));
            for (size_t i = 0; i < std::min(text.size(), static_cast<size_t>(200)); i++) lower_text += std::tolower(static_cast<unsigned char>(text[i]));
            if (lower_text.find(qt) != std::string::npos) tf_val *= 1.5;
            score += tf_val; matched++;
        }
    }
    score = (score / query_tokens.size()) * 100.0;
    score += (static_cast<double>(matched) / query_tokens.size()) * 20.0;
    return std::min(100.0, score);
}

static std::string canonical_url(const std::string& url) {
    std::string result = url;
    size_t frag = result.find('#'); if (frag != std::string::npos) result = result.substr(0, frag);
    size_t q = result.find('?');
    if (q != std::string::npos) {
        std::string path = result.substr(0, q), params = result.substr(q + 1), clean_params, param;
        std::istringstream ss(params);
        while (std::getline(ss, param, '&')) if (param.find("utm_") != 0 && param.find("ref=") != 0 && param.find("fbclid=") != 0) { if (!clean_params.empty()) clean_params += "&"; clean_params += param; }
        result = clean_params.empty() ? path : path + "?" + clean_params;
    }
    while (result.length() > 1 && result.back() == '/') result.pop_back();
    return result;
}

std::vector<RankedHit> dedup_hits(const std::vector<RankedHit>& hits) {
    std::vector<RankedHit> result; std::unordered_set<std::string> seen;
    for (const auto& hit : hits) { std::string key = canonical_url(hit.url); if (seen.count(key)) continue; seen.insert(key); result.push_back(hit); }
    return result;
}

std::vector<RankedHit> rank_hits(const std::string& query, const std::vector<RankedHit>& hits, int limit) {
    std::vector<RankedHit> ranked = hits;
    for (auto& hit : ranked) { std::string combined = hit.title + " " + hit.snippet; hit.score = tfidf_score(query, combined); }
    std::sort(ranked.begin(), ranked.end(), [](const RankedHit& a, const RankedHit& b) { return a.score > b.score; });
    ranked = dedup_hits(ranked);
    if (static_cast<int>(ranked.size()) > limit) ranked.resize(limit);
    return ranked;
}

}  // namespace sweep
