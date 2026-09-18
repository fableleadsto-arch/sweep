#include "cpp_engine/regex_engine.h"
#include <algorithm>
#include <cctype>

namespace sweep {

PatternSet::PatternSet()
    : email_re_(R"([a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,})", std::regex::icase)
    , phone_re_(R"(\+?\d[\d\s().\-]{7,17}\d)", std::regex::icase)
    , url_re_(R"(https?://[^\s"'<>]+)", std::regex::icase)
    , social_re_(R"(https?://(?:www\.)?(?:linkedin\.com|x\.com|twitter\.com|instagram\.com|facebook\.com|github\.com|tiktok\.com|youtube\.com)/[^\s"'<>]+)", std::regex::icase)
    , captcha_markers_{
        "captcha", "are you a robot", "unusual traffic",
        "verify you are human", "cf-browser-verification",
        "checking your browser before", "access denied",
        "enable javascript and cookies to continue"
    }
{
    injection_res_.emplace_back(R"(ignore\s+(all\s+)?previous\s+instructions)", std::regex::icase);
    injection_res_.emplace_back(R"(disregard\s+(all\s+)?prior)", std::regex::icase);
    injection_res_.emplace_back(R"(you\s+are\s+now\s+(a|an|the))", std::regex::icase);
    injection_res_.emplace_back(R"(system\s*prompt\s*(override|injection|bypass))", std::regex::icase);
    injection_res_.emplace_back(R"(<\|im_start\|>system)", std::regex::icase);
    injection_res_.emplace_back(R"(IMPORTANT:\s*you\s+must)", std::regex::icase);
    injection_res_.emplace_back(R"(new\s+instructions?:)", std::regex::icase);
    injection_res_.emplace_back(R"(forget\s+(everything|all))", std::regex::icase);
    injection_res_.emplace_back(R"(act\s+as\s+if\s+you\s+(have|are))", std::regex::icase);
}

std::vector<std::string> PatternSet::find_emails(const std::string& text) const {
    std::vector<std::string> results; std::smatch m; auto si = text.cbegin();
    while (std::regex_search(si, text.cend(), m, email_re_)) {
        std::string email = m[0].str(); std::transform(email.begin(), email.end(), email.begin(), ::tolower);
        if (email.find(".png") == std::string::npos && email.find(".jpg") == std::string::npos && email.find(".jpeg") == std::string::npos && email.find(".gif") == std::string::npos && email.find(".webp") == std::string::npos) results.push_back(email);
        si = m.suffix().first; if (results.size() >= 20) break;
    }
    return results;
}

std::vector<std::string> PatternSet::find_phones(const std::string& text) const {
    std::vector<std::string> results; std::smatch m; auto si = text.cbegin();
    while (std::regex_search(si, text.cend(), m, phone_re_)) {
        const std::string phone = m[0].str(); std::string digits;
        for (char c : phone) if (std::isdigit(static_cast<unsigned char>(c)) || c == '+') digits += c;
        if (digits.length() >= 8 && digits.length() <= 18) results.push_back(phone);
        si = m.suffix().first; if (results.size() >= 10) break;
    }
    return results;
}

std::vector<std::string> PatternSet::find_urls(const std::string& text) const {
    std::vector<std::string> results; std::smatch m; auto si = text.cbegin();
    while (std::regex_search(si, text.cend(), m, url_re_)) { results.push_back(m[0].str()); si = m.suffix().first; if (results.size() >= 50) break; }
    return results;
}

std::vector<std::string> PatternSet::find_social_urls(const std::string& text) const {
    std::vector<std::string> results; std::smatch m; auto si = text.cbegin();
    while (std::regex_search(si, text.cend(), m, social_re_)) {
        std::string url = m[0].str();
        while (!url.empty() && (url.back() == ')' || url.back() == '"' || url.back() == '\'')) url.pop_back();
        results.push_back(url); si = m.suffix().first; if (results.size() >= 20) break;
    }
    return results;
}

bool PatternSet::has_injection_signals(const std::string& text, std::vector<std::string>& signals) const {
    signals.clear();
    const std::string sample = text.substr(0, std::min(text.size(), static_cast<size_t>(5000)));
    for (const auto& re : injection_res_) {
        std::smatch match;
        if (std::regex_search(sample, match, re) && match.size() > 0) signals.push_back(match[0].str());
    }
    return !signals.empty();
}

bool PatternSet::looks_blocked(int status, const std::string& text) const {
    if (status == 403 || status == 429 || status == 503) return true;
    std::string head = text.substr(0, std::min(text.size(), static_cast<size_t>(4000)));
    std::transform(head.begin(), head.end(), head.begin(), ::tolower);
    if (head.length() < 200) return false;
    for (const auto& marker : captcha_markers_) if (head.find(marker) != std::string::npos) return true;
    return false;
}

}  // namespace sweep
