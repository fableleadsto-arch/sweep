#include "cpp_engine/html_parser.h"
#include <algorithm>
#include <cctype>
#include <regex>
#include <sstream>
#include <unordered_set>

namespace sweep {

static std::string trim_copy(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return "";
    const auto last = value.find_last_not_of(" \t\n\r");
    return value.substr(first, last - first + 1);
}

static std::string decode_entities(const std::string& input) {
    std::string out = input;
    const std::vector<std::pair<std::string, std::string>> replacements = {
        {"&nbsp;", " "}, {"&amp;", "&"}, {"&quot;", "\""}, {"&lt;", "<"},
        {"&gt;", ">"}, {"&mdash;", "—"}, {"&ndash;", "–"}, {"&hellip;", "…"},
        {"&#039;", "'"}, {"&apos;", "'"}
    };
    for (const auto& [from, to] : replacements) {
        size_t pos = 0;
        while ((pos = out.find(from, pos)) != std::string::npos) {
            out.replace(pos, from.size(), to);
            pos += to.size();
        }
    }
    return out;
}

static std::string strip_noise(const std::string& html) {
    const std::vector<std::pair<std::string, std::string>> blocks = {
        {"<!--", "-->"}, {"<script", "</script>"}, {"<style", "</style>"},
        {"<noscript", "</noscript>"}, {"<svg", "</svg>"}, {"<iframe", "</iframe>"},
        {"<form", "</form>"}
    };
    std::string out = html;
    for (const auto& [open, close] : blocks) {
        size_t pos = 0;
        while ((pos = out.find(open, pos)) != std::string::npos) {
            const size_t close_pos = out.find(close, pos + open.size());
            const size_t end = close_pos == std::string::npos ? pos + open.size() : close_pos + close.size();
            out.erase(pos, end - pos);
        }
    }
    return out;
}

static std::string strip_tags(const std::string& value) {
    std::string out;
    bool in_tag = false;
    for (char c : value) {
        if (c == '<') in_tag = true;
        else if (c == '>') in_tag = false;
        else if (!in_tag) out += c;
    }
    return decode_entities(trim_copy(out));
}

std::string html_to_text(const std::string& html) {
    const std::string cleaned = strip_noise(html);
    std::string result;
    bool in_tag = false;
    for (char c : cleaned) {
        if (c == '<') { in_tag = true; result += ' '; }
        else if (c == '>') in_tag = false;
        else if (!in_tag) result += c;
    }
    std::string collapsed;
    bool previous_space = false;
    for (char c : result) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!previous_space) collapsed += ' ';
            previous_space = true;
        } else { collapsed += c; previous_space = false; }
    }
    return decode_entities(trim_copy(collapsed));
}

std::string extract_main_region(const std::string& html) {
    const std::vector<std::regex> patterns = {
        std::regex(R"re(<article[^>]*>([\s\S]*?)</article>)re", std::regex::icase),
        std::regex(R"re(<main[^>]*>([\s\S]*?)</main>)re", std::regex::icase),
        std::regex(R"re(<div[^>]+(?:id|class)="[^"]*(?:post|content|entry|story|body)[^"]*"[^>]*>([\s\S]*?)</div>)re", std::regex::icase),
        std::regex(R"re(<body[^>]*>([\s\S]*?)</body>)re", std::regex::icase)
    };
    std::string best;
    size_t best_length = 0;
    for (const auto& pattern : patterns) {
        std::smatch match;
        auto start = html.cbegin();
        while (std::regex_search(start, html.cend(), match, pattern)) {
            if (match.size() > 1) {
                const std::string candidate = match[1].str();
                const size_t length = html_to_text(candidate).size();
                if (length > best_length) { best_length = length; best = candidate; }
            }
            start = match.suffix().first;
        }
    }
    return best.empty() ? html : best;
}

static std::string get_attr(const std::string& tag, const std::string& attr) {
    const std::regex pattern(attr + R"re(\s*=\s*["']([^"']*)["'])re", std::regex::icase);
    std::smatch match;
    return std::regex_search(tag, match, pattern) && match.size() > 1 ? match[1].str() : "";
}

PageMeta extract_meta(const std::string& html) {
    PageMeta meta;
    std::smatch match;
    const std::regex title_pattern(R"re(<title[^>]*>([\s\S]*?)</title>)re", std::regex::icase);
    if (std::regex_search(html, match, title_pattern) && match.size() > 1) meta.title = strip_tags(match[1].str());
    const std::regex meta_pattern(R"re(<meta\s+(?:name|property)=["']([^"']*)["']\s+content=["']([^"']*)["'])re", std::regex::icase);
    auto start = html.cbegin();
    while (std::regex_search(start, html.cend(), match, meta_pattern)) {
        std::string name = match[1].str();
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        const std::string content = match[2].str();
        if (name == "description") meta.description = content;
        else if (name == "og:site_name" || name == "site-name") meta.site_name = content;
        else if (name == "author") meta.author = content;
        else if (name == "lang") meta.lang = content;
        start = match.suffix().first;
    }
    if (meta.lang.empty()) {
        const std::regex lang_pattern(R"re(<html[^>]+lang=["']([^"']*)["'])re", std::regex::icase);
        if (std::regex_search(html, match, lang_pattern) && match.size() > 1) meta.lang = match[1].str();
    }
    const std::regex canonical_pattern(R"re(<link[^>]+rel=["']canonical["'][^>]+href=["']([^"']*)["'])re", std::regex::icase);
    if (std::regex_search(html, match, canonical_pattern) && match.size() > 1) meta.canonical = match[1].str();
    return meta;
}

static std::string resolve_url(const std::string& href, const std::string& base_url) {
    if (href.rfind("http://", 0) == 0 || href.rfind("https://", 0) == 0) return href;
    if (href.empty() || href[0] != '/') return "";
    const auto scheme_end = base_url.find("://");
    if (scheme_end == std::string::npos) return "";
    const auto host_end = base_url.find('/', scheme_end + 3);
    return base_url.substr(0, host_end == std::string::npos ? base_url.size() : host_end) + href;
}

std::vector<Link> extract_links(const std::string& html, const std::string& base_url) {
    std::vector<Link> links;
    std::unordered_set<std::string> seen;
    const std::regex link_pattern(R"re(<a[^>]+href=["']([^"']*)["'][^>]*>([\s\S]*?)</a>)re", std::regex::icase);
    const std::vector<std::pair<std::regex, std::string>> intents = {
        {std::regex(R"re(\bpricing\b)re", std::regex::icase), "pricing"},
        {std::regex(R"re(\bdocs?\b|documentation|api\s*reference)re", std::regex::icase), "documentation"},
        {std::regex(R"re(\bfaq\b|frequently\s+asked)re", std::regex::icase), "faq"},
        {std::regex(R"re(\babout\b|our\s+story)re", std::regex::icase), "about"},
        {std::regex(R"re(\bcontact\b|support\b)re", std::regex::icase), "contact"},
        {std::regex(R"re(\bgithub\.com\b|source\s+code)re", std::regex::icase), "github"}
    };
    std::smatch match;
    auto start = html.cbegin();
    while (std::regex_search(start, html.cend(), match, link_pattern)) {
        const std::string href = resolve_url(match[1].str(), base_url);
        const std::string text = strip_tags(match[2].str());
        if (!href.empty() && seen.insert(href).second) {
            std::string intent;
            const std::string haystack = text + " " + href;
            for (const auto& [pattern, label] : intents) if (std::regex_search(haystack, pattern)) { intent = label; break; }
            const auto base_scheme = base_url.find("://");
            const auto link_scheme = href.find("://");
            const auto base_host_start = base_scheme == std::string::npos ? 0 : base_scheme + 3;
            const auto link_host_start = link_scheme == std::string::npos ? 0 : link_scheme + 3;
            const auto base_host_end = base_url.find('/', base_host_start);
            const auto link_host_end = href.find('/', link_host_start);
            const std::string base_host = base_url.substr(base_host_start, base_host_end == std::string::npos ? base_url.size() : base_host_end - base_host_start);
            const std::string link_host = href.substr(link_host_start, link_host_end == std::string::npos ? href.size() : link_host_end - link_host_start);
            links.push_back({href, text, intent, base_host != link_host});
            if (links.size() >= 200) break;
        }
        start = match.suffix().first;
    }
    return links;
}

static std::vector<Heading> headings_from_markdown(const std::string& markdown) {
    std::vector<Heading> headings;
    std::istringstream stream(markdown);
    std::string line;
    while (std::getline(stream, line)) {
        size_t level = 0;
        while (level < line.size() && line[level] == '#') level++;
        if (level == 0 || level > 6 || level >= line.size() || line[level] != ' ') continue;
        const std::string text = trim_copy(line.substr(level + 1));
        std::string id;
        for (char c : text) if (std::isalnum(static_cast<unsigned char>(c)) || c == ' ' || c == '-') id += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        std::replace(id.begin(), id.end(), ' ', '-');
        headings.push_back({static_cast<int>(level), text, id});
        if (headings.size() >= 80) break;
    }
    return headings;
}

ParseResult html_to_markdown(const std::string& html, const std::string& url, int max_chars) {
    ParseResult result;
    result.meta = extract_meta(html);
    result.links = extract_links(html, url);
    const std::string region = extract_main_region(html);
    const std::regex element_pattern(R"re(<(h[1-6]|p|li|pre|blockquote|td|th|tr|br|hr|strong|em|b|i|a|code)[^>]*>([\s\S]*?)</\1>)re", std::regex::icase);
    std::string markdown;
    std::smatch match;
    auto start = region.cbegin();
    while (std::regex_search(start, region.cend(), match, element_pattern)) {
        std::string tag = match[1].str();
        std::transform(tag.begin(), tag.end(), tag.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        const std::string content = strip_tags(match[2].str());
        if (!content.empty()) {
            if (tag.size() == 2 && tag[0] == 'h' && std::isdigit(static_cast<unsigned char>(tag[1]))) markdown += std::string(tag[1] - '0', '#') + " " + content + "\n\n";
            else if (tag == "li") markdown += "- " + content + "\n";
            else if (tag == "pre") markdown += "```\n" + content + "\n```\n\n";
            else if (tag == "blockquote") markdown += "> " + content + "\n\n";
            else markdown += content + "\n\n";
        }
        start = match.suffix().first;
    }
    result.markdown = markdown;
    result.text = html_to_text(markdown);
    if (max_chars < 1) max_chars = 1;
    if (static_cast<int>(result.markdown.size()) > max_chars) { result.markdown.resize(static_cast<size_t>(max_chars)); result.truncated = true; }
    if (static_cast<int>(result.text.size()) > max_chars) { result.text.resize(static_cast<size_t>(max_chars)); result.truncated = true; }
    std::istringstream words(result.text);
    std::string word;
    while (words >> word) result.word_count++;
    result.headings = headings_from_markdown(result.markdown);
    return result;
}

ParseResult json_to_markdown(const std::string& json_str, const std::string& url, int max_chars) {
    (void)url;
    ParseResult result;
    result.meta.title = "JSON Response";
    if (max_chars < 1) max_chars = 1;
    result.text = json_str.substr(0, static_cast<size_t>(max_chars));
    result.markdown = result.text;
    result.truncated = json_str.size() > static_cast<size_t>(max_chars);
    std::istringstream words(result.text);
    std::string word;
    while (words >> word) result.word_count++;
    return result;
}

}  // namespace sweep
