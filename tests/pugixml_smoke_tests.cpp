#include <pugixml.hpp>

#include <cstdio>
#include <string>

namespace {
int g_failures = 0;
void check(bool condition, const char* expr, const char* file, int line) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s (%s:%d)\n", expr, file, line);
        g_failures++;
    }
}
} // namespace
#define CHECK(cond) check((cond), #cond, __FILE__, __LINE__)

void test_parsesSimpleView() {
    pugi::xml_document doc;
    const pugi::xml_parse_result result = doc.load_string(
        R"(<View><Image name="image" value="$image"/><RectangleLabels name="label" toName="image">)"
        R"(<Label value="Person"/><Label value="Car"/></RectangleLabels></View>)");
    CHECK(result.status == pugi::status_ok);

    const pugi::xml_node imageNode = doc.select_node("//Image").node();
    CHECK(std::string(imageNode.attribute("name").value()) == "image");

    const pugi::xml_node rectNode = doc.select_node("//RectangleLabels").node();
    CHECK(std::string(rectNode.attribute("toName").value()) == "image");

    int labelCount = 0;
    for (pugi::xml_node label : rectNode.children("Label")) {
        (void)label;
        labelCount++;
    }
    CHECK(labelCount == 2);
}

int main() {
    test_parsesSimpleView();
    if (g_failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::printf("%d test(s) failed.\n", g_failures);
    return 1;
}
