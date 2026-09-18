#include "cpp_engine/data_engine.h"
#include <cassert>
#include <string>

int main() {
    const auto df = sweep::parse_csv("name,value,flag,empty\nalpha,1,true,\nbeta,2,false,NULL\n", true);
    assert(df.num_rows() == 2);
    assert(df.num_cols() == 4);

    const auto csv = sweep::dataframe_to_csv(df);
    assert(csv.find("alpha") != std::string::npos);
    assert(csv.find("beta") != std::string::npos);
    assert(csv.find("true") != std::string::npos);
    assert(csv.find("false") != std::string::npos);

    auto duplicated = df;
    duplicated.rows.push_back(duplicated.rows.front());
    const auto unique = sweep::df_drop_duplicates(duplicated);
    assert(unique.num_rows() == 2);

    const auto grouped = sweep::df_group_by(df, "name", "sum");
    assert(grouped.num_rows() == 2);
    return 0;
}
