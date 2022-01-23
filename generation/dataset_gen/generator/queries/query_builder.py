from generator.queries.sub_graph import SubGraph
from generator.utils import extract_all_root_to_leaves_paths, extract_flattened_elements


class QueryBuilder:
    @staticmethod
    def _build_match_query(where: SubGraph, split, prefix=""):
        if where.multi_count and where.multi_count > 1:
            # limiting due to performance issues with this query
            where.multi_count = min(where.multi_count, 2)
            repeat = where.multi_count
            cypher_query_str = ""
            where_elements = []
            for i in range(repeat):
                prefix_i = prefix if i == 0 else f"multi_{i}_{prefix}"
                cypher_query_str_i, where_elements_i = QueryBuilder.build_match_clause(where, prefix_symbol=prefix_i)
                cypher_query_str += cypher_query_str_i + "\n"
                where_elements += where_elements_i
                if i > 0:
                    where_elements.append(f"{where[0].char_symbol} <> "
                                          f"multi_{i}_{where[0].char_symbol}")
        else:
            cypher_query_str, where_elements = QueryBuilder.build_match_clause(where, prefix, prefix)

        cypher_query_str += f" WHERE {prefix}scene.split = '{split}' "
        if where_elements:
            cypher_query_str += " AND (" + " AND ".join(where_elements) + ") "

        return cypher_query_str

    @staticmethod
    def build_match_clause(sub_graph, prefix_scene="", prefix_symbol=""):
        paths_to_match = extract_all_root_to_leaves_paths(sub_graph.root)
        cypher_query_matches = []
        where_elements = set()
        for path in paths_to_match:
            match_str = f'MATCH ({prefix_scene}scene:Scene)-[{prefix_symbol}r:IN]-'
            for i, query_element in enumerate(path):
                node_query_str, node_where_elements = query_element.as_cyhper_element(char_symbol_prefix=prefix_symbol)
                match_str += node_query_str
                if i < len(path) - 1:
                    match_str += query_element.get_trailing_symbol()
                if node_where_elements:
                    where_elements.add(node_where_elements)

                if query_element.parallel_element:
                    parallel = query_element.parallel_element
                    if not query_element.name or not parallel.name:
                        pair = sorted((query_element.char_symbol, parallel.char_symbol))
                        where_elements.add(f"({pair[0]} <> {pair[1]})")
            cypher_query_matches.append(match_str)
        cypher_query_str = "\n".join(cypher_query_matches)
        return cypher_query_str, sorted(list(where_elements))

    @staticmethod
    def build_query(sub_graph, split):
        positive_match = QueryBuilder._build_match_query(sub_graph, split)

        cypher_query_str = positive_match + "\n\n"

        returned_elements_str = ', '.join([n.char_symbol for n in extract_flattened_elements(sub_graph.root)])
        cypher_query_str += f" RETURN {returned_elements_str}, scene.scene_id LIMIT 500"

        return cypher_query_str
