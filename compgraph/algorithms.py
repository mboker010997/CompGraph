import math
import json

from .graph import Graph
from . import operations


def word_count_graph(input_stream_name: str, text_column: str = 'text', count_column: str = 'count',
                     from_file: bool = False) -> Graph:
    """Constructs graph which counts words in text_column of all rows passed"""
    if from_file:
        input_graph = Graph.graph_from_file(input_stream_name, lambda string: json.loads(string))
    else:
        input_graph = Graph.graph_from_iter(input_stream_name)

    return input_graph \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column)) \
        .sort([text_column]) \
        .reduce(operations.Count(count_column), [text_column]) \
        .sort([count_column, text_column])


def inverted_index_graph(input_stream_name: str, doc_column: str = 'doc_id', text_column: str = 'text',
                         result_column: str = 'tf_idf', from_file: bool = False) -> Graph:
    """Constructs graph which calculates td-idf for every word/document pair"""
    if from_file:
        graph = Graph.graph_from_file(input_stream_name, lambda string: json.loads(string))
    else:
        graph = Graph.graph_from_iter(input_stream_name)

    input_graph = graph \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column))

    tf_graph = input_graph \
        .sort([doc_column]) \
        .reduce(operations.TermFrequency(text_column, result_column='tf'), [doc_column])

    count_graph = input_graph \
        .sort([doc_column]) \
        .reduce(operations.FirstReducer(), [doc_column]) \
        .reduce(operations.Count('total_docs_count'), [])

    df_graph = input_graph \
        .sort([doc_column, text_column]) \
        .reduce(operations.FirstReducer(), [doc_column, text_column]) \
        .sort([text_column]) \
        .reduce(operations.Count('docs_count'), [text_column])

    idf_graph = count_graph.join(operations.InnerJoiner(), df_graph, []) \
        .map(operations.FuncMapper('docs_count', lambda x: 1 / x)) \
        .map(operations.Product(['docs_count', 'total_docs_count'], 'idf')) \
        .map(operations.FuncMapper('idf', lambda x: math.log(x)))

    res_graph = tf_graph.sort([text_column]) \
        .join(operations.InnerJoiner(), idf_graph, [text_column]) \
        .map(operations.Product(['idf', 'tf'], result_column)) \
        .sort([text_column]) \
        .reduce(operations.TopN(result_column, 3), [text_column]) \
        .map(operations.Project([doc_column, text_column, result_column])) \
        .sort([doc_column])

    return res_graph


def pmi_graph(input_stream_name: str, doc_column: str = 'doc_id', text_column: str = 'text',
              result_column: str = 'pmi', from_file: bool = False) -> Graph:
    """Constructs graph which gives for every document the top 10 words ranked by pointwise mutual information"""
    if from_file:
        graph = Graph.graph_from_file(input_stream_name, lambda string: json.loads(string))
    else:
        graph = Graph.graph_from_iter(input_stream_name)

    input_graph = graph \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column)) \
        .sort([doc_column, text_column])

    filter_graph = input_graph \
        .map(operations.Filter(lambda row: len(row[text_column]) > 4)) \
        .sort([doc_column, text_column]) \
        .reduce(operations.Count('words_count'), [doc_column, text_column]) \
        .map(operations.Filter(lambda row: row['words_count'] >= 2)) \
        .map(operations.Project([doc_column, text_column])) \
        .sort([doc_column, text_column]) \
        .join(operations.InnerJoiner(), input_graph, [doc_column, text_column])

    tf_graph = filter_graph \
        .sort([doc_column]) \
        .reduce(operations.TermFrequency(text_column, result_column='tf'), [doc_column])

    words_in_docs_count = filter_graph \
        .sort([text_column]) \
        .reduce(operations.Count('docs_count'), [text_column])

    total_words_count = filter_graph \
        .reduce(operations.Count('total_words_count'), [])

    word_freq = words_in_docs_count.join(operations.InnerJoiner(), total_words_count, []) \
        .map(operations.FuncMapper('total_words_count', lambda x: 1 / x)) \
        .map(operations.Product(['docs_count', 'total_words_count'], 'word_freq'))

    res_graph = tf_graph.sort([text_column]).join(operations.InnerJoiner(), word_freq, [text_column]) \
        .map(operations.FuncMapper('word_freq', lambda x: 1 / x)) \
        .map(operations.Product(['word_freq', 'tf'], result_column)) \
        .map(operations.FuncMapper(result_column, lambda x: math.log(x))) \
        .map(operations.Project([doc_column, text_column, result_column])) \
        .sort([doc_column]) \
        .reduce(operations.TopN(result_column, 10), [doc_column])

    return res_graph


def yandex_maps_graph(input_stream_name_time: str, input_stream_name_length: str,
                      enter_time_column: str = 'enter_time', leave_time_column: str = 'leave_time',
                      edge_id_column: str = 'edge_id', start_coord_column: str = 'start', end_coord_column: str = 'end',
                      weekday_result_column: str = 'weekday', hour_result_column: str = 'hour',
                      speed_result_column: str = 'speed', from_file: bool = False) -> Graph:
    """Constructs graph which measures average speed in km/h depending on the weekday and hour"""
    if from_file:
        input_time_graph = Graph.graph_from_file(input_stream_name_time, lambda string: json.loads(string))
        input_length_graph = Graph.graph_from_file(input_stream_name_length, lambda string: json.loads(string))
    else:
        input_time_graph = Graph.graph_from_iter(input_stream_name_time)
        input_length_graph = Graph.graph_from_iter(input_stream_name_length)

    time_graph = input_time_graph \
        .map(operations.DateMapper(enter_time_column, leave_time_column,
                                   hour_result_column, weekday_result_column, 'delta')) \
        .sort([edge_id_column])

    length_graph = input_length_graph \
        .map(operations.DistanceMapper(start_coord_column, end_coord_column, 'dist')) \
        .sort([edge_id_column])

    info_graph = time_graph \
        .join(operations.InnerJoiner(), length_graph, [edge_id_column]) \
        .map(operations.FuncMapper('delta', lambda x: 3600 / x)) \
        .map(operations.Product(['delta', 'dist'], 'sum_speed')) \
        .sort([weekday_result_column, hour_result_column])

    sum_graph = info_graph \
        .reduce(operations.Sum('sum_speed'), [weekday_result_column, hour_result_column])

    count_graph = info_graph \
        .reduce(operations.Count('count'), [weekday_result_column, hour_result_column])

    result_graph = sum_graph \
        .join(operations.InnerJoiner(), count_graph, [weekday_result_column, hour_result_column]) \
        .map(operations.FuncMapper('count', lambda x: 1 / x)) \
        .map(operations.Product(['sum_speed', 'count'], speed_result_column)) \
        .map(operations.Project([weekday_result_column, hour_result_column, speed_result_column]))

    return result_graph
