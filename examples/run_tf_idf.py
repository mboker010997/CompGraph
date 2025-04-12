import click

from compgraph.algorithms import inverted_index_graph


@click.command()
@click.argument('input_filename')
@click.argument('output_filename')
def tf_idf(input_filename: str, output_filename: str) -> None:
    graph = inverted_index_graph(input_stream_name=input_filename, from_file=True)

    result = graph.run()
    with open(output_filename, "w") as out:
        for row in result:
            print(row, file=out)


if __name__ == "__main__":
    tf_idf()
