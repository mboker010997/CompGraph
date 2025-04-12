import click

from compgraph.algorithms import word_count_graph


@click.command()
@click.argument('input_filename')
@click.argument('output_filename')
def word_count(input_filename: str, output_filename: str) -> None:
    graph = word_count_graph(input_stream_name=input_filename,
                             text_column='text',
                             count_column='count',
                             from_file=True)

    result = graph.run()
    with open(output_filename, "w") as out:
        for row in result:
            print(row, file=out)


if __name__ == "__main__":
    word_count()
