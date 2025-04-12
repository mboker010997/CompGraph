import click

from compgraph.algorithms import pmi_graph


@click.command()
@click.argument('input_filename')
@click.argument('output_filename')
def pmi(input_filename: str, output_filename: str) -> None:
    graph = pmi_graph(input_stream_name=input_filename, from_file=True)

    result = graph.run()
    with open(output_filename, "w") as out:
        for row in result:
            print(row, file=out)


if __name__ == "__main__":
    pmi()
