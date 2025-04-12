import click

from compgraph.algorithms import yandex_maps_graph


@click.command()
@click.argument('time_input')
@click.argument('length_input')
@click.argument('output_filename')
def maps(time_input: str, length_input: str, output_filename: str) -> None:
    graph = yandex_maps_graph(input_stream_name_time=time_input,
                              input_stream_name_length=length_input,
                              from_file=True)

    result = graph.run()
    with open(output_filename, "w") as out:
        for row in result:
            print(row, file=out)


if __name__ == "__main__":
    maps()
