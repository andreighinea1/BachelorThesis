from torchview import draw_graph
import pygraphviz as pgv
import os


def convert_dot_to_png(input_file_name: str, dpi=500, *, remove_dot_file=True):
    # Load the .dot file
    graph = pgv.AGraph(input_file_name)

    # Set the DPI (resolution)
    graph.layout(prog='dot')  # 'dot', 'neato', 'twopi', 'circo', etc.

    output_fn_no_ext = input_file_name.removesuffix('.dot')
    graph.draw(output_fn_no_ext + '.png', prog='dot', format='png', args=f"-Gdpi={dpi}")

    # Remove the input file
    if remove_dot_file:
        os.remove(input_file_name)


def make_graph(
        _model,
        mine,
        detailed,
        *,
        fontname="Arial",  # "Times-Roman", "Linux libertine"
        batch_size=1,
        channel_size=3,
        img_size=(224, 224),
        return_graph=False,
        depth=None,
):
    temp_name = f"model_{'mine' if mine else 'loaded'}_{'detailed' if detailed else 'simple'}.dot"
    if depth is None:
        depth = 4 if detailed else 3

    # device="meta" -> no memory is consumed for visualization
    model_graph = draw_graph(_model, input_size=(batch_size, channel_size, *img_size), device="meta",
                             expand_nested=True, hide_inner_tensors=True, hide_module_functions=False,
                             depth=depth, roll=True,
                             graph_dir="TB")
    model_graph.visual_graph.node_attr["fontname"] = fontname
    model_graph.visual_graph.save(filename=temp_name)
    convert_dot_to_png(temp_name)

    if return_graph:
        return model_graph.visual_graph
