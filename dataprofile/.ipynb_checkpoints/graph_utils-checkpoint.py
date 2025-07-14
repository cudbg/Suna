import networkx as nx
import matplotlib.pyplot as plt

def plot_graph(G, plot_path=None, coeffs=None):
    """
    Plot a given graph, with edge weights equal to coeffs.
    """
    plt.figure(figsize=(12, 8))
    pos = nx.kamada_kawai_layout(G, dist=None, weight='weight', scale=2)
    nx.draw(G, pos, with_labels=True, arrows=True, node_size=700, font_size=10)

    # Draw edge labels only if coefficients are provided
    if coeffs is not None:
        edge_labels = {k: round(v, 2) for k, v in coeffs.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    if plot_path:
        plt.savefig(plot_path)

# TODO: KeyError: 'type' in d['type']
def print_graph_edges(G):
    """
    Print out edges of a given graph.
    """
    for u, v, d in G.edges(data=True):
        edge_type = d['type']
        if edge_type == 'directed_XY':
            print(f"{u} -> {v}")
        elif edge_type == 'directed_YX':
            print(f"{v} -> {u}")
        elif edge_type == 'bidirected':
            print(f"{u} <-> {v}")
        elif edge_type == 'combined_XY':
            print(f"{u} -> {v} (with bi-directional component)")
        elif edge_type == 'combined_YX':
            print(f"{v} -> {u} (with bi-directional component)")


if __name__ == "__main__":
    print("Testing Graph Utils")
    from dataprofile.data_profile import DataProfile

    cur_dataprofile = DataProfile()
    cur_dataprofile.set_full_dag()
    cur_dataprofile.set_full_data_from_full_dag()
    # print_graph_edges(cur_dataprofile.G)
    plot_graph(cur_dataprofile.G, cur_dataprofile.edge_coeffs)
    plt.show()
