from abc import ABC, abstractmethod
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn as nn

import torch


class BasePolicyNetwork(ABC,torch.nn.Module):
    @abstractmethod
    def forward(self):
        pass
    
    def draw_graph(self):
        return self.__visualize_network__()
    
    def __visualize_network__(self):
        G = nx.DiGraph()
        layers = []
        node_positions = {}
        node_labels = {}
        node_colors = []
        edge_weights = {}
        
        def add_layer(layer_index, node_counts , center_position):
            nonlocal G, node_positions, node_labels, node_colors
            
            layer_name = f"Layer {layer_index}"
            
            for node_index in range(node_counts):
                node_name = f"{layer_name}_Node {node_index}"
                G.add_node(node_name)
                y_position = center_position - (node_counts - 1) * 1 + node_index * 2
                
                node_positions[node_name] = (layer_index, y_position)
                node_labels[node_name] = f"{node_index+1}"
                node_colors.append(layer_index) 

        total_layers = sum(1 for _ in self.children() if isinstance(_, nn.Linear))
        center_position = (total_layers - 1) * 2 / 2
        
        for layer_index, layer in enumerate(self.children()):
            if isinstance(layer, nn.Linear):
                if layer_index == 0:
                    layers.append(layer)
                    node_counts = layer.in_features
                    add_layer(layer_index, node_counts , center_position)
                    
                layers.append(layer)
                node_counts = layer.out_features
                add_layer(layer_index + 1, node_counts, center_position)

        for layer_index,layer in enumerate(self.children()):
            if isinstance(layer, nn.Linear):
                weights = layer.weight.cpu().detach().numpy()
                
                layer1_name = f"Layer {layer_index}"
                layer2_name = f"Layer {layer_index + 1}"
                layer1_size = layers[layer_index].out_features
                layer2_size = layers[layer_index + 1].out_features
                
                if layer_index == 0:
                    layer1_size = layers[layer_index].in_features
                    layer2_size = layers[layer_index].out_features
                    

                for i in range(layer1_size):
                    for j in range(layer2_size):
                        weight = abs(weights[j, i])
                        
                        if weight > 0.2:
                            weight = weights[j, i]
                            node1 = f"{layer1_name}_Node {i}"
                            node2 = f"{layer2_name}_Node {j}"
                            
                            edge_weights[(node1, node2)] = f"{weight:.2f}"
                            G.add_edge(node1, node2 , weight=abs(weight))

        
        pos = {node: (x, -y) for (node, (x, y)) in node_positions.items()}
        
        edges = G.edges(data=True)
        weights = [data['weight'] for _, _, data in edges]
        
        edge_weight_pos = []
        edge_labels = []
        
        for (u, v, d) in G.edges(data=True):
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            edge_labels.append(f'{d["weight"]:.2f}')
            edge_weight_pos.append(((x1 + x2) / 2, (y1 + y2) / 2))
        
        nx.draw(G, pos, with_labels=False, node_size=700, node_color=node_colors, edge_color=weights, width=2, edge_cmap=plt.cm.Blues, cmap=plt.cm.viridis)
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights, font_size=8, font_color='red')

        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color="black")
        
        return G