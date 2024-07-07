# Detection-of-DDOS
Using GNN
Introduction
The goal is to overwhelm a target's resources (such as servers, bandwidth, or network components) by flooding it with a massive amount of malicious traffic. To maximize the impact of their attack, attackers often use amplification techniques. These techniques involve sending a small request to a vulnerable server or device and causing it to generate a much larger response, which is then directed towards the target.

![image](https://github.com/KPchandra18/Detection-of-DDOS/assets/93926748/5af90453-29be-4916-9aa8-c8d54d76d5e7)
Graphs:
A graph can be mathematically defined as G=(V, E) where V = Vertex (or node) attributes and E = Edges (or link) attributes and directions. It is basically a structure that contains nodes and edges which interconnect these. Both nodes and edges can have features that describe them.
A graph can be mathematically defined as G=(V, E) where V = Vertex (or node) attributes and E = Edges (or link) attributes and directions. It is basically a structure that contains nodes and edges which interconnect these. Both nodes and edges can have features that describe them.


There are three main types of graphs: undirected, directed and weighted
![image](https://github.com/KPchandra18/Detection-of-DDOS/assets/93926748/6975054a-967b-40af-8c8f-3c75f014b815)
Introduction To GNN(Graph Neural Network):
GNNs are a type of advanced computer system that are really good at working with data that looks like a network or a map. They can understand the connections between things in a network, which is perfect for IoT networks.
It can work with data that has a graph structure, which is common in various real-world applications like social networks, biology, and telecommunications. .
GNNs begin by assigning initial feature vectors to nodes, such as attributes in a social network like age and interests. Then, they update these node features by gathering information from nearby nodes, aiding in understanding connections. This is achieved through message passing: nodes exchange messages with their neighbors, combining neighbor information (e.g., sum, mean) to refine their own feature vectors, allowing GNNs to learn about the graph's structure and relationships.
GNNs often consist of multiple layers, similar to deep neural networks, with each layer refining node embeddings by considering information from a broader neighborhood, allowing them to capture complex patterns in the graph. These refined embeddings can then be applied to various tasks; for instance, in recommendation systems, they can predict user interests based on their interactions in the graph. Additionally, there are various GNN variants, like GCNs, GraphSAGE, and GGNNs, each with its specific methods for aggregating information and handling different types of graphs or tasks.
We can do multiple types of classifications within a graph: 
             1. Graph-level: we can leverage a GNN to classify an entire graph. 
             2. Node-level: we can leverage a GNN to classify only certain nodes of the graph. 
             3. Edge-level: we can leverage a GNN to predict new edges within a graph

Uses Of Endpoint Traffic Graph: The endpoint traffic graph captures the entire interaction process between clients and servers. This graph includes information about traffic patterns, which can be categorized into two types: relationships among packets and relationships among flows. Packet relationships represent the structural details within a flow, whereas flow relationships convey details about bursts and recurring patterns in flows. Both of these types of information are valuable for distinguishing Distributed Denial of Service (DDoS) attacks.



