import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List

from utils import seed_it

# Constants
CPU = 0
MEM = 1
STOR = 2

seed_it(646)

class FogNode:
    """Class to represent a fog node in the network"""
    def __init__(self, node_id, total_resources):
        self.id = node_id
        self.total_resources = np.array(total_resources)  # [CPU, MEM, STOR]
        self.available_resources = np.array(total_resources.copy())
        self.placed_microservices = []
        self.idle_power = random.uniform(10, 20)  # Watts
        self.max_power = random.uniform(80, 120)  # Watts at full utilization
        
    def can_host(self, microservice):
        """Check if the fog node can host the microservice"""
        return all(microservice.resource_demand <= self.available_resources)
    
    def place_microservice(self, microservice):
        """Place a microservice on the fog node"""
        if not self.can_host(microservice):
            return False
        
        self.available_resources -= microservice.resource_demand
        self.placed_microservices.append(microservice)
        return True
    
    def remove_microservice(self, microservice):
        """Remove a microservice from the fog node"""
        if microservice in self.placed_microservices:
            self.available_resources += microservice.resource_demand
            self.placed_microservices.remove(microservice)
            return True
        return False
    
    def get_cpu_utilization(self):
        """Calculate CPU utilization of the node"""
        used_cpu = self.total_resources[CPU] - self.available_resources[CPU]
        return used_cpu / self.total_resources[CPU]
    
    def get_mem_utilization(self):
        """Calculate memory utilization of the node"""
        used_mem = self.total_resources[MEM] - self.available_resources[MEM]
        return used_mem / self.total_resources[MEM]
    
    def get_computational_energy(self, delta_t=1):
        """Calculate energy consumption for computation"""
        cpu_util = self.get_cpu_utilization()
        energy = 0
        if len(self.placed_microservices) > 0:  # If the node is active
            energy = (self.idle_power + (self.max_power - self.idle_power) * cpu_util) * delta_t
        return energy

class Link:
    """Class to represent a network link between fog nodes"""
    def __init__(self, link_id, source, destination, bandwidth, energy_per_bit):
        self.id = link_id
        self.source = source
        self.destination = destination
        self.total_bandwidth = bandwidth  # in Mbps
        self.available_bandwidth = bandwidth
        self.energy_per_bit = energy_per_bit  # in J/bit
        self.delay = random.uniform(1, 10)  # ms
        self.routed_edges = []
    
    def can_route(self, invocation_edge):
        """Check if the link can route the invocation edge"""
        return invocation_edge.data_size <= self.available_bandwidth
    
    def route_edge(self, invocation_edge):
        """Route an invocation edge through this link"""
        if not self.can_route(invocation_edge):
            return False
        
        self.available_bandwidth -= invocation_edge.data_size
        self.routed_edges.append(invocation_edge)
        return True
    
    def get_communication_energy(self):
        """Calculate energy consumption for communication"""
        energy = 0
        for edge in self.routed_edges:
            energy += self.energy_per_bit * edge.data_size * 1e6  # Convert Mb to bits
        return energy

class Microservice:
    """Class to represent a microservice in a service DAG"""
    def __init__(self, ms_id, service_id, resource_demand, computational_intensity):
        self.id = ms_id
        self.service_id = service_id
        self.resource_demand = np.array(resource_demand)  # [CPU, MEM, STOR]
        self.computational_intensity = computational_intensity  # cycles/request
        self.placed_on = None  # The fog node this microservice is placed on
    
    def get_processing_time(self, fog_node):
        """Calculate processing time on a given fog node"""
        # Processing time = computational_intensity / CPU throughput
        cpu_throughput = fog_node.total_resources[CPU]  # Simplified, assuming CPU units are in cycles/s
        return self.computational_intensity / cpu_throughput

class InvocationEdge:
    """Class to represent a dependency edge between microservices"""
    def __init__(self, edge_id, service_id, source_ms, dest_ms, data_size):
        self.id = edge_id
        self.service_id = service_id
        self.source_ms = source_ms
        self.dest_ms = dest_ms
        self.data_size = data_size  # in Mbps
        self.routed_through = []  # List of physical links this edge is routed through

class Service:
    """Class to represent a service request as a DAG"""
    def __init__(self, service_id, deadline, lifetime, arrival_rate=1.0):
        self.id = service_id
        self.microservices: List[Microservice] = []
        self.invocation_edges: List[InvocationEdge] = []
        self.deadline = deadline  # maximum tolerable end-to-end delay
        self.lifetime = lifetime  # duration this service exists in the system # TODO: not yet used!
        self.arrival_rate = arrival_rate  # Poisson arrival rate of requests
        self.entry_microservice = None  # First microservice in the DAG
    
    def add_microservice(self, resource_demand, computational_intensity):
        """Add a microservice to the service DAG"""
        ms_id = f"ms_{self.id}_{len(self.microservices)}"
        microservice = Microservice(ms_id, self.id, resource_demand, computational_intensity)
        self.microservices.append(microservice)
        
        # If this is the first microservice, set it as the entry point
        if len(self.microservices) == 1:
            self.entry_microservice = microservice
            
        return microservice
    
    def add_invocation_edge(self, source_ms, dest_ms, data_size):
        """Add an invocation edge between microservices"""
        edge_id = f"edge_{self.id}_{len(self.invocation_edges)}"
        edge = InvocationEdge(edge_id, self.id, source_ms, dest_ms, data_size)
        self.invocation_edges.append(edge)
        return edge
    
    def calculate_end_to_end_latency(self, source_device_location=None):
        """Calculate the total end-to-end latency for the service"""
        # 1. Calculate access delay (if entry microservice is placed)
        access_delay = 0
        if self.entry_microservice.placed_on and source_device_location:
            # Simplified access delay calculation
            distance = np.linalg.norm(np.array(source_device_location) - 
                                     np.array([self.entry_microservice.placed_on.id % 10, 
                                              self.entry_microservice.placed_on.id // 10]))
            access_delay = distance * 2  # Simplified model: 2ms per unit distance
        
        # 2. Calculate processing delay
        processing_delay = 0
        for ms in self.microservices:
            if ms.placed_on:
                processing_delay += ms.get_processing_time(ms.placed_on)
        
        # 3. Calculate queuing delay (simplified model)
        queuing_delay = 0
        for ms in self.microservices:
            if ms.placed_on:
                # Simple M/M/1 queuing model
                service_rate = ms.placed_on.total_resources[CPU] / ms.computational_intensity
                utilization = min(0.99, self.arrival_rate / service_rate)  # Cap at 0.99 to avoid division by 0
                if utilization < 1:  # Stability condition
                    queue_delay = (utilization / (service_rate * (1 - utilization)))
                    queuing_delay += queue_delay
                else:
                    queuing_delay += float('inf')  # Queue is unstable
        
        # 4. Calculate network delay
        network_delay = 0
        for edge in self.invocation_edges:
            path_delay = 0
            for link in edge.routed_through:
                path_delay += link.delay
            network_delay += path_delay
        
        total_latency = access_delay + processing_delay + queuing_delay + network_delay
        return total_latency

class FogNetwork:
    """Class to represent the fog computing network"""
    def __init__(self):
        self.fog_nodes: List[FogNode] = []
        self.links = []
        self.services: List[Service] = []
        self.graph = nx.Graph()
        
    def generate_random_topology(self, num_nodes=20, connectivity=0.3):
        """Generate a random network topology"""
        # Create fog nodes with random resources
        for i in range(num_nodes):
            cpu = random.uniform(1000, 5000)  # CPU in cycles/s
            mem = random.uniform(4, 16)      # Memory in GB
            stor = random.uniform(50, 200)    # Storage in GB
            
            # Convert to common units
            cpu_units = cpu
            mem_units = mem * 1024  # Convert GB to MB
            stor_units = stor * 1024  # Convert GB to MB
            
            node = FogNode(i, [cpu_units, mem_units, stor_units])
            self.fog_nodes.append(node)
            self.graph.add_node(i, resources=node.total_resources, node=node)
        
        # Create links between nodes
        link_id = 0
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if random.random() < connectivity:
                    bandwidth = random.uniform(100, 1000)  # Bandwidth in Mbps
                    energy_per_bit = random.uniform(1e-9, 5e-9)  # Energy in J/bit
                    
                    link = Link(link_id, i, j, bandwidth, energy_per_bit)
                    self.links.append(link)
                    self.graph.add_edge(i, j, bandwidth=bandwidth, delay=link.delay, link=link)
                    
                    link_id += 1
        
        # Ensure the graph is connected
        if not nx.is_connected(self.graph):
            components = list(nx.connected_components(self.graph))
            for i in range(len(components) - 1):
                # Connect components by adding edges
                node1 = random.choice(list(components[i]))
                node2 = random.choice(list(components[i+1]))
                
                bandwidth = random.uniform(100, 1000)
                energy_per_bit = random.uniform(1e-9, 5e-9)
                
                link = Link(link_id, node1, node2, bandwidth, energy_per_bit)
                self.links.append(link)
                self.graph.add_edge(node1, node2, bandwidth=bandwidth, delay=link.delay, link=link)
                
                link_id += 1
    
    def generate_random_service(self, num_microservices=5, connectivity=0.4):
        """Generate a random service DAG"""
        service_id = len(self.services)
        deadline = random.uniform(100, 500)  # ms
        lifetime = 10 #random.uniform(100, 1000)  # s
        arrival_rate = random.uniform(0.5, 5)  # requests/s
        
        service = Service(service_id, deadline, lifetime, arrival_rate)
        
        # Add microservices
        for _ in range(num_microservices):
            cpu_demand = random.uniform(100, 500)  # CPU in cycles/s
            mem_demand = random.uniform(128, 1024)  # Memory in MB
            stor_demand = random.uniform(256, 1024)  # Storage in MB
            comp_intensity = random.uniform(1e6, 5e6)  # Computational intensity in cycles/request
            
            service.add_microservice([cpu_demand, mem_demand, stor_demand], comp_intensity)
        
        # Add invocation edges to create a DAG
        # Ensure the graph is connected and acyclic
        # for i in range(num_microservices):
        #     for j in range(i+1, num_microservices):  # Ensure acyclic (i -> j where i < j)
        #         if random.random() < connectivity:
        #             data_size = random.uniform(1, 20)  # Data size in Mbps
        #             service.add_invocation_edge(service.microservices[i], service.microservices[j], data_size)
        
        # just chain the microservices for simplicity
        # The assumption is if a ms is to be reused for that specific service, let it be redefined as a different microservice
        for i in range(num_microservices - 1):
            data_size = random.uniform(1, 20)
            service.add_invocation_edge(service.microservices[i], service.microservices[i+1], data_size)


        self.services.append(service)
        return service
    
    def place_microservices_random(self, service):
        """Place microservices randomly on fog nodes"""
        for ms in service.microservices:
            # Try to find a node that can host the microservice
            eligible_nodes = [node for node in self.fog_nodes if node.can_host(ms)]
            
            if not eligible_nodes:
                print(f"No eligible nodes found for microservice {ms.id}")
                return False
            
            # Randomly select a node
            selected_node = random.choice(eligible_nodes)
            selected_node.place_microservice(ms)
            ms.placed_on = selected_node
        
        return True
    
    def route_invocation_edges_random(self, service):
        """Route invocation edges randomly through the network"""
        for edge in service.invocation_edges:
            # Get the fog nodes where the source and destination microservices are placed
            source_node = edge.source_ms.placed_on
            dest_node = edge.dest_ms.placed_on
            
            if not source_node or not dest_node:
                print(f"Source or destination not placed for edge {edge.id}")
                return False
            
            # Find a path from source to destination
            try:
                path = nx.shortest_path(self.graph, source=source_node.id, target=dest_node.id, 
                                      weight='delay')
                
                # Route the edge through each link in the path
                for i in range(len(path) - 1):
                    # Find the corresponding link
                    for link in self.links:
                        if ((link.source == path[i] and link.destination == path[i+1]) or 
                            (link.source == path[i+1] and link.destination == path[i])):
                            
                            if not link.can_route(edge):
                                print(f"Link {link.id} cannot route edge {edge.id}")
                                return False
                            
                            link.route_edge(edge)
                            edge.routed_through.append(link)
                            break
                    
            except nx.NetworkXNoPath:
                print(f"No path found for edge {edge.id}")
                return False
        
        return True
    
    def calculate_load_balance_metric(self):
        """Calculate the load balance metric"""
        # Calculate average CPU and memory utilization
        avg_cpu_util = 0
        avg_mem_util = 0
        
        for node in self.fog_nodes:
            avg_cpu_util += node.get_cpu_utilization()
            avg_mem_util += node.get_mem_utilization()
        
        avg_cpu_util /= len(self.fog_nodes)
        avg_mem_util /= len(self.fog_nodes)
        
        # Calculate variance
        var_cpu = 0
        var_mem = 0
        
        for node in self.fog_nodes:
            var_cpu += (node.get_cpu_utilization() - avg_cpu_util) ** 2
            var_mem += (node.get_mem_utilization() - avg_mem_util) ** 2
        
        var_cpu /= len(self.fog_nodes)
        var_mem /= len(self.fog_nodes)
        
        # Load balance metric
        lb_metric = var_cpu + var_mem
        return lb_metric
    
    def calculate_total_energy(self):
        """Calculate total energy consumption"""
        comp_energy = 0
        comm_energy = 0
        
        # Computational energy
        for node in self.fog_nodes:
            comp_energy += node.get_computational_energy()
        
        # Communication energy
        for link in self.links:
            comm_energy += link.get_communication_energy()
        
        return comp_energy + comm_energy
    
    def evaluate_objective_function(self, service, alpha=0.4, beta=0.3, gamma=0.3):
        """Evaluate the objective function"""
        # 1. End-to-end latency term
        latency = service.calculate_end_to_end_latency()
        norm_latency = latency / service.deadline  # Normalize by deadline
        
        # 2. Load balance term
        lb_metric = self.calculate_load_balance_metric()
        
        # 3. Energy consumption term
        energy = self.calculate_total_energy()
        # Assuming Emax is 1000J for normalization
        norm_energy = energy / 1000
        
        # Combined objective function
        objective_value = alpha * norm_latency + beta * lb_metric + gamma * norm_energy
        
        return objective_value, latency, lb_metric, energy
    
    def reset_placements(self):
        """Reset all placements and routings"""
        # Reset fog node resources
        for node in self.fog_nodes:
            node.available_resources = np.array(node.total_resources.copy())
            node.placed_microservices = []
        
        # Reset link bandwidths
        for link in self.links:
            link.available_bandwidth = link.total_bandwidth
            link.routed_edges = []
        
        # Reset microservice placements
        for service in self.services:
            for ms in service.microservices:
                ms.placed_on = None
            
            for edge in service.invocation_edges:
                edge.routed_through = []
    
    def visualize_network(self):
        """Visualize the network topology"""
        plt.figure(figsize=(12, 10))
        
        # Create position layout
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Draw nodes
        node_colors = []
        node_sizes = []
        
        for node in self.fog_nodes:
            cpu_util = node.get_cpu_utilization()
            # Color based on utilization (green->yellow->red)
            r = min(1.0, 2 * cpu_util)
            g = min(1.0, 2 * (1 - cpu_util))
            node_colors.append((r, g, 0))
            
            # Size based on total resources
            total_res = sum(node.total_resources)
            node_sizes.append(100 + total_res / 100)
        
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=node_sizes)
        
        # Draw edges with width based on bandwidth
        edge_widths = []
        for u, v in self.graph.edges():
            edge_data = self.graph[u][v]
            # Width based on used bandwidth percentage
            for link in self.links:
                if ((link.source == u and link.destination == v) or 
                    (link.source == v and link.destination == u)):
                    used_bw = (link.total_bandwidth - link.available_bandwidth) / link.total_bandwidth
                    edge_widths.append(1 + 4 * used_bw)
                    break
        
        nx.draw_networkx_edges(self.graph, pos, width=edge_widths, alpha=0.7)
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos)
        
        plt.title("Fog Network Topology")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def visualize_service(self, service):
        """Visualize a service DAG"""
        G = nx.DiGraph()
        
        # Add nodes for microservices
        for ms in service.microservices:
            G.add_node(ms.id)
        
        # Add edges for invocations
        for edge in service.invocation_edges:
            G.add_edge(edge.source_ms.id, edge.dest_ms.id, weight=edge.data_size)
        
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)
        
        node_colors = []
        for ms in service.microservices:
            if ms.placed_on:
                # Color based on which node it's placed on
                node_colors.append(f"C{ms.placed_on.id % 10}")
            else:
                node_colors.append('lightgray')
        
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=1000, 
               font_size=10, font_weight='bold', arrows=True, arrowsize=20)
        
        # Add edge labels for data sizes
        edge_labels = {(edge.source_ms.id, edge.dest_ms.id): f"{edge.data_size:.1f} Mbps" 
                      for edge in service.invocation_edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title(f"Service {service.id} DAG")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def save_to_json(self, file_path):
        """Save the network and services to a JSON file"""
        """
        network has the following attributes:
        - fog_nodes: list of fog nodes
        - links: list of links
        - services: list of services
        save it in a way that the fognework object can be reconstructed
        (ignore the graph since it can be reconstructed from the object attrs)
        """
        import json
        network_data = {
            "fog_nodes": [{
                "id": node.id,
                "total_resources": node.total_resources.tolist(),
                "idle_power": node.idle_power,
                "max_power": node.max_power
            } for node in self.fog_nodes],
            "links": [{
                "id": link.id,
                "source": link.source,
                "destination": link.destination,
                "bandwidth": link.total_bandwidth,
                "energy_per_bit": link.energy_per_bit,
                "delay": link.delay
            } for link in self.links],
            "services": [{
                "id": service.id,
                "deadline": service.deadline,
                "lifetime": service.lifetime,
                "arrival_rate": service.arrival_rate,
                "microservices": [{
                    "id": ms.id,
                    "resource_demand": ms.resource_demand.tolist(),
                    "computational_intensity": ms.computational_intensity
                } for ms in service.microservices],
                "invocation_edges": [{
                    "id": edge.id,
                    "source_ms_id": edge.source_ms.id,
                    "dest_ms_id": edge.dest_ms.id,
                    "data_size": edge.data_size
                } for edge in service.invocation_edges]
            } for service in self.services]
        }
        with open(file_path, 'w') as f:
            json.dump(network_data, f, indent=4)


    def load_from_json(self, file_path):
        """Load the network and services from a JSON file"""
        import json
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Load fog nodes
        self.fog_nodes = [FogNode(node['id'], node['total_resources']) for node in data['fog_nodes']]
        
        # Load links
        self.links = [Link(link['id'], link['source'], link['destination'], link['bandwidth'], 
                          link['energy_per_bit']) for link in data['links']]
        
        # Load services
        self.services = []
        for service_data in data['services']:
            service = Service(service_data['id'], service_data['deadline'], service_data['lifetime'], 
                              service_data['arrival_rate'])
            for ms_data in service_data['microservices']:
                ms = Microservice(ms_data['id'], service.id, ms_data['resource_demand'], 
                                  ms_data['computational_intensity'])
                service.microservices.append(ms)
            
            for edge_data in service_data['invocation_edges']:
                source_ms = next(ms for ms in service.microservices if ms.id == edge_data['source_ms_id'])
                dest_ms = next(ms for ms in service.microservices if ms.id == edge_data['dest_ms_id'])
                edge = InvocationEdge(edge_data['id'], service.id, source_ms, dest_ms, edge_data['data_size'])
                service.invocation_edges.append(edge)
            
            self.services.append(service)



def run_simulation():
    # Create a fog network
    network = FogNetwork()
    network.generate_random_topology(num_nodes=15, connectivity=0.3)
    
    # Generate a service
    
    service = network.generate_random_service(num_microservices=6, connectivity=0.3)
    
    # Visualize the network and service
    network.visualize_network()
    network.visualize_service(service)

    # save the network (including the fognetwork and services) to a json file
    # network.save_to_json("fog_network.json")
    
 
def returnFogNetwork(seed=646):
    """Return a FogNetwork object with a random topology and list of services to sample from later"""
    np.random.seed(seed)
    random.seed(seed)

    network = FogNetwork()
    num_nodes = 10
    network.generate_random_topology(num_nodes=num_nodes, connectivity=0.3)
    print(f"Generated a FogNetwork of {num_nodes} nodes")

    # add 50 POSSIBLE services (SERVICE SPACE)
    SERVICE_SPACE_SIZE = 50
    for i in range(SERVICE_SPACE_SIZE):
        network.generate_random_service(num_microservices=5)        
    print(f"Generated {SERVICE_SPACE_SIZE} potential services to sample from")
    
    # TODO: save to JSON and load from that 
    return network
    
# Run the simulation
if __name__ == "__main__":
    run_simulation()