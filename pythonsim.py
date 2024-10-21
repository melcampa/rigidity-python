import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import pandas as pd

class Edge:
    def __init__(self, start, velocity=None, buffer_distance=2.0, separation_distance=1.5):
        self.start = np.array(start, dtype=float)
        self.velocity = np.array(velocity, dtype=float) if velocity is not None else np.random.uniform(-0.1, 0.1, size=2)
        self.rect_size = (1.0, 1.0)  # Size of the rectangle
        self.buffer_distance = buffer_distance  # Buffer distance to avoid overlaps
        self.separation_distance = separation_distance  # Minimum distance to maintain between edges

    def move(self, edges):
        """Move the edge towards the centroid of all other edges and avoid collision."""
        target_position = self.calculate_centroid(edges)

        if target_position is not None:
            # Calculate movement vector towards the centroid
            direction = target_position - self.start
            distance_to_target = np.linalg.norm(direction)

            # Normalize the direction
            if distance_to_target > 0:
                direction /= distance_to_target  
                # Apply attraction force
                attraction_move = 0.1 * direction

                # Calculate repulsion force
                repulsion_move = self.calculate_repulsion(edges)

                # Combine attraction and repulsion
                self.start += attraction_move + repulsion_move

    def calculate_centroid(self, edges):
        """Calculate the centroid of all other edges."""
        positions = [edge.start for edge in edges if edge is not self]
        if positions:
            return np.mean(positions, axis=0)  # Return the mean position of all other edges
        return None

    def calculate_repulsion(self, edges):
        """Calculate repulsion vector from other edges that are too close."""
        repulsion_vector = np.array([0.0, 0.0])
        for edge in edges:
            if edge is not self:
                distance = np.linalg.norm(edge.start - self.start)
                if distance < self.separation_distance:
                    # Calculate the direction to push away
                    direction = self.start - edge.start  # Move away from the other edge
                    if np.linalg.norm(direction) > 0:
                        direction /= np.linalg.norm(direction)  # Normalize the direction

                    # Calculate the overlap distance and create a stronger repulsion force
                    overlap = self.separation_distance - distance
                    repulsion_strength = 0.5 * overlap  # Increase repulsion strength
                    repulsion_vector += direction * repulsion_strength

        return repulsion_vector

def calculate_average_distance(edges):
    """Calculate the average distance between all edges."""
    distances = []
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            distance = np.linalg.norm(edges[i].start - edges[j].start)
            distances.append(distance)
    return np.mean(distances) if distances else 0

def simulate_with_visualization(edges, steps=100, interval=50, stability_threshold=0.01, save_path='animation.gif'):
    """Simulate the motion of the edges with silent visualization, saved as a GIF using Pillow."""
    fig, ax = plt.subplots()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    # Create a rectangle object for each edge
    rects = []
    for edge in edges:
        rect = Rectangle(edge.start, edge.rect_size[0], edge.rect_size[1], fill=True, color="blue")
        ax.add_patch(rect)
        rects.append(rect)

    # Store results for each iteration
    results = []
    previous_average_distance = None

    def init():
        """Initialize the plot."""
        for rect in rects:
            rect.set_xy((0, 0))  # Reset position
        return rects

    def update(frame):
        """Update the positions of edges and plot them."""
        nonlocal previous_average_distance
        
        for edge, rect in zip(edges, rects):
            edge.move(edges)  # Pass all edges for collision checking
            rect.set_xy(edge.start)  # Update rectangle position

        # Calculate average distance and store it
        average_distance = calculate_average_distance(edges)
        results.append((frame, average_distance))  # Store iteration number and average distance
        
        # Check for stopping conditions
        if frame > 0 and previous_average_distance is not None:
            if abs(average_distance - previous_average_distance) < stability_threshold:
                print(f"Stopping at frame {frame}: Stabilized.")
                ani.event_source.stop()  # Stop the animation

        previous_average_distance = average_distance

        return rects

    # Create the animation and save it as a GIF using Pillow
    ani = FuncAnimation(fig, update, frames=steps, init_func=init, interval=interval)
    
    # Save animation as a GIF using Pillow
    ani.save(save_path, writer='pillow', dpi=100)  # Use 'pillow' for saving as GIF

    plt.close(fig)  # Close the plot after saving the animation

    return results  # Return results

num_edges = 20  
for i in range(10):  # Running 10 simulations
    edges = [Edge(start=[np.random.uniform(-10, 10), np.random.uniform(-10, 10)]) for _ in range(num_edges)]
    results = simulate_with_visualization(edges, steps=100, interval=50, save_path=f'C:/Users/chipr/OneDrive/Desktop/sim/rigidity-python/results/animation_{i+1}.gif')

    # Create a DataFrame for results and save to CSV
    results_df = pd.DataFrame(results, columns=['Iteration', 'Average Distance'])
    results_df.to_csv(f'C:/Users/chipr/OneDrive/Desktop/sim/rigidity-python/results/resultsaverage_distances_{i+1}.csv', index=False)  # Save to CSV file
