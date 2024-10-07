import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

class Edge:
    def __init__(self, start, velocity, target, buffer_distance=2.0):
        self.start = np.array(start, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.target = np.array(target, dtype=float)  # Target position for the rectangle
        self.rect_size = (1.0, 1.0)  # Size of the rectangle
        self.buffer_distance = buffer_distance  # Buffer distance to avoid overlaps

    def move(self, edges):
        """Move the edge towards the target and avoid collision with other edges."""
        direction = self.target - self.start
        distance_to_target = np.linalg.norm(direction)

        if distance_to_target > 0.05:  # If not close to the target, keep moving
            direction = direction / distance_to_target  # Normalize the direction
            new_position = self.start + 0.1 * direction  # Move toward the target

            # Check for potential collisions with buffer
            if not self.check_collision(new_position, edges):
                self.start = new_position  # Move if no collision
        else:
            self.start = self.target  # Snap to the target when close enough

    def check_collision(self, new_position, edges):
        """Check if the new position of the rectangle collides with other rectangles considering the buffer."""
        for edge in edges:
            if edge is not self:  # Avoid checking against itself
                # Calculate the coordinates of the buffer zone around this edge
                buffer_x_min = edge.start[0] - edge.buffer_distance
                buffer_x_max = edge.start[0] + edge.rect_size[0] + edge.buffer_distance
                buffer_y_min = edge.start[1] - edge.buffer_distance
                buffer_y_max = edge.start[1] + edge.rect_size[1] + edge.buffer_distance

                # Check if the new position is within the buffer zone of the other edge
                if (new_position[0] < buffer_x_max and
                    new_position[0] + self.rect_size[0] > buffer_x_min and
                    new_position[1] < buffer_y_max and
                    new_position[1] + self.rect_size[1] > buffer_y_min):
                    return True  # Collision detected
        return False  # No collision

def simulate_with_visualization(edges, steps=100, interval=50):
    """Simulate the motion of the edges with real-time visualization using rectangles."""
    fig, ax = plt.subplots()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    # Create a rectangle object for each edge
    rects = []
    for edge in edges:
        rect = Rectangle(edge.start, edge.rect_size[0], edge.rect_size[1], fill=True, color="blue")
        ax.add_patch(rect)
        rects.append(rect)

    # Create a red rectangle for the target position
    target_rect = Rectangle((0, 0), 1.0, 1.0, fill=True, color="red")
    ax.add_patch(target_rect)

    # Initialize target position
    target_position = np.array([0.0, 0.0], dtype=float)  # Keep target position as an array

    def init():
        """Initialize the plot."""
        for rect in rects:
            rect.set_xy((0, 0))  # Reset position
        target_rect.set_xy((0, 0))  # Reset target position
        return rects + [target_rect]

    def update(frame):
        """Update the positions of edges and plot them."""
        for edge, rect in zip(edges, rects):
            edge.move(edges)  # Pass all edges for collision checking
            rect.set_xy(edge.start)  # Update rectangle position

        target_rect.set_xy(target_position - (0.5, 0.5))  # Adjust for rectangle size
        return rects + [target_rect]

    def on_click(event):
        """Update the target position based on mouse click."""
        nonlocal target_position  # Allow modification of the target_position
        if event.inaxes:  # Check if the click is within the axes
            target_position = np.array([event.xdata, event.ydata], dtype=float)  # Update target position
            # Update target position for each edge
            for edge in edges:
                edge.target = target_position  # Update target for each edge

    fig.canvas.mpl_connect('button_press_event', on_click)  # Connect mouse click event

    ani = FuncAnimation(fig, update, frames=steps, init_func=init, interval=interval)
    plt.show()

# Example usage

# Number of edges
num_edges = 5
# Initial positions and velocities for the edges with random starting points
edges = [
    Edge(start=[np.random.uniform(-10, 10), np.random.uniform(-10, 10)], 
         velocity=[np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05)], 
         target=[0, 0])
    for _ in range(num_edges)
]

simulate_with_visualization(edges, steps=200, interval=50)
