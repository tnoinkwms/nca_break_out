import pygame
import numpy as np
import onnxruntime

# GNCA Class Definition (as provided)
class GNCA():
    def __init__(self, height=72, width=72, n_channels=16, model_path=r"./growing-neural-cellular-automata.onnx"):
        self.height: int = height
        self.width: int = width
        self.n_channels: int = n_channels
        self.session = onnxruntime.InferenceSession(model_path)
        self.input: np.ndarray
        self.output: np.ndarray
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def to_alpha(self, x):
        return np.clip(x[..., 3:4], 0, 0.9999)

    def to_rgba(self, x):
        rgb, a = x[..., :3], self.to_alpha(x)
        return np.concatenate((np.clip(1.0 - a + rgb, 0, 0.9999), a), axis=3)

    def write_alpha_tolist(self, x):
        alpha = self.to_alpha(x).tolist()
        return alpha

    def make_seeds(self):
        x = np.zeros([1, self.height, self.width, self.n_channels], np.float32)
        x[:, self.height // 2 - 10:self.height // 2 + 10, self.width // 2 - 10:self.width // 2 + 10, 3:] = 1.0
        self.input = x
        return x

    def run(self) -> np.ndarray:
        out = self.session.run([self.output_name], {self.input_name: self.input})
        self.output = out[0].astype(np.float32)
        self.input = out[0].astype(np.float32)
        return self.input

# Initialize pygame
pygame.init()

# Set up display
width = 800
height = 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Breakout")

# Define colors
black = (0, 0, 0)
white = (255, 255, 255)
blue = (0, 0, 255)
red = (255, 0, 0)

# Define paddle
paddle_width = 100
paddle_height = 10
paddle_speed = 10
paddle = pygame.Rect(width // 2 - paddle_width // 2, height - 30, paddle_width, paddle_height)

# Define ball
ball_size = 20
ball_speed = [5, -5]
ball = pygame.Rect(width // 2 - ball_size // 2, height - 40 - ball_size // 2, ball_size, ball_size)
ball_in_motion = False

# Initialize GNCA
gnca = GNCA(height=72, width=72, n_channels=16)  # Adjust GNCA dimensions to match brick layout
gnca.make_seeds()

# Main game loop
running = True
clock = pygame.time.Clock()

brick_width = 5
brick_height = 5

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                ball_in_motion = True
                # Reset ball position if not in motion
                if not ball_in_motion:
                    ball.topleft = (width // 2 - ball_size // 2, height//2 - ball_size // 2)
                    ball_speed = [5, -5]

    # Move paddle
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and paddle.left > 0:
        paddle.left -= paddle_speed
    if keys[pygame.K_RIGHT] and paddle.right < width:
        paddle.right += paddle_speed

    # Move ball if in motion
    if ball_in_motion:
        ball.left += ball_speed[0]
        ball.top += ball_speed[1]

        # Ball collision with walls
        if ball.left <= 0 or ball.right >= width:
            ball_speed[0] = -ball_speed[0]
        if ball.top <= 0:
            ball_speed[1] = -ball_speed[1]

        # Ball collision with paddle
        if ball.colliderect(paddle):
            ball_speed[1] = -ball_speed[1]

        # Ball collision with bricks (GNCA cells)

        for row in range(72):
            for col in range(72):
                brick_x = col * brick_width + 220 
                brick_y = row * brick_height 
                brick = pygame.Rect(brick_x, brick_y, brick_width, brick_height)
                if ball.colliderect(brick) and gnca.input[0, row, col, 3] != 0:
                    ball_speed[1] = -ball_speed[1]
                    ball_speed[0] = -ball_speed[0]
                    gnca.input[0, row-15: row + 15, col-15 : col + 15, 3] = 0 
                    break

        # Reset ball if it falls below the screen
        if ball.top >= height:
            ball_in_motion = False
            ball.topleft = (paddle.left +paddle_width//2  - ball_size // 2, height - 40 - ball_size // 2)
            ball_speed = [5, -5]

    # Update GNCA
    gnca.run()

    # Draw everything
    screen.fill(black)
    pygame.draw.rect(screen, white, paddle)
    pygame.draw.ellipse(screen, red, ball)

    # Draw GNCA bricks
    gnca_output = gnca.to_rgba(gnca.input)[0]  # Get RGBA output
    for row in range(72):
        for col in range(72):
            brick_x = col * brick_width +220
            brick_y = row * brick_height
            alpha = gnca_output[row, col, 3]
            if alpha > 0:
                color = (int(gnca_output[row, col, 0] * 255),
                         int(gnca_output[row, col, 1] * 255),
                         int(gnca_output[row, col, 2] * 255))
                pygame.draw.rect(screen, color, pygame.Rect(brick_x, brick_y, brick_width, brick_height))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()