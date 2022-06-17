import pygame
import numpy as np

pygame.init()

W = 50  # in number of patches
H = 25  # in number of patches
PATCH_SIZE = 20  # thus window width and height are W * PATCH_SIZE and H * PATCH_SIZE
TURTLE_SIZE = PATCH_SIZE - 1  # turtles must be slightly smaller

N_TURTLES = 10

SHOW_TURTLES = True
SHOW_PATCHES = True
MOVEMENT = True

BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

coords = []
offset = PATCH_SIZE // 2
W_pixels = W * PATCH_SIZE
H_pixels = H * PATCH_SIZE
for x in range(offset, (W_pixels - offset) + 1, PATCH_SIZE):
    for y in range(offset, (H_pixels - offset) + 1, PATCH_SIZE):
        coords.append((x, y))  # "centre" of the patch or turtle (also ID of the patch)

# nel dizionario associato alle coordinate x,y della patch puoi mettere i dati che vuoi, come
#  - quantita di feromone,
#  - lista degli ID delle turtle che sono sulla patch
#  - ...(tutto quello che ti puo servire)
patches = {coords[i]: {"id": i} for i in range(len(coords))}
# stesso discorso per il dizionario associato all'ID della turtle
turtles = {i: {"pos": coords[np.random.randint(len(coords))]} for i in range(N_TURTLES)}

screen = pygame.display.set_mode((W_pixels, H_pixels))
pygame.display.set_caption("PATCH TEST")

clock = pygame.time.Clock()

playing = True
while playing:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # chiusura finestra -> termina il programma
            playing = False
    screen.fill(BLACK)
    if SHOW_TURTLES:
        # print("turtles:", end=" ")
        for t in turtles:  #  una per patch
            # print(t, end=" ")
            pygame.draw.circle(screen, BLUE, turtles[t]["pos"],
                               TURTLE_SIZE // 2)  # ultimo parametro è il raggio del cerchio
        # print()
    if SHOW_PATCHES:
        # mostra le patch come quadrati
        # print("patches:", end=" ")
        for p in patches:
            # print(patches[p]["id"], end=" ")
            pygame.draw.rect(screen, WHITE, pygame.Rect(p[0] - offset, p[1] - offset, PATCH_SIZE - 1, PATCH_SIZE - 1),
                             width=1)
        # print()
        # mostra la griglia che evidenzia le patch
        # for p in range(PATCH_SIZE, W_pixels, PATCH_SIZE):
        #     pygame.draw.line(screen, WHITE, (p, 0), (p, H_pixels))
        # for p in range(PATCH_SIZE, H_pixels, PATCH_SIZE):
        #     pygame.draw.line(screen, WHITE, (0, p), (W_pixels, p))
    if MOVEMENT:
        choice = [PATCH_SIZE, -PATCH_SIZE, 0]
        # choice = [PATCH_SIZE]
        for t in turtles:
            x, y = turtles[t]["pos"]
            x2, y2 = x + np.random.choice(choice), y + np.random.choice(choice)
            if x2 < 0:
                x2 = W_pixels - offset
            if x2 > W_pixels:
                x2 = 0 + offset
            if y2 < 0:
                y2 = H_pixels - offset
            if y2 > H_pixels:
                y2 = 0 + offset
            turtles[t]["pos"] = (x2, y2)

    pygame.display.flip()
    # clock.tick(1)

pygame.quit()
