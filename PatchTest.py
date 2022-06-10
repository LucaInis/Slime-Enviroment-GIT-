import pygame
import numpy as np

pygame.init()

W = 50  # in number of patches
H = 25  # in number of patches
PATCH_SIZE = 20  # thus window width and height are W * PATCH_SIZE and H * PATCH_SIZE
TURTLE_SIZE = PATCH_SIZE-1  # turtles must be slightly smaller

N_TURTLES = 50

SHOW_PATCHES = True
SHOW_TURTLES = True
MOVEMENT = True

BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

coords = []
offset = PATCH_SIZE // 2
W_pixels = W * PATCH_SIZE
H_pixels = H * PATCH_SIZE
for x in range(offset, (W_pixels - offset)+1, PATCH_SIZE):
    for y in range(offset, (H_pixels - offset)+1, PATCH_SIZE):
        coords.append((x, y))  # "centre" of the patch or turtle is the ID of the patch or turtle

# nel dizionario associato alle coordinate x,y della patch o turtle puoi mettere i dati che vuoi, come
#  - quantita di feromone,
#  - lista degli ID delle turtle che sono sulla patch
#  - se la turtle è in un cluster o meno
#  - ...(tutto quello che ti puo servire)
patches = {coords[i]: {"id": i} for i in range(len(coords))}
turtles = {coords[np.random.randint(len(coords))]: {"id": i} for i in range(N_TURTLES)}

screen = pygame.display.set_mode((W_pixels, H_pixels))
pygame.display.set_caption("PATCH TEST")

pygame.time.Clock()
SPEED = 60  # QUESTION never used...

playing = True
while playing:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # chiusura finestra -> termina il programma
            playing = False
    screen.fill(BLACK)
    if SHOW_TURTLES:
        #print("turtles:", end=" ")
        for t in turtles:  # una per patch
            #print(turtles[t]["id"], end=" ")
            pygame.draw.circle(screen, BLUE, t, TURTLE_SIZE // 2)  # ultimo parametro è il raggio del cerchio
        #print()
    if SHOW_PATCHES:
        # mostra le patch come quadrati
        #print("patches:", end=" ")
        for p in patches:
            #print(patches[p]["id"], end=" ")
            pygame.draw.rect(screen, WHITE, pygame.Rect(p[0]-offset, p[1]-offset, PATCH_SIZE-1, PATCH_SIZE-1), width=1)
        #print()
        # mostra la griglia che evidenzia le patch
        # for p in range(PATCH_SIZE, W_pixels, PATCH_SIZE):
        #     pygame.draw.line(screen, WHITE, (p, 0), (p, H_pixels))
        # for p in range(PATCH_SIZE, H_pixels, PATCH_SIZE):
        #     pygame.draw.line(screen, WHITE, (0, p), (W_pixels, p))
    pygame.display.flip()

pygame.quit()
