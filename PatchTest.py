import pygame

pygame.init()

# possono anche avere dimensione diversa...
W = 500
H = 500
PATCH_SIZE = W // 10  # ...purchè divisibile per PATCH_SIZE
TURTLE_SIZE = PATCH_SIZE-1  # le turtle devono essere un po piu piccole
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

SHOW_PATCHES = True
SHOW_TURTLES = True

coords = []
for x in range(0 + PATCH_SIZE // 2, (W - PATCH_SIZE // 2)+1, PATCH_SIZE):
    for y in range(0 + PATCH_SIZE // 2, (H - PATCH_SIZE // 2)+1, PATCH_SIZE):
        coords.append((x, y))
# nel dizionario associato alle coordinate x,y della patch puoi mettere i dati che vuoi, come
#  - quantita di feromone,
#  - lista degli ID delle turtle che sono sulla patch
#  - ...(tutto quello che tui puo servire)
patches = {e: {} for e in coords}

screen = pygame.display.set_mode((W, H))
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
        for c in patches:  # una per patch
            pygame.draw.circle(screen, BLUE, c, TURTLE_SIZE // 2)
    if SHOW_PATCHES:  # mostra la griglia che evidenzia le patch
        for p in range(PATCH_SIZE, W, PATCH_SIZE):
            pygame.draw.line(screen, WHITE, (p, 0), (p, H))
        for p in range(PATCH_SIZE, H, PATCH_SIZE):
            pygame.draw.line(screen, WHITE, (0, p), (W, p))
    pygame.display.flip()

pygame.quit()
