def manhatten_distance(point_a: tuple[int, int], point_b: tuple[int, int]):
    ax, ay = point_a
    bx, by = point_b
    return abs(ax - bx) + abs(ay - by)