from build123d import BuildPart, Box, Cylinder, Locations, Mode


PLATE_LENGTH = 100.0
PLATE_WIDTH = 60.0
PLATE_THICKNESS = 8.0

M5_CLEARANCE_DIAMETER = 5.5
CENTER_HOLE_DIAMETER = 20.0
CORNER_HOLE_EDGE_OFFSET = 10.0


def corner_hole_locations() -> list[tuple[float, float, float]]:
    x = PLATE_LENGTH / 2.0 - CORNER_HOLE_EDGE_OFFSET
    y = PLATE_WIDTH / 2.0 - CORNER_HOLE_EDGE_OFFSET
    return [
        (-x, -y, 0.0),
        (-x, y, 0.0),
        (x, -y, 0.0),
        (x, y, 0.0),
    ]


def gen_step():
    through_height = PLATE_THICKNESS + 2.0

    with BuildPart() as plate:
        Box(PLATE_LENGTH, PLATE_WIDTH, PLATE_THICKNESS)

        for location in corner_hole_locations():
            with Locations(location):
                Cylinder(
                    radius=M5_CLEARANCE_DIAMETER / 2.0,
                    height=through_height,
                    mode=Mode.SUBTRACT,
                )

        Cylinder(
            radius=CENTER_HOLE_DIAMETER / 2.0,
            height=through_height,
            mode=Mode.SUBTRACT,
        )

    part = plate.part
    part.label = "mounting_plate_100x60x8"
    return part
