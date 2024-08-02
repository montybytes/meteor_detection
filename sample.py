from shapes import Box


class Sample:
    image_name = bbox = None

    def __init__(self, image_name, x1, y1, x2, y2):
        self.image_name = image_name
        self.bbox = Box(x1, y1, x2, y2)
