class Box:
    x1 = y1 = x2 = y2 = None

    def __init__(self, x1, y1, x2, y2):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)

    @property
    def points(self):
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def midpoint(self):
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


class Line:
    x1 = y1 = x2 = y2 = None

    def __init__(self, x1, y1, x2, y2):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)

    @property
    def points(self):
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def midpoint(self):
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
