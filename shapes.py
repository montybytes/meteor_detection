class Box:
    x1 = y1 = x2 = y2 = 0

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @property
    def points(self):
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def midpoint(self):
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


class Line:
    x1 = y1 = x2 = y2 = 0

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @property
    def points(self):
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def midpoint(self):
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
