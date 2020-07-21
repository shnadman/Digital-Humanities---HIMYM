class Character(object):
    def __init__(self, name, aliases, gender):
        self.name = name
        self.aliases = aliases
        self.lines = {str(i): [] for i in range(1, 10)}
        self.gender = gender
        self.season_counter = {str(i): [] for i in range(1, 10)}
        self.total_words = 0
        self.polarity = {str(i): [] for i in range(1, 10)}
        self.subjectivity = {str(i): [] for i in range(1, 10)}


    def __str__(self):
        return f'Name: {self.name}, Aliases: {self.aliases}, Gender: {self.gender}'



class Line(object):
    def __init__(self, season, ep, text):
        self.season = season
        self.ep = ep
        self.text = text
        self.wordCounter = len(text.split(" "))


    def __dir__(self):
        return ['text']

class Location(object):
    def __init__(self, season, ep, loc):
        self.season = season
        self.ep = ep
        self.loc = loc
        self.latitude = 0
        self.longitude = 0


    def setCoords(self,latitude, longitude):
        self.latitude=latitude
        self.longitude=longitude


