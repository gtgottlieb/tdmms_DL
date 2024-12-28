"""Module that contains custom exceptions."""

class WeightsNotFound(Exception):
    """Weights for material not found."""

class IncorrectDataSplit(Exception):
    """Data set directory contain incorrect amount of images."""

class DirectoryAlreadyExists(Exception):
    """Directory already exists."""