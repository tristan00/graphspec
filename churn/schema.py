from graphspec.features import encoder

from .spec import CATEGORIES, NUMBERS, WIDTH


encode = encoder(WIDTH, NUMBERS, CATEGORIES)
