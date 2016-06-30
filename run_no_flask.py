from math import*
import pandas as pd
import numpy as np
import random as rd

from recommendation_engine import RecommendationEngine

engine = RecommendationEngine()
#found, recs = engine.generate_recommendations(7)

for i in range(1,11):
    user_exists, rec_movie = engine.generate_top_recommendation(i)
    print rec_movie




