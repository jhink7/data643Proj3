from app import app
from flask import jsonify, request, abort
from recommendation_engine import RecommendationEngine

BASE_URL = '/rec-engine/api/v1.0/'

@app.route(BASE_URL + 'users/<int:user_id>/recommendations', methods=['GET'])
def get_recs_for_user(user_id):
    recEngine = RecommendationEngine()
    user_exists, recs = recEngine.generate_recommendations(user_id)
    if not user_exists:
        abort(404)
    return jsonify({'top_picks': recs})

@app.route(BASE_URL + 'users/<int:user_id>/top-rec', methods=['GET'])
def get_top_rec_for_user(user_id):
    recEngine = RecommendationEngine()
    user_exists, rec = recEngine.generate_top_recommendation(user_id)
    if not user_exists:
        abort(404)
    return jsonify({'top_pick': rec})