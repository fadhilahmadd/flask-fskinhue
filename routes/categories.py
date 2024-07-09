from flask import Blueprint, jsonify

categories_blueprint = Blueprint('categories', __name__)


@categories_blueprint.route('/api/json/categories/<categories>', methods=['GET'])
def get_categories(categories):
    if categories == 'Cool':
        outfit = [
            {
                "idOutfit": "1001",
                "strOutfit": "Outfit Cool 1",
                "strOutfitThumb": "http://192.168.26.16:5000/img/outfit-1.jpg",
            },
            {
                "idOutfit": "1002",
                "strOutfit": "Outfit Cool 2",
                "strOutfitThumb": "http://192.168.26.16:5000/img/35.jpg",
            },
        ]
    elif categories == 'Neutral':
        outfit = [
            {
                "idOutfit": "2001",
                "strOutfit": "Outfit Neutral 1",
                "strOutfitThumb": "http://192.168.26.16:5000/img/outfit-1.jpg",
            },
            {
                "idOutfit": "2002",
                "strOutfit": "Outfit Neutral 2",
                "strOutfitThumb": "http://192.168.26.16:5000/img/outfit-1.jpg",
            },
        ]
    elif categories == 'Warm':
        outfit = [
            {
                "idOutfit": "3001",
                "strOutfit": "Outfit Warm 1",
                "strOutfitThumb": "http://192.168.26.16:5000/img/outfit-1.jpg",
            },
            {
                "idOutfit": "3002",
                "strOutfit": "Outfit Warm 2",
                "strOutfitThumb": "http://192.168.26.16:5000/img/outfit-1.jpg",
            },
        ]
    else:
        return jsonify({"message": "Category not found"})

    return jsonify({"outfit": outfit})