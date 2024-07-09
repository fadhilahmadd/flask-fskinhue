from flask import Blueprint, jsonify

detail_blueprint = Blueprint('detail', __name__)


@detail_blueprint.route('/api/json/v1/detail/<id>', methods=['GET'])
def get_recipe_id(id):
    if id == '1001':
        detail = [
            {
                "idOutfit": "1001",
                "strOutfit": "Outfit Cool 1",
                "strCategory": "Cool",
                "strDescription": "deskripsi outfit cool 1\n\nasdaskjdhkahd\n\nasdaskjdhkahd\n\nasdaskjdhkahd\n\nasdaskjdhkahd",
                "strOutfitThumb": "http://192.168.26.16:5000/img/outfit-1.jpg",
                "hexColor1":"8da7b4",
                "hexColor2":"9cb5c1",
                "hexColor3":"ddd9c9",
            }
        ]
    elif id == '1002':
        detail = [
            {
                "idOutfit": "1002",
                "strOutfit": "Outfit Cool 2",
                "strCategory": "Neutral",
                "strDescription": "deskripsi outfit cool 2",
                "strOutfitThumb": "http://192.168.26.16:5000/img/35.jpg",
                "hexColor1":"ddd9c9",
                "hexColor2":"cc785c",
                "hexColor3":"c1644d",
            }
        ]
    elif id == '2001':
        detail = [
            {
                "idOutfit": "2001",
                "strOutfit": "Outfit Neutral 1",
                "strCategory": "Neutral",
                "strDescription": "deskripsi outfit Neutral 1",
                "strOutfitThumb": "http://192.168.26.16:5000/img/outfit-1.jpg",
                "hexColor1":"517d98",
                "hexColor2":"b6b9b8",
                "hexColor3":"c18178",
            }
        ]
    elif id == '2002':
        detail = [
            {
                "idOutfit": "2002",
                "strOutfit": "Outfit Neutral 2",
                "strCategory": "Neutral",
                "strDescription": "deskripsi Outfit Neutral 2",
                "strOutfitThumb": "http://192.168.26.16:5000/img/outfit-1.jpg",
                "hexColor1":"c18178",
                "hexColor2":"4f5d71",
                "hexColor3":"493b4c",
            }
        ]
    elif id == '3001':
        detail = [
            {
                "idOutfit": "3001",
                "strOutfit": "Outfit Warm 1",
                "strCategory": "Warm",
                "strDescription": "deskripsi outfit Warm 1",
                "strOutfitThumb": "http://192.168.26.16:5000/img/outfit-1.jpg",
                "hexColor1":"01002f",
                "hexColor2":"38220f",
                "hexColor3":"070707",
            }
        ]
    elif id == '3002':
        detail = [
            {
                "idOutfit": "3002",
                "strOutfit": "Outfit Warm 2",
                "strCategory": "Warm",
                "strDescription": "deskripsi outfit Warm 2",
                "strOutfitThumb": "http://192.168.26.16:5000/img/outfit-1.jpg",
                "hexColor1":"1c4a2d",
                "hexColor2":"124b18",
                "hexColor3":"125c25",
            }
        ]
    else:
        return jsonify({"message": "Detail not found"})

    return jsonify({"detail": detail})