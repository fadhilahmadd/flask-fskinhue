from flask import Blueprint, jsonify

categories_blueprint = Blueprint('categories', __name__)


@categories_blueprint.route('/api/json/categories/<categories>', methods=['GET'])
def get_categories(categories):
    if categories == 'Cool':
        outfit = [
            {
                "idOutfit": "1001",
                "strOutfit": "Outfit Cool 1",
                "strOutfitThumb": "http://192.168.48.55:5000/img/C1.jpg",
            },
            {
                "idOutfit": "1002",
                "strOutfit": "Outfit Cool 2",
                "strOutfitThumb": "http://192.168.48.55:5000/img/C2.jpg",
            },
            {
                "idOutfit": "1003",
                "strOutfit": "Outfit Cool 3",
                "strOutfitThumb": "http://192.168.48.55:5000/img/C3.jpg",
            },
            {
                "idOutfit": "1004",
                "strOutfit": "Outfit Cool 4",
                "strOutfitThumb": "http://192.168.48.55:5000/img/C4.jpg",
            },
            {
                "idOutfit": "1005",
                "strOutfit": "Outfit Cool 5",
                "strOutfitThumb": "http://192.168.48.55:5000/img/C5.jpg",
            },
            {
                "idOutfit": "1006",
                "strOutfit": "Outfit Cool 6",
                "strOutfitThumb": "http://192.168.48.55:5000/img/C6.jpg",
            },
            {
                "idOutfit": "1007",
                "strOutfit": "Outfit Cool 7",
                "strOutfitThumb": "http://192.168.48.55:5000/img/C7.jpg",
            },
            {
                "idOutfit": "1008",
                "strOutfit": "Outfit Cool 8",
                "strOutfitThumb": "http://192.168.48.55:5000/img/C8.jpg",
            },
            {
                "idOutfit": "1009",
                "strOutfit": "Outfit Cool 9",
                "strOutfitThumb": "http://192.168.48.55:5000/img/C9.jpg",
            },
            {
                "idOutfit": "1010",
                "strOutfit": "Outfit Cool 10",
                "strOutfitThumb": "http://192.168.48.55:5000/img/C10.jpg",
            },
        ]
    elif categories == 'Neutral':
        outfit = [
            {
                "idOutfit": "2001",
                "strOutfit": "Outfit Neutral 1",
                "strOutfitThumb": "http://192.168.48.55:5000/img/N1.jpg",
            },
            {
                "idOutfit": "2002",
                "strOutfit": "Outfit Neutral 2",
                "strOutfitThumb": "http://192.168.48.55:5000/img/N2.jpg",
            },
            {
                "idOutfit": "2003",
                "strOutfit": "Outfit Neutral 3",
                "strOutfitThumb": "http://192.168.48.55:5000/img/N3.jpg",
            },
            {
                "idOutfit": "2004",
                "strOutfit": "Outfit Neutral 4",
                "strOutfitThumb": "http://192.168.48.55:5000/img/N4.jpg",
            },
            {
                "idOutfit": "2005",
                "strOutfit": "Outfit Neutral 5",
                "strOutfitThumb": "http://192.168.48.55:5000/img/N5.jpg",
            },
            {
                "idOutfit": "2006",
                "strOutfit": "Outfit Neutral 6",
                "strOutfitThumb": "http://192.168.48.55:5000/img/N6.jpg",
            },
            {
                "idOutfit": "2007",
                "strOutfit": "Outfit Neutral 7",
                "strOutfitThumb": "http://192.168.48.55:5000/img/N7.jpg",
            },
            {
                "idOutfit": "2008",
                "strOutfit": "Outfit Neutral 8",
                "strOutfitThumb": "http://192.168.48.55:5000/img/N8.jpg",
            },
            {
                "idOutfit": "2009",
                "strOutfit": "Outfit Neutral 9",
                "strOutfitThumb": "http://192.168.48.55:5000/img/N9.jpg",
            },
            {
                "idOutfit": "2010",
                "strOutfit": "Outfit Neutral 10",
                "strOutfitThumb": "http://192.168.48.55:5000/img/N10.jpg",
            },
        ]
    elif categories == 'Warm':
        outfit = [
            {
                "idOutfit": "3001",
                "strOutfit": "Outfit Warm 1",
                "strOutfitThumb": "http://192.168.48.55:5000/img/W1.jpg",
            },
            {
                "idOutfit": "3002",
                "strOutfit": "Outfit Warm 2",
                "strOutfitThumb": "http://192.168.48.55:5000/img/W2.jpg",
            },
             {
                "idOutfit": "3003",
                "strOutfit": "Outfit Warm 3",
                "strOutfitThumb": "http://192.168.48.55:5000/img/W3.jpg",
            },
              {
                "idOutfit": "3004",
                "strOutfit": "Outfit Warm 4",
                "strOutfitThumb": "http://192.168.48.55:5000/img/W4.jpg",
            },
               {
                "idOutfit": "3005",
                "strOutfit": "Outfit Warm 5",
                "strOutfitThumb": "http://192.168.48.55:5000/img/W5.jpg",
            },
                {
                "idOutfit": "3006",
                "strOutfit": "Outfit Warm 6",
                "strOutfitThumb": "http://192.168.48.55:5000/img/W6.jpg",
            },
                 {
                "idOutfit": "3007",
                "strOutfit": "Outfit Warm 7",
                "strOutfitThumb": "http://192.168.48.55:5000/img/W7.jpg",
            },
                  {
                "idOutfit": "3008",
                "strOutfit": "Outfit Warm 8",
                "strOutfitThumb": "http://192.168.48.55:5000/img/W8.jpg",
            },
                   {
                "idOutfit": "3009",
                "strOutfit": "Outfit Warm 9",
                "strOutfitThumb": "http://192.168.48.55:5000/img/W9.jpg",
            },
                    {
                "idOutfit": "30010",
                "strOutfit": "Outfit Warm 10",
                "strOutfitThumb": "http://192.168.48.55:5000/img/W10.jpg",
            },
        ]
    else:
        return jsonify({"message": "Category not found"})

    return jsonify({"outfit": outfit})