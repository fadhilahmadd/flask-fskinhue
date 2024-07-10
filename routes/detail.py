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
                "strDescription": "Warna cokelat, peach, dan putih bisa menjadi pilihan yang bagus untuk melengkapi warna kulit cool. Cokelat memberikan sentuhan hangat yang kontras, peach memberikan kesan lembut yang menyenangkan\nsementara putih memberikan kesan bersih dan elegan\nsemuanya cocok untuk menonjolkan keanggunan warna kulit.",
                "strOutfitThumb": "http://192.168.48.55:5000/img/C1.jpg",
                "hexColor1":"D2B48C",
                "hexColor2":"FFE5B4",
                "hexColor3":"F5F5F5",
            }
        ]
    elif id == '1002':
        detail = [
            {
                "idOutfit": "1002",
                "strOutfit": "Outfit Cool 2",
                "strCategory": "Cool",
                "strDescription": "Putih tulang, putih, dan pink adalah kombinasi yang cocok untuk menonjolkan keanggunan warna kulit cool. Putih tulang memberikan nuansa netral yang hangat dan elegan\nputih memberikan kesan bersih dan modern\nsementara pink memberikan sentuhan feminin yang lembut dan menyegarkan\nsemuanya secara harmonis melengkapi karakteristik warna kulit cool.",
                "strOutfitThumb": "http://192.168.48.55:5000/img/C2.jpg",
                "hexColor1":"FFFFFF",
                "hexColor2":"EAE0C8",
                "hexColor3":"FF69B4",
            }
        ]
    elif id == '1003':
        detail = [
            {
                "idOutfit": "1003",
                "strOutfit": "Outfit Cool 3",
                "strCategory": "Cool",
                "strDescription": "Warna beige, hijau pastel, dan biru menciptakan kombinasi yang menenangkan dan harmonis untuk warna kulit cool.",
                "strOutfitThumb": "http://192.168.48.55:5000/img/C3.jpg",
                "hexColor1":"F5F5DC",
                "hexColor2":"77DD77",
                "hexColor3":"ADD8E6",
            }
        ]
    elif id == '1004':
        detail = [
            {
                "idOutfit": "1004",
                "strOutfit": "Outfit Cool 4",
                "strCategory": "Cool",
                "strDescription": "Kombinasi lavender, dark purple, dan white dapat menciptakan palet yang elegan dan kontras untuk menonjolkan kecantikan warna kulit cool. Lavender memberikan sentuhan lembut dan menenangkan\ndark purple menambahkan elemen dramatis dan misterius\nsementara white memberikan kesan bersih dan murni yang menyoroti kejernihan warna kulit cool.",
                "strOutfitThumb": "http://192.168.48.55:5000/img/C4.jpg",
                "hexColor1":"E6E6FA",
                "hexColor2":"483D8B",
                "hexColor3":"FFFFFF",
            }
        ]
    elif id == '1005':
        detail = [
            {
                "idOutfit": "1005",
                "strOutfit": "Outfit Cool 5",
                "strCategory": "Cool",
                "strDescription": "Kombinasi beige, khaki, dan black menciptakan palet yang serbaguna dan elegan untuk warna kulit cool. Beige memberikan sentuhan hangat yang netral\nkhaki menambahkan nuansa tanah yang tenang\nsementara black memberikan kontras yang tegas dan serbaguna.\nKombinasi ini cocok untuk menciptakan tampilan yang stylish dan sesuai dengan karakteristik warna kulit cool.",
                "strOutfitThumb": "http://192.168.48.55:5000/img/C5.jpg",
                "hexColor1":"F5F5DC",
                "hexColor2":"C3B091",
                "hexColor3":"000000",
            }
        ]
    elif id == '1006':
        detail = [
            {
                "idOutfit": "1006",
                "strOutfit": "Outfit Cool 6",
                "strCategory": "Cool",
                "strDescription": "Kombinasi warna beige, pink, dan white menawarkan palet yang lembut dan feminin, sempurna untuk menampilkan kelembutan dan keanggunan warna kulit cool dengan sentuhan hangat dan elegan.",
                "strOutfitThumb": "http://192.168.48.55:5000/img/C6.jpg",
                "hexColor1":"FFFDD0",
                "hexColor2":"FFC0CB",
                "hexColor3":"FFFFFF",
            }
        ]
    elif id == '1007':
        detail = [
            {
                "idOutfit": "1007",
                "strOutfit": "Outfit Cool 7",
                "strCategory": "Cool",
                "strDescription": "Kombinasi ini menciptakan palet yang harmonis\nmenampilkan kelembutan alami dari green pastel dan khaki dengan kontras yang diberikan oleh white\ncocok untuk berbagai jenis desain yang membutuhkan kesan tenang dan elegan",
                "strOutfitThumb": "http://192.168.48.55:5000/img/C7.jpg",
                "hexColor1":"77DD77",
                "hexColor2":"C3B091",
                "hexColor3":"FFFFFF",
            }
        ]
    elif id == '1008':
        detail = [
            {
                "idOutfit": "1008",
                "strOutfit": "Outfit Cool 8",
                "strCategory": "Cool",
                "strDescription": "Kombinasi warna ini menciptakan palet yang halus dan elegan/nsempurna untuk menonjolkan keanggunan dan kelembutan warna kulit cool.\nOff white dan beige memberikan kehangatan sementara white memberikan kontras yang bersih\nmenciptakan kesan yang seimbang dan harmonis.",
                "strOutfitThumb": "http://192.168.48.55:5000/img/C8.jpg",
                "hexColor1":"F5F5F5",
                "hexColor2":"F5F5DC",
                "hexColor3":"FFFFFF",
            }
        ]
    elif id == '1009':
        detail = [
            {
                "idOutfit": "1009",
                "strOutfit": "Outfit Cool 9",
                "strCategory": "Cool",
                "strDescription": "Kombinasi grey, beige, dan white menciptakan palet warna yang elegan dan seimbang untuk warna kulit cool.\nGrey memberikan nuansa netral yang modern dan tenang,\nbeige menambahkan sentuhan hangat yang menenangkan, sementara white memberikan kontras yang bersih dan murni.Ketiganya bekerja bersama untuk menonjolkan keanggunan dan kesejukan warna kulit cool dengan kesan yang harmonis dan menyeluruh.",
                "strOutfitThumb": "http://192.168.48.55:5000/img/C9.jpg",
                "hexColor1":"808080",
                "hexColor2":"F5F5DC",
                "hexColor3":"FFFFFF",
            }
        ]
    elif id == '1010':
        detail = [
            {
                "idOutfit": "1010",
                "strOutfit": "Outfit Cool 10",
                "strCategory": "Cool",
                "strDescription": "Kombinasi ini menciptakan palet yang lembut dan harmonis,\nideal untuk menonjolkan keanggunan dan kelembutan warna kulit cool.\nSalt dan broken white memberikan kesan bersih dan elegan,\nsementara dusty pink menambahkan sentuhan hangat yang menenangkan.",
                "strOutfitThumb": "http://192.168.48.55:5000/img/C10.jpg",
                "hexColor1":"FAF9F6",
                "hexColor2":"D3A297",
                "hexColor3":"F0EAD6",
            }
        ]
    elif id == '2001':
        detail = [
            {
                "idOutfit": "2001",
                "strOutfit": "Outfit Neutral 1",
                "strCategory": "Neutral",
                "strDescription": "Navy, beige, dan black adalah pilihan yang cocok untuk melengkapi warna kulit neutral. Navy memberikan kesan formal dan tegas\n\nbeige memberikan sentuhan hangat yang netral\n\nsementara black menambahkan kontras yang tajam dan serbaguna\n\nsemuanya cocok untuk menonjolkan keanggunan warna kulit.",
                "strOutfitThumb": "http://192.168.48.55:5000/img/N1.jpg",
                "hexColor1":"000080",
                "hexColor2":"F5F5DC",
                "hexColor3":"000000",
            }
        ]
    elif id == '2002':
        detail = [
            {
                "idOutfit": "2002",
                "strOutfit": "Outfit Neutral 2",
                "strCategory": "Neutral",
                "strDescription": "Kombinasi warna hitam, putih, dan lerek hitam-putih memberikan kontras yang kuat dan elegan,\nideal untuk warna kulit netral. Hitam menambahkan kedalaman dan kekuatan,\nputih memberikan kesan bersih dan murni,\nsementara pola lerek hitam-putih menambahkan dinamika visual yang menarik,\nsemuanya bersama-sama menciptakan tampilan yang seimbang dan gaya yang berani.",
                "strOutfitThumb": "http://192.168.48.55:5000/img/N2.jpg",
                "hexColor1":"000000",
                "hexColor2":"FFFFFF",
                "hexColor3":"FFFFFF",
            }
        ]
    elif id == '2003':
        detail = [
            {
                "idOutfit": "2003",
                "strOutfit": "Outfit Neutral 3",
                "strCategory": "Neutral",
                "strDescription": "Kombinasi blue pastel,\ngrey pastel, dan brown pastel memberikan palet warna yang menenangkan dan serasi untuk menonjolkan keanggunan warna kulit neutral.\nBlue pastel memberikan nuansa biru lembut yang menyegarkan,\ngrey pastel memberikan kesan netral yang stabil,\nsementara brown pastel memberikan sentuhan hangat dan alami,\nsemuanya bersama menciptakan tampilan yang lembut dan harmonis dalam desain.",
                "strOutfitThumb": "http://192.168.48.55:5000/img/N3.jpg",
                "hexColor1":"B0C4DE",
                "hexColor2":"D3D3D3",
                "hexColor3":"BC8F8F",
            }
        ]
    elif id == '2004':
        detail = [
            {
                "idOutfit": "2004",
                "strOutfit": "Outfit Neutral 4",
                "strCategory": "Neutral",
                "strDescription": "Kombinasi coksu, maroon, dan frapuccino menciptakan palet warna hangat dan mewah yang menonjolkan keanggunan warna kulit neutral dengan nuansa yang lembut dan alami.",
                "strOutfitThumb": "http://192.168.48.55:5000/img/N4.jpg",
                "hexColor1":"B87333",
                "hexColor2":"800000",
                "hexColor3":"A15F3A",
            }
        ]
    elif id == '2005':
        detail = [
            {
                "idOutfit": "2005",
                "strOutfit": "Outfit Neutral 5",
                "strCategory": "Neutral",
                "strDescription": "Kombinasi corn yellow, white, dan pastel peach\ncocok digunakan untuk menciptakan desain yang ceria dan menyegarkan,\nmemberikan kesan hangat dan lembut yang ideal untuk menonjolkan keanggunan warna kulit neutral.",
                "strOutfitThumb": "http://192.168.48.55:5000/img/N5.jpg",
                "hexColor1":"FFD700",
                "hexColor2":"FFFFFF",
                "hexColor3":"FFDAB9",
            }
        ]
    elif id == '2006':
        detail = [
            {
                "idOutfit": "2006",
                "strOutfit": "Outfit Neutral 6",
                "strCategory": "Neutral",
                "strDescription": "Kombinasi ini tidak hanya cocok untuk menciptakan tampilan yang hangat dan seimbang,\ntetapi juga memberikan kesan yang kuat dan maskulin dalam desain,\nideal untuk berbagai aplikasi yang menginginkan kombinasi warna yang kokoh dan menarik.",
                "strOutfitThumb": "http://192.168.48.55:5000/img/N6.jpg",
                "hexColor1":"FFFDD0",
                "hexColor2":"FFFFFF",
                "hexColor3":"4B5320",
            }
        ]
    elif id == '2007':
        detail = [
            {
                "idOutfit": "2007",
                "strOutfit": "Outfit Neutral 7",
                "strCategory": "Neutral",
                "strDescription": "Kombinasi cream, yellow, dan white memberikan sentuhan\nhangat dan ceria yang cocok untuk menonjolkan\nkeanggunan warna kulit neutral dengan kesan yang bersih dan ringan.",
                "strOutfitThumb": "http://192.168.48.55:5000/img/N7.jpg",
                "hexColor1":"FFFDD0",
                "hexColor2":"FFFF00",
                "hexColor3":"FFFFFF",
            }
        ]
    elif id == '2008':
        detail = [
            {
                "idOutfit": "2008",
                "strOutfit": "Outfit Neutral 8",
                "strCategory": "Neutral",
                "strDescription": "Kombinasi walnut dan coksu menciptakan palet yang hangat dan alami,\ncocok untuk desain yang menginginkan nuansa tradisional dan kehangatan alam.",
                "strOutfitThumb": "http://192.168.48.55:5000/img/N8.jpg",
                "hexColor1":"6E4826",
                "hexColor2":"B87333",
                "hexColor3":"6E4826",
            }
        ]
    elif id == '2009':
        detail = [
            {
                "idOutfit": "2009",
                "strOutfit": "Outfit Neutral 9",
                "strCategory": "Neutral",
                "strDescription": "Kombinasi black, broken white,\ndan dark brown menciptakan palet yang elegan dan hangat,\nideal untuk menonjolkan keanggunan dan kehangatan alami warna kulit netral dalam desain interior atau fashion.",
                "strOutfitThumb": "http://192.168.48.55:5000/img/N9.jpg",
                "hexColor1":"000000",
                "hexColor2":"F0EAD6",
                "hexColor3":"654321",
            }
        ]
    elif id == '2010':
        detail = [
            {
                "idOutfit": "2010",
                "strOutfit": "Outfit Neutral 10",
                "strCategory": "Neutral",
                "strDescription": "Kombinasi ini bisa dimaksimalkan\ndengan paduan yang seimbang antara coksu dan maroon, mungkin dengan\nmenggunakan coksu sebagai warna dominan dan aksen maroon untuk memberikan kontras yang elegan.",
                "strOutfitThumb": "http://192.168.48.55:5000/img/N10.jpg",
                "hexColor1":"AF8F6F",
                "hexColor2":"A91D3A",
                "hexColor3":"AF8F6F",
            }
        ]
    elif id == '3001':
        detail = [
            {
                "idOutfit": "3001",
                "strOutfit": "Outfit Warm 1",
                "strCategory": "Warm",
                "strDescription": "Kuning, putih, dan hitam dapat melengkapi warna kulit hangat dengan indah. Kuning menambahkan keceriaan dan kehangatan\n\nputih memberikan kontras yang bersih dan segar\n\nsedangkan hitam menambahkan kedalaman dan keserbagunaan\n\nsemuanya meningkatkan keindahan warna kulit hangat.",
                "strOutfitThumb": "http://192.168.48.55:5000/img/W1.jpg",
                "hexColor1":"FFD700",
                "hexColor2":"FFFFFF",
                "hexColor3":"000000",
            }
        ]
    elif id == '3002':
        detail = [
            {
                "idOutfit": "3002",
                "strOutfit": "Outfit Warm 2",
                "strCategory": "Warm",
                "strDescription": "Kombinasi blue, red, dan black cocok untuk menonjolkan kehangatan dan keceriaan warna kulit warm. Blue dapat dipilih dalam nuansa biru langit untuk memberikan kedalaman yang menenangkan,\nred bisa dalam nuansa merah cerah untuk energi yang menyenangkan,\ndan black sebagai elemen kontras yang elegan,\nmenciptakan tampilan yang berimbang dan menarik untuk warna kulit warm.",
                "strOutfitThumb": "http://192.168.48.55:5000/img/W2.jpg",
                "hexColor1":"0000FF",
                "hexColor2":"8B0000",
                "hexColor3":"000000",
            }
        ]
    elif id == '3003':
        detail = [
            {
                "idOutfit": "3003",
                "strOutfit": "Outfit Warm 3",
                "strCategory": "Warm",
                "strDescription":"Kombinasi biru muda, orange jeruk, dan white menciptakan palet yang ceria dan menyegarkan, ideal untuk menonjolkan kehangatan alami pada warna kulit warm dengan sentuhan yang energik dan harmonis.",
                "strOutfitThumb": "http://192.168.48.55:5000/img/W3.jpg",
                "hexColor1":"ADD8E6",
                "hexColor2":"FFA500",
                "hexColor3":"FFFFFF",
            }
        ]
    elif id == '3004':
        detail = [
            {
                "idOutfit": "3004",
                "strOutfit": "Outfit Warm 4",
                "strCategory": "Warm",
                "strDescription":"Kombinasi dark brown dan hijau army menciptakan palet warna yang hangat dan alami,\nideal untuk menonjolkan keanggunan serta\nkarakter maskulin dari warna kulit warm dengan kontras yang menenangkan dan stabil.",
                "strOutfitThumb": "http://192.168.48.55:5000/img/W4.jpg",
                "hexColor1":"FDAF7B",
                "hexColor2":"254336",
                "hexColor3":"76453B",
            }
        ]
    elif id == '3005':
        detail = [
            {
                "idOutfit": "3005",
                "strOutfit": "Outfit Warm 5",
                "strCategory": "Warm",
                "strDescription":"Kombinasi hijau army, taupe, dan brown menghasilkan palet warna yang hangat dan alami.\nHijau army memberikan nuansa alamiah yang maskulin,\nsementara taupe memberikan sentuhan lembut dan elegan.\nBrown menambahkan kedalaman dan kehangatan yang cocok untuk menonjolkan keanggunan pada warna kulit warm.",
                "strOutfitThumb": "http://192.168.48.55:5000/img/W5.jpg",
                "hexColor1":"43766C",
                "hexColor2":"B19470",
                "hexColor3":"E7D4B5",
            }
        ]
    elif id == '3006':
        detail = [
            {
                "idOutfit": "3006",
                "strOutfit": "Outfit Warm 6",
                "strCategory": "Warm",
                "strDescription":"Kombinasi light grey, dark brown,\ndan light blue menciptakan palet yang menyeimbangkan kehangatan alami\ndari warna kulit hangat dengan sentuhan segar dan kedalaman yang elegan.",
                "strOutfitThumb": "http://192.168.48.55:5000/img/W6.jpg",
                "hexColor1":"D3D3D3",
                "hexColor2":"654321",
                "hexColor3":"87CEEB",
            }
        ]
    elif id == '3007':
        detail = [
            {
                "idOutfit": "3007",
                "strOutfit": "Outfit Warm 7",
                "strCategory": "Warm",
                "strDescription":"Kombinasi hitam, brown,\ndan hitam cocok untuk menampilkan kehangatan dan keanggunan pada warna kulit warm.\nHitam memberikan kontras dramatis yang menonjolkan elegansi, sementara nuansa brown menambahkan kedalaman alami yang mempertegas karakter hangat dari warna kulit.",
                "strOutfitThumb": "http://192.168.48.55:5000/img/W7.jpg",
                "hexColor1":"000000",
                "hexColor2":"ECB176",
                "hexColor3":"000000",
            }
        ]
    elif id == '3008':
        detail = [
            {
                "idOutfit": "3008",
                "strOutfit": "Outfit Warm 8",
                "strCategory": "Warm",
                "strDescription":"Kombinasi hitam, brown,\ndan hitam cocok untuk menampilkan kehangatan dan keanggunan pada warna kulit warm.\nHitam memberikan kontras dramatis yang menonjolkan elegansi, sementara nuansa brown menambahkan kedalaman alami yang mempertegas karakter hangat dari warna kulit.",
                "strOutfitThumb": "http://192.168.48.55:5000/img/W8.jpg",
                "hexColor1":"1A1A1A",
                "hexColor2":"FFA500",
                "hexColor3":"1A1A1A",
            }
        ]
    elif id == '3009':
        detail = [
            {
                "idOutfit": "3009",
                "strOutfit": "Outfit Warm 9",
                "strCategory": "Warm",
                "strDescription":"Memakai kombinasi outfit orange, black,\ndan brown dapat menciptakan tampilan yang berani dan hangat, menonjolkan\nkeceriaan dari warna orange serta kehangatan\ndan kedalaman dari black dan brown, cocok untuk warna kulit warm.",
                "strOutfitThumb": "http://192.168.48.55:5000/img/W9.jpg",
                "hexColor1":"8FD694",
                "hexColor2":"000000",
                "hexColor3":"654321",
            }
        ]
    elif id == '3009':
        detail = [
            {
                "idOutfit": "3009",
                "strOutfit": "Outfit Warm 9",
                "strCategory": "Warm",
                "strDescription":"Kombinasi gray, black, dan red\nmenciptakan tampilan yang kuat dan berkesan,\nmenonjolkan kehangatan alami dan keanggunan\npada warna kulit warm dengan kontras yang dramatis dan energik.",
                "strOutfitThumb": "http://192.168.48.55:5000/img/W9.jpg",
                "hexColor1":"808080",
                "hexColor2":"000000",
                "hexColor3":"FF0000",
            }
        ]
    else:
        return jsonify({"message": "Detail not found"})

    return jsonify({"detail": detail})