"""Symptom descriptions for the PlantDoc tomato experiment (field C).

Written on 11 Nov 2026 from standard plant-pathology descriptions, BEFORE any tomato image was embedded or any result was seen,
and not changed afterwards. Two descriptions per class, in the same style as the okra descriptions.
Order of TOMATO_CLASSES = label index in the tomato index (prepare_tomato.py).
"""
TOMATO_CLASSES = ["Healthy", "Early blight", "Septoria leaf spot", "Bacterial spot", "Late blight", "Mosaic virus",
                  "Yellow leaf curl virus", "Leaf mold"]
PLANTDOC_FOLDER = {"Healthy": "Tomato leaf", "Early blight": "Tomato Early blight leaf", "Septoria leaf spot": "Tomato Septoria leaf spot",
                   "Bacterial spot": "Tomato leaf bacterial spot", "Late blight": "Tomato leaf late blight",
                   "Mosaic virus": "Tomato leaf mosaic virus", "Yellow leaf curl virus": "Tomato leaf yellow virus", "Leaf mold": "Tomato mold leaf"}
TOMATO_PROMPTS = {
    "Healthy": ["a photo of a healthy green tomato leaf with no spots or lesions",
                "healthy tomato leaflet: uniformly green, smooth surface, no discoloration"],
    "Early blight": ["a photo of a tomato leaf with brown spots showing concentric rings like a target",
                     "tomato early blight: dark brown lesions with concentric rings and a yellow halo on older leaves"],
    "Septoria leaf spot": ["a photo of a tomato leaf with many small circular spots with grey centres and dark borders",
                           "tomato septoria leaf spot: numerous small round lesions with tan-grey centres, dark margins and tiny black dots"],
    "Bacterial spot": ["a photo of a tomato leaf with small dark greasy water-soaked spots",
                       "tomato bacterial spot: small dark brown to black angular lesions, sometimes with yellow halos"],
    "Late blight": ["a photo of a tomato leaf with large irregular dark brown greasy patches",
                    "tomato late blight: large water-soaked grey-green to brown blotches, white mould on the underside in humid weather"],
    "Mosaic virus": ["a photo of a tomato leaf with a light and dark green mottled mosaic pattern",
                     "tomato mosaic virus: mottled light and dark green areas, distorted, puckered or fern-like leaflets"],
    "Yellow leaf curl virus": ["a photo of a small tomato leaf curled upwards with yellow margins",
                               "tomato yellow leaf curl virus: upward cupping, yellowing of leaflet edges, stunted crinkled leaves"],
    "Leaf mold": ["a photo of a tomato leaf with pale yellow patches on top and olive-green velvety mould underneath",
                  "tomato leaf mold: diffuse yellow spots on the upper surface with olive to brown fuzzy growth below"],
}
# plausible tomato disorders that are NOT in the data: distractors for naming (bank = 8 classes + 8 distractors = 16 entries)
TOMATO_EXTRA_BANK = {
    "Powdery mildew": ["a photo of a tomato leaf covered with white powdery patches",
                       "tomato powdery mildew: white to grey powdery fungal growth on the leaf surface"],
    "Fusarium wilt": ["a photo of a yellowing wilting tomato leaf",
                      "tomato fusarium wilt: yellowing and wilting of leaves, often on one side of the plant"],
    "Leaf miner": ["a photo of a tomato leaf with winding white tunnels",
                   "leaf miner damage on tomato: pale serpentine mines and blotches inside the leaf"],
    "Spider mite damage": ["a photo of a tomato leaf with fine pale stippling and webbing",
                           "spider mite damage on tomato: tiny yellow speckles, bronzing and fine webs"],
    "Nitrogen deficiency": ["a photo of a uniformly pale yellow-green tomato leaf",
                            "nitrogen deficiency in tomato: general yellowing of older leaves without spots"],
    "Magnesium deficiency": ["a photo of a tomato leaf yellow between the veins while the veins stay green",
                             "magnesium deficiency in tomato: interveinal chlorosis of older leaves, veins remain green"],
    "Target spot": ["a photo of a tomato leaf with brown spots with light centres and faint rings",
                    "tomato target spot: brown necrotic lesions with pale centres and concentric zonation"],
    "Gray leaf spot": ["a photo of a tomato leaf with small grey-brown spots that crack in the centre",
                       "tomato gray leaf spot: small dark specks enlarging to grey-brown spots with shot-hole centres"],
}
