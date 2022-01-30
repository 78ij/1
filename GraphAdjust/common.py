

FRONT_PATH = 'H:\\GRAINS-master\\0-data\\3D-FRONT'
FUTURE_PATH = 'H:\\GRAINS-master\\0-data\\3D-FUTURE-model'


categories = {
    "Children Cabinet" : "Storage_Furniture",
    "Nightstand" : "Table",
    "Bookcase / jewelry Armoire" : "Storage_Furniture",
    "Wardrobe" : "Storage_Furniture",
    "Coffee Table" : "Table",
    "Corner/Side Table" : "Table",
    "Sideboard / Side Cabinet / Console Table" : "Storage_Furniture",
    "Wine Cabinet" : "Storage_Furniture",
    "TV Stand" : "Table",
    "Drawer Chest / Corner cabinet" : "Storage_Furniture",
    "Shelf" : "Storage_Furniture",
    "Round End Table" : "Table",
    "King-size Bed" : "Bed",
    "Bunk Bed" : "Bed",
    "Bed Frame" : "Bed",
    "Single bed" : "Bed",
    "Kids Bed" : "Bed",
    "Dining Chair" : "Chair",
    "Lounge Chair / Cafe Chair / Office Chair" : "Chair",
    "Dressing Chair" : "Chair",
    "Classic Chinese Chair" : "Chair",
    "Barstool" : "Chair",
    "Dressing Table" : "Table",
    "Dining Table" : "Table",
    "Desk" : "Table",
    "Three-Seat / Multi-seat Sofa" : "Chair",
    "armchair" : "Chair",
    "Loveseat Sofa" : "Chair",
    "L-shaped Sofa" : "Chair",
    "Lazy Sofa" : "Chair",
    "Chaise Longue Sofa" : "Chair",
    "Footstool / Sofastool / Bed End Stool / Stool" : "Chair",
    "Pendant Lamp" : "Lamp",
    "Ceiling Lamp" : "Lamp"
}

category_class = [
    "Storage_Furniture", 
    "Table",
    "Chair", 
    "Bed", 
   # "Sofa",
    "Lamp",
    "Moveable_Slider",
    "Moveable_Revolute"
]

edge_category = [
    'Same',
    'Spatial',
    "Moveable"
]

visualize_color = {
    "Storage_Furniture" : [0.503981, 0.530892, 0.238707, 0.500000 ],
    "Table": [0.336221, 0.925177, 0.077883, 0.500000],
    "Chair": [0.937088, 0.615881, 0.401896, 0.500000 ], 
    "Bed": [0.703355, 0.101756, 0.980487, 0.500000 ], 
   # "Sofa",
    "Lamp": [0.728417, 0.556356, 0.508547, 0.500000],
    "Moveable_Slider": [0, 0, 0, 0.500000],
    "Moveable_Revolute": [0, 0, 0, 0.500000]
}