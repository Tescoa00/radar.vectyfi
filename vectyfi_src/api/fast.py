from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# --- Modèle de données ---
class Item(BaseModel):
    name: str
    price: float
    in_stock: bool = True

# Fausse base de données en mémoire
fake_db = [
    {"id": 1, "name": "Apple", "price": 1.5, "in_stock": True},
    {"id": 2, "name": "Banana", "price": 0.8, "in_stock": False},
]

# --- Routes ---

# GET tous les items
@app.get("/items")
def get_items():
    return fake_db

# GET un item par ID
@app.get("/items/{item_id}")
def get_item(item_id: int):
    item = next((i for i in fake_db if i["id"] == item_id), None)
    return item or {"error": "Not found"}

# POST créer un item
@app.post("/items")
def create_item(item: Item):
    new_item = {"id": len(fake_db) + 1, **item.dict()}
    fake_db.append(new_item)
    return new_item

# PUT modifier un item
@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    for i, existing in enumerate(fake_db):
        if existing["id"] == item_id:
            fake_db[i] = {"id": item_id, **item.dict()}
            return fake_db[i]
    return {"error": "Not found"}

# DELETE supprimer un item
@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    for i, existing in enumerate(fake_db):
        if existing["id"] == item_id:
            fake_db.pop(i)
            return {"message": "Deleted"}
    return {"error": "Not found"}
