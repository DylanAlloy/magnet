from fastapi import FastAPI

# The file where NeuralSearcher is stored
from neural_search import NeuralSearcher

app = FastAPI()

# Create a neural searcher instance
neural_searcher = NeuralSearcher(collection_name='filings')

@app.get("/api/search")
def search_startup(q: str):
    return {
        "result": neural_search.search(text=q)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

 