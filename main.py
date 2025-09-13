
from fastapi import FastAPI
from pydantic import BaseModel

from service.rewrite import rewrite

from service.scraper import scraper

app = FastAPI()
def executeAgents(initialInput: str):
    variable = scraper(initialInput)
    variable2 = rewrite(variable)
    return {'variable': variable, 'variable2': variable2}

class InputSchema(BaseModel):
    initialInput: str

@app.post("/execute-agent-workflow")
def executeAgentWorkflow(request: InputSchema):
    print(request)
    try:
        response = executeAgents(request.initialInput)
        return response
    except Exception as e:
        print(e)
        raise Exception(e)

@app.get("/health")
def health():
    try:
        return { "message": "Hi HackTheNorth! The container is healthy!"}
    except Exception as e:
        raise Exception(e)
