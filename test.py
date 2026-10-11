from openai import OpenAI
import os

client = OpenAI(
    base_url="http://127.0.0.1:1234/",
    api_key = "random-key" #It is local anyways, so no need to worry about security 
)

