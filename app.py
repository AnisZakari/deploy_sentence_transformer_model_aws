from sentence_transformers import SentenceTransformer
import os 

# Check if the environment variable ENV_NAME is set and equal to 'Docker'
# Otherwise if it's not set, we are working locally.
if os.environ.get('ENV_NAME') == 'Docker':
    model_name_or_path = 'model/all-MiniLM-L6-v2'
    os.environ['TRANSFORMERS_CACHE'] = './model'
else:
    model_name_or_path = 'all-MiniLM-L6-v2'

model = SentenceTransformer(model_name_or_path)

def my_app(my_input):
    """Simple app that tranforms text(s) into vector(s) using a sentence transformer"""
    output = model.encode(my_input)
    return output

def lambda_handler(event:dict, context:str):
    # unpacking my_input
    my_input = event['my_input']
    # don't forget to convert the numpy array to list so it can be JSON Serializable
    output = my_app(my_input).tolist()
    return output