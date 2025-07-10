import base64
from io import BytesIO
from PIL import Image


def format_api_content(type: str, input: str):
    '''Converts input text to a format suitable for API requests
    
    Parameters
    ----------
    type : str
        The type of content to be converted. Can be "text" or "image".
    input : strself._self._
        The input content to be converted.'''

    if type == "text":
        return {"type": "text", "text": str(input)}
    elif type == "image_url":
        return {"type": "image_url", "image_url": {"url": base64_to_url(input)}}
    else:
        raise ValueError(f"Unsupported type: {type}")

def is_base64(s: str)->bool:
    """
    Check if a string is in valid Base64 format.

    Parameters:
    -----------
    s : str
        The string to check.

    Returns:
    --------
    bool
        True if the string is in valid Base64 format, False otherwise.
    
    Notes:
    ------
    Non-zero chance that the model will return a base64 string that is not a
    valid image. However, since spaces aren't allowed, the model would need to
    return a one word response.
    """
    # Try to decode the string
    try:
        decoded = base64.b64decode(s)
        # Check if the decoded string can be encoded back to the original
        return base64.b64encode(decoded).decode() == s
    except Exception:
        return False

def base64_to_url(base64_str: str):
    '''Converts a base64 string to a URL
    
    Parameters
    ----------
    base64_str : str
        The base64 string to be converted.'''
    
    return "data:image/jpeg;base64," + base64_str

def image2str(image)->str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image = base64.b64encode(buffer.read()).decode('ascii')
    return(image)

def str2image(image_str):
    #Converts a Base64 encoded string to an image.
    img_bytes = base64.b64decode(image_str)
    img_buffer = BytesIO(img_bytes)
    img = Image.open(img_buffer)
    return img