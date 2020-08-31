import requests

def do_get(url):
    rsp = requests.get(url)
    return rsp.text
