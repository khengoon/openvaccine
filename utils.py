import requests

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_vaccine = load_lottieurl('https://assets9.lottiefiles.com/packages/lf20_p2evb1ab.json')