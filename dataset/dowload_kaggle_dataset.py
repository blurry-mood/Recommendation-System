import gdown
from os.path import join, split

_HERE = split(__file__)[0]

url = 'https://drive.google.com/uc?id=1h8T4_N9EwqpoQ0D0I1PKiOMzAuMZbn5i'
output = join(_HERE, '..', 'zip1.zip')
gdown.download(url, output, quiet=False)


url = 'https://drive.google.com/uc?id=1TVcTf8EtZuWhekiw3tYxCtM6lwuzT9kz'
output = join(_HERE, '..', 'zip2.zip')
gdown.download(url, output, quiet=False)


url = 'https://drive.google.com/uc?id=1E0diCEpyGnFhZGc-fhMvIDB4SqTVEBSp'
output = join(_HERE, '..', 'zip3.zip')
gdown.download(url, output, quiet=False)
