import gdown
from os.path import join, split

_HERE = split(__file__)[0]

url = 'https://drive.google.com/uc?id=1-FG8wF1DKr5TsXL0TCvvmDOC-egy2FkM'
output = join(_HERE, '..', 'VASP_ML20_1.data-00000-of-00001')
gdown.download(url, output, quiet=False)


url = 'https://drive.google.com/uc?id=1-7uKUakqU4_yePZtFbuQyRxlGO0SeHzl'
output = join(_HERE, '..', 'VASP_ML20_1.index')
gdown.download(url, output, quiet=False)


url = 'https://drive.google.com/uc?id=1-7qD0EujpZm5arJm49ltxDBu0TnOmFZT'
output = join(_HERE, '..', 'checkpoint')
gdown.download(url, output, quiet=False)
