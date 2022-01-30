import os
import sys
import shutil
import tarfile
import requests
from tqdm import tqdm

def download(url: str):
    print('\n')
    fname = url.split('/')[-1]
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc= f'Downloading {fname}',
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    print('\n')


def unzip(name='images.tar'):
    with tarfile.open(name) as tar:

        for member in tqdm(
            desc= f'Unzipping {name}',
            iterable=tar.getmembers(), 
            total=len(tar.getmembers()),
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ):
            tar.extract(member=member, path='./')
    print('\n')


def organise():
    files = os.listdir('./Images')
    if not os.path.exists('./Images/train'):
        os.mkdir('./Images/train')
    for i in tqdm(
        desc      = f'Organising Folders',
        iterable  = files,
        total     = len(files)

    ):
        if i[-4:] != '.jpg':
            for j in os.listdir(f'./Images/{i}'):
                shutil.move(f'./Images/{i}/{j}','./Images/train/')
            os.rmdir(f'./Images/{i}')
    print('\n')
    os.rename('./Images', './images')
    os.remove('./images.tar')
    os.mkdir('./images/test')


def namer(x:int, length=6, ext='.jpg'):
    diff = length-len(str(x))
    return '0'*diff+str(x)+ext
    


def rename(path='./images/train'):
    files = os.listdir(path)
    for i, name in tqdm(
        desc     = 'Renaming files',
        iterable = enumerate(files),
        total    = len(files)
        ):
        os.rename(f'{path}/{name}',f'{path}/{namer(i)}' )
    print('\n All done!')
    

if __name__ == '__main__':

    if os.path.exists('./Images') or os.path.exists('./images'):
        print("You might already have the dataset. Check your directory! If not delete any directory named 'Images' or 'images' ")
        sys.exit()
    
    url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'
    
    download(url)
    unzip()
    organise()
    rename()