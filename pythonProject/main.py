import sys
from pip._internal import main as pip_main

def install(package):
    print(['--default-timeout=1000','install','-U', package])
    pip_main(['--default-timeout=1000','install','-U', package,'-i','http://pypi.douban.com/simple', '--trusted-host','pypi.douban.com'])

if __name__=='__main__':
    with open("./requirements.txt",'r') as f:
        for line in f:
            install(str(line.strip()))
