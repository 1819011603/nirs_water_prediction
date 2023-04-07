import random
import numpy as np
random.seed(1121)

def save_to_file(lines,path):
    with open(path,"a") as f:
        f.writelines(lines)


def f1():
    with open("./test_all_reflect1.txt", "r") as f:
        lines = f.readlines()
        lines = np.array(lines)
        # print(len(lines))
        for i in range(int(len(lines) / 10)):
            p = list(range(i * 10, (i + 1) * 10))
            random.shuffle(p)
            # print(p)
            save_to_file(lines[p[:2]], "./test_imp.txt")
            save_to_file(lines[p[2:]], "./train_imp.txt")
if __name__ == '__main__':
    with open("./train.txt","r") as f:
        lines = f.readlines()
        lines = np.array(lines)
        f = list(range(0,len(lines)))

        random.shuffle(f)
        save_to_file(lines[f],"./train_imp.txt")




