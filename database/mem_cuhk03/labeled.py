for l in open('list.txt','r').readlines():
    print l[:-1], int(l[:4])-1
