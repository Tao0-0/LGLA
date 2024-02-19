f = open('ImageNet_LT_backward50_.txt','r')
f1 = open('ImageNet_LT_backward50.txt','w')

line=f.readline()
while(line):
    line_new = line.split('/')
    f1.write(line_new[0]+'/'+line_new[2])
    line=f.readline()