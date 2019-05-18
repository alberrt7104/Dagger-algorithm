from sklearn import svm  
import os
word=0
inputx=[]
outputy=[]
convert={'a':1,'b':2,'c':3,'d':4,'e':5,'f':6,'g':7,'h':8,'i':9,'j':10,'k':11,'l':12,'m':13,'n':14,'o':15,'p':16,'q':17,'r':18,'s':19,'t':20,'u':21,'v':22,'w':23,'x':24,'y':25,'z':26}
seq=[]
with open("ocr_fold0_sm_train.txt","r") as f:
    lines=f.readlines()
    for line in lines:
        if len(line)!=4:
            c=line.split()            
            inputx.append(list(c[1]))
            outputy.append(c[2])
            

#print(outputy)
for data in inputx:
    data.remove("i")
    data.remove("m")

#print(inputx)
for data in inputx:
    data=data+[0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    #print(data)
replace=0
word=0
with open("ocr_fold0_sm_train.txt","r") as f:
    lines=f.readlines()
    for line in lines:
        if len(line)!=4:
            if c[0]=='1':
                seq.clear()
            c=line.split()
            seq.append(c[2])
            for i in range(len(seq)):
                #print(i)
                if i==0:
                    word+=1
                else:
                    inputx[replace][-14+i]=convert[seq[i]]

            

            replace=replace+1



#print(datax)
count=0
word=0
testinput=[]
testoutput=[]
with open("ocr_fold0_sm_test.txt","r") as f:
    lines=f.readlines()
    for line in lines:
        if len(line)!=4:
            c=line.split()
            testinput.append(list(c[1]))
            testoutput.append(c[2])
        	#print(c[1])
        	#print(c[2])
#print(inputx)
#print(testoutput)
testx=[]
for data in testinput:
    data.remove("i")
    data.remove("m")

for data in testinput:
    data=data+[0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    #print(data)
replace=0
word=0
with open("ocr_fold0_sm_test.txt","r") as f:
    lines=f.readlines()
    for line in lines:
        if len(line)!=4:
            
            if c[0]=='1':
                seq.clear()
            c=line.split()
            seq.append(c[2])
            for i in range(len(seq)):
                #print(i)
                if i==0:
                    word+=1
                else:
                    testinput[replace][-14+i]=convert[seq[i]]
            

            replace=replace+1

#print(testx)
#using sklearn svm for linear classifier
x=inputx
y=outputy 
r=[]
check=[]
clf = svm.SVC(probability=True)  # class   
clf.fit(x, y)  # training the svc model  
z = testinput
k=testoutput
result = clf.predict(z) # predict the target of testing samples   
r=result  # target   
#print(r)
for i in range(len(r)):
    check.append(r[i])
#print(check)
#print(len(r))
# print (clf.support_vectors_)  #support vectors  
  
# print (clf.support_)  # indeices of support vectors  
#print(clf.predict_proba(z))
# print (clf.n_support_)  # number of support vectors for each class  
print("Recurrent Classifier Learning via Exact Imitation:")
print("training accuracy:",clf.score(x,y)) #algo1 train
print("testing accuracy:",clf.score(z,k)) #algo1 test

counterror=0
for i in range(len(check)):
    if check[i] != testoutput[i]:
        x.append(testinput[i])
        y.append(testoutput[i])
        counterror=counterror+1
        #print(inputx)
        #print(outputy)
#print(counterror)
#print(len(x)," +",len(y))
#print(len(inputx)," +",len(outputy))

csvm=svm.SVC()
csvm.fit(x,y)
dr=csvm.predict(z)
print("Dagger:")

print("iteration 1:")
print("training accuracy:",csvm.score(inputx,outputy)) #alg2 train
print("testing accuracy:",csvm.score(z,k)) #alg2 test
da=[]
for i in range(4):
    check.clear()
    for j in range(len(dr)):
        if i==0:
            check.append(dr[j])
        else:
            check.append(da[j])
    da=[]
    for j in range(len(check)):
        if check[j] != testoutput[j]:
            if check[j] in x:
                counterror=counterror+1
                print("alredy")
            else:
                x.append(testinput[j])
                y.append(testoutput[j])
    dsvm=svm.SVC()
    dsvm.fit(x,y)
    da=dsvm.predict(z)
    print("iteration",i+2)
    print("training accuracy:",dsvm.score(inputx,outputy)) 
    print("testing accuracy",dsvm.score(z,k))






