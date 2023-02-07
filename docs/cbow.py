
### SK_011_NLP_03.ipynb
### CBOW implementation

###########################################################################

import pandas as pd
import numpy as np
import sys

###########################################################################

l=[
"c++ programming language",
"java programming language",
"python programming language",
"javascript programming language",
"php programming language",
"csharp programming language",
"go-language programming language",
"c-language programming language",
"kotlin programming language",
"swift programming language",

"c++ software language",
"java software language",
"python software language",
"javascript software language",
"php software language",
"csharp software language",
"go-language software language",
"c-language software language",
"kotlin software language",
"swift software language",

"csharp development language",
"java development language",
"python development language",
"javascript development language",
"php development language",
"csharp development language",
"go-language development language",
"c-language development language",
"kotlin development language",
"swift development language",

"data structures",
"data algorithm",
"machine learning",
"data-science algorithm",
"searching algorithm",
"graph algorithm",
"pattern searching",
"geometric algorithm",
"mathematical",
"bitwise algorithm",
"randomized algorithm",
"greedy algorithm",
"dynamic programming"
]


l=[
   "This is a test message", "taking same Count Vectorizer matrix calculated earlier",
   "replacing each cell of it by score for this term ", "this is a from online document"]

data = []
unique=[]
for i in l:
    k = []
    for j in i.split(" "):
        j = j.lower()
        if j not in unique:
            unique.append(j)
        k.append(j)
    data.append(k)

one_hot = pd.get_dummies(unique)

col = one_hot.columns
one_hot_dict = {}
for i in col:
    one_hot_dict[i] = np.array(one_hot[i], dtype=float)

###########################################################################

#embedding size
m = 25
#input vector size
n = len(one_hot_dict)
#matrix u with weights 
u = np.random.rand(n,m)
#matrix v with weights
v = np.random.rand(m,n)
#learning rate
alpha = 0.05
#window size exp 2 x1,x2,y,x3,x4
window_size = 3
#number of epochs 
epoch = 50
###########################################################################
def sbyafunc2(s,i,j):
    if (i==j):
        return (s[i]*(1-s[i])) 
    else:
        return (-1*s[i]*s[j])
###########################################################################	
def ebys(s,y):
    l=[]
    for i in range(n):
        l.append(y[i]/s[i]-((1-y[i])/(1-s[i])))
    l=np.array(l)
    return l
###########################################################################
def sbya(s):
    l=[[0 for i in range(n)]for j in range(n)]
    for i in range(n):
        for j in range(n):
            l[i][j]=sbyafunc2(s,i,j)
    return np.array(l)
###########################################################################
def abyv(z):
    l=[[0 for i in range(n*m)]for j in range(n)]
    c=0
    for i in range(n):
        index=0
        for j in range(c*m,(c*m)+m):
            l[i][j]=z[index]
            index=index+1
        c=c+1
    return np.array(l)
###########################################################################
def abyz():
    return np.transpose(v)
###########################################################################
def zbyu(x):
    l=[[0 for i in range(n*m)]for j in range(m)]
    c=0
    for i in range(m):
        index=0
        for j in range(c*n,(c*n)+n):
            l[i][j]=x[index]
            index=index+1
        c=c+1
    return np.array(l)
###########################################################################
def backward(s,y,z,x):
    global u,v
    de_ds=ebys(s,y)
    #print(de_ds.shape)
    ds_da=sbya(s)
    #print(ds_da.shape)
    da_dv=abyv(z)
    #print(da_dv.shape)
    da_dz=abyz()
    #print(da_dz.shape)
    dz_du=zbyu(x)
    #print(dz_du.shape)
    #updating v
    de_dv=np.dot(np.dot(de_ds,ds_da),da_dv)
    de_dv=de_dv.reshape(m,n)
    #print(de_dv.shape)
    v=v-alpha*de_dv
    #print(v.shape)
    #updating u
    de_du=np.dot(np.dot(np.dot(de_ds,ds_da),da_dz),dz_du)
    de_du=de_dv.reshape(n,m)
    #print(de_du.shape)
    u=u-alpha*de_du
    #print(u.shape)
###########################################################################
def softmax(x):
    exp=np.exp(x)
    sum_exp=sum(exp)
    return exp/sum_exp
###########################################################################
# def forward(z,y):
def forward(x,y):
    z=np.dot(x,u)
    a=np.dot(z,v)
    a=np.transpose(a)
    s=softmax(a)
#     print(x.shape)
#     print(u.shape)
#     print(z.shape)
#     print(v.shape)
#     print(a.shape)
#     print(s.shape)
    backward(s,y,z,x)
    #s=np.softmax(a)
###########################################################################	
# def create_input_x(l,y):
#     print(u)
#     z=np.array([0.0 for i in range (m)])
#     for i in l:
#         a=np.dot(one_hot_dict[i],u)
#         z=z+a
#     z=z/len(l)
#     forward(z,y)
###########################################################################
def create_input_x(l,y):
    #print(":", l)
    #print(y)
    x = np.array([0.0 for i in range (n)])
    for i in l:
        #print(i)
        x = x + one_hot_dict[i]
    #print(x)
    forward(x/len(l), y)
###########################################################################
def start_training(data):
    for i in range(len(data)):
        t = len(data[i])
        if (t <= 1): return;
        for j in range(t):
            str = data[i][j]
            y = one_hot_dict[data[i][j]]
            if(j >= window_size and j + window_size < t):
                find = data[i][j-window_size:j] + data[i][j+1:j+window_size+1]
            elif (j < window_size):
                find = data[i][0:j] + data[i][j+1:j+window_size+1]           
            elif(j + window_size >= t):
                find = data[i][j-window_size:j] + data[i][j+1:]
            create_input_x(find,y)
        
        
                
#         z=np.array([0.0 for i in range (n)])
#         t=len(data[i])
#         for k in range(len(data[i])):
#             if(k-window_size=>0):
#                 for j in range(k-window_size,k):
#                     print(data[i][j],end=" ")
#                     #a=one_hot_dic[l[i][j]]
#                     #print(one_hot_dict[data[i][j]],end=" ")
#             if(k+window_size+1<=t):
#                 for j in range(k+1,window_size+k+1):
#                     print(data[i][j],end=" ")
#                     #print(one_hot_dict[data[i][j]],end=" ")
#             print("\n")
###########################################################################
def train(data):
    for i in range(epoch):
        print(i)
        print(u)
        print(v)
        print("*************************************")
        start_training(data)
#     print(u)

def predict(l):
    x_pred=np.array([0.0 for i in range (n)])
    for i in l:
        x_pred=x_pred+one_hot_dict[i]
    z=np.dot(x_pred/len(l),u)
    a=np.dot(z,v)
    a=np.transpose(a)
    s=softmax(a)
    print(s)
    result = np.where(s == np.amax(s))
    print("**",result)
    print(unique[int(result[0])])

train(data)

l=[ "taking","the","count","vectorizer"]
predict(l)
