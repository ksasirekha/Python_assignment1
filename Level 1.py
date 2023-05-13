#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1
print("Hello World")


# In[1]:


#2
a=int(input('First number:'))
b=int(input('Second number:'))
c=a+b
print("Sum",c)


# In[2]:


#3
a=int(input('First number:'))
b=int(input('Second number:'))
a,b=b,a
print('First number:',a)
print('Second number:',b)


# In[9]:


#4
a=input("Kilometers:")
b=float(a)*.621371
print(f'Miles:{b}')


# In[12]:


#5
a=int(input('Number:'))
if a>0:
    print("Positive number")
elif a<0:
    print("Negative number")
else:
    print("Zero")


# In[16]:


#6
a=int(input('Year:'))
if (a%4==0) and (a%100!=0) or (a%400==0):
    print("Leap year")
else:
    print("Normal year")


# In[5]:


#7
b=int(input("Begin:"))
e=int(input("End:"))
for i in range(b,e+1):
    #prime numbers will be positive
    if i>1:
        for j in range(2,i):
            if i%j==0:
                break
        else:
            print(i)


# In[6]:


#8
n=int(input('nth term:'))
n1,n2=0,1
c=1
if n<=0:
    print("Please enter a positive integer")
elif n==1:
    print("Fibonacci sequence upto",n,':')
    print(n1)
else:
    print("Fibonacci sequence:")
    while c<=n:
        print(n1)
        nth=n1+n2
        n1=n2
        n2=nth
        c+=1


# In[6]:


#9
a=int(input('Number:'))
s=0
temp=a
while temp>0:
    d=temp%10
    s+=d**3
    temp//=10
if a==s:
    print("Armstrong number")
else:
    print("Not a Armstrong number")


# In[2]:


#10
a=int(input("Range:"))
s=0
for i in range(1,a+1):
    s+=i
print("Sum=",s)


# In[11]:


#11
def show_stars(row):
    for i in range(row):
        for j in range(i+1):
            print('*',end='')
        print("\n")    
show_stars(5)


# In[21]:


#12
a='Python_programming'
n=int(input("Enter the value of n:"))
print(a)
res=a[n:]
print('Resultant string',res)


# In[17]:


#13
l=[1,5,26,34,50,10,20,23,620,25,96]
for i in l:
    if i%5==0:
        print(i)


# In[13]:


#14
a="Hi everyone,Hi ladies and gentlemen,Hi everybody,Hi one and all,Hi!Hi!Hi!"
print(a.count("Hi"))


# In[9]:


#15
for i in range(1,6):
        for j in range(i):
            print(i,end='')
        print('\n')


# In[16]:


#16
def palindrome(n):
    temp=n
    rev=0
    while n>0:
        d=n%10
        rev=rev*10+d
        n//=10
    if rev==temp:
        print("Palindrome")
    else:
        print("Not a palindrome")
palindrome(545)


# In[1]:


#17
l=[1,2,3,4,7,"hi"]
length=len(l)
print(length)


# In[2]:


#18
l=[1,2,3,4,7,"hi"]
l[0],l[5]=l[5],l[0]
print(l)


# In[4]:


#19
l=[1,2,3,4,7,"hi"]
l[0],l[1]=l[1],l[0]
print(l)


# In[10]:


#20
a=int(input('Enter first number:'))
b=int(input('Enter second number:'))
if (a>b):
    print('{} is maximum'.format(a))
else:
    print('{} is maximum'.format(b))


# In[20]:


#21
a=int(input('Enter first number:'))
b=int(input('Enter second number:'))
if (b>a):
    print('{0} is minimum'.format(a))
else:
    print('{0} is minimum'.format(b))


# In[25]:


#22
a="Hi everyone,Hi ladies and gentlemen,Hi everybody,Hi one and all,Hi!Hi!Hi!"
b="".join(reversed(a))
print("Reversed string")
print(b)
d="1234567890"
c=d[::-1]
print(c)


# In[28]:


#23
a="Sasirekha"
b=''.join([a[i] for i in range (len(a)) if i !=5])
print("Resultant string:"+b)


# In[32]:


#24
a="Sasirekha"
print("Length of string 'Sasirekha' is",len(a))


# In[37]:


#25
a="amaama"
h=int(len(a)/2)
if a[:h]==a[h:]:
    print("Symmetrical")
else:
    print("Not a symmetrical")
if a[:h]==''.join(reversed(a[h:])):
    print("Palindrome")
else:
    print("Not a palindrome")


# In[38]:


#26
a="Sasirekha is studyding Bachelor of Technology in Artificial Intelligence and Data Science"
b=a.split()
for i in b:
    if len(i)%2==0:
        print(i)


# In[15]:


#27
b=((23,12,44),(37,293,28))
print('Size of tuple b: ',str(b.__sizeof__()),'bytes')


# In[29]:


#28
import heapq
a=(5,20,3,7,6,8)
print("Original tuple:",a)
K=2
smallest=heapq.nsmallest(K,a)
largest=heapq.nlargest(K,a)
r=tuple(sorted(smallest+largest))
print("Extracted values:"+str(r))


# In[30]:


#29
b=(52,45,86,1,2,4,3,6,9,8)
a=sum(b)
print("Sum=",a)


# In[39]:


#30
t=((9,8,7,6),(5,4,3,3),(7,1,2,3,9))
print("Row-wise sum:")
for i in t:
    s=sum(i)
    print(s)

