##class Animal(object):
##    def __init__(self,name,age=None):
##        self.name = name
##        self.age = age
##
##    def sound(self):
##        print('Animal can sound ...')

   
##Dude = Animal('Dude')
##print(Dude.name)
##Dude.sound()



class Animal(object):
    def __init__(self,name,age=None):
        self.__name = name
        self.__age = age

    def sound(self):
        print('Animal can sound ...')

    def print_age(self):
        print('%s is an animal with age:%s.'%(self.__name,self.__age))


class Dog(Animal):
    def sound(self):
        print('Dog can sound: wang,wang,wang....')

class Cat(Animal):
    def sound(self):
        print('Cat can sound: miao,miao,miao....')


Dude = Animal('Dude',2)
Dude.print_age()
##print(Dude.name)

try:
    print('try...')
    r = 10 / 0
    print('result:', r)
except ZeroDivisionError as e:
    print('except:', e)
finally:
    print('finally...')






##if __name__ =='__main__':






