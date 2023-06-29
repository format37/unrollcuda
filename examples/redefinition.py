class MyClass:
    def __init__(self):
        self.value = 1
    def sum(self,a,b):
        return a+b


def main():
    myclass = MyClass()
    print(myclass.sum(3,4))
    # Redefine the inference method without lambda
    def mul(self,a,b):
        return a*b
    MyClass.sum = mul
    print(myclass.sum(3,4))
    

if __name__ == '__main__':
    main()
