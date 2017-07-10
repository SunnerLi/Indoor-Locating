from abc import ABCMeta, abstractmethod

class A():
    __metaclass__ = ABCMeta
    def __init__(self):
        pass
    
    @abstractmethod
    def Foo(self):
        pass

class B():
    def __init__(self):
        pass

    

_obj1 = B()
_obj1.Foo()