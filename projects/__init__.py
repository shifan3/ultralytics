from abc import ABC, abstractmethod

class ProjectBase(ABC):

    @abstractmethod
    def cls_to_color(self):
        pass

    @abstractmethod
    def cls_to_name(self):
        pass
    
    
    @abstractmethod
    def cls_to_raw(self):
        pass
    
    @abstractmethod
    def raw_to_cls(self):
        pass

    def no_merge_cls(self):
        return []

    @abstractmethod
    def image_size(self):
        pass
    
    @abstractmethod
    def cls_selector(self):
        pass

    def all_cls(self):
        return []
