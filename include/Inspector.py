import os
import json
import include.network.net_constants as netco
class Inspector():
    """
        @cass Base Inspector
    """
    
    def __init__(self,
                folder_path=None,
                indent=True):
        self.inspect_path = folder_path
        if indent:
            self.space = 4
        else:
            self.space = 0


class ModelsInspector(Inspector):
    """
        @cass Models Inspector

        --Monitors Models and Networks for Version Control
    """
    
    def __init__(self,a,b,
                folder_path=None,
                indent=True):
        super(ModelsInspector,self).__init__(os.path.join(os.getcwd(),netco.CLASSIFIERS))
        self.a = a
        self.b = b
        
    def listNested(self):
        print(self.inspect_path)
        '''
        for nested in os.listdir(self.inspect_path):
            print(nested)
            print('->')
            if os.listdir(os.path.join(self.inspect_path,nested))==[]:
                print(os.listdir(os.path.join(self.inspect_path,nested)))
                pass
            else:
                print(os.listdir(os.path.join(self.inspect_path,nested)))
                print(50*'*')
                self.listNested()
        '''

    def writeJSON(self):
        pass
