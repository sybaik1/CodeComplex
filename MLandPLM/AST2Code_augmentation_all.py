import random
import javalang
import operator
valid_type={'char':chr,
            'double':float,
            'int':int,
            'long':str,
            'String':str}

def noraml_assignment(val1,val2): return val2 
assigmen_type={'%=':operator.mod,
               '*=':operator.mul,
               '+=':operator.add,
               '-=':operator.sub,
               '/=':operator.truediv,
               '=':noraml_assignment}

class AST2Code_module():
    def __init__(self,variable=0.0,folding=0.0,bracket=0.0,for_while=0.0,ternary=0.0,dead_code=1):
        r'''
        init module for ast -> code
        Arguments:
        
        dead_code (:obj:`float`):
            probability of code dead code elimination.
        '''

        #function dictornary for node-function matching.key: node name , value : node convert function
        self.FUNC_DICT={
        "ClassCreator":self.ClassCreator2Code,
        "Literal":self.Literal2Code,
        'BinaryOperation':self.BinaryOperation2Code,
        'MemberReference':self.MemberReference2Code,
        'BlockStatement':self.BlockStatement2Code,
        'list':self.List2Code,
        'LocalVariableDeclaration':self.LocalVariableDeclaration2Code, #variavle Declaration
        'VariableDeclaration':self.VariableDeclaration2Code,
        'VariableDeclarator':self.VariableDeclarator2Code,
        'ArrayCreator': self.ArrayCreator2Code,
        'StatementExpression':self.StatementExpression2Code,
        'ArraySelector':self.ArraySelector2Code,
        'MethodInvocation':self.MethodInvocation2Code,
        'Assignment':self.Assignment2Code,
        'IfStatement':self.IfStatement2Code,
        'ForStatement':self.ForStatement2Code,
        'MethodDeclaration':self.MethodDeclaration2Code,
        'Cast':self.Cast2Code,
        'ReturnStatement':self.ReturnStatement2Code,
        'ClassDeclaration':self.ClassDeclaration2Code,
        'FieldDeclaration':self.FieldDeclaration2Code,
        'NoneType':self.Nonetype,
        'ConstructorDeclaration':self.ConstructorDeclaration2Code,
        'WhileStatement':self.WhileStatement2Code,
        'TryStatement':self.TryStatement2Code,
        'CatchClauseParameter':self.CatchClauseParameter2Code,
        'ThrowStatement':self.ThrowStatement2Code,
        'CatchClause':self.CatchClause2Code,
        'TryResource':self.TryResource2Code,
        'This':self.This2Code,
        'TernaryExpression':self.TernaryExpression2Code,
        'BreakStatement':self.BreakStatement2Code,
        'ExplicitConstructorInvocation':self.ExplicitConstructorInvocation2Code,
        'DoStatement':self.DoStatement2Code,
        'InterfaceDeclaration':self.InterfaceDeclaration2Code,
        'ContinueStatement':self.ContinueStatement2Code,
        'Statement':self.Statement2Code,
        'ArrayInitializer':self.ArrayInitializer2Code,
        'SuperConstructorInvocation':self.SuperConstructorInvocation2Code,
        'SwitchStatement':self.SwitchStatement2Code,
        'EnhancedForControl':self.EnhancedForControl2Code,
        'ForControl':self.ForControl2Code,
        'ReferenceType':self.Type2Code,
        'AssertStatement':self.AssertStatement2Code,
        'SuperMethodInvocation':self.SuperMethodInvocation2Code,
        'BasicType':self.BasicType2Code,
        'LambdaExpression':self.LambdaExpression2Code,
        'FormalParameter':self.FormalParameter2Code,
        'InferredFormalParameter':self.InferredFormalParameter2Code,
        'MethodReference':self.MethodReference2Code,
        'ConstantDeclaration':self.ConstantDeclaration2Code,
        'EnumDeclaration':self.EnumDeclaration2Code,
        'EnumBody':self.EnumBody2Code,
        'EnumConstantDeclaration':self.EnumConstantDeclaration2Code,
        'InnerClassCreator':self.InnerClassCreator2Code,
        'SynchronizedStatement':self.SynchronizedStatement2Code,
        'ElementArrayValue':self.ElementArrayValue2Code,
        'ClassReference':self.ClassReference2Code,
        'AnnotationDeclaration':self.AnnotationDeclaration2Code

        }
        self.variable,self.folding,self.bracket,self.for_while,self.ternary,self.dead_code= \
        variable,folding,bracket,for_while,ternary,dead_code
    def check_binary(self,node:javalang.tree.BinaryOperation,value_dict:dict):
        ops = {
        '+' : operator.add,
        '-' : operator.sub,
        '*' : operator.mul,
        '/' : operator.truediv,  # use operator.div for Python 2
        '%' : operator.mod,
        # '^' : operator.xor,
        }
        valid_type=['Literal','MemberReference']

        nodes=[node.operandl,node.operandr]
        results=[]
        values=[]
        if node.operator not in ops.keys():
            return False
        for n in nodes:
            if type(n).__name__ =='BinaryOperation':
                results.append(self.check_binary(n,value_dict))
                if results[-1]==False:
                    return False
            if type(n).__name__ not in valid_type:
                return False
            if type(n).__name__ =='MemberReference' and n.member in value_dict.keys():
                #cast
                values.append(value_dict[n.member]['type'](value_dict[n.member]['value']))
            else:
                values.append(n.value)
            

        return str(ops[node.operator](values[0],values[1]))

    def parse_variable(self,node:javalang.tree.LocalVariableDeclaration,value_dict:dict):
        if node.type.name in valid_type.keys():
            for  declarator in node.declarators:
                if type(declarator.initializer).__name__=='NoneType':
                    value_dict[declarator.name]={'type':valid_type[node.type.name],'value':None}

                elif type(declarator.initializer).__name__=='Literal':
                    value_dict[declarator.name]={'type':valid_type[node.type.name],'value':declarator.initializer.value}

    def check_valid_assignment(self,node):
        if type(node).__name__=="Literal":
            return True,node.value
        
        if type(node).__name__=='BinaaryOperation':
            result,value=self.check_binary(node,self.fold_variables)
            return result,value
        
        return False,0
    def AST2Code(self,root):

        code=''
        self.check_variables(root)
        self.check_methods(root)
        self.check_class(root)

        if self.dead_code>0:
            self.check_dead_code(root)
        #start convert
        if root.package:
            code+= 'package '+root.package.name+';\n'
        for import_item in root.imports:
            code+= 'import '+('static ' if import_item.static else '') +import_item.path+ ('.*' if import_item.wildcard else '') + ';\n'

        for child in root.types:
            code+=self.FUNC_DICT[type(child).__name__](node=child)
        return code
    def check_variables(self,tree):
        self.variable_dict=dict()
        for _,node in tree:
            if type(node).__name__=='VariableDeclarator':
                self.variable_dict[node.name]='VAR'+str(len(self.variable_dict))

    def check_methods(self,tree):
        self.method_dict=dict()
        for _,node in tree:
            if type(node).__name__=='MethodDeclaration' and node.name != 'main':
                self.method_dict[node.name]='METHOD'+str(len(self.method_dict))   
    def check_class(self,tree):
        self.class_dict=dict()
        for _,node in tree:
            if type(node).__name__=='ClassDeclaration':
                self.class_dict[node.name]='CLASS'+str(len(self.class_dict)) 

    def get_variable_name(self,node,attr):
        result= self.variable_dict[getattr(node,attr)] if getattr(node,attr) in self.variable_dict.keys() else getattr(node,attr)
        return result
    
    def get_method_name(self,node,attr):
        result= self.method_dict[getattr(node,attr)] if getattr(node,attr) in self.method_dict.keys() else getattr(node,attr)
        return result
    
    def get_class_name(self,node,attr):
        result= self.class_dict[getattr(node,attr)] if getattr(node,attr) in self.class_dict.keys() else getattr(node,attr)
        return result
    
    def check_dead_code(self,tree):
        self.valid_method=set()
        self.valid_class=set()
        self.valid_method.add('main')
        for _,node in tree:
            if type(node).__name__ == 'ClassDeclaration':
                for class_body in node.body:
                    if type(class_body).__name__ == 'MethodDeclaration' and class_body.name=='main':
                        self.valid_class.add(node.name)
            elif type(node).__name__=='MethodInvocation':
                self.valid_method.add(node.member)
                
            elif type(node).__name__=='ClassCreator':
                self.valid_class.add(self.Type2Code(node=node.type))

    def split_method(self,root):
        '''
        for new model. code split to class,method
        '''
        code_array=[]
        for node in root.types:
            method_array=[]
            if type(node).__name__=='ClassDeclaration': # find class 
                for node_child in node.body:
                    if type(node_child).__name__=='MethodDeclaration' and node_child.name =='main':
                        method_index=len(method_array)
                        cla_index=len(code_array)
                    method_array.append(self.FUNC_DICT[type(node_child).__name__](node=node_child))
            else:
                for node_child in node.body:
                    method_array.append(self.FUNC_DICT[type(node_child).__name__](node=node_child))
            if len(method_array)>0:
                code_array.append(method_array)
        return code_array,{"method":method_index,"class":cla_index}

    def Nonetype(self,**kwargs):
        return ''

    def FieldDeclaration2Code(self,**kwargs):
        node=kwargs['node']
    
        code_snippet=(node.documentation+'\n' if node.documentation else '')
        for annot in node.annotations:
            code_snippet+='@'+annot.name+'('+self.FUNC_DICT[type(annot.element).__name__](node=annot.element)+')\n'
        modifier_arr=[]
        for modifier in node.modifiers:
            modifier_arr.append(modifier)
        code_snippet+=' '.join(reversed(modifier_arr)) +' '
        
        code_snippet+=self.Type2Code(node=node.type)+' '
        for declarator in node.declarators:
            code_snippet+= self.FUNC_DICT[type(declarator).__name__](node=declarator)+','
        code_snippet=code_snippet.rstrip(',')

        return code_snippet+';\n'

    def WhileStatement2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=''
        if node.label:
            code_snippet+=node.label+':'

        code_snippet+='while('+self.FUNC_DICT[type(node.condition).__name__](node=node.condition,inline=False)+')'
        code_snippet+=self.FUNC_DICT[type(node.body).__name__](node=node.body)
        return code_snippet

    def ConstructorDeclaration2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=node.documentation +'\n'if node.documentation != None else ''
        code_snippet+=' '.join(reversed(list(node.modifiers)))
        if node.type_parameters:
            code_snippet+='<'
            for param in node.type_parameters:
                code_snippet+=self.get_variable_name(param,'name')
                if param.extends:
                    code_snippet+=' extends '
                    for extend in param.extends:
                        code_snippet+=' ' + self.Type2Code(node=extend)
                code_snippet+=','
            code_snippet=code_snippet.rstrip(',') +'>'
        code_snippet+=' '+ self.get_class_name(node,'name')
        code_snippet=code_snippet.lstrip()
        paramter=''
    
        for pram in node.parameters:
            paramter+=' '.join(reversed(list(pram.modifiers))) +' '
            paramter+=self.Type2Code(node=pram.type)
            paramter+=('...' if pram.varargs else '')
            paramter+=' '+ pram.name +','
        
        code_snippet+='('+paramter.rstrip(',')+')'
        code_snippet+= ('throws ' +','.join(node.throws) if node.throws else '')
        code_snippet+='{\n'
    
        
        for child in node.body:
            code_snippet+= self.FUNC_DICT[type(child).__name__](node=child)
        return code_snippet+'}\n'

    def ClassDeclaration2Code(self,**kwargs):
        node=kwargs['node']
        if (random.random() < self.dead_code and node.name not in self.valid_class)\
        or node.name in self.valid_class:
            code_snippet=''
            for annot in node.annotations:
                code_snippet+='@'+annot.name+'('+self.FUNC_DICT[type(annot.element).__name__](node=annot.element)+')\n'
            code_snippet+=(node.documentation+'\n' if node.documentation else '')

            code_snippet+=' '.join(reversed(list(node.modifiers))) +' '
    
            code_snippet+=' class '+node.name
            if node.type_parameters:
                code_snippet+='<'
                for param in node.type_parameters:
                    code_snippet+=self.get_variable_name(param,'name')
                    if param.extends:
                        code_snippet+=' extends '
                        for extend in param.extends:
                            code_snippet+=' ' + self.Type2Code(node=extend)
                    code_snippet+=','
                code_snippet=code_snippet.rstrip(',') +'>'

            if node.extends:
                code_snippet+=' extends '+self.Type2Code(node=node.extends)
            if node.implements:
                code_snippet+=' implements '
                code_snippet+=','.join(self.Type2Code(node=implement)for implement in node.implements)
            code_snippet+='{\n'
            for class_item in node.body:
                code_snippet+=self.FUNC_DICT[type(class_item).__name__](node=class_item)
            code_snippet+='}\n'
        else:
            code_snippet=''
        return code_snippet
    # def process_modifier(node):
    #     if ''

    def MethodDeclaration2Code(self,**kwargs):
        node=kwargs['node']
        
        if (random.random() < self.dead_code and node.name not in self.valid_method)\
              or node.name in self.valid_method:
            code_snippet=(node.documentation+'\n' if node.documentation else '')
            if node.annotations:
                for annotation in node.annotations:
                    code_snippet+='@'+annotation.name
                    if annotation.element:
                        code_snippet+='('+self.FUNC_DICT[type(annotation.element).__name__](node=annotation.element)+')'
                    code_snippet+='\n'
            #modifiers processing
            
            if 'abstract' in node.modifiers:
                code_snippet+='abstract '
            for atr in ['public', 'private', 'protected']:
                if atr in node.modifiers:code_snippet+=f'{atr} '
            for atr in ['static','final']:
                if atr in node.modifiers:code_snippet+=f'{atr} '
            # code_snippet+=' '.join(reversed(list(node.modifiers))) +' '

            
            if node.type_parameters:
                code_snippet+='<'
                for param in node.type_parameters:
                    code_snippet+=self.get_variable_name(param,'name')
                    if param.extends:
                        code_snippet+=' extends '
                        for extend in param.extends:
                            code_snippet+=' ' + self.Type2Code(node=extend)
                    code_snippet+=','
                code_snippet=code_snippet.rstrip(',') +'>'
                
            code_snippet+=(self.Type2Code(node=node.return_type) if node.return_type else 'void')+' '+self.get_method_name(node,'name')+'('
            paramter=''
            
            for pram in node.parameters:
                paramter+=' '.join(reversed(list(pram.modifiers))) +' '
                paramter+=self.Type2Code(node=pram.type)
                paramter+=('...' if pram.varargs else '')
                paramter+=' '+ pram.name+','

            code_snippet+= paramter.rstrip(',')+')'+('throws ' +','.join(node.throws)+' ' if node.throws else '')
            if node.body != None:
                code_snippet += '{\n'
                
                for method_item in node.body:
                    code_snippet+=self.FUNC_DICT[type(method_item).__name__](node=method_item)
                code_snippet+= '}\n'
            else:
                code_snippet+= ';\n'
        else:
            code_snippet=''
        return code_snippet
  

    def InterfaceDeclaration2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=(node.documentation+'\n' if node.documentation else '')
        if node.annotations:
            for annotation in node.annotations:
                code_snippet+='@'+annotation.name+'\n'
        for modifier in node.modifiers:
            code_snippet+=modifier+' ' 

        code_snippet+=' interface '+node.name
        if node.type_parameters:
            code_snippet+='<'
            code_snippet+=','.join(self.get_variable_name(pram,'name') for pram in node.type_parameters)
            code_snippet+='>'
            
        if node.extends:
            code_snippet+=' extends '+','.join(self.Type2Code(node=extend) for extend in node.extends)
        
        code_snippet+='{\n'
        for b in node.body:
            if type(b).__name__=='MethodDeclaration' and b.body==None:
                code_snippet+=self.FUNC_DICT[type(b).__name__](node=b).rstrip('{\n}\n')+';\n'
            else:
                code_snippet+=self.FUNC_DICT[type(b).__name__](node=b)
        code_snippet+='}\n'
        return code_snippet

    def ConstantDeclaration2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=(node.documentation+'\n' if node.documentation else '')
        for annot in node.annotations:
            code_snippet+='@'+annot.name+'('+self.FUNC_DICT[type(annot.element).__name__](node=annot.element)+')\n'
        modifier_arr=[]
        for modifier in node.modifiers:
            modifier_arr.append(modifier)
        code_snippet+=' '.join(reversed(modifier_arr)) +' '
        code_snippet+=self.Type2Code(node=node.type)
        code_snippet += ' ' +','.join(self.FUNC_DICT[type(declarator).__name__](node=declarator)  for declarator in node.declarators)

        return code_snippet+';\n'

    def LocalVariableDeclaration2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=''
        if node.annotations != None:
            for annot in node.annotations:
                code_snippet+='@'+annot.name+'('+self.FUNC_DICT[type(annot.element).__name__](node=annot.element)+')\n'            
        code_snippet+=' '.join(reversed(list(node.modifiers))) +' '
        code_snippet+=self.Type2Code(node=node.type)+' '

        for declarator in node.declarators:
            code_snippet += self.FUNC_DICT[type(declarator).__name__](node=declarator) +','

        code_snippet=code_snippet.rstrip(',')
        return code_snippet+'; \n'

    def VariableDeclarator2Code(self,**kwargs):
        node=kwargs['node']
        
        name_modifier=''
        code_snippet=''

        '''
        if node.qualifier in self.object_variable:
            code_snippet+=name_modifier+self.variable_dict[node.name]
        else:    
            code_snippet+=node.name # normal option
        '''
        code_snippet+=name_modifier+(self.get_variable_name(node,'name') if node.name else '')
        #argmentation option
        code_snippet+=('[]'*len(node.dimensions) if node.dimensions else '')+' ' 
     
        if node.initializer:
            code_snippet+= ' = '+self.FUNC_DICT[type(node.initializer).__name__](node=node.initializer)

        return code_snippet

    def VariableDeclaration2Code(self,**kwargs):
        node=kwargs['node']

        code_snippet=''
        code_snippet+=self.Type2Code(node=node.type)+' '
        operandr=''
        operandl=''

        for declarator in node.declarators:
            operandl += self.FUNC_DICT[type(declarator).__name__](node=declarator)+','
        
        code_snippet +=operandr.rstrip(',')+operandl.rstrip(',')
        return code_snippet
        
    def Type2Code(self,**kwargs):
        node=kwargs['node']

        if type(node).__name__ == 'BasicType':
            return node.name +('[]'*len(node.dimensions) if node.dimensions else '')
        elif type(node).__name__ == 'ReferenceType':
            code_snippet=self.get_class_name(node,'name')
            if node.arguments != None:
                code_snippet+='<'
                for arg in node.arguments:
                    if arg.pattern_type:
                        if arg.pattern_type in ['extends','super']:
                            code_snippet+='? ' +arg.pattern_type+' '
                        else:
                            code_snippet+=arg.pattern_type+','
                    if arg.type:
                        code_snippet+=self.Type2Code(node=arg.type)+','
                code_snippet=code_snippet.rstrip(',')+'>'
            
            if node.sub_type:
                code_snippet+='.'+self.Type2Code(node=node.sub_type)
            code_snippet+=('[]'*len(node.dimensions) if node.dimensions else '')

            return code_snippet
        else:
            return self.FUNC_DICT[type(node).__name__](node=node)
    def ClassCreator2Code(self,**kwargs):
        node=kwargs['node']
        
        argments='('
        for arg in node.arguments:
            argments+=self.FUNC_DICT[type(arg).__name__](node=arg)+','

        code_snippet=(''.join(node.prefix_operators) if node.prefix_operators else '')
        argments=argments.rstrip(',')+')'
        code_snippet+='new '+self.Type2Code(node=node.type)+argments

        if node.body != None:
            code_snippet+='{'
            for b in node.body:
                code_snippet+=self.FUNC_DICT[type(b).__name__](node=b)
            code_snippet+='}'
        if node.selectors:
            for index in node.selectors:
                if type(index).__name__ != 'ArraySelector':
                    code_snippet+='.'    
                code_snippet+=self.FUNC_DICT[type(index).__name__](node=index,inline=False)    
        return code_snippet

    def MemberReference2Code(self,**kwargs):
        node=kwargs['node']
        
        code_snippet=(''.join(node.prefix_operators) if node.prefix_operators else '')+(self.get_class_name(node,'qualifier')+'.' if node.qualifier else '')
        if getattr(node,'member') in self.class_dict:
            code_snippet+=self.get_class_name(node,'member') 
        else:
            code_snippet+=self.get_variable_name(node,'member') 
        selector=''
        if node.selectors:
            for index in node.selectors:
                if type(index).__name__ != 'ArraySelector':
                    selector+='.'    
                selector+=self.FUNC_DICT[type(index).__name__](node=index)
        code_snippet+=selector
        return code_snippet+(''.join(node.postfix_operators) if node.postfix_operators else '')
    def MethodReference2Code(self,**kwargs):
        r"""
        Methodreference type to code snippet

        Arguments:
            node (:obj:`javalang.node.tree.MethodReference`):
                target node
            
        Returns:
            :Boolean : if match condtions return True else False 
        """
    
        node=kwargs['node']
        code_snippet=''
        
        code_snippet+=self.FUNC_DICT[type(node.expression).__name__](node=node.expression)
        code_snippet+='::'
        code_snippet+=self.FUNC_DICT[type(node.method).__name__](node=node.method)
        

        return code_snippet
    def For2While(self,node):
        new_node  = javalang.tree.WhileStatement(body=None, condition=None, label=None)
        code_snippet=''
        if node.control.init:
            if type(node.control.init) == list:
                code_snippet+= ','.join([self.FUNC_DICT[type(i).__name__](node=i).rstrip(';\n') for i in node.control.init])
            else:
                code_snippet+=self.FUNC_DICT[type(node.control.init).__name__](node=node.control.init).rstrip(';\n')
        new_node.body=node.body
        new_node.condition=node.control.condition
        code_snippet+=self.WhileStatement2Code(node=new_node)

        if node.control.update:
            if code_snippet[-1]=='}':
                code_snippet=code_snippet[:-1]
                code_snippet+= ','.join([self.FUNC_DICT[type(update).__name__](node=update).rstrip(';\n') for update in node.control.update])
                code_snippet+=':'
            else:
                code_snippet+= ','.join([self.FUNC_DICT[type(update).__name__](node=update).rstrip(';\n') for update in node.control.update])
        return code_snippet
    
    def ForStatement2Code(self,**kwargs):
        node=kwargs['node']
        if self.for_while > random.random():
            return self.For2While(node)
        else:

            code_snippet=''
            if node.label:
                code_snippet+=node.label+':'
            code_snippet+='for ( '
            code_snippet+=self.FUNC_DICT[type(node.control).__name__](node=node.control)
            code_snippet+= ')\n'
            for_body=self.FUNC_DICT[type(node.body).__name__](node=node.body)
            if type(node.body).__name__ == "StatementExpression":
                for_body='{\n'+for_body+'\n}'
            if type(node.body).__name__ == "BlockStatement" and len(node.body.statements)== 1 and random.random() <self.bracket:
                for_body=for_body[1:-1]

        return code_snippet

    def ForControl2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=''
        if node.init:
            if type(node.init) == list:
                code_snippet+= ','.join([self.FUNC_DICT[type(i).__name__](node=i).rstrip(';\n') for i in node.init])
            else:
                code_snippet+=self.FUNC_DICT[type(node.init).__name__](node=node.init).rstrip(';\n')
        
        code_snippet+=';'
        if node.condition:
            code_snippet+=self.FUNC_DICT[type(node.condition).__name__](node=node.condition)
        code_snippet +=';'
        if node.update:
            
            code_snippet+= ','.join([self.FUNC_DICT[type(update).__name__](node=update).rstrip(';\n') for update in node.update])
        return code_snippet

    def EnhancedForControl2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=''
        code_snippet+=self.FUNC_DICT[type(node.var).__name__](node=node.var)+':'
        code_snippet+=self.FUNC_DICT[type(node.iterable).__name__](node=node.iterable)
        return code_snippet

    def IfStatement2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='if ( '
        code_snippet+= self.FUNC_DICT[type(node.condition).__name__](node=node.condition)+')'
        code_snippet+='\n'+self.FUNC_DICT[type(node.then_statement).__name__](node=node.then_statement)+'\n'
        if node.else_statement:
            code_snippet+='else '+self.FUNC_DICT[type(node.else_statement).__name__](node=node.else_statement)
        return code_snippet
        
    def BlockStatement2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet="{"
        for block_item in node.statements:
            code_snippet+=self.FUNC_DICT[type(block_item).__name__](node=block_item)
        code_snippet+="}"
        return code_snippet

    def List2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet="{"
        for list_item in node:
            code_snippet+=self.FUNC_DICT[type(list_item).__name__](node=list_item)
        code_snippet+="}"
        return code_snippet
        
    def BinaryOperation2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=''
        operandl=self.FUNC_DICT[type(node.operandl).__name__](node=node.operandl,replace=type(node.operandl).__name__ == 'MemberReference')
        if type(node.operandl).__name__== 'Assignment':
            code_snippet+='('+operandl+')'
        else:
            code_snippet+= operandl
        code_snippet+=' ' +node.operator +' '
        
        operandr=self.FUNC_DICT[type(node.operandr).__name__](node=node.operandr,replace=type(node.operandr).__name__ == 'MemberReference')
        if type(node.operandr).__name__== 'Assignment':
            code_snippet+='('+operandr+')'
        else:
            code_snippet+= operandr
        return '('+code_snippet+')'

    def Literal2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=node.value
        prefix=''.join(node.prefix_operators if node.prefix_operators else '')
        postfix=''.join(node.postfix_operators if node.postfix_operators else '')
        for s in node.selectors:
            if not type(s).__name__ =='ArraySelector':
                code_snippet+= '.'    
            code_snippet+=self.FUNC_DICT[type(s).__name__](node=s)
        return prefix+code_snippet+postfix

    def ArrayCreator2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='new '
        code_snippet+= self.Type2Code(node=node.type)
        for dim in node.dimensions:
            code_snippet+= '['+self.FUNC_DICT[type(dim).__name__](node=dim)+']'
        if node.initializer:
            code_snippet+=self.FUNC_DICT[type(node.initializer).__name__](node=node.initializer)
        return code_snippet
 
    def ArrayInitializer2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='{'
        for init in node.initializers:
            code_snippet+=self.FUNC_DICT[type(init).__name__](node=init)+','
        return code_snippet.rstrip(',')+'}'

    def Cast2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='('+self.Type2Code(node=node.type)+')'

        code_snippet+=self.FUNC_DICT[type(node.expression).__name__](node=node.expression)
        return code_snippet

    def StatementExpression2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=''
        if node.label:
            code_snippet+=node.label+':'
        code_snippet+=self.FUNC_DICT[type(node.expression).__name__](node=node.expression,end=';\n')

        return code_snippet+';\n'

    def Assignment2Code(self,**kwargs):
        node=kwargs['node']

        code_snippet=self.FUNC_DICT[type(node.expressionl).__name__](node=node.expressionl)

        code_snippet+=' ' +node.type+' '
        code_snippet+= self.FUNC_DICT[type(node.value).__name__](node=node.value)
        return code_snippet

    def MethodInvocation2Code(self,**kwargs):
        
        node=kwargs['node']
        args=[self.FUNC_DICT[type(arg).__name__](node=arg) for arg in node.arguments]
        args=','.join(args)
        code_snippet=''
        if getattr(node,'qualifier') in self.class_dict:
            qualifier= self.get_class_name(node,'qualifier')
        else:
            qualifier= self.get_variable_name(node,'qualifier')

        code_snippet+=(qualifier+'.' if node.qualifier else '')
        code_snippet+= self.get_method_name(node,'member')+'('
        code_snippet+=args+')'

        if node.selectors:
            for s in node.selectors:
                if not type(s).__name__ in ['ArraySelector']:
                    code_snippet+= '.'
                code_snippet+=self.FUNC_DICT[type(s).__name__](node=s)
        prefix_operators=node.prefix_operators if node.prefix_operators else []
        postfix_operators=node.postfix_operators if node.postfix_operators else []
        
        return ''.join(prefix_operators)+code_snippet+''.join(postfix_operators)

    def ArraySelector2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='['
        code_snippet+=self.FUNC_DICT[type(node.index).__name__](node=node.index)
        code_snippet+=']'
        
        return code_snippet

    def ReturnStatement2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='return '
        expression=self.FUNC_DICT[type(node.expression).__name__](node=node.expression)
        
        return code_snippet+expression+';'

    def TryStatement2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='try'
        if node.resources:
            code_snippet+='('
            resource_code=[]
            for resource in node.resources:
                resource_code.append(self.FUNC_DICT[type(resource).__name__](node=resource))
            code_snippet+=';'.join(resource_code)+')'
            
        code_snippet+='{'
        for b in node.block:
            code_snippet+=self.FUNC_DICT[type(b).__name__](node=b)
        code_snippet+='\n}'
        if node.catches:
            for catch in node.catches:
                code_snippet+= self.FUNC_DICT[type(catch).__name__](node=catch)
        
        if node.finally_block != None:
            code_snippet+='finally{'    
            for b in node.finally_block:
                code_snippet+=self.FUNC_DICT[type(b).__name__](node=b)
            code_snippet+='\n}'
        return code_snippet

    def CatchClause2Code(self,**kwargs):
        node=kwargs['node']

        code_snippet= 'catch ('+self.FUNC_DICT[type(node.parameter).__name__](node=node.parameter) +')'
        code_snippet+='{\n'
        for catch_block in node.block:
            code_snippet+=self.FUNC_DICT[type(catch_block).__name__](node=catch_block)
        code_snippet+='}\n'

        return code_snippet

    def CatchClauseParameter2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=''
        for p in node.types:
            code_snippet+=p+' '
        name=self.get_variable_name(node,'name') if getattr(node,'name') in self.variable_dict else self.get_method_name(node,'name')
        code_snippet+=name
        return code_snippet

    def ThrowStatement2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='throw ('
        code_snippet+=self.FUNC_DICT[type(node.expression).__name__](node=node.expression)
        code_snippet+=')'
        return code_snippet +';'

    def TryResource2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=''
        if 'final' in node.modifiers:
            code_snippet+='final '
        code_snippet+=self.Type2Code(node=node.type)+' '
        code_snippet+= self.get_variable_name(node,'name')

        code_snippet+='='+self.FUNC_DICT[type(node.value).__name__](node=node.value)
        return code_snippet

    def This2Code(self,**kwargs):
        node=kwargs['node']
        prefix=''.join(node.prefix_operators if node.prefix_operators else '')
        postfix=''.join(node.postfix_operators if node.postfix_operators else '')
        code_snippet=''
        if node.qualifier != None:
            code_snippet+=node.qualifier+'.'
        code_snippet+='this'
        for s in node.selectors:
            if not type(s).__name__ =='ArraySelector':
                code_snippet+= '.'    
            code_snippet+=self.FUNC_DICT[type(s).__name__](node=s)
        return prefix+code_snippet+postfix
    def Ternary2If(self,node):

        new_node=javalang.tree.IfStatement()
        new_node.condition =node.conditionm
        new_node.else_statement = node.if_true
        new_node.then_statement =node.if_false
        return self.Ifstatement2Code(node=new_node)


    def TernaryExpression2Code(self,**kwargs):
        node=kwargs['node']
        if self.ternary > random.random:
            return self.Ternary2If(node)
        else:

            code_snippet=''    
            code_snippet+='('
            if node.condition:
                if type(node.condition).__name__ ==' MethodInvocation':
                    code_snippet=self.FUNC_DICT[type(node.condition).__name__](node=node.condition,method_target=code_snippet)
                else:
                    code_snippet+=self.FUNC_DICT[type(node.condition).__name__](node=node.condition)
            code_snippet+='?'
            if node.if_true:
                if type(node.if_true).__name__ == 'Assignment':
                    code_snippet+='('+self.FUNC_DICT[type(node.if_true).__name__](node=node.if_true,inline=False) +')'
                else:
                    code_snippet+=self.FUNC_DICT[type(node.if_true).__name__](node=node.if_true,inline=False)
            code_snippet+=':'
            if node.if_false:
                if type(node.if_false).__name__ == 'Assignment':
                    code_snippet+='('+self.FUNC_DICT[type(node.if_false).__name__](node=node.if_false,inline=False) +')'
                else:
                    code_snippet+=self.FUNC_DICT[type(node.if_false).__name__](node=node.if_false,inline=False)
            code_snippet+=')'
        return code_snippet

    def BreakStatement2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='break'
        if node.goto != None:
            code_snippet+=' '+node.goto
        return code_snippet+';'

    def ExplicitConstructorInvocation2Code(self,**kwargs):
        node=kwargs['node']
        
        code_snippet='this('
        for argument in node.arguments:
            code_snippet+=self.FUNC_DICT[type(argument).__name__](node=argument)+','
        code_snippet=code_snippet.rstrip(',')
        code_snippet+=')'
        return code_snippet

    def DoStatement2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=''
        if node.label:
            code_snippet+=node.label+':'
        code_snippet+='do '+self.FUNC_DICT[type(node.body).__name__](node=node.body)
        code_snippet+='while('+self.FUNC_DICT[type(node.condition).__name__](node=node.condition,inline=False)+')'
        return code_snippet+';'

    def ContinueStatement2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='continue' 
        if node.goto:
            code_snippet+=' '+node.goto
        return code_snippet+';'

    def Statement2Code(self,**kwargs):

        return ';'

    def SuperConstructorInvocation2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='super'
        code_snippet+='('
        if node.arguments:
            code_snippet+=','.join([self.FUNC_DICT[type(arg).__name__](node=arg) for arg in node.arguments])

        code_snippet+=')'
            
        return code_snippet

    def SwitchStatement2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='switch('+self.FUNC_DICT[type(node.expression).__name__](node=node.expression,inline=False)+'){\n'

        for case in node.cases:
            if case.case:
                code_snippet+='case '
                for cons in case.case:
                    code_snippet+=self.FUNC_DICT[type(cons).__name__](node=cons)+','
                code_snippet=code_snippet.rstrip(',')
            else:
                code_snippet+='default '
            code_snippet+=':'
            for statement in case.statements:
                code_snippet+=self.FUNC_DICT[type(statement).__name__](node=statement)
            code_snippet+='\n'

        return code_snippet+'}\n'

    def AssertStatement2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='assert '
        code_snippet+=self.FUNC_DICT[type(node.condition).__name__](node=node.condition)
        return code_snippet+';\n'

    def SuperMethodInvocation2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='super.'
        code_snippet+=self.get_method_name(node,'member')
        code_snippet+='('
        arg_lst=[]
        for arg in node.arguments:
            arg_lst.append(self.Type2Code(node=arg))
        return code_snippet+','.join(arg_lst)+')'

    def BasicType2Code(self,**kwargs):
        node=kwargs['node']
    
        return node.name +('[]'*len(node.dimensions) if node.dimensions else '')

    def LambdaExpression2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=''
        code_snippet+=','.join(self.FUNC_DICT[type(pram).__name__](node=pram) for pram in node.parameters)
        code_snippet='('+code_snippet+')'

        code_snippet+='->'
        if type(node.body) ==list:
                code_snippet+='{'
                code_snippet+=''.join(self.FUNC_DICT[type(b).__name__](node=b,inline=False) for b in node.body)
                code_snippet+='}'
        else:
            code_snippet+=self.FUNC_DICT[type(node.body).__name__](node=node.body)
        return code_snippet

    def FormalParameter2Code(self,**kwargs):
        node=kwargs['node']
        name= self.get_class_name(node,'name') if getattr(node,'name') in self.class_dict else self.get_variable_name(node,'name')
        code_snippet=self.Type2Code(node=node.type)+' '+name
        return code_snippet

    def InferredFormalParameter2Code(self,**kwargs):
        node=kwargs['node']
        return self.get_variable_name(node,'name') if getattr(node,'name') in self.variable_dict else self.get_method_name(node,'name')

    def EnumDeclaration2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='enum '+node.name+'{'
        code_snippet+=self.FUNC_DICT[type(node.body).__name__](node=node.body)
        code_snippet+='}'
        return code_snippet

    def EnumBody2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=''
        constants_items=[self.FUNC_DICT[type(constant).__name__](node=constant) for constant in node.constants]
        code_snippet+=','.join(constants_items)
        return code_snippet

    def EnumConstantDeclaration2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=node.name
        return code_snippet

    def InnerClassCreator2Code(self,**kwargs):
        node=kwargs['node']
        
        argments='('
        for arg in node.arguments:
            argments+=self.FUNC_DICT[type(arg).__name__](node=arg)+','
        
        argments=argments.rstrip(',')+')'
        
        code_snippet=(self.get_variable_name(node,'qualifier')+'.' if node.qualifier else '')+'new '+self.Type2Code(node=node.type)+argments
        if node.body:
            code_snippet+='{'
            for b in node.body:
                code_snippet+=self.FUNC_DICT[type(b).__name__](node=b)
            code_snippet+='}'
        if node.selectors:
            for index in node.selectors:
                if type(index).__name__ != 'ArraySelector':
                    code_snippet+='.'    
                code_snippet+=self.FUNC_DICT[type(index).__name__](node=index,inline=False)    
        return code_snippet
    def SynchronizedStatement2Code(self,**kwargs):
        node=kwargs['node']

        code_snippet='synchronized('+self.FUNC_DICT[type(node.lock).__name__](node=node.lock)+'){'

        for b in node.block:
            code_snippet+=self.FUNC_DICT[type(b).__name__](node=b)
        return code_snippet+'}'
    def ClassReference2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet=self.FUNC_DICT[type(node.type).__name__](node=node.type)
        code_snippet+='.class'
        if node.selectors:
            for s in node.selectors:
                code_snippet+='.'
                code_snippet+=self.FUNC_DICT[type(s).__name__](node=s,inline=False)

        return code_snippet
    def ElementArrayValue2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='{'
        for item in node.values:
            code_snippet+=self.FUNC_DICT[type(item).__name__](node=item)+','
        code_snippet=code_snippet.rstrip(',')
        return code_snippet+'}'
    def AnnotationDeclaration2Code(self,**kwargs):
        node=kwargs['node']
        code_snippet='@'+node.name+'('
        code_snippet+=','.join([self.FUNC_DICT[type(annot.element).__name__](node=annot.element) for annot in node.annotations])
        code_snippet+=')\n'

        return code_snippet
