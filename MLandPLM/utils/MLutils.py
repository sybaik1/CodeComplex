import javalang
import string
from javalang.ast import Node
import ast
from vulture import core

def python_code_depth(code):
    code = code.splitlines()
    max_depth = 0
    indent = 0
    max_iter = 0
    depth = []
    for line in code:
        if line.strip() == "":
            continue
        for char in line:
            if char not in string.whitespace:
                break
            indent += 1
        else:
            indent = 0
        if indent!=0: 
            break
    if indent == 0:
        indent = 4
    for line in code:
        line_indent = 0
        if line.strip() == "":
            continue
        for char in line:
            if char not in string.whitespace:
                break
            line_indent += 1
        while len(depth)>0 and line_indent//indent <= depth[-1]:
            del depth[-1]
        if 'def' in line or 'for' in line or 'while' in line:
            depth.append(line_indent//indent)
        max_depth = max(max_depth, line_indent//indent)
        max_iter = max(max_iter, len(depth))
    return max_depth, max_iter
def get_children(root):
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    yield sub_item
            elif item:
                yield item

    return list(expand(children))


class method_feature_extractor():
    def __init__(self):

        self.num_if = 0
        self.num_switch = 0
        self.num_loop = 0
        self.num_break = 0
        self.num_Priority = 0
        self.num_sort = 0
        self.num_hash_set = 0
        self.num_hash_map = 0
        self.num_recursive = 0
        self.num_nasted_loop = 0
        self.num_vari = 0
        self.num_method = 0
        self.num_state = 0

    def count_feature(self, node, method_feature):

        children = get_children(node)

        if isinstance(node, javalang.tree.Statement):
            self.num_state += 1

        if type(node) == javalang.tree.MethodDeclaration:
            self.num_method += 1
            temp_name = node.name
            if node.name in str(node).replace('name=' + temp_name, ''):
                self.num_recursive += 1

        if type(node) in [javalang.tree.ForStatement, javalang.tree.WhileStatement]:
            self.num_loop += 1
            tmp_depth = nested_depth_v2(node, method_feature, 1)
            if self.num_nasted_loop < tmp_depth:
                self.num_nasted_loop = tmp_depth

        if type(node) == javalang.tree.LocalVariableDeclaration:
            try:
                if node.declarators[0].initializer.type.name == 'HashMap':
                    self.num_hash_map += 1
                elif node.declarators[0].initializer.type.name == 'HashSet':
                    self.num_hash_set += 1
                elif node.declarators[0].initializer.type.name == 'PriorityQueue':
                    self.num_Priority += 1
            except:
                pass

        if type(node) == javalang.tree.MethodInvocation:
            if node.member in method_feature.keys():
                temp_list = method_feature[node.member]
                self.num_if += temp_list[0]
                self.num_switch += temp_list[1]
                self.num_loop += temp_list[2]
                self.num_break += temp_list[3]
                self.num_Priority += temp_list[4]
                self.num_sort += temp_list[5]
                self.num_hash_set += temp_list[6]
                self.num_hash_map += temp_list[7]
                self.num_nasted_loop = max([self.num_nasted_loop, temp_list[9]])

                self.num_vari += temp_list[10]
                self.num_method += temp_list[11]
                self.num_state += temp_list[12]
            if node.member == 'sort':
                self.num_sort += 1

        if type(node) == javalang.tree.IfStatement:
            self.num_if += 1
        if type(node) == javalang.tree.BreakStatement:
            self.num_break += 1
        if type(node) in [javalang.tree.VariableDeclaration, javalang.tree.LocalVariableDeclaration]:
            self.num_vari += 1
        if type(node) == javalang.tree.SwitchStatement:
            self.num_switch += 1

        for child in children:
            self.count_feature(child, method_feature)

    def get_feature(self):
        tmp = [self.num_if, self.num_switch, self.num_loop, self.num_break, self.num_Priority, self.num_sort,
               self.num_hash_set, self.num_hash_map, self.num_recursive, self.num_nasted_loop, self.num_vari,
               self.num_method, self.num_state]
        self.__init__()
        return tmp


def nested_depth_v2(node, method_feature, depth):
    children = get_children(node)
    tmp_depth1, tmp_depth2, tmp_depth3 = 0, 0, 0

    max_depth = depth
    for child in children:
        if type(child) == javalang.tree.MethodInvocation:
            if child.member in method_feature.keys():
                tmp_depth1 = depth + method_feature[child.member][9]
        elif type(child) in [javalang.tree.ForStatement, javalang.tree.WhileStatement]:
            tmp_depth2 = nested_depth_v2(child, method_feature, depth + 1)
        else:
            tmp_depth3 = nested_depth_v2(child, method_feature, depth)

        max_depth = max([tmp_depth1, tmp_depth2, tmp_depth3])

    return max_depth


def nested_depth_v1(node, depth):
    children = get_children(node)
    tmp_depth1, tmp_depth2 = 0, 0

    max_depth = depth
    for child in children:
        if type(child) in [javalang.tree.ForStatement, javalang.tree.WhileStatement]:
            tmp_depth1 = nested_depth_v1(child, depth + 1)
        else:
            tmp_depth2 = nested_depth_v1(child, depth)

        max_depth = max([tmp_depth1, tmp_depth2])

    return max_depth


def get_method_info(node, called_method):
    children = get_children(node)

    if type(node) == javalang.tree.MethodInvocation:
        called_method.add(node.member)

    for child in children:
        get_method_info(child, called_method)


def get_feature_v2(tree):
    num_recursive = 0
    method_feature = {}
    MFE = method_feature_extractor()

    called_method = dict()
    method_dict = dict()
    for _, node in tree:
        if type(node) == javalang.tree.MethodDeclaration:
            called_method[node.name] = set()
            get_method_info(node, called_method[node.name])
            method_dict[node.name] = node
            if node.name in called_method[node.name]:
                num_recursive += 1
            # print(node.name, called_method[node.name])

    called_method_stack = list()

    called_method_stack.append(set(['main']))  # methods in depth 1 by BFS
    called_method_stack.append(called_method['main'])  # methods in depth 2 by BFS

    methods_examined = set()  # Methods seen so far
    methods_examined.add('main')
    methods_examined |= called_method['main']

    while True:
        method_in_new_depth = set()

        for method in called_method_stack[-1]:
            if method in called_method.keys():
                method_in_new_depth = set()
                method_in_new_depth |= called_method[method]

        if len(method_in_new_depth - methods_examined) == 0:
            break

        methods_examined |= method_in_new_depth
        called_method_stack.append(method_in_new_depth - methods_examined)

    for methods in reversed(called_method_stack):
        for name in methods:
            if name in method_dict.keys():
                MFE.count_feature(method_dict[name], method_feature)
                method_feature[name] = MFE.get_feature()
                # print(method_feature)
    # print(method_feature)
    method_feature['main'][8] = num_recursive
    return method_feature['main']


def feature_Extractor(source, lang='java', version=1):
    # f = open(path + source_file)
    # source = f.read()
    # f.close()

    if lang=='java':
        tree = javalang.parse.parse(source)
        if version == 1:
            return get_feature_v1(tree)
        return get_feature_v2(tree)

    else:
        v = core.Vulture()
        v.scan(source)
        a,b= python_code_depth(source)
        return [source.count('if'),source.count('while')+source.count('for'),source.count('break'),source.count('sort'),source.count('list')+source.count('split'),len(source.splitlines()), len(v.used_names), len(v.defined_attrs), a,b, len(v.defined_vars), len(v.defined_methods), len(v.defined_funcs)]

        #tree = ast.parse(source)
        #feature_extractor = PythonFeature()
        #return feature_extractor.get_feature_python(tree) + [len(v.used_names), len(v.defined_attrs), len(v.defined_classes), len(v.defined_props), len(v.defined_vars), len(v.defined_funcs), len(v.defined_methods)]


def get_feature_v1(tree):
    num_if = 0
    num_switch = 0
    num_loof = 0
    num_break = 0
    num_Priority = 0
    num_sort = 0
    num_hash_map = 0
    num_hash_set = 0
    num_recursive = 0
    num_nasted_loop = 0
    num_vari = 0
    num_method = 0
    num_state = 0

    for path, node in tree:

        called_method = dict()
        if type(node) == javalang.tree.MethodDeclaration:
            num_method += 1
            called_method[node.name] = set()
            get_method_info(node, called_method[node.name])
            if node.name in called_method[node.name]:
                num_recursive += 1

        if isinstance(node, javalang.tree.Statement):
            num_state += 1

        if type(node) in [javalang.tree.ForStatement, javalang.tree.WhileStatement]:
            num_loof += 1
            num_nasted_loop = max([num_nasted_loop, nested_depth_v1(node, 0)])

        if type(node) == javalang.tree.LocalVariableDeclaration:
            try:
                if node.declarators[0].initializer.type.name == 'HashMap':
                    num_hash_map += 1
                elif node.declarators[0].initializer.type.name == 'HashSet':
                    num_hash_set += 1
                elif node.declarators[0].initializer.type.name == 'PriorityQueue':
                    num_Priority += 1
            except:
                pass

        if type(node) == javalang.tree.MethodInvocation:
            if node.member == 'sort':
                num_sort += 1

        if type(node) == javalang.tree.IfStatement:
            num_if += 1

        if type(node) == javalang.tree.BreakStatement:
            num_break += 1

        if type(node) in [javalang.tree.VariableDeclaration, javalang.tree.LocalVariableDeclaration]:
            num_vari += 1
        if type(node) == javalang.tree.SwitchStatement:
            num_switch += 1

    return [num_if, num_switch, num_loof, num_break, num_Priority, num_sort, num_hash_map, num_hash_set, num_recursive,
            num_nasted_loop, num_vari, num_method, num_state]


class PythonFeature(ast.NodeVisitor):
    num_if = 0
    num_loop = 0
    num_break = 0
    num_recursive = 0
    num_vari = 0
    num_method = 0
    num_state = 0
    names = set()

    def visit(self, node):
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        if isinstance(node, ast.stmt):
            self.num_state += 1
        return visitor(node)


    def visit_If(self, node):
        self.num_if += 1
        return node

    def visit_While(self, node):
        self.num_loop += 1
        return node

    def visit_For(self, node):
        self.num_loop += 1
        return node

    def visit_Break(self, node):
        self.num_break += 1
        return node

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)
        return node
    
    def visit_FunctionDef(self, node):
        self.num_method += 1
        return node

    
    def get_feature_python(self,node):
        self.visit(node)
        self.num_vari = len(self.names)
        return [self.num_if, self.num_loop, self.num_break,
                self.num_vari, self.num_method, self.num_state]
