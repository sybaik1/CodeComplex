"""
    pythoncodeaug
    ~~~
    python code augmentation
    provides the following features
    -- for to while ('for2while')
    -- ternary to if ('ternary2if')
    -- function outline for comprehension expression ('comp2genfunc')
    -- comprehension expression to statment ('comp2stmt')
    -- function inlining ('inline')
    -- dead code elimination
    TODO
    -- folding constant values
    -- function outline
"""
import ast
import re
from itertools import count
import os
import random
import argparse
from copy import deepcopy
from typing import Any, TypeVar
import builtins

T = TypeVar('T')


class PythonCodeAug():
    """augmentor of python codes"""

    def __init__(self, probs: dict = dict(),
                 seed: int = None, debug: bool = False):
        # set probs
        self.probs = probs
        # set seed for random
        self.seed = seed
        self.debug = debug
        self.ast_augmentor = ASTAug(probs, seed, debug)

    def python_ast2code(self, module: ast.Module):
        """python AST to python code"""
        return ast.unparse(module)

    def python_code_aug(self, code: str):
        """augments the given code and returns it"""
        python_ast = ast.parse(code)
        new_python_ast = ast.fix_missing_locations(
                self.ast_augmentor.visit(python_ast))
        deadcoderemove = DeadCodeRemoval(new_python_ast)
        dead_removed_ast = deadcoderemove.remove()
        final_python_ast = ast.fix_missing_locations(dead_removed_ast)
        return self.python_ast2code(final_python_ast)


class ASTAugBase(ast.NodeTransformer):
    def __init__(self, default_info: dict = dict()) -> None:
        super().__init__()
        self._defined_vars = set()
        self._default_info = default_info

    def update(self, base, update_info):
        for key, val in update_info.items():
            if isinstance(val, bool):
                if self._default_info[key]:
                    base[key] &= val
                else:
                    base[key] |= val
            elif isinstance(val, list):
                base[key].extend(val)
            elif isinstance(val, set):
                base[key].update(val)
            elif isinstance(val, dict):
                base[key].update(val)
            elif isinstance(val, int):
                base[key] = val
            else:
                raise TypeError("Unknow info type")
        return base

    def _name_gen(self, base: str) -> str:
        """gives a variable name for replacing expressions"""
        for i in count():
            if f"_{base}_{i}" not in self._defined_vars:
                self._defined_vars.add(f"__{base}_{i}")
                return f"_{base}_{i}"

    def visit(self, node: ast.AST, *val, **kwargs) -> Any:
        """Visit a node"""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node, *val, **kwargs)

    def generic_visit(self, node: ast.AST, *val, **kwargs) \
            -> tuple[ast.AST, Any]:
        """
        Called if no explicit visitor function exists for a node
        provokes visit on each of its child nodes
        """
        info = deepcopy(self._default_info)
        for field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast.AST):
                        value, node_info = self.visit(value, *val, **kwargs)
                        self.update(info, node_info)
                        if value is None:
                            continue
                        elif not isinstance(value, ast.AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                new_node, node_info = self.visit(old_value, *val, **kwargs)
                self.update(info, node_info)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node, info

    def visit_Constant(self, node: ast.Constant, *val, **kwargs) -> Any:
        return super().visit_Constant(node)


class NameAug(ASTAugBase):
    def __init__(self, name_map: dict = dict()):
        super().__init__()
        self._name_map = name_map

    def visit_Name(self, node: ast.Name) -> ast.Name:
        node, info = self.generic_visit(node)
        node.id = self._name_map.get(node.id, node.id)
        return node, info


class FuncReturnAug(ASTAugBase):
    def __init__(self, retvalname, retflagname) -> None:
        super().__init__({"return": False})
        self.retvalname = retvalname
        self.retflagname = retflagname

    def visit_For(self, node: ast.For, depth: int = 0):
        node, info = self.generic_visit(node, depth + 1)
        if info["return"] and depth > 0:
            node = [node, ast.If(
                test=ast.Name(
                        id=self.retflagname, ctx=ast.Load()),
                body=ast.Break(),
                orelse=[])]
        return node, info

    def visit_While(self, node: ast.While, depth: int = 0) -> Any:
        node, info = self.generic_visit(node, depth + 1)
        if depth > 0:
            node = [node, ast.If(
                test=ast.Name(
                    id=self.retflagname, ctx=ast.Load()),
                body=ast.Break(),
                orelse=[])]
        return node, info

    def visit_Return(self, node: ast.Return, depth: int = 0) -> Any:
        node, info = self.generic_visit(node, depth)
        info["return"] = True
        # chage flag val
        retnode = [
            ast.Assign(
                targets=[ast.Name(
                    id=self.retflagname, ctx=ast.Load())],
                value=ast.Constant(
                    value=True))]
        # return Value
        if node.value is not None:
            retnode.append(
                ast.Assign(
                    targets=[ast.Name(
                        id=self.retvalname, ctx=ast.Load())],
                    value=node.value))
        else:
            retnode.append(
                ast.Assign(
                    targets=[ast.Name(
                        id=self.retvalname, ctx=ast.Load())],
                    value=ast.Constant(
                        value=None)))
        # Add break
        if depth > 0:
            retnode.append(ast.Break())
        return retnode, info


class DeadCodeRemoval(ASTAugBase):
    def __init__(self, ast: ast.AST):
        super().__init__({"delete_assign": False})
        self.ast = ast
        self._using_vars = set()
        self._using_funcs = set()
        self._defined_name_used = dict()
        self._remove_var = False

    def remove(self):
        # get defined vars and variable dependencies
        self.visit(self.ast)
        self._remove_var = True
        self.visit(self.ast)
        return self.ast

    def generic_visit(self, node: ast.AST, *val, **kwargs) \
            -> tuple[ast.AST, Any]:
        """
        Called if no explicit visitor function exists for a node
        provokes visit on each of its child nodes
        """
        info = deepcopy(self._default_info)
        for field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast.AST):
                        value, node_info = self.visit(value, *val, **kwargs)
                        self.update(info, node_info)
                        if value is None:
                            info["delete_assign"] = True
                            continue
                        elif not isinstance(value, ast.AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                new_node, node_info = self.visit(old_value, *val, **kwargs)
                self.update(info, node_info)
                if new_node is None:
                    return None, info
                else:
                    setattr(node, field, new_node)
        return node, info
    
    def visit_FunctionDef(self, node: ast.FunctionDef) \
            -> tuple[ast.Name, bool, set, set]:
        node, info = self.generic_visit(node)
        if self._remove_var:
            if not self._defined_name_used.get(node.name, True):
                return None, info
            else:
                return node, info
        if not node.name.startswith("_"):
            self._defined_name_used[node.name] = False
        return node, info

    def visit_Name(self, node: ast.Name, *val, **kwargs) -> Any:
        node, info = self.generic_visit(node)
        id = node.id
        ctx = node.ctx
        if self._remove_var:
            if self._defined_name_used[id]:
                return node, info
            else:
                return None, info
        if isinstance(ctx, ast.Store):
            if not self._defined_name_used.get(id, False):
                self._defined_name_used[id] = False
        elif isinstance(ctx, ast.Load):
            self._defined_name_used[id] = True
        return node, info


class ASTAug(ASTAugBase):
    """augmentor class of python AST"""

    def __init__(self, probs: dict = dict(),
                 seed: int = None, debug: bool = False):
        super().__init__({"modify_stmt": False,
                          "exists_yield": False,
                          "using_vars": set(),
                          "def_vars": set()})
        self.probs = probs
        # set random
        self.random = random.Random()
        self.random.seed(seed)
        self.debug = debug
        self._prefix_stmt = list()
        self._suffix_stmt = list()
        self._defined_funcs = dict(map(lambda x: (x, None), dir(builtins)))
        self._defined_funcs_yield = dict(map(
            lambda x: (x, None), dir(builtins)))

    def _release_names(self, node: ast.AST):
        """release generated names in nodes"""
        if self.debug:
            return
        for name, field in ast.iter_fields(node):
            if isinstance(field, ast.AST):
                self._release_names(field)
            elif isinstance(field, list):
                for item in field:
                    if isinstance(item, ast.AST):
                        self._release_names(item)
            elif name == 'id':
                id_num_match = re.fullmatch(r'__[a-zA-Z0-9]+_\d+', field)
                if id_num_match is not None:
                    try:
                        self._defined_vars.remove(int(field.split('_')[-1]))
                    except KeyError:
                        pass

    def _build_stmt_and_release_names(self, node: ast.AST) -> list:
        body = node
        if len(self._prefix_stmt) > 0 or len(self._suffix_stmt) > 0:
            body = self._prefix_stmt + [node] + self._suffix_stmt
        for node in self._prefix_stmt:
            self._release_names(node)
        for node in self._suffix_stmt:
            self._release_names(node)
        self._prefix_stmt = []
        self._suffix_stmt = []
        return body

    def _arg_parse(self, node: ast.arguments):
        """
        Returns argument name string list of function arguments
        """
        arg_list = []
        for pos_args in node.posonlyargs:
            arg_list.append(pos_args.arg)
        for args in node.args:
            arg_list.append(args.arg)
        for kw_args in node.kwonlyargs:
            arg_list.append(kw_args.arg)
        return arg_list

    def generic_visit(self, node, *args, **kwargs):
        node, info = super().generic_visit(node, *args, **kwargs)
        if (isinstance(node, ast.stmt)
            and (info["modify_stmt"]
                 or len(self._prefix_stmt) > 0
                 or len(self._suffix_stmt) > 0)):
            # need to modify or add statement blocks
            node = self._build_stmt_and_release_names(node)
            info["modify_stmt"] = False
        return node, info

    def visit_Module(self, node: ast.Module) -> ast.Module:
        node, _ = self.generic_visit(node)
        return node

    def visit_alias(self, node: ast.alias) -> tuple[ast.alias, bool, set, set]:
        node, info = self.generic_visit(node)
        if hasattr(node, 'asname'):
            self._defined_funcs[node.asname] = None
        else:
            self._defined_funcs[node.name] = None
        return node, info

    def visit_Name(self, node: ast.Name) -> tuple[ast.Name, bool, set, set]:
        node, info = self.generic_visit(node)
        id = node.id
        ctx = node.ctx
        if isinstance(ctx, ast.Store):
            self._defined_vars.add(id)
            info["def_vars"].add(id)
        elif isinstance(ctx, ast.Load):
            if id not in self._defined_funcs:
                info["using_vars"].add(id)
        return node, info

    def visit_FunctionDef(self, node: ast.FunctionDef) \
            -> tuple[ast.Name, bool, set, set]:
        node, info = self.generic_visit(node)
        self._defined_funcs[node.name] = node
        self._defined_funcs_yield[node.name] = info["exists_yield"]
        info["exists_yield"] = False
        return node, info

    def visit_Yield(self, node: ast.Yield) -> tuple[ast.Yield, bool, set, set]:
        node, info = self.generic_visit(node)
        info['exists_yield'] = True
        return node, info

    def visit_Call(self, node: ast.Call) \
            -> tuple[list[ast.stmt | ast.expr], bool, set, set]:
        node, info = self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            if node.func.id in self._defined_funcs:
                if (self._defined_funcs[node.func.id] is not None
                        and self._defined_funcs_yield[node.func.id] is False):
                    if self.random.random() <= self.probs.get('inline', 0):
                        # map from argument variable string
                        # to variable(Constant) object
                        arg_map = dict()
                        # map from argument variable string
                        # to augmented variable string
                        inline_func = self._defined_funcs[node.func.id]
                        # argument name string list
                        func_args = self._arg_parse(inline_func.args)
                        # return variable
                        ret_val = self._name_gen('ret')
                        ret_flag = self._name_gen('retflag')
                        # augmented name for each argument
                        aug_arg_map = dict()
                        for arg in func_args:
                            aug_arg_map[arg] = f"{inline_func.name}_{arg}"
                        # default value for pos and args
                        for default, arg in zip(
                                reversed(inline_func.args.defaults),
                                reversed(inline_func.args.posonlyargs
                                         + inline_func.args.args)):
                            arg_map[arg.arg] = default
                        # default value for kw_args
                        for default, arg in zip(
                                inline_func.args.kw_defaults,
                                inline_func.args.kwonlyargs):
                            arg_map[arg.arg] = default
                        # function call variable position match
                        for arg, call_arg in zip(func_args, node.args):
                            arg_map[arg] = call_arg
                        # function call variable keword match
                        for kw_call_arg in node.keywords:
                            arg_map[kw_call_arg.arg] = kw_call_arg.value
                        # assign called arguments to augmented variables
                        # info['modify_stmt'] = True
                        for arg in func_args:
                            self._prefix_stmt.append(
                                ast.Assign(
                                    targets=[ast.Name(
                                        id=aug_arg_map[arg],
                                        ctx=ast.Store())],
                                    value=arg_map[arg]))
                        # augment names
                        copy_inline_func = deepcopy(inline_func)
                        name_aug, _ = \
                            NameAug(aug_arg_map).visit(copy_inline_func)
                        aug_inline_func, _ = \
                            FuncReturnAug(ret_val, ret_flag).visit(name_aug)
                        body = aug_inline_func.body
                        self._prefix_stmt.append(
                            ast.Assign(
                                targets=[ast.Name(
                                    id=ret_flag, ctx=ast.Store())],
                                value=ast.Constant(value=False)))
                        self._prefix_stmt.extend(body)
                        node = ast.Name(
                            id=ret_val, ctx=ast.Load())
                        info["modify_stmt"] = True
        return node, info

    def visit_For(self, node: ast.AST) \
            -> tuple[ast.AST | list, bool, set, set]:
        """
        transforms for statment to while statement
        with the given probability
        """
        node, info = self.generic_visit(node)
        if self.random.random() <= self.probs.get('for2while', 0):
            target = node.target
            compare_target = deepcopy(target)
            setattr(compare_target, 'ctx', ast.Load())
            iter_id = self._name_gen("iter_id")
            iter_obj = node.iter
            body = node.body
            orelse = node.orelse
            node = [
                ast.Assign(
                    targets=[
                        ast.Name(id=iter_id, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id='iter', ctx=ast.Load()),
                        args=[
                            iter_obj],
                        keywords=[])),
                ast.Assign(
                    targets=[target],
                    value=ast.Call(
                        func=ast.Name(id='next', ctx=ast.Load()),
                        args=[
                            ast.Name(id=iter_id, ctx=ast.Load()),
                            ast.Constant(value=None)],
                        keywords=[])),
                ast.While(
                    test=ast.Compare(
                            left=compare_target,
                            ops=[ast.IsNot()],
                            comparators=[ast.Constant(value=None)]),
                    body=body + [
                        ast.Assign(
                            targets=[target],
                            value=ast.Call(
                                func=ast.Name(id='next', ctx=ast.Load()),
                                args=[
                                    ast.Name(id=iter_id, ctx=ast.Load()),
                                    ast.Constant(value=None)],
                                keywords=[]))],
                    orelse=orelse)]
        return node, info

    def visit_While(self, node: ast.AST) \
            -> tuple[ast.AST | list, bool, set, set]:
        node, info = self.generic_visit(node)
        if self.random.random() <= self.probs.get('while2for', 0):
            test = node.test
            body = node.body
            orelse = node.orelse
        return node, info

    def visit_IfExp(self, node: ast.AST) -> tuple[ast.AST, bool, set, set]:
        node, info = self.generic_visit(node)
        if (info["modify_stmt"] or self.random.random()
                <= self.probs.get('ternary2if', 0)):
            test = node.test
            body = node.body
            orelse = node.orelse
            ifexp_name = self._name_gen('ifexp')
            self._prefix_stmt.append(
                ast.If(
                    test=test,
                    body=ast.Assign(
                        targets=[
                            ast.Name(id=ifexp_name, ctx=ast.Store())],
                        value=body),
                    orelse=[
                        ast.Assign(
                            targets=[
                                ast.Name(id=ifexp_name, ctx=ast.Store())],
                            value=orelse)]))
            node = ast.Name(id=ifexp_name, ctx=ast.Load())
            info["modify_stmt"] = True
        return node, info

    def _decompose_chain_ifs(self, ifs: list) -> tuple[ast.If, ast.If]:
        if not isinstance(ifs, list):
            raise TypeError("ifs not given as list")
        elif len(ifs) == 0:
            raise ValueError("ifs empty")
        ifs_iter = iter(ifs)
        node = next(ifs_iter)
        cur = ast.If(
            test=node,
            body=[],
            orelse=[])
        root = cur
        for node in ifs_iter:
            cur.body = [
                ast.If(
                    test=node,
                    body=[],
                    orelse=[])]
            cur = cur.body[0]
        return root, cur

    def _decompose_chain_generators(self, generators: list) \
            -> tuple[ast.For, ast.For | ast.If]:
        if not isinstance(generators, list):
            raise TypeError("generators not given as list")
        elif len(generators) == 0:
            raise ValueError("generators empty")
        gen_iter = iter(generators)
        node = next(gen_iter)
        cur = ast.For(
            target=node.target,
            iter=node.iter,
            body=[],
            orelse=[])
        root = cur
        if len(node.ifs) > 0:
            ifs_root, ifs_cur = self._decompose_chain_ifs(node.ifs)
            cur.body = [ifs_root]
            cur = ifs_cur
        for node in gen_iter:
            cur.body = [
                ast.For(
                    target=node.target,
                    iter=node.iter,
                    body=[],
                    orelse=[])]
            cur = cur.body[0]
            if len(node.ifs) > 0:
                ifs_root, ifs_cur = self._decompose_chain_ifs(node.ifs)
                cur.body = [ifs_root]
                cur = ifs_cur
            else:
                cur = cur.body[0]
        return root, cur

    def visit_ListComp(self, node: ast.ListComp) -> tuple[Any, bool, set, set]:
        """
        If a expression inside the ListComp transformed
        into a statement, ListComp is also transformed
        to a statement
        """
        node, info = self.generic_visit(node)
        if self.debug:
            print(info["using_vars"], info["def_vars"])
        modify_comp2stmt = (info["modify_stmt"]
                            | (self.random.random()
                               <= self.probs.get('comp2stmt', 0)))
        modify_comp2func = (self.random.random()
                            <= self.probs.get('comp2genfunc', 0))
        if modify_comp2stmt or modify_comp2func:
            listcomp_name = self._name_gen("listcomp")
            elt = node.elt
            generators = node.generators
        # both comp2stmt and comp2func applied
        if modify_comp2func:
            func_vars = info["using_vars"].difference(info["def_vars"])
            # compile arguments and make function
            args = []
            for var in func_vars:
                args.append(ast.arg(arg=var))
            func_args = ast.arguments(
                    posonlyargs=[],
                    args=args,
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[])
            func_def = ast.FunctionDef(
                name=listcomp_name,
                args=func_args,
                body=[],
                decorator_list=[])
            # make function body
            body = [ast.Return(value=deepcopy(node))]
            # prepend statement
            func_def.body = body
            # include defined func
            if not modify_comp2stmt:
                self._prefix_stmt.append(func_def)
                self._defined_funcs[listcomp_name] = func_def
            # make return node
            node = ast.Call(
                func=ast.Name(
                    id='list', ctx=ast.Load()),
                args=[
                    ast.Call(
                        func=ast.Name(
                            id=listcomp_name, ctx=ast.Load()),
                        args=list(map(
                            lambda var: ast.Name(id=var,
                                                 ctx=ast.Load()),
                            func_vars)),
                        keywords=[])],
                keywords=[])
        if modify_comp2stmt:
            gen_root, gen_cur = self._decompose_chain_generators(generators)
            if not modify_comp2func:
                assignbody = ast.Assign(
                    targets=[
                        ast.Name(
                            id=listcomp_name,
                            ctx=ast.Store())],
                    value=ast.List(
                        elts=[],
                        ctx=ast.Load()))
                body = ast.Expr(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id=listcomp_name,
                                               ctx=ast.Load()),
                                attr='append',
                                ctx=ast.Load()),
                            args=[elt],
                            keywords=[]))
                # append the statements to thier position
                gen_cur.body = self._build_stmt_and_release_names(body)
                self._prefix_stmt.append(assignbody)
                self._prefix_stmt.append(gen_root)
                node = ast.Name(id=listcomp_name,
                                ctx=ast.Load())
        if modify_comp2stmt and modify_comp2func:
            body = ast.Expr(
                value=ast.Yield(
                    value=elt))
            gen_cur.body = self._build_stmt_and_release_names(body)
            func_def.body = [gen_root]
            # include defined func
            self._prefix_stmt.append(func_def)
            self._defined_funcs[listcomp_name] = func_def
            info["modify_stmt"] = False
        return node, info

    def visit_SetComp(self, node: ast.SetComp) -> Any:
        """
        If a expression inside the SetComp transformed
        into a statement, SetComp is also transformed
        to a statement
        """
        node, info = self.generic_visit(node)
        if info["modify_stmt"]:
            setcomp_name = self._name_gen("setcomp")
            elt = node.elt
            generators = node.generators
            gen_root, gen_cur = self._decompose_chain_generators(generators)
            body = ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id=setcomp_name,
                                           ctx=ast.Load()),
                            attr='add',
                            ctx=ast.Load()),
                        args=[elt],
                        keywords=[]))
            gen_cur.body = self._build_stmt_and_release_names(body)
            # append the statements to thier position
            self._prefix_stmt.append(
                ast.Assign(
                    targets=[
                        ast.Name(id=setcomp_name,
                                 ctx=ast.Store())],
                    value=ast.Set(
                        elts=[],
                        ctx=ast.Load())))
            self._prefix_stmt.append(gen_root)
            node = ast.Name(id=setcomp_name,
                            ctx=ast.Load())
        return node, info

    def visit_DictComp(self, node: ast.DictComp) -> Any:
        """
        If a expression inside the DictComp transformed
        into a statement, ListComp is also transformed
        to a statement
        """
        node, info = self.generic_visit(node)
        if info["modify_stmt"]:
            dictcomp_name = self._name_gen("dictcomp")
            key = node.key
            value = node.value
            generators = node.generators
            gen_root, gen_cur = self._decompose_chain_generators(generators)
            body = ast.Assign(
                targets=[
                    ast.Subscript(
                        value=ast.Name(id=dictcomp_name,
                                       ctx=ast.Load()),
                        slice=key,
                        ctx=ast.Store())],
                value=value)
            gen_cur.body = self._build_stmt_and_release_names(body)
            # append the statements to thier position
            self._prefix_stmt.append(
                ast.Assign(
                    targets=[
                        ast.Name(id=dictcomp_name,
                                 ctx=ast.Store())],
                    value=ast.Dict(
                        keys=[],
                        values=[],
                        ctx=ast.Load())))
            self._prefix_stmt.append(gen_root)
            node = ast.Name(id=dictcomp_name,
                            ctx=ast.Load())
        return node, info

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> Any:
        """
        If a expression inside the GeneratorExp transformed
        into a statement, ListComp is also transformed
        to a statement
        """
        node, info = self.generic_visit(node)
        if info["modify_stmt"]:
            genexp_name = self._name_gen("genexp2list")
            elt = node.elt
            generators = node.generators
            gen_root, gen_cur = self._decompose_chain_generators(generators)
            body = ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id=genexp_name,
                                           ctx=ast.Load()),
                            attr='append',
                            ctx=ast.Load()),
                        args=[elt],
                        keywords=[]))
            gen_cur.body = self._build_stmt_and_release_names(body)
            # append the statements to their position
            self._prefix_stmt.append(
                ast.Assign(
                    targets=[
                        ast.Name(id=genexp_name,
                                 ctx=ast.Store())],
                    value=ast.List(
                        elts=[],
                        ctx=ast.Load())))
            self._prefix_stmt.append(gen_root)
            node = ast.Name(id=genexp_name,
                            ctx=ast.Load())
        return node, info


def test(test_type, root="test", probs: dict = dict(), debug: bool = False) \
        -> bool:
    test_result = True
    astAugmentor = ASTAug(probs)
    try:
        for i in count(1):
            with open(f"{root}/{test_type}/test{i}.py", 'r') as testfile:
                test_ast = ast.parse(testfile.read())
            if debug:
                test_name = f"{test_type} test {i}"
                print(f"{test_name:=^24}")
                print(ast.dump(test_ast, indent=2))
                print("<" * 24)
            with open(f"{root}/{test_type}/ans{i}.py", 'r') as ansfile:
                ans_ast = ast.parse(ansfile.read())
            aug_test_ast = ast.fix_missing_locations(
                    astAugmentor.visit(test_ast))
            if debug:
                print(">" * 24)
                print(ast.dump(aug_test_ast, indent=2))
                debug_path = f"debug/{test_type}"
                if not os.path.exists(debug_path):
                    os.makedirs(debug_path)
                with open(f"{debug_path}/testaug{i}.py", 'w') as resultfile:
                    resultfile.write(ast.unparse(aug_test_ast))
            test_i_result = ast.unparse(aug_test_ast) == ast.unparse(ans_ast)
            test_result &= test_i_result
            if debug:
                if test_i_result:
                    test_i_result_string = f"passed test {i}"
                else:
                    test_i_result_string = f"failed test {i}"
                print(f"{test_i_result_string:=^24}")
    except FileNotFoundError:
        pass
    if test_result:
        print(f"passed {test_type} test")
    else:
        print(f"failed {test_type} test")
    return test_result


def test_for2while(debug=False) -> bool:
    probs = dict()
    probs['for2while'] = 1
    return test('for2while', 'test', probs, debug)


def test_ternary2if(debug=False) -> bool:
    probs = dict()
    probs['ternary2if'] = 1
    return test('ternary2if', 'test', probs, debug)


if __name__ == "__main__":
    # parse argments
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()
    test_flag = True
    test_flag &= test_for2while(args.debug)
    test_flag &= test_ternary2if(args.debug)
    if test_flag:
        print("passed all tests")
    else:
        print("failed some tests")
