from collections import defaultdict
from os.path import splitext
from contextlib import contextmanager
from pathlib import Path
import inspect, itertools, collections
import re, gzip, bz2


class argmap:
    def __init__(self, func, *args, try_finally=False):
        self._func = func
        self._args = args
        self._finally = try_finally

    @staticmethod
    def _lazy_compile(func):
        """Compile the source of a wrapped function

        Assemble and compile the decorated function, and intrusively replace its
        code with the compiled version's.  The thinly wrapped function becomes
        the decorated function.

        Parameters
        ----------
        func : callable
            A function returned by argmap.__call__ which is in the process
            of being called for the first time.

        Returns
        -------
        func : callable
            The same function, with a new __code__ object.

        Notes
        -----
        It was observed in NetworkX issue #4732 [1] that the import time of
        NetworkX was significantly bloated by the use of decorators: over half
        of the import time was being spent decorating functions.  This was
        somewhat improved by a change made to the `decorator` library, at the
        cost of a relatively heavy-weight call to `inspect.Signature.bind`
        for each call to the decorated function.

        The workaround we arrived at is to do minimal work at the time of
        decoration.  When the decorated function is called for the first time,
        we compile a function with the same function signature as the wrapped
        function.  The resulting decorated function is faster than one made by
        the `decorator` library, so that the overhead of the first call is
        'paid off' after a small number of calls.

        References
        ----------

        [1] https://github.com/networkx/networkx/issues/4732

        """
        real_func = func.__argmap__.compile(func.__wrapped__)
        func.__code__ = real_func.__code__
        func.__globals__.update(real_func.__globals__)
        func.__dict__.update(real_func.__dict__)
        return func

    def __call__(self, f):
        """Construct a lazily decorated wrapper of f.

        The decorated function will be compiled when it is called for the first time,
        and it will replace its own __code__ object so subsequent calls are fast.

        Parameters
        ----------
        f : callable
            A function to be decorated.

        Returns
        -------
        func : callable
            The decorated function.

        See Also
        --------
        argmap._lazy_compile
        """

        if inspect.isgeneratorfunction(f):

            def func(*args, __wrapper=None, **kwargs):
                yield from argmap._lazy_compile(__wrapper)(*args, **kwargs)

        else:

            def func(*args, __wrapper=None, **kwargs):
                return argmap._lazy_compile(__wrapper)(*args, **kwargs)

        # standard function-wrapping stuff
        func.__name__ = f.__name__
        func.__doc__ = f.__doc__
        func.__defaults__ = f.__defaults__
        func.__kwdefaults__.update(f.__kwdefaults__ or {})
        func.__module__ = f.__module__
        func.__qualname__ = f.__qualname__
        func.__dict__.update(f.__dict__)
        func.__wrapped__ = f

        # now that we've wrapped f, we may have picked up some __dict__ or
        # __kwdefaults__ items that were set by a previous argmap.  Thus, we set
        # these values after those update() calls.

        # If we attempt to access func from within itself, that happens through
        # a closure -- which trips an error when we replace func.__code__.  The
        # standard workaround for functions which can't see themselves is to use
        # a Y-combinator, as we do here.
        func.__kwdefaults__["_argmap__wrapper"] = func

        # this self-reference is here because functools.wraps preserves
        # everything in __dict__, and we don't want to mistake a non-argmap
        # wrapper for an argmap wrapper
        func.__self__ = func

        # this is used to variously call self.assemble and self.compile
        func.__argmap__ = self

        return func

    __count = 0

    @classmethod
    def _count(cls):
        """Maintain a globally-unique identifier for function names and "file" names

        Note that this counter is a class method reporting a class variable
        so the count is unique within a Python session. It could differ from
        session to session for a specific decorator depending on the order
        that the decorators are created. But that doesn't disrupt `argmap`.

        This is used in two places: to construct unique variable names
        in the `_name` method and to construct unique fictitious filenames
        in the `_compile` method.

        Returns
        -------
        count : int
            An integer unique to this Python session (simply counts from zero)
        """
        cls.__count += 1
        return cls.__count

    _bad_chars = re.compile("[^a-zA-Z0-9_]")

    @classmethod
    def _name(cls, f):
        """Mangle the name of a function to be unique but somewhat human-readable

        The names are unique within a Python session and set using `_count`.

        Parameters
        ----------
        f : str or object

        Returns
        -------
        name : str
            The mangled version of `f.__name__` (if `f.__name__` exists) or `f`

        """
        f = f.__name__ if hasattr(f, "__name__") else f
        fname = re.sub(cls._bad_chars, "_", f)
        return f"argmap_{fname}_{cls._count()}"

    def compile(self, f):
        """Compile the decorated function.

        Called once for a given decorated function -- collects the code from all
        argmap decorators in the stack, and compiles the decorated function.

        Much of the work done here uses the `assemble` method to allow recursive
        treatment of multiple argmap decorators on a single decorated function.
        That flattens the argmap decorators, collects the source code to construct
        a single decorated function, then compiles/executes/returns that function.

        The source code for the decorated function is stored as an attribute
        `_code` on the function object itself.

        Note that Python's `compile` function requires a filename, but this
        code is constructed without a file, so a fictitious filename is used
        to describe where the function comes from. The name is something like:
        "argmap compilation 4".

        Parameters
        ----------
        f : callable
            The function to be decorated

        Returns
        -------
        func : callable
            The decorated file

        """
        sig, wrapped_name, functions, mapblock, finallys, mutable_args = self.assemble(
            f
        )

        call = f"{sig.call_sig.format(wrapped_name)}#"
        mut_args = f"{sig.args} = list({sig.args})" if mutable_args else ""
        body = argmap._indent(sig.def_sig, mut_args, mapblock, call, finallys)
        code = "\n".join(body)

        locl = {}
        globl = dict(functions.values())
        filename = f"{self.__class__} compilation {self._count()}"
        compiled = compile(code, filename, "exec")
        exec(compiled, globl, locl)
        func = locl[sig.name]
        func._code = code
        return func

    def assemble(self, f):
        """Collects components of the source for the decorated function wrapping f.

        If `f` has multiple argmap decorators, we recursively assemble the stack of
        decorators into a single flattened function.

        This method is part of the `compile` method's process yet separated
        from that method to allow recursive processing. The outputs are
        strings, dictionaries and lists that collect needed info to
        flatten any nested argmap-decoration.

        Parameters
        ----------
        f : callable
            The function to be decorated.  If f is argmapped, we assemble it.

        Returns
        -------
        sig : argmap.Signature
            The function signature as an `argmap.Signature` object.
        wrapped_name : str
            The mangled name used to represent the wrapped function in the code
            being assembled.
        functions : dict
            A dictionary mapping id(g) -> (mangled_name(g), g) for functions g
            referred to in the code being assembled. These need to be present
            in the ``globals`` scope of ``exec`` when defining the decorated
            function.
        mapblock : list of lists and/or strings
            Code that implements mapping of parameters including any try blocks
            if needed. This code will precede the decorated function call.
        finallys : list of lists and/or strings
            Code that implements the finally blocks to post-process the
            arguments (usually close any files if needed) after the
            decorated function is called.
        mutable_args : bool
            True if the decorator needs to modify positional arguments
            via their indices. The compile method then turns the argument
            tuple into a list so that the arguments can be modified.
        """

        # first, we check if f is already argmapped -- if that's the case,
        # build up the function recursively.
        # > mapblock is generally a list of function calls of the sort
        #     arg = func(arg)
        # in addition to some try-blocks if needed.
        # > finallys is a recursive list of finally blocks of the sort
        #         finally:
        #             close_func_1()
        #     finally:
        #         close_func_2()
        # > functions is a dict of functions used in the scope of our decorated
        # function. It will be used to construct globals used in compilation.
        # We make functions[id(f)] = name_of_f, f to ensure that a given
        # function is stored and named exactly once even if called by
        # nested decorators.
        if hasattr(f, "__argmap__") and f.__self__ is f:
            (
                sig,
                wrapped_name,
                functions,
                mapblock,
                finallys,
                mutable_args,
            ) = f.__argmap__.assemble(f.__wrapped__)
            functions = dict(functions)  # shallow-copy just in case
        else:
            sig = self.signature(f)
            wrapped_name = self._name(f)
            mapblock, finallys = [], []
            functions = {id(f): (wrapped_name, f)}
            mutable_args = False

        if id(self._func) in functions:
            fname, _ = functions[id(self._func)]
        else:
            fname, _ = functions[id(self._func)] = self._name(self._func), self._func

        # this is a bit complicated -- we can call functions with a variety of
        # nested arguments, so long as their input and output are tuples with
        # the same nested structure. e.g. ("a", "b") maps arguments a and b.
        # A more complicated nesting like (0, (3, 4)) maps arguments 0, 3, 4
        # expecting the mapping to output new values in the same nested shape.
        # The ability to argmap multiple arguments was necessary for
        # the decorator `nx.algorithms.community.quality.require_partition`, and
        # while we're not taking full advantage of the ability to handle
        # multiply-nested tuples, it was convenient to implement this in
        # generality because the recursive call to `get_name` is necessary in
        # any case.
        applied = set()

        def get_name(arg, first=True):
            nonlocal mutable_args
            if isinstance(arg, tuple):
                name = ", ".join(get_name(x, False) for x in arg)
                return name if first else f"({name})"
            # if arg in applied:
            #     raise nx.NetworkXError(f"argument {arg} is specified multiple times")
            applied.add(arg)
            if arg in sig.names:
                return sig.names[arg]
            elif isinstance(arg, str):
                # if sig.kwargs is None:
                #     raise nx.NetworkXError(
                #         f"name {arg} is not a named parameter and this function doesn't have kwargs"
                #     )
                return f"{sig.kwargs}[{arg!r}]"
            else:
                # if sig.args is None:
                #     raise nx.NetworkXError(
                #         f"index {arg} not a parameter index and this function doesn't have args"
                #     )
                mutable_args = True
                return f"{sig.args}[{arg - sig.n_positional}]"

        if self._finally:
            # here's where we handle try_finally decorators.  Such a decorator
            # returns a mapped argument and a function to be called in a
            # finally block.  This feature was required by the open_file
            # decorator.  The below generates the code
            #
            # name, final = func(name)                   #<--append to mapblock
            # try:                                       #<--append to mapblock
            #     ... more argmapping and try blocks
            #     return WRAPPED_FUNCTION(...)
            #     ... more finally blocks
            # finally:                                   #<--prepend to finallys
            #     final()                                #<--prepend to finallys
            #
            for a in self._args:
                name = get_name(a)
                final = self._name(name)
                mapblock.append(f"{name}, {final} = {fname}({name})")
                mapblock.append("try:")
                finallys = ["finally:", f"{final}()#", "#", finallys]
        else:
            mapblock.extend(
                f"{name} = {fname}({name})" for name in map(get_name, self._args)
            )

        return sig, wrapped_name, functions, mapblock, finallys, mutable_args

    @classmethod
    def signature(cls, f):
        r"""Construct a Signature object describing `f`

        Compute a Signature so that we can write a function wrapping f with
        the same signature and call-type.

        Parameters
        ----------
        f : callable
            A function to be decorated

        Returns
        -------
        sig : argmap.Signature
            The Signature of f

        Notes
        -----
        The Signature is a namedtuple with names:

            name : a unique version of the name of the decorated function
            signature : the inspect.signature of the decorated function
            def_sig : a string used as code to define the new function
            call_sig : a string used as code to call the decorated function
            names : a dict keyed by argument name and index to the argument's name
            n_positional : the number of positional arguments in the signature
            args : the name of the VAR_POSITIONAL argument if any, i.e. \*theseargs
            kwargs : the name of the VAR_KEYWORDS argument if any, i.e. \*\*kwargs

        These named attributes of the signature are used in `assemble` and `compile`
        to construct a string of source code for the decorated function.

        """
        sig = inspect.signature(f, follow_wrapped=False)
        def_sig = []
        call_sig = []
        names = {}

        kind = None
        args = None
        kwargs = None
        npos = 0
        for i, param in enumerate(sig.parameters.values()):
            # parameters can be position-only, keyword-or-position, keyword-only
            # in any combination, but only in the order as above.  we do edge
            # detection to add the appropriate punctuation
            prev = kind
            kind = param.kind
            if prev == param.POSITIONAL_ONLY != kind:
                # the last token was position-only, but this one isn't
                def_sig.append("/")
            if prev != param.KEYWORD_ONLY == kind != param.VAR_POSITIONAL:
                # param is the first keyword-only arg and isn't starred
                def_sig.append("*")

            # star arguments as appropriate
            if kind == param.VAR_POSITIONAL:
                name = "*" + param.name
                args = param.name
                count = 0
            elif kind == param.VAR_KEYWORD:
                name = "**" + param.name
                kwargs = param.name
                count = 0
            else:
                names[i] = names[param.name] = param.name
                name = param.name
                count = 1

            # assign to keyword-only args in the function call
            if kind == param.KEYWORD_ONLY:
                call_sig.append(f"{name} = {name}")
            else:
                npos += count
                call_sig.append(name)

            def_sig.append(name)

        fname = cls._name(f)
        def_sig = f'def {fname}({", ".join(def_sig)}):'

        if inspect.isgeneratorfunction(f):
            _return = "yield from"
        else:
            _return = "return"

        call_sig = f"{_return} {{}}({', '.join(call_sig)})"

        return cls.Signature(fname, sig, def_sig, call_sig, names, npos, args, kwargs)

    Signature = collections.namedtuple(
        "Signature",
        [
            "name",
            "signature",
            "def_sig",
            "call_sig",
            "names",
            "n_positional",
            "args",
            "kwargs",
        ],
    )

    @staticmethod
    def _flatten(nestlist, visited):
        """flattens a recursive list of lists that doesn't have cyclic references

        Parameters
        ----------
        nestlist : iterable
            A recursive list of objects to be flattened into a single iterable

        visited : set
            A set of object ids which have been walked -- initialize with an
            empty set

        Yields
        ------
        Non-list objects contained in nestlist

        """
        for thing in nestlist:
            if isinstance(thing, list):
                if id(thing) in visited:
                    raise ValueError("A cycle was found in nestlist.  Be a tree.")
                else:
                    visited.add(id(thing))
                yield from argmap._flatten(thing, visited)
            else:
                yield thing

    _tabs = " " * 64

    @staticmethod
    def _indent(*lines):
        """Indent list of code lines to make executable Python code

        Indents a tree-recursive list of strings, following the rule that one
        space is added to the tab after a line that ends in a colon, and one is
        removed after a line that ends in an hashmark.

        Parameters
        ----------
        *lines : lists and/or strings
            A recursive list of strings to be assembled into properly indented
            code.

        Returns
        -------
        code : str

        Examples
        --------

            argmap._indent(*["try:", "try:", "pass#", "finally:", "pass#", "#",
                             "finally:", "pass#"])

        renders to

            '''try:
             try:
              pass#
             finally:
              pass#
             #
            finally:
             pass#'''
        """
        depth = 0
        for line in argmap._flatten(lines, set()):
            yield f"{argmap._tabs[:depth]}{line}"
            depth += (line[-1:] == ":") - (line[-1:] == "#")
