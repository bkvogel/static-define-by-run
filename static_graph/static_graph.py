import contextlib
import threading

import chainer
from chainer import function
import numpy as np


_thread_local = threading.local()


class StaticScheduleFunction(function.Function):
    """A function that executes the static schedule of a Chain.

    An instance of this class executes the static schedule of computations
    that are equivalent to executing the define-by-run code of a Chain.

    This class is used by the `static_graph` decorator to wrap the define-by-run
    computations of a chain into two static schedules:
    - The forward schedule corresponds to the computations that are executed by
    the define-by-run code of the `__call__()` method of a chain. The static
    schedule corresponding to these computations can be executed by calling the
    `forward()` method of this class.
    - The backward schedule corresponds to the computations that are executed
    by the sequence of calls to `Function.backward()` that occur during when
    backpropagating the gradients through the same chain. That is, for each
    `Function.forward()` that was called during the forward propagation,
    there will be a corresponding call to `Function.backward()` (of the
    same Function object) in the backward schedule. This backward schedule
    can be executed by calling the `backward()` method of this class.

    Note the intended usage of this class:
    During the first forward pass of a static chain (that is, a chain that is
    decorated by `static_graph`) the define-by-run code is executed. However,
    for subsequent iterations, that define-by-run code is replaced by an instance
    of this function and this function will be called instead. Since the static
    schedules contained by this function perform the same computations, it is
    safe (and potentially much more efficient) to simply execute the static schedule instead
    of the define-by-run code. See `static_graph` for details.

    Args:
            in_vars (tuple of Variable): The tuple of input variables that is supplied to
            `__call__()` method of the chain that this schedule corresponds to. 
    """

    def __init__(self, in_vars):
        self._static_schedule_forward = []
        self._static_schedule_backward = []
        self._chain_in_vars = in_vars
        self._in_arrays = tuple([x.data for x in in_vars])
        self._backward_out_arrays = None
        self._backward_grad_outputs = None

    def set_return_vars(self, out_vars):
        """Set the return variables of the chain.

        This method must be called before the schedule can be executed.

        Args:
            out_vars (Variable or tuple of Varible): The output variables that
            are returned by the `__call__()` method of the chain that this schedule
            corresponds to.
        """
        # Force self._chainer_return_vars to be a tuple of variables.
        if isinstance(out_vars, tuple):
            self._chainer_return_vars = out_vars
        else:
            self._chainer_return_vars = out_vars,
        self._forward_out_arrays = tuple([y.data for y in (out_vars,)])

    def append_forward_function(self, forward):
        """Append a function to the forward static schedule.

        Args:
            forward: The function to append to the schedule. The function
            should not take any arguments and should not return any results.

        """
        self._static_schedule_forward.append(forward)

    def append_backward_function(self, backward):
        """Append a function to the backward static schedule.

        Args:
            backward: The function to append to the schedule. The function
            should not take any arguments and should not return any results.

        """
        self._static_schedule_backward.append(backward)

    def forward(self, inputs):
        # Note: This method will be invoked every iteration starting from the second
        # iteration. That is because the corresponding define-by-run code runs instead
        # during the first iteration.
        if self._backward_grad_outputs is None:
            self._backward_grad_outputs = tuple([x.grad for x in self._chainer_return_vars])
        # Copy any external input arrays into statically-allocated arrays:
        assert len(self._in_arrays) == len(inputs)
        for n in range(len(inputs)):
            # Debug:
            assert self._in_arrays[n].shape == inputs[n].shape
            assert self._in_arrays[n][0].dtype == inputs[n][0].dtype
            self._in_arrays[n][:] = inputs[n]

        # The following line should be the new performance bottleneck after the first iteration
        # has completed. Note that we have several options for the implementation:
        # - Simply iterate through the schedule (in Python), calling each function.
        # - Export the schedule to C/C++ code. The forward computation can then
        # potentially be performed without any dependency on Python.
        # - Optimize the schedule code in Cython, calling optimized C/C++ code.
        [x() for x in self._static_schedule_forward]
        return self._forward_out_arrays

    def backward(self, inputs, grad_outputs):
        # Note: This method will be invoked every iteration starting from the second
        # iteration. That is because the corresponding define-by-run code runs instead
        # during the first iteration.
        print('StaticScheduleFunction: backward()...')
        # Copy `grad_outputs` into statically-allocated storage that is used by
        # the backward schedule.
        assert len(self._backward_grad_outputs) == len(grad_outputs)
        for n in range(len(grad_outputs)):
            # Debug:
            assert self._backward_grad_outputs[n].shape == grad_outputs[n].shape
            assert self._backward_grad_outputs[n][0].dtype == grad_outputs[n][0].dtype
            self._backward_grad_outputs[n][:] = grad_outputs[n]

        # The following line should be the new performance bottleneck after the first iteration
        # has completed. It can potentially be optimized similar to the forward schedule.
        [x() for x in self._static_schedule_backward]
        if self._backward_out_arrays is None:
            self._backward_out_arrays = tuple([x.grad for x in self._chain_in_vars])
        return self._backward_out_arrays


@contextlib.contextmanager
def static_schedule_scope(static_sched_func):
    """Returns a static scope with the current StaticScheduleFunction.

    Calling the `__call__()` method of a chain in this scope will cause
    all functions that are decorated with `static_forward` or
    `static_backward` to be added to the corresponding static schedules
    of the supplied static schedule instance.

    Only a single chain instance should be called within this scope.
    For example usage, refer to the `static_graph` implementation.

    Args:
        static_sched_func (StaticScheduleFunction): The static schedule
        object that will contain both the forward and backward
        static schedules for the chain that is called
        within this scope.

    """
    default = getattr(_thread_local, 'schedule_func', None)
    _thread_local.schedule_func = static_sched_func
    yield
    _thread_local.schedule_func = default


def static_forward(func):
    """Decorator to mark a function for inclusion in the forward schedule.

    This decorator is used to wrap a function `func` that is a forward-pass
    method of a sub-class of Function. This will cause it to be added to
    the forward static schedule when the `static_graph` feature is
    enabled on a Chain that deeply calls it from the chain's
    `__call__()` method.

    The function to be wrapped should only return `None` because any return value
    will be ignored. Instead of returning its results, any result arrays must
    be supplied as input arguments and must have already have been initialized
    to the appropriate dimensions and data types.

    Usage:

    Typical usage is to allocate any required arrays (Numpy or Cupy) in Python
    code in an instance of Function (See `LinearFunction` function for an example).
    Generally, this will involve first allocating storage for the results arrays
    in the `forward()` method of a sub-class of Function. Then, the foward()
     method should call another
    (private) method that is wrapped by this decorator. The
    decorated function will take all required input and output arrays as
    arguments and will not return anything (that is, `None` will be implicitly 
    returned). 
    
    Note that by following this usage convention, all input and output activations,
    along with any parameter arrays will have been statically allocated by the
    end of the first forward pass. Since the the forward-pass functions that
    are used inside the forward static schedule (that is, the functions that
    use this decorator) do not allocate any arrays, this results in code that
    looks like 'define-by-run' to the user, and which can be debugged during
    the first iteration, but then becomes static in terms of memory allocations and
    scheduling starting from the second iteration. Thus, we get the benifit of
    both ease of use and optimized performance.

    It is important that all of the required computations that occur during the
    second  and later forward passes must be contained inside the functions 
    that are use this decorator. That is, any other code (that is not wrapped inside this
    decorator) in the various Function and Link instances can be viewed as
    setup code that only checks types, allocates result arrays, initializes
    parameters etc., but does not perform any computations that must
    be repeated after the first forward pass.

    The reason for this is that after the first iteration (that is, starting
    from the second forward pass), when the chain's `__call__()` is called,
    the forward static schedule will be invoked and it will only call the
    functions that were wrapped with this decorator. Note that this can potentially
    lead to difficult to find bugs if one forgets to decorate a required function,
    since the corresponding computations would no longer execute after the
    first iteration.

    Restrictions:

    This feature currently assumes that the inputs to the wrapped function
    Will continue to have the same shapes and types each time it is called.
    There are currently no checks to ensure that this constraint is satisfied.
    Such a type check may be added in the future. For example, the current code
    will break if the mini-batch size changes at any point. 
    todo: add checks for this and run the define-by-run code again whenever any
    input sizes change. If such changes are frequent, we can consider caching multiple
    static schedules and using the appropriate one for the current input sizes. 

    Args:
        func: A a forward-pass method of a sub-class of Function that should be inserted
        into the static forward schedule when `static_graph` is enabled. The function
        must not return anything because any return values will be ignored.

    Returns: The wrapped function.

    """
    def wrapped_func(*args, **kwargs):
        # Save arguments, function, and results pointers/references to the schedule list:
        def no_arg_func():
            #print('In no_arg_func: Calling: ', func)
            func(*args, **kwargs)
            #print("Arguments were: %s, %s" % (args, kwargs))

        # no_arg_func() requires no arguments to call since the arguments of the decorated function
        # are captured by the closure.
        no_arg_func()

        schedule_function = getattr(_thread_local, 'schedule_func', None)
        # If trace mode is on, add to schedule.
        if schedule_function is not None:
            schedule_function.append_forward_function(no_arg_func)
            # Add the schedule function as an attribute of the Function instance
            # that contains the wrapped function as a method
            # This attribute will be needed by the corresponding @static_backward
            # function.
            instance = args[0]
            assert isinstance(instance, function.Function)
            print('Adding function to the forward static schedule.')
            #print('static_forward: instance: ', instance)
            instance.schedule_func = schedule_function

    return wrapped_func


def static_backward(func):
    """Decorator to mark a function for inclusion in the backward schedule.

    The decorator is used in the same way as `static_forward` except that
    it is used to decorate the functions that should be added to the static
    backward-pass schedule. The wrapped function implements the
    computations of the `backward()` method of a Function instance that
    must be executed during every backward pass.

    Similarly to `static_forward`, the wrapped function should not return
    a result because it will be ignored.

    Args:
        func: A a backward-pass method of a sub-class of Function that should be inserted
        into the static backward schedule when `static_graph` is enabled. The function
        must not return a value because any return values will be ignored.

    Returns: The wrapped function.

    """
    def wrapped_func(*args, **kwargs):
        # Save arguments, function, and results pointers/references to the schedule list:
        def no_arg_func():
            #print('In no_arg_func: Calling: ', func)
            func(*args, **kwargs)
            #print("Arguments were: %s, %s" % (args, kwargs))

        # no_arg_func() requires no arguments to call since the arguments of the decorated function
        # are captured by the closure.
        no_arg_func()
        inst = args[0]
        assert isinstance(inst, function.Function)
        schedule_function = getattr(inst, 'schedule_func', None)
        # If trace mode is on, add to schedule.
        if schedule_function is not None:
            print('Adding function to the backward static schedule.')
            schedule_function.append_backward_function(no_arg_func)

    return wrapped_func

def static_graph(forward):
    """Decorator to mark a Chain's __call__() as a static sub-graph.

    This decorator marks the define-by-run code inside the `__call__()`
    method of a Chain instance as corresponding to a static computation
    graph or sub-graph. Only the top-most (that is, largest) static
    sub-graph should be decorated.

    This decorator will cause the wrapped `__call__()` method to
    execute its define-by-run code once (the first time it is called). 
    Subsequent calls
    will then invoke optimized code that performs the same computations
    but without the Python overhead of define-by-run. Such optimized
    code can potentially be executed as optimized C or C++ code, and
    potentially deployed to platforms that do not support Python (todo).

    Usage:

    Apply this decorator only to the top-most Chain in the hierarchy that
    contains a static sub-graph. It is not necessary (and not allowed) to
    mark a chain as static if it is contained within
    another chain that is also marked as being static (todo: this is not checked yet). 
    That is, suppose a
    static graph `A` contains a static sub-graph `B`. Then, only the chain
    corresponding to `A` should be marked as static and the chain corresponding
    to `B` should not be marked as static.

    Notes:
        It is required to set retain_grad=True when calling loss.backward()
        on a model that uses the static graph feature. This is because
        the gradient arrays that were allocated during the first backward
        pass will be reused in the backward static schedule. If retain_grad
        were set to False, then these arrays would be set to None in
        `Variable.backward()` which would break the functionality.

    Args:
        forward: The forward `__call__()` method of an instance of Chain
        that is wrapped by this decorator.

    Returns:

    """
    # Check if the first iteration has completed. If not, call the define-by-run
    # code inside the chain's __call__(). Otherwise, call the static schedule
    # function (which is an instance of Function).

    def wrapped_func(*args, **kwargs):
        #print('Inside the wrapped Chain.__call__()...')
        #print('Chain instance: ', args[0])

        chain = args[0]
        in_vars = args[1:]
        if hasattr(chain, 'static_schedule'):
            # Call the optimized static schedule code.
            print('This is the 2nd or greater iteration. Calling the optimized schedule...')
            return chain.static_schedule(*in_vars, **kwargs)
        else:
            # This is the first iteration. Calling the define-by-run code.
            assert isinstance(chain, chainer.Chain)
            print('This is the first iteration. Calling the define-by-run code.: ', forward)
            chain.static_schedule = StaticScheduleFunction(in_vars)
            with static_schedule_scope(chain.static_schedule):
                out_vars = forward(*args, **kwargs)

            chain.static_schedule.set_return_vars(out_vars)
            #print("Arguments were: %s, %s" % (args, kwargs))
            return out_vars

    return wrapped_func