
#### Demonstration of  "static define-by-run" feature

This is prototype code that shows how Chainer can be refactored to support both the user convenience of define-by-run and the runtime performance of a static computation graph. It is a hybrid "define-by-run"-"static graph" feature that we call "static define-by-run." This is a partial implementation that is only intended to provide a single concrete example to illustrate the idea.  

This feature intends to support optimized execution of Chainer models, as long as the model (or part of the model) corresponds to a static computation graph. This feature can potentially also support exporting a Chainer model to C/C++ code that can be executed on platforms that do not support Python (although this is not implemented yet, but seems conceptually straightforward).

The current code is limited to supporting only models using the Linear link and/or function, since the code is only intended to illustrate the user-facing API and show a concrete example of how to modify Chainer to support such a feature. It will still require some further work to actually create optimized Chainer functions (such as by using MKL-DNN), as well as refactoring the many existing links and functions to be compatible with the requirements of this feature. The current code shows that, at least for the case of Linear link, that the only change to the public API that is needed is for the user to decorate the `__call__()` method of each chain that contains a static sub-graph.

### What it does

This feature can be used to speedup execution of a Chainer model as long as the model corresponds to a static computation graph or contains static sub-graphs. Chainer can be considered a "dynamic graph deep learning framework" that supports "define-by-run", which is a form of automatic differentiation. This makes it easy for the user to define a model by writing Python code that imperatively executes the forward-pass commutations. This is convenient for describing computation graphs in which the graph potentially changes from iteration to iteration.

It seems, however,  that the majority of the currently popular deep learning models correspond to either a static graph or a mostly static graph. The assumption of a static computation graph is made by other frameworks such as Tensorflow. If the graph can be assumed to be static, many optimizations then become possible. This current code only performs the most basic of optimizations, which consists of removing most dynamic type checking, using statically-allocated arrays, and using a static schedule. However, these kinds of optimizations could still be sufficient to reduce the most significant performance bottlenecks observed in Chainer models, which are often in the Python interpreter rather than in executing the deep learning functions. It is expected that this approach could be very effective in reducing Python interpreter overhead from Chainer models, which has been particularly challenging to optimize when a dynamic graph must be assumed.

Another advantage of dynamic graph frameworks is that they can support easier debugging because if there is an error in the model code, it will often cause the Python interpreter to throw an exception and print a stack trace on or soon after the line of code with the bug. A static graph framework might instead first optimize the graph, so that finding the bug in optimized code might be considerably more difficult.

In designing this feature, we therefore wish to retain the ease of debugging provided by Chainer while also making the runtime performance more competitive with static graph frameworks. We think a reasonable compromise consists of the following design choices.

* The user-facing API should be almost unchanged compared to existing Chainer. Ideally, the user should only need to decorate each `Chain` in the model that corresponds to a static sub-graph.
* The first iteration of the model (that is, the first forward and backward pass) should execute as define-by-run to allow for easy debugging. That is, there should be essentially no change in how the code executes compared to existing Chainer.
* Starting from the second iteration, the execution mode will change so that an optimized static schedule will be used, potentially making the runtime performance close to static graph frameworks. This switch from define-by-run to static mode should be invisible to the user (but the user should see their model run much faster). Note that we assume that if the execution finishes the first iteration without a problem, there is probably no bug and it should be safe to run the optimized schedule.
* It should be possible to export the static sub-graph to optimized C or C++ code that can execute without any dependencies on Python. This is intended to support easier deployment of Chainer models to platforms that do not support Python. The current prototyping code does not implement this capability, but it should hopefully be clear to the reader of this code how such capability can potentially be added.

Note that conceptually, this optimization strategy corresponds to optimization by removal of redundant code. That is, during the first iteration, as the define-by-run code executes, the functions that perform the necessary deep learning computations are executed, but so are the various other functions that perform various dynamic type-checking, allocation of Python objects to build the backward computation graph (which is needed for backprop), etc. If we must assume a dynamic graph, then such dynamic code must execute during every iteration, slowing down performance. However, if a static graph can instead be assumed, then there is no longer any reason to run the dynamic allocations and/or checks once the first iteration has completed. Thus, it seems that a reasonable optimization method can simply consist of saving only the static schedule of the needed deep learning computations to run in subsequent iterations, and discard the redundant dynamic code.


The current implementation provides the following features:

* Optimization-by-removal: The same deep learning functions are called whether this static graph is enabled or not. Therefore, only a single implementation of these functions are required, easing maintainability.
* Memory efficient: The same arrays that were allocated in the various Chainer variables during the first forward and backward pass are reused during subsequent iterations. That is, the memory for the arrays is allocated statically, even though it appears to be dynamic in the define-by-run code.


#### Usage

First install Chainer v1. Then run the example:
```
python test_static_graph.py
```

To use this feature, the computation graph must be static. Even if the complete Chainer model is not static, this feature can still be used on a static sub-graphs in the computation graph. The complete computation graph should first be partitioned into static sub-graphs that are as large as possible. The user should then write a chain that computes the forward pass of the static graph in its `__call()__` method and also decorate the method with the `@static_graph` decorator.

It is important to note that this decorated `__call__()` method will only be called once during execution of the model. Therefore, the user must be careful not place any code inside this method other than the define-by-run code. For example, Python code that increments a counter variable or prints to stdout would then only be called once during execution of the model.



#### How it works

For details, see `static_graph.py`, which implements this feature, and also see the slightly modified Chainer linear link and function in the `functions` and `links` directories. Note that the example model, `test_static_graph.py`, uses these modified links and functions, not the Chainer versions (notice the different imports).

Basically, the performance optimization is achieved by disabling redundant code. A user can specify that a Chainer model is static by using the `@static_graph` decorator on the `__call__()` method of the `Chain`. This will cause any functions that are decorated with `@static_forward` in the define-by-run code to be added to a forward-pass static schedule object. Likewise, when the first backward pass is run (such as by calling `loss.backward()` on the output `Variable` of the model), any functions that are decorated with `@static_backward` will be added to a corresponding backward static schedule. Although it is not implemented yet, we can also consider creating a static schedule for the optimizer and parameter updates on the static graph.

It is important to note that memory for the various activation and parameter arrays of the static graph should be statically allocated during the first iteration (that is, while the define-by-run code is executing). Therefore, code that allocates arrays, such as parameters and results arrays, should **not** be placed inside functions decorated with `@static_forward`/`@static_backward`. To help enforce this, we do not even allow such decorated functions to return a value (other than the implicit `None`). Thus, if a layer such as a linear layer needs to return a results array `y`, that array will first need to be allocated in a function that is not decorated as static, such as the `forward()` method of `Function`. This will cause the allocation code to only be called during the first iteration. The (initially empty) results array `y` can then be passed as an argument to a function decorated by `@static_forward` that actually performs the linear computation and writes the results into the supplied `y`. Note that since a function decorated by `@static_forward` is not allowed to return a result value, any memory that is allocated inside the function will not be able to persist to the next iteration. Thus, we require that any results be "returned" by writing the values into one or more of the already-allocated "input" arrays to the function.The general usage convention for this feature thus involves first performing any necessary setup and initialization of arrays in code that is not decorated as static and then placing all of the code that needs to execute each iteration (and that reuses the statically-allocated arrays) in functions decorated as static. For a concrete example of all this, please refer to the `linear.py` file in the `functions` directory.

If a performance library such as MKL-DNN is used, any other necessary resources that need to be statically allocated can also occur during the first iteration.

Once the forward and backward pass of the first iteration has completed, the execution mode automatically changes to static scheduling for the static graph. Specifically, starting from the second iteration, calling the  `__call__()` method of the `Chain` that was decorated with `@static_graph` will no longer execute the define-by-run code inside the method. Instead, the `@static_graph` decorator will effectively cause the method to call a `Function` object that executes the forward static schedule. The `backward()` method of this function object implements the backward static schedule so that the correct behavior occurs when calling `loss.backward()` on the output `Variable`. Thus, we can see that the `@static_graph` decorator causes the code in its `__call__()` method (that is, the define-by-run code) to execute during the first iteration and create a corresponding static schedule. Then, starting from the second iteration, this decorator basically causes the code inside of `__call__()` to be replaced with a `Function` object that implements the forward and backward static schedules. Note that this change in execution mode is automatic and invisible to the user, since the user only has to apply the decorator. To understand the code, it may be helpful to see it in action by stepping through the code in a debugger.

Since most of the redundant code is Python code that performs dynamic checking and graph construction (which is redundant for a static graph) and most or all of the necessary non-redundant deep learning computations can potentially be optimized using MKL-DNN or CUDA, this method of optimization provides two key benefits:

* Can be used to speedup Chainer models without introducing external dependencies (although we still might want external dependencies such as CUDA or MKL-DNN for performance reasons). The optimized static graph implementation can call the same Chainer functions as before, but without the Python overhead that is needed for dynamic graphs. Actually, recall that we do still have such overhead during the first iteration only.
* With some further work, it is expected that this feature can potentially support deployment to C or C++ for platforms that do not support Python. For example, we could consider adding "export to C/C++" capability to the `StaticScheduleFunction` in `static_graph.py` that would export the static schedules (and possibly also the activation and parameter arrays) C or C++ code that executes the schedule.

### Other notes

Chainer code that only ever runs during the first iteration does not need to be optimized since it will not be needed again after the first iteration completes. Examples of such code include the various initializers that set the initial parameter values.

#### Limitations

* Currently, this is a proof-of-concept only, and so only the linear link and function is supported in models. Both forward and backward passes are supported, but the backward pass has some limitations and does not yet support general graphs.

* An assumption is that if the first iteration completes successfully, then it is safe to switch to an optimized static schedule for subsequent iterations. However, we must be careful that there are no bugs in the framework code to support this, since any bugs could cause very confusing behavior for users.

* The user needs to be careful that they understand what the `@static_graph` decorator does so that they do not get unexpected results. For example, they need to understand that using this decorator causes the corresponding code in the function to only execute once, not once per iteration. They should think of it as containing define-by-run code that builds a static graph, and therefore only needs to be called once.

* The current Chainer functions are not yet well-optimized for multi-core CPU, and so even if this feature is used to reduce the Python interpreter overhead, the current implementations will still lead to poor performance in many cases. An effort is underway to optimize for Intel CPUs at [Intel Chainer](https://github.com/intel/chainer) and hopefully combining this method with optimized MKL-DNN versions of the Chainer functions could lead to improved performance on many-core CPUs.

* This feature is not yet supported in the optimizers, but should be conceptually straightforward to add (no additional user-facing API changes should be necessary).

* If the computation graph is actually dynamic, this feature cannot be used.

* The static schedule that is currently created corresponds to the functions being called in the same order that they were called in the imperative define-by-run code. This was chosen because of the straightforward implementation. However, Chainer actually already creates the full computation dataflow graph during the first iteration. Using this graph, it seems possible in principle to perform more sophisticated kinds of optimizations, such as kernel fusion, exploiting spatial parallelism, distributed execution, etc., which are already supported in some other static-graph frameworks. As future work, it could be interesting to also consider some of these optimizations.
