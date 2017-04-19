
#### Demonstration of  "static define-by-run" feature

This is prototype code that shows how Chainer can be refactored to support both the user convenience of define-by-run and the runtime performance of a static computation graph. It is a hybrid "define-by-run"-"static graph" feature that we call "static define-by-run." This is a partial implementation that is only intended to provide a single concrete example to illustrate the idea.  

This feature intends to support optimized execution of Chainer models, as long as the model (or part of the model) corresponds to a static computation graph. This feature can potentially also support exporting a Chainer model to C/C++ code that can be executed on platforms that do not support Python (although this is not implemented yet, but seems conceptually straightforward).

The current code is limited to supporting only models using the Linear link and/or function, since the code is only intended to illustrate the user-facing API and show an concrete example of how to modify Chainer to support such a feature. It will still require some further work to actually create optimized Chainer functions (such as by using MKL-DNN), as well as refactoring the many existing links and functions to be compatible with the requirements of this feature. The current code shows that, at least for the case of Linear link, that the only change to the public API that is needed is for the user to decorate the `__call__()` method of each chain that contains a static sub-graph.

### What it does

This feature can be used to speedup execution of a Chainer model as long as the model corresponds to a static computation graph or contains static sub-graphs. Chainer can be considered a "dynamic graph deep learning framework" that supports "define-by-run", which is a form of automatic differentiation. This makes it easy for the user to define a model by writing Python code that imperatively executes the forward-pass commutations. This is convenient for describing computation graphs in which the graph changes from iteration to iteration.

It seems, however,  that the majority of the currently popular models correspond to either a static graph or a mostly static graph. The assumption of a static computation graph is made by other frameworks such as Tensorflow. If the graph can be assumed to be static, many optimizations then become possible. This current code only performs the most basic of optimizations, which consists of removing most dynamic type checking, using statically-allocated arrays, and using a static schedule. However, these kinds of optimizations could still be sufficient to remove the most significant performance bottlenecks observed in Chainer models. It is expected that this approach could be very effective in reducing Python interpreter overhead from Chainer models, which has been particularly challenging to optimize when a dynamic graph must be assumed.

Another advantage of dynamic graph frameworks is that they can support easier debugging because if there is an error in the model code, it will often cause the Python interpreter to throw an exception and print a stack trace on or soon after the line of code with the bug. A static graph framework might instead first optimize the graph, and finding the bug in optimized code might be considerably more difficult.

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

To use this feature, the computation graph must be static. Even if the complete Chainer model is not static, this feature can still be used on a static sub-graphs in the computation graph. The complete computation graph should first be partitioned into static sub-graphs that are as large as possible. The user should then write a chain the computes the forward pass of the static graph in its `__call()__` method and also decorate the method with the `@static_graph` decorator.

It is important to note that this decorated `__call__()` method will only be called once during execution of the model. Therefore, the user must be careful not place any code inside this method other than the define-by-run code. For example, Python code that increments a counter variable or prints to stdout would then only be called once during execution of the model.



#### How it works

See the code for details. In particular, see `static_graph.py` which implements this feature, and also see the slightly modified Chainer linear link and function in the subfolders. Note that the test model uses these modified links and functions, not the Chainer versions (notice the different imports).

Basically, the performance optimization is achieved by disabling redundant code. Since most or all of the redundant code is Python and most or all of the necessary code can potentially be optimized using MKL-DNN or CUDA, this method of optimization provides two key benefits:

* Can be used to speedup Chainer models without introducing external dependencies. The optimized static graph implementation calls the same Chainer functions as before, but without the Python overhead that is needed for dynamic graphs. Actually, recall that we do still have such overhead during the first iteration only.
* With some further work, it is expected that this feature can potentially support deployment C or C++ for platforms that do not support Python.

### Other notes

Chainer code that only ever runs during the first iteration does not need to be optimized since it will not be needed again after the first iteration completes. Examples of such code include the various initializers that set the initial parameter values.

#### Limitations

* Currently, since this is a proof-of-concept only. The only link that is supported is the linear link. Both forward and backward passes are supported. However, the static schedule for the optimizer is not yet implemented. Also, the backward pass has some limitations and does not yet support general graphs (`Variable.py` of Chainer will also need to be slightly modified).

* An assumption is that if the first iteration completes successfully, then it is safe to switch to an optimized static schedule for subsequent iterations. However, we must be careful that there are no bugs in the framework code to support this, since any bugs could cause very confusing behavior for users.

* The user also needs to be careful that they understand what the `@static_graph` decorator does so that they do not get unexpected results. For example, they need to understand that using this decorator causes the corresponding code in the function to only execute once, not once per iteration.

* The current Chainer functions are not yet well-optimized for CPU, and so even if the various other Chainer links and functions are modified to use this feature, the removal of the Python interpreter overhead will likely be replaced by poorly-optimized Numpy code. An effort is underway to optimize for Intel CPUs at [Intel Chainer](https://github.com/intel/chainer) and hopefully combining this method with optimized MKL-DNN versions of the Chainer functions could lead to good performance on many-core CPUs.

* This feature is not yet supported in the optimizers, but should be conceptually straightforward to add (no additional user-facing API changes should be necessary).

* The static schedule that is currently created corresponds to the functions being called in the same order that they were called in the imperative define-by-run code. This was chosen because if the straightforward implementation. However, Chainer actuall already creates the full computation dataflow graph during the first iteration. Using this graph, it seems possible in principle to perform more sophisticated optimizations such as kernel fusion, exploiting spatial parallelism, distributed execution, etc., which are already supported in some other static-graph frameworks. As future work, it could be interesting to also consider some of these optimizations.
