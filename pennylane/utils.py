# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utilities
=========

**Module name:** :mod:`pennylane.utils`

.. currentmodule:: pennylane.utils

This module contains utilities and auxiliary functions, which are shared
across the PennyLane submodules.

.. raw:: html

    <h3>Summary</h3>

.. autosummary::
    flatten
    _unflatten
    unflatten

.. raw:: html

    <h3>Code details</h3>
"""
import collections
import numbers

import autograd.numpy as np

from .variable  import Variable


def flatten(x):
    """Iterate through an arbitrarily nested structure, flattening it in depth-first order.

    See also :func:`_unflatten`.

    Args:
        x (array, Iterable, other): each element of the Iterable may itself be an iterable object

    Yields:
        other: elements of x in depth-first order
    """
    if isinstance(x, np.ndarray):
        yield from flatten(x.flat)
    elif isinstance(x, collections.Iterable) and not isinstance(x, (str, bytes)):
        for item in x:
            yield from flatten(item)
    else:
        yield x


def _unflatten(flat, model):
    """Restores an arbitrary nested structure to a flattened iterable.

    See also :func:`flatten`.

    Args:
        flat (array): 1D array of items
        model (array, Iterable, Number): model nested structure

    Returns:
        (other, array): first elements of flat arranged into the nested
        structure of model, unused elements of flat
    """
    print("_unflatten("+str(flat)+", "+str(model)+")")
    # if isinstance(model, np.ndarray) and model.shape == ():
    #     model = (float)(model.item())
    if isinstance(model, np.ndarray) and model.shape == ():
        print("branch0: x")
        print("returning0: "+str(flat[0]))
        return flat[0], flat[1:]
    elif isinstance(model, np.ndarray) and model.dtype != object and model.shape != ():
        print("branch1: isinstance(model, np.ndarray) and model.dtype != object")
        idx = model.size
        res = np.array(flat)[:idx].reshape(model.shape)
        # if res.shape==():#not isinstance(res.item(), Variable) and
        #     print("type(res.item())="+str(type(res.item())))
        #     res = res.item()#(float)(res)
        print("returning1: "+str(res))
        return res, flat[idx:]
    #elif isinstance(model, np.ndarray) and model.shape == ():
    #    print("isinstance(model, np.ndarray) and model.shape == ()")
    #    return _unflatten(flat, model.item())
    elif isinstance(model, collections.Iterable) and (not isinstance(model, np.ndarray) or model.shape != ()):
        print("branch2: isinstance(model, collections.Iterable)")
        res = []
        for x in model:
            val, flat = _unflatten(flat, x)
            if isinstance(x, np.ndarray) and x.shape != () and not isinstance(model, tuple):
                print("Got_here: x="+str(x)+" model="+str(model))
                val = np.array(val)
            print("Appending: "+str(val))
            res.append(val)
        #if isinstance(model, np.ndarray):
            #print("isinstance(model, np.ndarray)")
            #res = np.array(res)
        print("returning2: "+str(res))
        return res, flat
    elif isinstance(model, (numbers.Number, Variable)):
        print("branch3: x")
        print("returning3: "+str(flat[0] if isinstance(model, Variable) or isinstance(flat[0], Variable) else type(model)(flat[0])))
        return flat[0] if isinstance(model, Variable) or isinstance(flat[0], Variable) else type(model)(flat[0]), flat[1:]
    else:
        raise TypeError('Unsupported type in the model: {}'.format(type(model)))


def unflatten(flat, model):
    """Wrapper for :func:`_unflatten`.
    """
    # pylint:disable=len-as-condition
    res, tail = _unflatten(np.asarray(flat), model)
    if len(tail) != 0:
        raise ValueError('Flattened iterable has more elements than the model.')
    return res
