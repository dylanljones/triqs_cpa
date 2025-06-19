# -*- coding: utf-8 -*-
# Author: Dylan Jones
# Date:   2025-06-13

from typing import List, Tuple, Union

import numpy as np
from numpy.typing import DTypeLike
from triqs.gf import BlockGf, Gf, MeshImFreq, MeshImTime, MeshReFreq, MeshReTime

GfStruct = List[Tuple[str, int]]
GfLike = Union[Gf, BlockGf]
MeshLike = Union[MeshReFreq, MeshImFreq, MeshReTime, MeshImTime]


def blockgf(
    mesh: MeshLike,
    names: List[str] = None,
    indices: List[int] = None,
    target_shape: List[int] = None,
    gf_struct: GfStruct = None,
    target_gf: BlockGf = None,
    blocks: Union[GfLike, List[GfLike]] = None,
    copy: bool = False,
    name: str = "G",
) -> BlockGf:
    """Create an empty or filled BlockGf.

    Parameters
    ----------
    mesh : MeshReFreq or MeshImFreq
        The mesh for the Green's functions.
    names : List[str], optional
        The names of the Green's functions.
    indices : List[int], optional
        The indices defining the target shape of the Green's functions. Either `indices` or
        `target_shape` has to be provided if no blocks are given.
    target_shape : List[int], optional
        The target shape of the Green's functions. Either `indices` or `target_shape` has to be
         provided if no blocks are given.
    gf_struct : GfStruct, optional
        The structure of the Green's functions. If provided, `names` and `indices` and
        `target_shape` are ignored.
    target_gf: BlockGf, optional
        Other block Green's function to copy the structure from. If provided, `names`, `indices` and
        `target_shape` are ignored.
    blocks : Gf or BlockGf or List[Gf] or List[BlockGf], optional
        The blocks used to fill the BlockGf. If a single Gf or BlockGf is given is used for all
        blocks of the resulting BlockGf.
    copy : bool, optional
        If True, the blocks are copied, otherwise they are used as is. Only applies if
        all blocks are passed.
    name : str, optional
        The name of the BlockGf.

    Returns
    -------
    BlockGf
        The new BlockGf object.
    """
    if target_gf is not None:
        names = list(target_gf.indices)
        target_shape = list(target_gf[names[0]].target_shape)
    elif gf_struct is not None:
        names = [name for name, _ in gf_struct]
        norbs = np.unique([norbs for _, norbs in gf_struct])
        assert len(norbs) == 1, "All Gfs must have the same number of orbitals."
        indices = list(range(norbs[0]))
    else:
        if names is None:
            raise ValueError("Either gf_struct or names must be provided!")
        if blocks is None and target_shape is None and indices is None:
            raise ValueError("Either gf_struct, blocks, target_shape or indices must be provided!")

    if blocks is None:
        copy = True  # Force copy
        blocks = [Gf(mesh=mesh, indices=indices, target_shape=target_shape) for _ in names]
    elif isinstance(blocks, (Gf, BlockGf)):
        copy = True  # Force copy
        blocks = [blocks] * len(names)

    return BlockGf(name_list=names, block_list=blocks, make_copies=copy, name=name)


def toarray(obj: Union[MeshLike, GfLike], dtype: DTypeLike = None) -> np.ndarray:
    """Converts a supported TRIQS object to an array.

    Currently, TRIQS Mesh and Gf objects are supported.

    Parameters
    ----------
    obj : Union[MeshLike, GfLike]
        The object to convert to an array. If the object is a TRIQS mesh or `Gf` the data
        of the object will be returned directly as an array. If the object is a (nested)
        TRIQS `BlockGf`, the data of all leaf Gfs will be returned as an array with a shape
        corresponding to the block sturcture and the leaf shape.
        *Note*: All leafs of the (nested) BlockGf need to have the same shape.
    dtype : DTypeLike, optional
        Data type to use for creating the array. If no dtpye is given the original
        datatype of the TRIQS object ios used.

    Returns
    -------
    np.ndarray
        The TRIQS object as numpy array.
    """
    if isinstance(obj, (MeshReFreq, MeshImFreq, MeshImTime, MeshReTime)):
        data = list(obj.values())
    elif isinstance(obj, Gf):
        data = obj.data
    elif isinstance(obj, BlockGf):

        def _bgf2arr(g: BlockGf) -> list:
            """Recursively convert the BlockGf to an array."""
            _data = list()
            for k in g.indices:
                _data.append(_bgf2arr(g[k]) if isinstance(g[k], BlockGf) else g[k].data)
            return _data

        data = _bgf2arr(obj)

    else:
        raise TypeError(f"Unsupported object type: {type(obj)}")

    return np.asarray(data, dtype=dtype)


def gf_shape(gf: GfLike) -> Tuple[int, ...]:
    """Return the shape of a Gf or (nested) BlockGf.

    If a BlockGf is provided, the shape of the block children is checked and an error
    is raised if the shapes do not match.
    """
    if isinstance(gf, Gf):
        return gf.data.shape
    else:
        queue = [(gf, [])]
        shapes = set()
        while queue:
            parent, shape = queue.pop(0)
            if isinstance(parent, Gf):
                shapes.add(tuple(shape) + parent.data.shape)
            else:
                for k, g in parent:
                    queue.append((g, shape + [len(parent)]))

    if len(shapes) > 1:
        raise ValueError("Shapes of the blocks do not match!")
    return shapes.pop()


def fill_gf(gf: GfLike, data: np.ndarray) -> None:
    """Fill a Gf or (nested) BlockGf with data.

    Parameters
    ----------
    gf : Gf or BlockGf
        The Gf or BlockGf to be filled.
    data : np.ndarray
        The data to be filled into the Gf. The shape must match the shape of the Gf.

    """
    # Check if the data shape matches the Gf shape
    shape = gf_shape(gf)
    if len(data.shape) < len(shape):
        data = data[..., np.newaxis, np.newaxis]
    elif len(data.shape) > len(shape) and data.shape[-1] == 1:
        data = data[..., 0, 0]

    if shape[:-1] != data.shape[:-1]:
        raise ValueError(f"Shape {data.shape} does not match shape of Gf {shape}!")

    if isinstance(gf, Gf):
        gf.data[...] = data[...]
    else:

        def _fill_block(g: BlockGf, d: np.ndarray) -> None:
            for k, v in zip(g.indices, d):
                if isinstance(g[k], BlockGf):
                    _fill_block(g[k], v)
                else:
                    g[k].data[...] = v[...]

        _fill_block(gf, data)
