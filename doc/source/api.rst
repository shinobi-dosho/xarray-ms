API
===

Opening Measurement Sets
------------------------

The standard :func:`xarray.open_datatree` method should
be used to open a :class:`~xarray.DataTree` interface
to the underlying Measurement Set data.

.. code-block:: python

    >>> datatree = xarray.open_datatree("/data/data.ms", partition_schema=["FIELD_ID"])

These methods defer to the relevant methods on the
`Entrypoint Class <entrypoint-class_>`_.
Consult the method signatures for information on extra
arguments that can be passed.

Creating a fresh Measurement Set
--------------------------------

Call :func:`xarray_ms.plan_fresh_msv2` to validate an MSv4 DataTree and
resolve its visibility destinations and metadata without creating tables.
Pass the resulting plan to :func:`xarray_ms.create_fresh_msv2` to create
and verify a new MSv2 skeleton with zero MAIN rows. Creation does not
replace an existing target, including one created concurrently; invalid
input raises :class:`~xarray_ms.errors.FreshMSv2ValidationError`, an
existing target raises :class:`~xarray_ms.errors.FreshMSv2TargetError`,
and verification failure does not publish the staging table.

.. autofunction:: xarray_ms.plan_fresh_msv2

.. autofunction:: xarray_ms.create_fresh_msv2


.. _entrypoint-class:

Entrypoint Class
----------------

Entrypoint class for the MSv2 backend.

.. autoclass:: xarray_ms.backend.msv2.entrypoint.MSv2EntryPoint
    :members: open_datatree, open_dataset

.. _partitioning-schema:
