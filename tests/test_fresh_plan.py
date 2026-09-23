import subprocess
import sys
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import xarray as xr

from xarray_ms import plan_fresh_msv2
from xarray_ms.errors import (
  FreshMSv2TargetError,
  FreshMSv2ValidationError,
  MeasureReferenceColumnRequired,
)

DIMS = ("time", "baseline_id", "frequency", "polarization")
COORDS = {
  "time": [1, 2],
  "baseline_id": [0],
  "frequency": [10, 20],
  "polarization": [0, 1],
  "uvw_label": ["u", "v", "w"],
  "baseline_antenna1_name": ("baseline_id", ["A"]),
  "baseline_antenna2_name": ("baseline_id", ["B"]),
  "field_name": ("time", ["field", "field"]),
  "scan_name": ("time", ["scan", "scan"]),
}


def make_tree(paths=("/part",), extra=False):
  datasets = {}
  for path in paths:
    groups = {
      "base": {
        "correlated_data": "VISIBILITY",
        "flag": "FLAG",
        "weight": "WEIGHT",
        "uvw": "UVW",
        "field_and_source": "field_and_source_base_xds",
      }
    }
    variables = {
      "VISIBILITY": (DIMS, np.ones((2, 1, 2, 2), dtype=np.complex64)),
      "FLAG": (DIMS, np.zeros((2, 1, 2, 2), dtype=bool)),
      "WEIGHT": (DIMS, np.ones((2, 1, 2, 2))),
      "UVW": (
        ("time", "baseline_id", "uvw_label"),
        np.zeros((2, 1, 3)),
        {"type": "uvw", "units": "m", "frame": "fk5"},
      ),
    }
    if extra:
      variables["CORRECTED"] = (DIMS, np.ones((2, 1, 2, 2), dtype=np.complex64))
      groups["corrected"] = {"correlated_data": "CORRECTED"}
    datasets[path] = xr.Dataset(
      variables,
      coords={
        **COORDS,
        "time": (
          "time",
          [1, 2],
          {"type": "time", "units": "s", "format": "unix", "scale": "utc"},
        ),
        "frequency": (
          "frequency",
          [10, 20],
          {"type": "spectral_coord", "units": "Hz", "observer": "TOPO"},
        ),
      },
      attrs={
        "type": "visibility",
        "data_groups": groups,
        "observation_info": {},
        "processor_info": {},
      },
    )
    datasets[f"{path}/field_and_source_base_xds"] = xr.Dataset(
      attrs={"type": "field_and_source"}
    )
    datasets[f"{path}/antenna_xds"] = xr.Dataset(attrs={"type": "antenna"})
  return xr.DataTree.from_dict(datasets)


def test_single_partition_is_pure_and_immutable(tmp_path):
  target = tmp_path / "new.ms"
  plan = make_tree().plan_msv2(target)
  assert plan.target == str(target)
  assert plan.data_group == "base"
  assert plan.visibility_mappings == (("base", "DATA"),)
  assert plan.partitions[0].path == "/part"
  assert plan.partitions[0].field_and_source == "/part/field_and_source_base_xds"
  assert plan.partitions[0].antenna == "/part/antenna_xds"
  assert plan.partitions[0].correlated_data == "VISIBILITY"
  assert [(m.column, m.frame) for m in plan.partitions[0].measures] == [
    ("MAIN::TIME", "UTC"),
    ("SPECTRAL_WINDOW::CHAN_FREQ", "TOPO"),
    ("MAIN::UVW", "J2000"),
  ]
  assert not target.exists()
  with pytest.raises(FrozenInstanceError):
    plan.target = "elsewhere"
  with pytest.raises(FrozenInstanceError):
    plan.partitions[0].path = "elsewhere"


def test_processing_set_shaped_tree_has_deterministic_partition_order(tmp_path):
  tree = make_tree(("/processing_set/b/part", "/processing_set/a/part"))
  plan = plan_fresh_msv2(tree, tmp_path / "new.ms")
  assert tuple(p.path for p in plan.partitions) == (
    "/processing_set/a/part",
    "/processing_set/b/part",
  )


def test_cross_partition_reference_conflict_requires_column(tmp_path):
  tree = make_tree(("/one", "/two"))
  tree["two"].ds["UVW"].attrs["frame"] = "icrs"
  with pytest.raises(
    MeasureReferenceColumnRequired, match=r"/one.*UVW.*/two.*UVW.*reference column"
  ):
    plan_fresh_msv2(tree, tmp_path / "fresh.ms")
  assert not (tmp_path / "fresh.ms").exists()


def test_optional_metadata_measures_are_encoded(tmp_path):
  tree = make_tree()
  tree["part/antenna_xds"].ds = xr.Dataset(
    {
      "ANTENNA_POSITION": (
        ("antenna", "cartesian"),
        np.zeros((1, 3)),
        {"type": "location", "units": "m", "frame": "ITRS"},
      )
    },
    attrs={"type": "antenna"},
  )
  tree["part/field_and_source_base_xds"].ds = xr.Dataset(
    {
      "FIELD_PHASE_CENTER_DIRECTION": (
        ("field", "sky"),
        np.zeros((1, 2)),
        {"type": "sky_coord", "units": "rad", "frame": "icrs"},
      )
    },
    attrs={"type": "field_and_source"},
  )
  measures = plan_fresh_msv2(tree, tmp_path / "fresh.ms").partitions[0].measures
  assert [(m.column, m.frame) for m in measures[-2:]] == [
    ("ANTENNA::POSITION", "ITRF"),
    ("FIELD::PHASE_DIR", "ICRS"),
  ]


def test_field_reference_root_relative_and_absolute(tmp_path):
  tree = make_tree(("/processing_set/part",))
  group = tree["/processing_set/part"].attrs["data_groups"]["base"]
  group["field_and_source"] = "processing_set/part/field_and_source_base_xds"
  assert plan_fresh_msv2(tree, tmp_path / "new.ms").partitions[0].field_and_source == (
    "/processing_set/part/field_and_source_base_xds"
  )
  group["field_and_source"] = "/processing_set/part/field_and_source_base_xds"
  assert plan_fresh_msv2(tree, tmp_path / "new.ms").partitions[0].field_and_source == (
    "/processing_set/part/field_and_source_base_xds"
  )


def test_existing_target_and_invalid_target(tmp_path):
  with pytest.raises(FreshMSv2TargetError, match="already exists"):
    plan_fresh_msv2(make_tree(), tmp_path)
  with pytest.raises(FreshMSv2TargetError, match="nonempty"):
    plan_fresh_msv2(make_tree(), "")
  with pytest.raises(FreshMSv2TargetError, match="nonempty"):
    plan_fresh_msv2(make_tree(), 123)
  with pytest.raises(FreshMSv2TargetError, match="parent is missing"):
    plan_fresh_msv2(make_tree(), tmp_path / "missing" / "new.ms")
  parent_file = tmp_path / "file"
  parent_file.write_text("not a directory")
  with pytest.raises(FreshMSv2TargetError, match="not a directory"):
    plan_fresh_msv2(make_tree(), parent_file / "new.ms")


def test_missing_correlated_dataset_group_role_and_reference(tmp_path):
  target = tmp_path / "new.ms"
  with pytest.raises(FreshMSv2ValidationError, match="No correlated datasets"):
    plan_fresh_msv2(xr.DataTree(), target)
  tree = make_tree()
  with pytest.raises(FreshMSv2ValidationError, match="no group"):
    plan_fresh_msv2(tree, target, data_group="missing")
  del tree["part"].attrs["data_groups"]["base"]["weight"]
  with pytest.raises(FreshMSv2ValidationError, match="lacks role 'weight'"):
    plan_fresh_msv2(tree, target)
  tree = make_tree()
  tree["part"].attrs["data_groups"]["base"]["flag"] = "MISSING"
  with pytest.raises(FreshMSv2ValidationError, match="missing data variable"):
    plan_fresh_msv2(tree, target)
  tree = make_tree()
  tree["part"].attrs["data_groups"]["base"]["field_and_source"] = "missing"
  with pytest.raises(FreshMSv2ValidationError, match="node 'missing' does not exist"):
    plan_fresh_msv2(tree, target)


def test_canonical_coordinates_dimensions_and_dtype(tmp_path):
  target = tmp_path / "new.ms"
  tree = make_tree()
  tree["part"].ds = tree["part"].ds.drop_vars("frequency")
  with pytest.raises(FreshMSv2ValidationError, match="coordinate 'frequency'"):
    plan_fresh_msv2(tree, target)
  for coord in (
    "baseline_antenna1_name",
    "baseline_antenna2_name",
    "field_name",
    "scan_name",
  ):
    tree = make_tree()
    tree["part"].ds = tree["part"].ds.drop_vars(coord)
    with pytest.raises(FreshMSv2ValidationError, match=coord):
      plan_fresh_msv2(tree, target)
  tree = make_tree()
  ds = tree["part"].to_dataset().drop_vars("scan_name")
  ds = ds.assign_coords(scan_name=("baseline_id", ["scan"]))
  tree["part"].ds = ds
  with pytest.raises(FreshMSv2ValidationError, match="scan_name.*time"):
    plan_fresh_msv2(tree, target)
  tree = make_tree()
  ds = tree["part"].to_dataset()
  ds["VISIBILITY"] = ds.VISIBILITY.transpose(
    "frequency", "time", "baseline_id", "polarization"
  )
  tree["part"].ds = ds
  with pytest.raises(
    FreshMSv2ValidationError, match="base visibility requires dimensions"
  ):
    plan_fresh_msv2(tree, target)
  tree = make_tree()
  ds = tree["part"].to_dataset()
  ds["VISIBILITY"] = ds.VISIBILITY.real
  tree["part"].ds = ds
  with pytest.raises(FreshMSv2ValidationError, match="complex64 or complex128 dtype"):
    plan_fresh_msv2(tree, target)


def test_additional_visibility_and_mapping_validation(tmp_path):
  target = tmp_path / "new.ms"
  tree = make_tree(extra=True)
  plan = plan_fresh_msv2(
    tree, target, additional_visibility={"corrected": "corrected_data"}
  )
  assert plan.visibility_mappings == (("base", "DATA"), ("corrected", "corrected_data"))
  assert plan.partitions[0].additional_correlated_data == (("corrected", "CORRECTED"),)
  for mappings, message in (
    ({"corrected": "DATA"}, "writer-controlled"),
    ({"corrected": "flag"}, "writer-controlled"),
    ({"corrected": "UVW"}, "writer-controlled"),
    ({"corrected": "FLAG"}, "writer-controlled"),
    ({"corrected": "WEIGHT_SPECTRUM"}, "writer-controlled"),
    ({"corrected": "corrected_data", "absent": "OTHER_DATA"}, "no group"),
  ):
    with pytest.raises(FreshMSv2ValidationError, match=message):
      plan_fresh_msv2(tree, target, additional_visibility=mappings)
  ds = tree["part"].to_dataset()
  ds["CORRECTED"] = ds.CORRECTED.real
  tree["part"].ds = ds
  with pytest.raises(FreshMSv2ValidationError, match="must match the base"):
    plan_fresh_msv2(tree, target, additional_visibility={"corrected": "CORRECTED_DATA"})


def test_additional_destination_collision(tmp_path):
  tree = make_tree(extra=True)
  tree["part"].attrs["data_groups"]["other"] = {
    **tree["part"].attrs["data_groups"]["corrected"]
  }
  with pytest.raises(FreshMSv2ValidationError, match="duplicated"):
    plan_fresh_msv2(
      tree,
      tmp_path / "new.ms",
      additional_visibility={"corrected": "MODEL_DATA", "other": "model_data"},
    )


def test_missing_additional_visibility_reference_and_invalid_mapping_name(tmp_path):
  tree = make_tree(extra=True)
  target = tmp_path / "new.ms"
  with pytest.raises(FreshMSv2ValidationError, match="names must be strings"):
    plan_fresh_msv2(tree, target, additional_visibility={1: "MODEL_DATA"})
  tree["part"].attrs["data_groups"]["corrected"]["correlated_data"] = "MISSING"
  with pytest.raises(FreshMSv2ValidationError, match="missing data variable"):
    plan_fresh_msv2(tree, target, additional_visibility={"corrected": "MODEL_DATA"})
  assert not target.exists()


def test_required_metadata_nodes_and_field_type(tmp_path):
  target = tmp_path / "new.ms"
  for name in ("observation_info", "processor_info"):
    tree = make_tree()
    tree["part"].attrs.pop(name)
    with pytest.raises(FreshMSv2ValidationError, match=name):
      plan_fresh_msv2(tree, target)
    tree["part"].attrs[name] = "not a mapping"
    with pytest.raises(FreshMSv2ValidationError, match=name):
      plan_fresh_msv2(tree, target)
  tree = make_tree()
  del tree["part"]["antenna_xds"]
  with pytest.raises(FreshMSv2ValidationError, match="antenna_xds"):
    plan_fresh_msv2(tree, target)
  tree = make_tree()
  tree["part/antenna_xds"].attrs["type"] = "other"
  with pytest.raises(FreshMSv2ValidationError, match="antenna_xds"):
    plan_fresh_msv2(tree, target)
  tree = make_tree()
  tree["part/field_and_source_base_xds"].attrs["type"] = "other"
  with pytest.raises(FreshMSv2ValidationError, match="invalid type"):
    plan_fresh_msv2(tree, target)
  tree["part/field_and_source_base_xds"].attrs["type"] = "field_and_source_ephemeris"
  with pytest.raises(FreshMSv2ValidationError, match="invalid type"):
    plan_fresh_msv2(tree, target)


@pytest.mark.parametrize(
  ("role", "replacement"),
  [
    ("FLAG", lambda ds: ds.FLAG.astype("int8")),
    (
      "FLAG",
      lambda ds: ds.FLAG.transpose("frequency", "time", "baseline_id", "polarization"),
    ),
    ("WEIGHT", lambda ds: ds.WEIGHT.astype("int64")),
    (
      "WEIGHT",
      lambda ds: ds.WEIGHT.transpose(
        "frequency", "time", "baseline_id", "polarization"
      ),
    ),
    ("UVW", lambda ds: ds.UVW.astype("complex64")),
    ("UVW", lambda ds: ds.UVW.transpose("uvw_label", "time", "baseline_id")),
  ],
)
def test_malformed_base_roles(tmp_path, role, replacement):
  tree = make_tree()
  ds = tree["part"].to_dataset()
  ds[role] = replacement(ds)
  tree["part"].ds = ds
  with pytest.raises(FreshMSv2ValidationError, match=role):
    plan_fresh_msv2(tree, tmp_path / "new.ms")


@pytest.mark.parametrize(
  ("role", "dtype", "accepted"),
  [
    ("VISIBILITY", "complex64", True),
    ("VISIBILITY", "complex128", True),
    ("VISIBILITY", "clongdouble", False),
    ("FLAG", "bool", True),
    ("FLAG", "uint8", True),
    ("FLAG", "int8", False),
    ("FLAG", "uint16", False),
    ("WEIGHT", "float32", True),
    ("WEIGHT", "float64", True),
    ("WEIGHT", "float16", False),
    ("WEIGHT", "complex64", False),
    ("UVW", "float32", True),
    ("UVW", "float64", True),
    ("UVW", "float16", False),
    ("UVW", "complex64", False),
  ],
)
def test_base_role_supported_dtypes(tmp_path, role, dtype, accepted):
  if dtype == "clongdouble" and np.dtype(dtype) == np.dtype("complex128"):
    pytest.skip("platform long complex has complex128 precision")
  tree = make_tree()
  ds = tree["part"].to_dataset()
  ds[role] = ds[role].astype(dtype)
  tree["part"].ds = ds
  if accepted:
    assert plan_fresh_msv2(tree, tmp_path / "new.ms").partitions[0].path == "/part"
  else:
    with pytest.raises(
      FreshMSv2ValidationError,
      match=role if role != "VISIBILITY" else "base visibility",
    ):
      plan_fresh_msv2(tree, tmp_path / "new.ms")


def test_real_backend_tree_can_be_planned_without_creation(simmed_ms, tmp_path):
  target = tmp_path / "fresh.ms"
  with xr.open_datatree(simmed_ms, auto_corrs=True) as tree:
    plan = tree.plan_msv2(target)
    assert plan.target == str(target)
    assert plan.partitions
    assert all(p.correlated_data == "VISIBILITY" for p in plan.partitions)
  assert not target.exists()


def test_missing_uvw_label_coordinate(tmp_path):
  tree = make_tree()
  tree["part"].ds = tree["part"].ds.drop_vars("uvw_label")
  with pytest.raises(FreshMSv2ValidationError, match="uvw_label"):
    plan_fresh_msv2(tree, tmp_path / "new.ms")


def test_lazy_visibility_payloads_are_not_computed(tmp_path):
  import dask.array as da
  from dask import delayed

  @delayed
  def fail_on_compute():
    raise AssertionError("planner computed visibility data")

  tree = make_tree(extra=True)
  ds = tree["part"].to_dataset()
  lazy = da.from_delayed(fail_on_compute(), shape=(2, 1, 2, 2), dtype=np.complex64)
  ds["VISIBILITY"] = (DIMS, lazy)
  ds["CORRECTED"] = (DIMS, lazy)
  tree["part"].ds = ds
  plan = plan_fresh_msv2(
    tree, tmp_path / "new.ms", additional_visibility={"corrected": "MODEL_DATA"}
  )
  assert plan.partitions[0].additional_correlated_data == (("corrected", "CORRECTED"),)


def test_lazy_measure_payload_is_not_computed(tmp_path):
  import dask.array as da
  from dask import delayed

  @delayed
  def fail_on_compute():
    raise AssertionError("planner computed UVW measure data")

  tree = make_tree()
  ds = tree["part"].to_dataset()
  attrs = dict(ds.UVW.attrs)
  ds["UVW"] = (
    ("time", "baseline_id", "uvw_label"),
    da.from_delayed(fail_on_compute(), shape=(2, 1, 3), dtype=np.float64),
    attrs,
  )
  tree["part"].ds = ds
  plan = plan_fresh_msv2(tree, tmp_path / "fresh.ms")
  assert plan.partitions[0].measures[2].frame == "J2000"


def test_plan_method_survives_legacy_writer_import_failure():
  script = """
import builtins
original_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == 'xarray_ms.backend.msv2.writes':
        raise ImportError('simulated legacy writer failure')
    return original_import(name, *args, **kwargs)
builtins.__import__ = blocked_import
import xarray_ms
from xarray import DataTree
assert not xarray_ms.HAS_WRITE_SUPPORT
assert callable(DataTree.plan_msv2)
assert callable(xarray_ms.plan_fresh_msv2)
"""
  result = subprocess.run(
    [sys.executable, "-c", script], capture_output=True, text=True
  )
  assert result.returncode == 0, result.stderr
