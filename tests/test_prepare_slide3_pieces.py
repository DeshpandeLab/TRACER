from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GBM = ROOT / "tutorials" / "gbm"
if str(GBM) not in sys.path:
    sys.path.insert(0, str(GBM))

import prepare_slide3_pieces as slide3


def _component(
    sample_name,
    component_id,
    assigned_label,
    *,
    qc_status="major",
    cell_count=10,
):
    return slide3.Component(
        sample_name=sample_name,
        raw_component_id=assigned_label,
        component_id=component_id,
        bin_size_um=50.0,
        cell_count=cell_count,
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        x_centroid=0.5,
        y_centroid=0.5,
        assigned_label=assigned_label,
        slide_tissue_id=slide3.SLIDE_TISSUE_IDS[assigned_label],
        qc_status=qc_status,
    )


def _clustered_cells(centers, points_per_cluster=4):
    rows = []
    offsets = np.linspace(-5.0, 5.0, points_per_cluster)
    for center_x, center_y in centers:
        for index, offset in enumerate(offsets):
            rows.append(
                {
                    "cell_id": f"cell_{len(rows)}",
                    "x_centroid": center_x + offset,
                    "y_centroid": center_y + offsets[-index - 1],
                }
            )
    return pd.DataFrame(rows)


def test_patient4_components_get_expected_labels_in_yx_order():
    centers = [
        (0, 0),
        (1000, 0),
        (2000, 0),
        (3000, 0),
        (0, 1000),
        (1000, 1000),
        (2000, 1000),
        (3000, 1000),
    ]
    cells = _clustered_cells(centers)

    components, messages = slide3.choose_component_resolution(
        cells,
        sample_name="Patient4",
        bin_sizes_um=(50.0, 75.0),
        min_component_cells=1,
        label_order="yx",
    )

    major = [component for component in components if component.qc_status == "major"]
    assert [component.assigned_label for component in major] == [
        1,
        2,
        4,
        3,
        6,
        5,
        7,
        8,
    ]
    assert [component.component_id for component in major] == [
        "Patient4_piece01",
        "Patient4_piece02",
        "Patient4_piece04",
        "Patient4_piece03",
        "Patient4_piece06",
        "Patient4_piece05",
        "Patient4_piece07",
        "Patient4_piece08",
    ]
    assert [component.slide_tissue_id for component in major] == [
        "DFCI4.S1.C4.L1",
        "DFCI4.S2.C4.L1",
        "DFCI4.S3.C4.L2",
        "DFCI4.S3.C4.L1",
        "DFCI4.S4.C4.L2",
        "DFCI4.S4.C4.L1",
        "DFCI4.S5.C4.L1",
        "DFCI4.S5.C4.L2",
    ]
    assert components[0].bin_size_um == 50.0
    assert "Patient4: 8 major components at 50 um" in messages[0]


def test_patient6_components_get_expected_labels_and_tissue_ids():
    centers = [
        (0, 0),
        (1000, 0),
        (0, 1000),
        (1000, 1000),
    ]
    cells = _clustered_cells(centers)

    components, messages = slide3.choose_component_resolution(
        cells,
        sample_name="Patient6",
        bin_sizes_um=(50.0, 75.0),
        min_component_cells=1,
        label_order="yx",
    )

    major = [component for component in components if component.qc_status == "major"]
    assert [component.assigned_label for component in major] == [9, 10, 11, 12]
    assert [component.component_id for component in major] == [
        "Patient6_piece09",
        "Patient6_piece10",
        "Patient6_piece11",
        "Patient6_piece12",
    ]
    assert [component.slide_tissue_id for component in major] == [
        "DFCI16.S1.C4.L1",
        "DFCI16.S2.C4.L1",
        "DFCI16.S3.C4.L1",
        "DFCI16.S4.C4.L1",
    ]
    assert "Patient6: 4 major components at 50 um" in messages[0]


def test_resection_sample_is_not_part_of_slide3_run_targets():
    assert "P4_resection" not in slide3.SAMPLE_LABELS
    discovered = slide3._discover_default_samples()
    assert all(sample.name != "P4_resection" for sample in discovered)


def test_approval_template_contains_major_components_only(tmp_path):
    components = [
        _component("Patient4", "Patient4_piece01", 1),
        _component(
            "Patient4",
            "Patient4_tiny001",
            1,
            qc_status="tiny",
            cell_count=1,
        ),
    ]
    output_path = tmp_path / "component_approval_template.csv"

    slide3.write_component_approval_template(components, output_path)

    df = pd.read_csv(output_path, keep_default_na=False)
    assert list(df.columns) == slide3.APPROVAL_TEMPLATE_COLUMNS
    assert df.to_dict("records") == [
        {
            "sample_name": "Patient4",
            "component_id": "Patient4_piece01",
            "assigned_label": 1,
            "slide_tissue_id": "DFCI4.S1.C4.L1",
            "approved": "",
        }
    ]


def test_validate_manifest_uses_only_approved_rows(tmp_path):
    components = [
        _component("Patient4", "Patient4_piece01", 1),
        _component("Patient4", "Patient4_piece02", 2),
    ]
    manifest_path = tmp_path / "component_approval_template.csv"
    pd.DataFrame(
        [
            {
                "sample_name": "Patient4",
                "component_id": "Patient4_piece01",
                "assigned_label": 1,
                "slide_tissue_id": "DFCI4.S1.C4.L1",
                "approved": "yes",
            },
            {
                "sample_name": "Patient4",
                "component_id": "Patient4_piece02",
                "assigned_label": 2,
                "slide_tissue_id": "DFCI4.S2.C4.L1",
                "approved": "",
            },
        ]
    ).to_csv(manifest_path, index=False)

    manifest = slide3.validate_manifest(manifest_path, components)

    assert manifest == {("Patient4", "Patient4_piece01"): 1}


def test_piece_run_manifest_uses_repo_relative_paths(tmp_path):
    components = [
        _component("Patient4", "Patient4_piece01", 1),
        _component("Patient4", "Patient4_piece02", 2),
    ]
    approved = {
        ("Patient4", "Patient4_piece01"): 1,
        ("Patient4", "Patient4_piece02"): 2,
    }
    counts = {
        ("Patient4", "Patient4_piece01"): 100,
        ("Patient4", "Patient4_piece02"): 200,
    }
    outdir = tmp_path / "tutorials" / "gbm" / "output" / "slide3_pieces"
    outdir.mkdir(parents=True)

    manifest_path = slide3.write_piece_run_manifest(
        outdir,
        components,
        approved,
        counts,
        repo_root=tmp_path,
    )

    df = pd.read_csv(manifest_path)
    assert list(df.columns) == slide3.PIECE_RUN_MANIFEST_COLUMNS
    assert df["task_id"].tolist() == [1, 2]
    assert df["input_piece_parquet"].tolist() == [
        "tutorials/gbm/output/slide3_pieces/slide3_piece_01_Patient4.parquet",
        "tutorials/gbm/output/slide3_pieces/slide3_piece_02_Patient4.parquet",
    ]
    assert df["output_tracer_parquet"].tolist() == [
        "tutorials/gbm/output/slide3_tracer/slide3_piece_01_Patient4_tracer.parquet",
        "tutorials/gbm/output/slide3_tracer/slide3_piece_02_Patient4_tracer.parquet",
    ]
    assert df["rows_written"].tolist() == [100, 200]
