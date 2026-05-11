# -*- coding: utf-8 -*-
"""Batch export XY data from all ODB files under `data/`.

Run inside Abaqus:
    abaqus cae noGUI=export_batch_xy_from_odb.py
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

from abaqus import session
from abaqusConstants import COMPONENT, INTEGRATION_POINT, NODAL
import xyPlot

ROOT = r'X:\crack_pridict'
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "rpt_data")

POINT_LIST = [
    {"name": "P001", "node_pick": (("A-SLAB2", 1, ("[#0:53 #20 ]",)),)},
    {"name": "P002", "node_pick": (("A-SLAB2", 1, ("[#0:36 #400000 ]",)),)},
    {"name": "P003", "node_pick": (("A-SLAB2", 1, ("[#0:36 #100 ]",)),)},
    {"name": "P004", "node_pick": (("A-SLAB2", 1, ("[#0:52 #800000 ]",)),)},
]

ACTIVE_FRAMES = (("Step-2", ("0:-1",)),)
VARIABLES = (
    ("A", NODAL, ((COMPONENT, "A3"),)),
    ("LE", INTEGRATION_POINT, ((COMPONENT, "LE11"),)),
    ("S", INTEGRATION_POINT, ((COMPONENT, "S11"),)),
)


def _safe_name(s: str) -> str:
    bad = '<>:"/\\|?* '
    for ch in bad:
        s = s.replace(ch, "_")
    return s


def _open_or_get_odb(odb_path: str):
    if odb_path in session.odbs:
        return session.odbs[odb_path]
    return session.openOdb(name=odb_path)


def export_one_point(odb, point_name, node_pick, out_dir):
    odb_name = session.viewports[session.currentViewportName].odbDisplay.name
    session.odbData[odb_name].setValues(activeFrames=ACTIVE_FRAMES)

    xy_list = xyPlot.xyDataListFromField(
        odb=odb,
        outputPosition=NODAL,
        variable=VARIABLES,
        nodePick=node_pick,
    )
    if not xy_list:
        print("[WARN] No XY data extracted for point: %s" % point_name)
        return

    if "XYPlot-1" not in session.xyPlots:
        session.XYPlot(name="XYPlot-1")
    xyp = session.xyPlots["XYPlot-1"]
    chart_name = xyp.charts.keys()[0]
    chart = xyp.charts[chart_name]
    curve_list = session.curveSet(xyData=xy_list)
    chart.setValues(curvesToPlot=curve_list)
    session.charts[chart_name].autoColor(lines=True, symbols=True)

    odb_base = _safe_name(os.path.splitext(os.path.basename(odb.path))[0])
    report_path = os.path.join(out_dir, "%s__%s.rpt" % (odb_base, point_name))
    session.writeXYReport(fileName=report_path, xyData=xy_list)
    print("[OK] Exported: %s" % report_path)


def main():
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    odb_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.odb")))
    if not odb_files:
        print("[WARN] No ODB files found under %s" % DATA_DIR)
        return

    for odb_path in odb_files:
        print("[INFO] Processing ODB: %s" % odb_path)
        try:
            odb = _open_or_get_odb(odb_path)
            vp_name = session.currentViewportName
            session.viewports[vp_name].setValues(displayedObject=odb)
            for point in POINT_LIST:
                try:
                    export_one_point(odb, point["name"], point["node_pick"], OUTPUT_DIR)
                except Exception as ex:
                    print("[ERROR] Failed point=%s odb=%s: %s" % (point["name"], odb_path, ex))
        except Exception as ex:
            print("[ERROR] Failed ODB=%s: %s" % (odb_path, ex))

    print("[DONE] Batch export finished.")


if __name__ == "__main__":
    main()
