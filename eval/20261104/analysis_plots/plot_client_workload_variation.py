#!/usr/bin/env python3
"""Replot the verified single-seed Pi5/NX 4/3/5/4 workload-development pilot."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_device_learning_efficiency import load_style, ROOT


SOURCE = Path("/Users/jang-won-u/Documents/Codex/2026-10-10/new-chat/outputs/cross_client_development_pilot_20261019")
ROUNDS_FILE = SOURCE / "policy_rounds.csv"
REPLIES_FILE = SOURCE / "policy_reply_barrier.csv"
MATRIX_FILE = SOURCE / "effective_run_matrix_retry3.csv"
MANIFEST_FILE = SOURCE / "source_manifest.json"
INTEGRITY_FILE = SOURCE / "integrity_report.md"
POLICIES = ("F", "A", "B", "C")
NAMES = {"F": "Fixed", "A": "Frozen cost", "B": "Last rate", "C": "LoRACast"}
STYLES = {"F": {"color": "#666666", "linestyle": "-", "marker": "o"},
          "A": {"color": "#999999", "linestyle": "--", "marker": "s"},
          "B": {"color": "#444444", "linestyle": "-.", "marker": "D"},
          "C": {"color": "#000000", "linestyle": ":", "marker": "^"}}
QUOTAS = {round_no: (4 if round_no <= 4 else 3 if round_no <= 8 else 5 if round_no <= 12 else 4)
          for round_no in range(1, 17)}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a: float, b: float, label: str, atol: float=1e-5) -> None:
    if not math.isclose(a,b,rel_tol=1e-10,abs_tol=atol):
        raise AssertionError(f"{label}: {a} != {b}")


def validate() -> tuple[dict, dict]:
    rows = read_csv(ROUNDS_FILE)
    replies = read_csv(REPLIES_FILE)
    matrix = read_csv(MATRIX_FILE)
    manifest = json.loads(MANIFEST_FILE.read_text())
    if len(rows) != 64 or len(replies) != 64 or len(matrix) != 4:
        raise AssertionError("expected 4 policies x 16 measured rounds")
    for name,path in (("policy_rounds.csv",ROUNDS_FILE),("policy_reply_barrier.csv",REPLIES_FILE)):
        if manifest["derived_files_sha256"][name] != sha(path):
            raise AssertionError(f"source manifest hash mismatch: {path}")
    if manifest["effective_run_matrix"]["sha256"] != sha(MATRIX_FILE):
        raise AssertionError("effective experiment mapping hash mismatch")
    if "Each run has 16 complete rounds" not in INTEGRITY_FILE.read_text():
        raise AssertionError("source run-integrity report missing")
    run_ids = {row["policy"]: row["run_id"] for row in matrix}
    if set(run_ids) != set(POLICIES) or len(set(run_ids.values())) != 4:
        raise AssertionError("unexpected F/A/B/C effective mapping")
    if [row["policy"] for row in sorted(matrix,key=lambda x:int(x["sequence"]))] != list(POLICIES):
        raise AssertionError("F/A/B/C execution order changed")
    point = {}
    reply = {}
    unique_raw_files = set()
    for row in rows:
        key=(row["policy"],int(row["round"]))
        if key in point or key[0] not in POLICIES or key[1] not in range(1,17):
            raise AssertionError(f"duplicate/invalid training round: {key}")
        if int(row["seed"]) != 42 or row["run_id"] != run_ids[key[0]]:
            raise AssertionError(f"seed/run mapping mismatch: {key}")
        if int(row["pi_scheduled_quota"]) != QUOTAS[key[1]]:
            raise AssertionError(f"Pi quota schedule mismatch: {key}")
        if int(row["pi_executed_samples"]) != QUOTAS[key[1]]:
            raise AssertionError(f"Pi actual quota mismatch: {key}")
        if int(row["jetson_executed_optimizer_steps"]) < 0:
            raise AssertionError(f"negative Jetson steps: {key}")
        tokens=float(row["aggregate_fl_input_tokens"])
        active=float(row["server_active_round_s"])
        if tokens<=0 or active<=0:
            raise AssertionError(f"invalid executed work/time: {key}")
        close(float(row["round_fl_tokens_per_active_s"]),tokens/active,f"actual active efficiency {key}")
        for field in ("decision_source","train_source","reply_source","cost_source"):
            path=Path(row[field])
            if not path.is_file():
                raise AssertionError(f"missing raw {field}: {key}: {path}")
            unique_raw_files.add(path)
        point[key]=row
    for row in replies:
        key=(row["policy"],int(row["round"]))
        if key in reply or key not in point:
            raise AssertionError(f"duplicate/orphan reply: {key}")
        close(float(row["server_dispatch_to_all_replies_s"]),
              float(row["all_replies_barrier_monotonic_s"])-float(row["server_dispatch_monotonic_s"]),
              f"server-monotonic reply barrier {key}",atol=1e-3)
        close(float(row["server_dispatch_to_all_replies_s"]),
              float(point[key]["server_dispatch_to_all_replies_s"]),f"joined reply barrier {key}",atol=1e-3)
        if row["run_id"] != point[key]["run_id"] or row["seed"] != "42":
            raise AssertionError(f"reply run identity mismatch: {key}")
        reply[key]=row
    if len(point)!=64 or len(reply)!=64:
        raise AssertionError("incomplete verified pilot rounds")
    hashes={str(path):sha(path) for path in sorted(unique_raw_files)}
    return point, {"raw_file_sha256":hashes,"run_ids":run_ids}


def plot_panel(ax, point: dict, *, field: str, ylabel: str) -> None:
    maxima=[]
    for policy in POLICIES:
        records=[point[(policy,round_no)] for round_no in range(1,17)]
        values=[float(row[field]) for row in records]
        maxima.extend(values)
        style=STYLES[policy]
        ax.plot(range(1,17), values, label=NAMES[policy], color=style["color"],
                linestyle=style["linestyle"], marker=style["marker"], markersize=3.2,
                markerfacecolor="white", markeredgewidth=.8, linewidth=1.4)
    for boundary in (4.5,8.5,12.5):
        ax.axvline(boundary,color="black",linestyle=":",linewidth=.7,alpha=.45,zorder=0)
    ax.set_xlim(.7,16.3)
    ax.set_ylim(0,max(maxima)*1.13)
    ax.set_xticks((1,4,8,12,16))
    ax.set_xlabel("FL round")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y",color="black",linestyle="--",linewidth=.5,alpha=.25)
    ax.tick_params(width=.8,length=3)
    ax.set_axisbelow(True)


def save(fig,stem:str,style_module) -> None:
    base=ROOT/"supplementary"/stem
    fig.savefig(str(base)+".svg",bbox_inches=None)
    style_module.savefig(fig,str(base),png_dpi=600)


def make_figure(point:dict,style_module,stem:str,panels:tuple[tuple[str,str],...]) -> None:
    fig,axes=plt.subplots(1,2,figsize=style_module.cm_to_inch(17.8,5.1))
    fig.subplots_adjust(left=.075,right=.985,bottom=.255,top=.75,wspace=.25)
    for ax,(field,label) in zip(axes,panels):
        plot_panel(ax,point,field=field,ylabel=label)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc="upper center",bbox_to_anchor=(.53,.98),frameon=False,
               handlelength=2.2,columnspacing=1.55)
    save(fig,stem,style_module)


def main() -> None:
    style=load_style()
    point,provenance=validate()
    make_figure(point,style,"figS2_workload_response",
                (("pi_local_compute_s","Pi compute (s)"),
                 ("jetson_executed_optimizer_steps","NX steps")))
    make_figure(point,style,"figS2_efficiency_and_reply_barrier",
                (("round_fl_tokens_per_active_s","FL input tokens / s"),
                 ("server_dispatch_to_all_replies_s","Reply barrier (s)")))
    plotted=[]
    for policy in POLICIES:
        for round_no in range(1,17):
            row=point[(policy,round_no)]
            plotted.append({"run_id":row["run_id"],"seed":42,"policy_code":policy,"method":NAMES[policy],
                            "round":round_no,"pi_scheduled_quota":QUOTAS[round_no],
                            "pi_actual_local_compute_s":row["pi_local_compute_s"],
                            "nx_executed_optimizer_steps":row["jetson_executed_optimizer_steps"],
                            "input_tokens_per_server_active_s":row["round_fl_tokens_per_active_s"],
                            "server_dispatch_to_all_replies_s":row["server_dispatch_to_all_replies_s"],
                            "raw_training_source":row["train_source"],"raw_reply_source":row["reply_source"],
                            "raw_decision_source":row["decision_source"]})
    output=ROOT/"verification/figS2_plot_data.csv"
    with output.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(plotted[0]))
        writer.writeheader();writer.writerows(plotted)
    audit={"status":"PASS","source_campaign":str(SOURCE),"seed":42,"policy_order":list(POLICIES),
           "Pi_quota_by_round":[QUOTAS[r] for r in range(1,17)],"observations":64,
           "experiment_class":"Single-seed, sequentially ordered hardware development pilot; not independent confirmation",
           "plotted_quantity_class":{"pi_scheduled_quota":"Prespecified request (phase descriptor only)",
                                     "pi_local_compute_s":"Measured on Pi client",
                                     "jetson_executed_optimizer_steps":"Actual executed training work",
                                     "round_fl_tokens_per_active_s":"Observed completed FL input tokens / measured server-active round time",
                                     "server_dispatch_to_all_replies_s":"Observed server-monotonic reply barrier"},
           "unplotted_allocation_predictions":"Estimator predictions and candidate search expectations are deliberately excluded from both plots",
           "policy_semantics":{"F":"Fixed NX S/H captured after R4","A":"Frozen cost model",
                               "B":"Last-round rate model","C":"Online LoRACast-NR allocator"},
           "time_scope":"Pi local compute measured on client; reply barrier is server-monotonic dispatch to first Flower poll observing all replies; includes noncompute request, handoff, transfer/wait, and polling envelope, not packet arrival or pure Wi-Fi transfer",
           "active_efficiency_scope":"Executed FL input tokens divided by measured server-active round seconds",
           "comparison_limits":"Single seed, fixed F/A/B/C sequence, mixed noncompute reply envelope; no causal policy superiority or GPU loss conclusion. There is no validation-loss score for this pilot.",
           "input_sha256":{str(path):sha(path) for path in (ROUNDS_FILE,REPLIES_FILE,MATRIX_FILE,MANIFEST_FILE,INTEGRITY_FILE)},
           **provenance}
    (ROOT/"verification/figS2_validation.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps({"status":"PASS","observations":len(plotted),"run_ids":provenance["run_ids"]},indent=2))


if __name__=="__main__":
    main()
