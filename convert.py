import hls4ml

CLOCK_NS = 5.0


def convert_and_synth(model, name, part, rf, root="hls_prjs"):
    cfg = hls4ml.utils.config_from_keras_model(model, granularity="name", backend="Vitis")
    cfg["Model"]["ReuseFactor"] = rf
    cfg["Model"]["Strategy"] = "Latency" if rf == 1 else "Resource"
    for layer_cfg in cfg["LayerName"].values():
        layer_cfg["ReuseFactor"] = rf
    if "softmax" in cfg["LayerName"]:  # from the official tutorial
        cfg["LayerName"]["softmax"]["exp_table_t"] = "ap_fixed<18,8>"
        cfg["LayerName"]["softmax"]["inv_table_t"] = "ap_fixed<18,8>"

    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=cfg, output_dir=f"{root}/{name}",
        backend="Vitis", part=part, clock_period=CLOCK_NS, io_type="io_parallel",
    )
    hls_model.compile()                                  # bit-accurate C++ emulation
    hls_model.build(csim=False, synth=True, vsynth=False)  # C-synthesis only
    return hls_model


def read_report(out_dir):
    r = hls4ml.report.parse_vivado_report(out_dir)["CSynthesisReport"]
    cycles = int(r["BestLatency"])
    return dict(
        latency_cycles=cycles,
        latency_worst_cycles=int(r["WorstLatency"]),
        latency_ns=cycles * CLOCK_NS,
        ii=int(r["IntervalMin"]),
        est_clock_ns=float(r["EstimatedClockPeriod"]),
        lut=int(r["LUT"]), ff=int(r["FF"]), dsp=int(r["DSP"]), bram18=int(r["BRAM_18K"]),
    )