import hls4ml

CLOCK_NS = 5.0


def convert_and_synth(model, name, part, rf, root="hls_prjs", synth=True):
    cfg = hls4ml.utils.config_from_keras_model(model, granularity="name", backend="Vitis")
    cfg["Model"]["ReuseFactor"] = rf
    cfg["Model"]["Strategy"] = "Latency" if rf == 1 else "Resource"
    for layer_cfg in cfg["LayerName"].values():
        layer_cfg["ReuseFactor"] = rf
    # hls4ml only accepts some reuse factors per layer (e.g. 15 inputs rule out 4) and otherwise picks the closest,
    # which can be smaller and so cost extra multipliers. Take the nearest allowed value at or above `rf` instead.
    backend, reuse_layers = hls4ml.backends.get_backend("Vitis"), {}
    for layer in model.layers:
        if layer.name in cfg["LayerName"] and layer.get_weights():
            n_in, n_out = layer.get_weights()[0].shape
            valid = backend.get_valid_reuse_factors(n_in, n_out)
            reuse_layers[layer.name] = min([v for v in valid if v >= rf], default=valid[-1])
            cfg["LayerName"][layer.name]["ReuseFactor"] = reuse_layers[layer.name]
    if "softmax" in cfg["LayerName"]:  # the circuit outputs the 5 raw scores; decisions use argmax or TriggerMenu
        cfg["LayerName"]["softmax"]["Skip"] = True

    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=cfg, output_dir=f"{root}/{name}",
        backend="Vitis", part=part, clock_period=CLOCK_NS, io_type="io_parallel",
    )
    hls_model.compile()                                  # bit-accurate C++ emulation
    if synth:
        hls_model.build(csim=False, synth=True, vsynth=False)  # C-synthesis only
    hls_model.reuse_layers = reuse_layers
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