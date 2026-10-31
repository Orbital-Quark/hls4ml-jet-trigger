# Drift-Robust, Self-Calibrating Jet Tagging for the LHC Trigger on a Low-Cost FPGA

Shreyas K · VNIT · Built on hls4ml (Duarte et al., arXiv:1804.06913) · Code: branch `main-dev`, tags `v1` … `v5-v5.2`

---

## 0. Summary

At the LHC, bunches of protons collide 40 million times a second, far too many to save. A first, very fast filter
called the trigger has to decide in about a microsecond which collisions to keep. Because it has to be that fast, it
runs on FPGAs: chips whose circuits you design yourself. A team of physicists (the hls4ml paper) showed that a
small neural network on an FPGA can look at a spray of particles called a jet and tell what made it (a quark, a
gluon, or a W, Z or top particle) in about 75 billionths of a second.

We rebuilt their work on a chip about 25 times smaller and cheaper, the Xilinx xc7z020. Their network needs almost
five times more multipliers than this chip has, so the first job was making it fit. Then we asked something the
paper doesn't: **real detectors slowly go out of calibration. What happens to the trigger then, and can it fix
itself?**

Over five versions and 404 chip designs, we found three things that help:

- **Training on imperfect data** (v2): show the network deliberately distorted jets while it learns, and it loses
  about half as much accuracy when the detector drifts.
- **A trigger that watches itself** (v4): each trigger counts how many jets it lets through and nudges its own
  threshold to stay on budget, like a thermostat. Fixed thresholds let through anywhere from 0.4 to 1.2 times their
  budget as the detector drifts; the self-correcting ones stay between 0.8 and 1.2.
- **Using physics to recalibrate** (v5.2): ordinary jets are so common and so predictable that they work as a
  reference. If their typical mass or particle count shifts, the detector has shifted, and the chip undoes it.
  This completely removes the damage from a 10% calibration slip (accuracy goes from 67% back to 75%), and works
  even for drifts bigger than anything the network was trained on.

**Our best design** gets 75.4% of jets right on a healthy detector and 73.1% on average across nine kinds of
drift. The version that copies the paper gets 70.2% under drift. Ours uses 15% of the chip's multipliers, gives
its answer in 195 billionths of a second, and can take a new jet every 5, five times faster than collisions arrive.

The paper's own approach (v1) is still the most accurate when the detector is perfect, by about half a point. Our
result is the trade: almost as good when everything is perfect, and clearly better, with a steady trigger rate,
when it isn't.

---

## 1. Goal and setup

### What the network has to do
When a heavy particle such as a W, Z or top quark is produced with a lot of energy, it decays into a narrow spray of
particles called a **jet**. Ordinary quarks and gluons make jets too, and far more often: roughly a thousand for
every interesting one. A jet tagger looks at a jet and says which of the five it came from: gluon (g), light quark
(q), W, Z or top (t). A trigger then keeps the jets that look like W, Z or top and throws away the rest.

### The data
We use the public hls4ml jet dataset: **830,000 simulated jets**, each about 1 TeV in energy, equally split between
the five types. Each jet is described by **16 numbers** that physicists already compute from the detector, for
example:
- the **jet mass** (a W jet sits near 80 GeV, a Z near 91, a top near 173),
- the **particle count** (gluon jets contain more particles than quark jets),
- several **shape numbers** that measure whether the energy is split into one, two or three clumps ("prongs"):
  W and Z decay into two prongs, top into three.

80% of the jets are used for training and 20% for testing.

### The network
The same small network as the paper: 16 inputs, three layers of 64, 32 and 32 "neurons", and 5 outputs, one score
per jet type. Running one jet through it takes 4,256 multiplications.

### The chip
| | Paper's chip (Xilinx KU115) | Our chip (Xilinx xc7z020) |
|---|---|---|
| Multipliers (**DSPs**: dedicated multiplier blocks) | 5,520 | **220** |
| Logic cells (**LUTs**: small blocks that can build any simple circuit) | 663,000 | **53,200** |
| Clock | 200 MHz | 200 MHz (one step every 5 ns) |

**Why this chip.** Real LHC trigger boards use large Virtex UltraScale+ chips, and the tools for those (and for the
paper's KU115) need a paid licence. AMD's free tools cover the smaller Zynq-7000 family, and the xc7z020 is its
common mid-sized member, found on cheap student boards. Installing support for just this family also kept the
tools around 50 GB instead of 200+. We turned that limit into the question: a design that works here leaves the
big chip's space free for the rest of the trigger, and DSP and LUT counts are absolute, so they compare directly
with bigger chips.

The paper's network needs **1,064 DSPs, 484% of what our chip has.** So the first question was: *can it fit at
all, and at what cost?*

### How a network becomes a circuit
1. **Train** in Python with QKeras, which can train a network that only uses a few bits per number (low precision).
2. **Convert** with hls4ml, which writes the trained network out as C++ code for a chip.
3. **Synthesise** with AMD's Vitis HLS 2024.1, which turns the C++ into a circuit and reports how many DSPs and
   LUTs it uses and how fast it runs.
4. **Test** on a bit-accurate emulation of that circuit: a simulation that gives exactly the answers the real chip
   would. Every accuracy in this report comes from this emulation, not from the Python model.

### The knobs we turned
Every version is built in 82 variations ("designs") from four settings:

| Knob | Values | What it trades |
|---|---|---|
| **Bits** per number | full precision, 16, 8, 6, 4, and mixed (different bits per layer) | Fewer bits: smaller multipliers, more rounding |
| **Size** | 100%, 50%, 25% of the paper's network | Smaller: fewer multiplications |
| **Pruning** (deleting the weakest connections) | 0%, 50%, 75% | More pruning: less work, risk of breaking the network |
| **Reuse factor** | 1 or 4 | 1: one multiplier per multiplication, fastest. 4: each multiplier does 4 jobs, fewer multipliers but slower |

A design's name spells these out. `v5_b6_s50_p50_r1` means version 5, 6 bits, 50% size, 50% pruned, reuse 1.

### The questions that drove the project
1. Can the paper's tagger fit on a chip 25 times smaller, and what does that cost? (v1)
2. Real detectors drift out of calibration. How much does that hurt, and can training fix it? (v2, v3)
3. Can the trigger keep its own rate steady while the detector drifts? (v4)
4. Can physics, not just machine learning, undo the drift? (v5.2)

---

## 2. How we measure

### Accuracy, and why it isn't enough
**Accuracy** is the share of test jets the circuit labels correctly, out of 50,000. It's the natural first score,
and the one we quote most. But a trigger doesn't care about labelling every jet right. In real collisions,
ordinary quark and gluon jets outnumber the interesting ones about a thousand to one, and the trigger can only pass
a small, fixed share of them: its **budget**. So the trigger question is:

> *If only 1% of ordinary jets may pass, how many real W, Z or top jets get through?*

We call that the **signal kept at 1% background**. We also quote the **AUC**, a single number that sums up how well
the network separates signal from ordinary jets over every possible threshold (100% = perfect separation). The
paper reports AUC, so it's our direct comparison with them.

To make trigger decisions possible, the chip outputs the five raw scores instead of just the winning type. Each
trigger (top, W, Z) passes a jet when that type's score beats all the others by more than a **threshold**, set on
healthy-detector data so that exactly 1% of ordinary jets pass.

### Detector drift
Real detectors don't stay perfect. We imitate three things that go wrong, applied only to the test jets. The
network and its thresholds stay exactly as they were trained.

| Drift | What we change | What it imitates | Strengths tested |
|---|---|---|---|
| **Mass shift** | Every jet's mass reads a few % high | The calorimeter's energy calibration slipping | +2%, +5%, +10% |
| **Smearing** | Random noise added to every input | Measurements getting blurrier as the detector ages | 5%, 10%, 20% |
| **Pileup** | Extra particles added to each jet | Stray particles from other collisions in the same bunch crossing | +2, +5, +10 |

For each design we record the accuracy left under each of the nine cases, and how far each trigger's background
rate moves away from its budget (1.00× = exactly on budget).

**Unseen drifts.** Later versions are trained on these same three kinds of drift, which would make the test too
easy. So from v5 on we also test drifts no version ever trains on: all three at once, and each one pushed beyond
the training range (mass +15%, smear 30%, pileup +15).

### Does it fit, and how fast is it?
A design **fits** if it stays within all of the xc7z020's limits: 220 DSPs, 53,200 LUTs, and its memory and
register limits. We also record the **latency** (how long one jet takes, start to answer) and how often the circuit
can **accept a new jet**. The LHC delivers a new collision every 25 ns, so anything at or below 25 ns keeps up.
These numbers are Vitis HLS estimates; a full chip build (place-and-route) can differ a little.

### Comparing fairly
- **Like for like.** When comparing versions, we compare the *same* design (same bits, size, pruning and reuse)
  in each version, and only designs that work in all of them.
- **Collapsed designs** (below 70% accuracy, usually from too much pruning) are left out of averages. Including them
  once fooled us: an early "v2 is 1.5 points more accurate than v1" came almost entirely from v1 having a few more
  collapsed designs. On working designs, v2 was in fact slightly *less* accurate.
- **Noise.** Training the same network twice gives results about 0.1 points apart, so smaller differences mean nothing.

---

## 3. v1: Rebuilding the paper

### What v1 is
v1 copies the paper as closely as we could: the same 16 inputs, each simply rescaled to a similar range, the same
network, and plain training on healthy jets. Everything later is measured against it.

### Did we reproduce the paper?
Yes. The full-size network gets **76.9%** on the chip emulation (the paper reports about 75–76%), and its AUCs are
95% for top, 94% for W and 94% for Z. Synthesis built exactly what we asked for: 4,256 multiplications at reuse 4
gave 4,256 ÷ 4 = 1,064 DSPs. That's the 484% from topic 1, so the paper's design can't be built on our chip.

### What fits
36 of v1's 75 designs fit the xc7z020. The two best:

| Design | Accuracy | DSPs | LUTs | Latency | New jet every |
|---|---|---|---|---|---|
| Full precision, 25% size, reuse 4 | 76.2% | 122 | 21,260 | 280 ns | 20 ns |
| 6 bits, 50% size, reuse 1 | 76.1% | 40 | 53,105 (99.8%) | 205 ns | 5 ns |

So shrinking costs under a point of accuracy. The second design is the more interesting one: a fifth of the DSPs
and four times the speed, but it uses almost every LUT on the chip, with no room to spare.

### Where v1 breaks: drift
On a drifting detector v1 falls apart quickly. The full-size network loses:

| Drift | Accuracy lost |
|---|---|
| Mass +10% | 13.4 points |
| Smear 20% | 23.0 points |
| Pileup +10 | 3.2 points |

Averaged over all nine drifts and every working design, v1 loses **5.5 points**. The trigger rates move too: a
5% mass shift lets the W and Z triggers pass 7–19% more ordinary jets than their budget (jets just below 80–90 GeV
get pushed into the W/Z mass window), while the top trigger passes 22–34% fewer. **This was the starting point for
everything after v1:** the paper's tagger works well on a perfect detector, and badly on a real one.

### Problems we found and fixed along the way
These were found while building v1 and v2. Each fix applies to every version, so all comparisons are fair.

- **Raw scores instead of a single answer.** At first the chip only output the winning jet type, which was simpler
  to build (the paper's full "softmax" step, which turns scores into probabilities, took the tools 25+ minutes and
  still failed). But a trigger needs the scores themselves to set thresholds. Switching to raw scores also made the
  chip twice as fast: a new jet every 4 clock steps instead of 8 (20 ns instead of 40). The final "pick the winner"
  step, not the network, had been the bottleneck.
- **Pruning deleted too much.** Low-bit training caps weights at ±1, so many weights end up exactly equal at the
  cap. Our pruning removed every weight below a cut-off, and with so many ties it sometimes removed whole layers,
  leaving pruned low-bit designs at 20% (pure guessing among five types). Fixed by keeping the strongest weights by
  rank; those designs went back up to about 74%.
- **The reuse-factor trap.** hls4ml only allows certain reuse factors for each layer, and when you ask for one it
  can't use, it quietly picks a *smaller* one, which means *more* multipliers. With 15 inputs (v2), asking for
  reuse 4 silently gave 3. Fixed by picking the nearest allowed value at or above the one asked for.
- **Thresholds that can't hit 1%.** Low-bit chips give coarse scores, so many jets tie on the same score and some
  exact rates, like 1%, can't be reached by any threshold. Thresholds now use the closest rate they can reach, and
  report it.
- **Training that never stopped.** Training is meant to stop once the network stops improving, but it kept
  improving by tiny amounts, so every model ran all 100 rounds. Requiring a minimum improvement cut training time
  roughly in half, with no change in accuracy.

---

## 4. v2: Training for an imperfect detector

### The question
v1 loses up to 23 points when the detector drifts. *Can we train a network that cares less, without making it
bigger?*

### What changed
| Change | What it does | Why |
|---|---|---|
| **15 inputs, not 16** | Drops one input that is an exact copy of another in this dataset | No information lost, 64 fewer multiplications |
| **Log scaling** | Takes the logarithm of inputs that span a huge range (0.00001 to 0.03) before rescaling | At 4–6 bits, tiny values would otherwise all round to zero |
| **Drift-aware training** | Half the training jets get a random drift (mass shift, smear or pileup) | The network sees imperfect jets while it learns |
| **Distillation** | The small network also learns from a big full-precision "teacher" network's scores, not only the right answers | The teacher's scores carry more information ("70% top, 25% W") than a plain label |
| **Mixed bits** | Different bits per layer, e.g. 8 bits at the input and output, 4 in the middle | The first and last layers are the most sensitive to rounding |

### Which change does what
We added each change on its own to v1 (6 bits, full size, tested in Python on 166,000 jets):

| Variant | Healthy | Mass +5% | Smear 10% | Pileup +5 |
|---|---|---|---|---|
| v1 | 76.6% | 74.4% | 66.1% | 75.6% |
| + 15 inputs | 76.7% | 74.6% | 66.1% | 75.7% |
| + log scaling | 76.8% | 74.5% | **60.4%** | 75.7% |
| + drift-aware training | 76.3% | 75.1% | **70.6%** | 75.8% |
| + distillation | 76.7% | 74.4% | 65.8% | 75.9% |
| **v2 (all)** | 76.4% | **75.1%** | **70.7%** | **76.2%** |

- **Drift-aware training does almost all the work.** It nearly halves the smearing loss for a quarter point of
  healthy accuracy.
- **The log-scaling trap.** On its own, log scaling made smearing *worse* (−16 points): the log magnifies noise on
  small values. Paired with drift-aware training the problem disappears, so we kept it, but it's a good example of
  a change that looks fine on healthy data and fails on drifted data.
- 15 inputs is free; distillation helps healthy accuracy a little and robustness not at all.

### Result on the chip
Across all working designs, v2 loses **3.0 points** under drift, against v1's 5.5. For the full-size network:
mass +10% costs 7.0 points instead of 13.4, smear 20% costs 12.8 instead of 23.0, pileup +10 costs 1.3 instead of
3.2. Healthy accuracy is about 0.7 points lower than v1 on the same designs.

**The hidden cost.** With 15 inputs, hls4ml can't use reuse 4 for the first layer (only 1, 3, 5, 15… divide
evenly), so it uses 5. v2 designs at reuse 4 therefore take a new jet every 25 ns instead of 20: still enough for
the LHC, but with no margin. v5 fixes this.

---

## 5. v3: Three ideas that mostly didn't work

### The question
v2 makes the network robust, but it's still trained as if every jet type were equally common. *Can we train it for
what a trigger actually needs?*

### What we tried (each added to v2 on its own)
| Idea | What it does | Result |
|---|---|---|
| **Rate weights** | Weights training jets by how often each type is really produced, so letting an ordinary jet through counts as the expensive mistake | **Kept.** +1.0 to +1.2 points of W and Z signal kept at 1% background, for 0.7 points of accuracy |
| **Flat mass** | Reweights jets so each type's mass spread looks flat, to push the network onto prong shape instead of mass, and make it immune to mass drift | **Rejected.** Mass-drift loss barely moved (1.0 → 0.9 points), top trigger lost 1 point. Several other inputs also depend on mass, so the network still reads it indirectly |
| **Input selector** | Scrambles each input in turn and drops the ones the network doesn't miss | **Rejected.** Every input mattered, so nothing was dropped |

All three together were worse than v2 by 1.8 points. **v3 was never built as a full version**; only rate weights
carried forward. We report it because negative results are results: they show the flat-mass idea doesn't work with
these inputs.

---

## 6. v4: A trigger that watches itself

### The question
Even a robust network shifts its scores when the detector drifts, so a threshold set on day one lets through the
wrong number of jets. Today, people fix this by recalibrating thresholds from monitoring data. *Can the trigger do
it on its own, on the chip?*

### The idea
At the LHC almost every jet is ordinary, so a trigger's **own pass rate is a good measure of its background rate**,
and it can count that without knowing which jets are which. Like a thermostat, every 256 jets each trigger counts
how many it passed, compares with its budget, and nudges its threshold: up if it passed too many, down if too few,
in proportion to the error. Statisticians call this **quantile tracking**: the threshold that passes a fixed share
of a stream is a quantile, and this kind of nudging follows it even as the stream changes.

**On the chip** it costs one counter, one subtraction and one add per trigger: tens of LUTs, no multipliers.
256 is a power of two, so no division is needed. At LHC rates it settles within about a millisecond, far faster
than real detector drift (minutes to months).

v4 = v2 + rate weights + this trigger.

### How we tested it
For each drift, we build a stream like real collisions (almost all ordinary jets, 1% signal), let the trigger run,
then measure its threshold on fresh jets it never saw. We compare three thresholds: **fixed** (set once on the
healthy detector), **self-correcting**, and a **perfect recalibration** (the best single threshold, set with the
true labels on the drifted jets).

### Result
| Threshold | Background passed, as a multiple of budget (middle 90% of cases) | Signal kept |
|---|---|---|
| Fixed | 0.42× to 1.20× | 54.4% |
| **Self-correcting** | **0.83× to 1.18×** | **56.3%** |
| Perfect recalibration | 1.00× by definition | 56.3% |

The self-correcting trigger keeps its rate near budget and keeps exactly as much signal as a perfect, labelled
recalibration, with no labels. **What it can't do:** it fixes the *rate*, not the network. If drift makes the
network worse at telling jets apart, signal is still lost, just no more than a perfect recalibration would lose.

---

## 7. Comparing v1, v2 and v4

Same 59 working designs in each version:

| | v1 | v2 | v4 |
|---|---|---|---|
| Accuracy, healthy | **75.8%** | 75.1% | 74.3% |
| Accuracy, average of 9 drifts | 70.1% | **72.1%** | 71.0% |
| Mass +10% | 63.4% | **69.5%** | 68.3% |
| Smear 20% | 55.5% | **62.8%** | 62.0% |
| Pileup +10 | 72.5% | **73.6%** | 71.2% |
| AUC top / W / Z | 94.2 / 93.6 / 92.1 | 93.7 / 93.7 / 92.2 | 94.1 / **93.8 / 93.1** |
| New jet every (reuse 4) | **20 ns** | 25 ns | 25 ns |

### Why each one wins where it does
- **v1 wins on a perfect detector and on speed.** It's trained only on clean jets, so all its effort goes into
  them. Its 16 inputs let the first layer use reuse 4, so it takes a new jet every 20 ns. At the mildest drifts
  (mass +2%, pileup +2) its head start is still enough to lead.
- **v2 wins once the detector drifts.** Drift-aware training pays off as soon as the drift is moderate.
- **v4 wins on AUC, mostly for Z.** Rate weights teach it that passing an ordinary jet is the costly mistake, so
  it separates signal from background better. The price is accuracy: it now leans towards calling jets ordinary,
  which costs points on our test set, where all five types are equally common. Rate weights also cost it some
  robustness compared with v2.

**The corrected comparison.** Our first v1-vs-v2 comparison said v2 was 1.5 points *more* accurate. That came from
v1 having more collapsed designs in the average. Comparing only working designs, v1 leads on a healthy detector by
0.7 points. We report the corrected number.

---

## 8. v5: Four fixes

### The questions
v4 had three loose ends: it was 0.7 points less accurate than v2, it took a new jet every 25 ns instead of 20, and
every drift we tested was one the network had trained on. v5 tries to fix all three.

| Fix | What it does | Why |
|---|---|---|
| **16th input, always zero** | Puts back a 16th input that is always zero | 16 inputs allow reuse 4 again: a new jet every 20 ns |
| **Softer rate weights** | Makes rate weights push less strongly towards "ordinary" | Get back accuracy v4 lost |
| **Fewer drifted jets** | 30% of training jets drifted instead of 50% | Get back healthy accuracy |
| **Unseen-drift test** | Tests drifts no version trains on (topic 2) | An honest test of robustness |

### Which fix did what (6 bits, full size, tested in Python)
| Variant | Healthy | Mass +5% |
|---|---|---|
| v4 | 76.0% | 74.6% |
| + softer rate weights | **76.2%** | **74.8%** |
| + fewer drifted jets | 75.9% | 74.3% |
| v5 (all) | **76.6%** | **74.9%** |

### Result on the chip
On the 59 matched designs, v5 beats v4 on **every one** (+0.9 points healthy), edges v2 (+0.2), and is 0.5
points behind v1. But two fixes had a cost:
- **Fewer drifted jets backfired.** Under mass +10%, v5 gets 67.4%, worse than v2's 69.5%. Our reason for this fix
  was wrong: we blamed drift-aware training for v4's accuracy loss, but the ablation shows rate weights caused it,
  and halving the drifted jets only made v5 less robust. We'd go back to 50%.
- **The zero input costs DSPs.** The extra input brings back the first-layer multipliers that v2 had saved, so some
  designs no longer fit: only 33 of v5's designs fit, against 40 for v4.

The unseen drifts show how much robustness is really learned: mass +15% drops v5 to 59.7%, and all three drifts
together to 66.7%.

---

## 9. v5.2: Using physics to recalibrate

### The question
Everything up to v5 is machine learning: train the network to tolerate drift. But physicists recalibrate
detectors with physics, using known quantities as references. *Can the chip do that to its own inputs?*

### The idea: a "standard candle"
Astronomers measure distances with stars of known brightness, called standard candles. Ordinary quark and gluon
jets can play the same role here: they make up almost all of the stream, and their typical (median) mass and
particle count are very stable. So:

1. On healthy data, record the median jet mass and median particle count. That's the reference.
2. While running, keep a **running median** of both over the incoming jets. Every 256 jets it moves a tiny step
   up if more jets were above it than below, and down otherwise.
3. If the running median mass is 10% above the reference, the energy scale has slipped by 10%: divide every jet's
   mass by 1.10. If the median particle count is 5 above the reference, pileup has added about 5 particles:
   subtract 5.

The network itself is unchanged: v5.2 is v5's networks with this correction in front. It needs no labels, only
the stream itself.

### Result (same 59 designs)
| Test | v5 | **v5.2** |
|---|---|---|
| Healthy | 75.2% | 75.2% |
| Mass +10% | 67.4% | **75.2%** |
| Pileup +10 | 72.7% | **75.0%** |
| Smear 20% | 62.0% | 61.7% |
| Average of 9 drifts | 71.6% | **72.9%**, best of all versions |
| *Unseen:* mass +15% | 59.7% | **75.2%** |
| *Unseen:* pileup +15 | 69.2% | **74.9%** |
| *Unseen:* all three at once | 66.7% | **69.9%** |
| *Unseen:* smear 30% | 56.2% | 55.6% |

- **Mass and pileup drift are gone**, even far beyond anything the network trained on. Accuracy under drift equals
  healthy accuracy.
- **Healthy accuracy doesn't change**: on a healthy detector the correction does nothing.
- **Trigger rates stay on budget even with fixed thresholds** (0.79× to 1.10×, against 0.40× to 1.25× for v5),
  because the inputs going into the network look healthy again.
- **Smearing can't be fixed this way** (slightly worse, −0.3 to −0.6 points). Smearing is random noise, not a
  shift, so the median doesn't move and there's nothing to undo.

### Caveats
- The correction isn't yet part of the synthesised circuit; v5.2's chip numbers are v5's. It's one running median,
  one multiply and one subtract per input, so the extra cost should be small, but it hasn't been built.
- Real calibration slips aren't always a clean percentage, and pileup doesn't only add particles. Our drift model
  is simpler than reality.
- Like the self-correcting trigger, it relies on ordinary jets dominating the stream.

---

## 10. What we learned about the hardware

- **Pruning only saves hardware at reuse 1.** At reuse 4, hls4ml keeps a multiplier slot for every weight, even
  zero ones, so 0%, 50% and 75% pruning give identical DSP counts. At reuse 1, zero weights are removed: one 8-bit
  network went from 342 to 279 to 160 DSPs.
- **At reuse 1, fewer bits moves the work from DSPs to LUTs.** 6-bit and 4-bit multiplications are built out of
  logic cells instead of DSPs. One 4-bit design used **zero DSPs**, but too many LUTs to fit. So at reuse 1, the
  LUTs become the limit, not the DSPs.
- **This is why the best designs are 6 bits, half size, 50% pruned, reuse 1.** It's the same design in every
  version, using 28–40 DSPs each time: few bits and pruning keep the LUTs in check, and reuse 1 means a new jet
  every clock step (5 ns).
- **8-bit designs miss the clock.** The tools estimate 5.4 ns per step for them, slower than the 5 ns target, so
  their real speed would be about 8% lower.
- **Latency barely depends on network size.** It's set mostly by the reuse factor and the number of layers.
- **Allowed reuse factors depend on layer sizes** (topic 4), which quietly decides how fast a design can be.
- **Too much pruning breaks small networks.** At 25% size, 75% pruning leaves about 100 weights and accuracy
  collapses. 50% pruning was safe everywhere.

---

## 11. Final comparison

### All versions, same 59 working designs
| | v1 | v2 | v4 | v5 | **v5.2** |
|---|---|---|---|---|---|
| Accuracy, healthy | **75.8%** | 75.1% | 74.3% | 75.2% | 75.2% |
| Accuracy, average of 9 drifts | 70.1% | 72.1% | 71.0% | 71.6% | **72.9%** |
| Mass +10% | 63.4% | 69.5% | 68.3% | 67.4% | **75.2%** |
| Smear 20% | 55.5% | **62.8%** | 62.0% | 62.0% | 61.7% |
| Pileup +10 | 72.5% | 73.6% | 71.2% | 72.7% | **75.0%** |
| Designs that fit the chip | 36 | 42 | 40 | 33 | 33 |

### Best design that fits, per version
| Version | Design | Healthy | Drifted | DSPs | LUTs | Latency | New jet every |
|---|---|---|---|---|---|---|---|
| v1 | full precision, 25% size, reuse 4 | **76.2%** | 70.2% | 122 | 21,260 | 280 ns | 20 ns |
| v2 | 8 bits, 50% size, reuse 4 | 76.1% | 72.9% | 202 | 47,158 | 370 ns | 25 ns |
| v4 | 8 bits, 50% size, reuse 4 | 75.6% | 71.7% | 202 | 47,153 | 370 ns | 25 ns |
| v5 | 6 bits, 50% size, 50% pruned, reuse 1 | 75.4% | 71.8% | 32 | 36,430 | 195 ns | 5 ns |
| **v5.2** | **6 bits, 50% size, 50% pruned, reuse 1** | 75.4% | **73.1%** | **32** | 36,430 | **195 ns** | **5 ns** |

**Our final design is v5.2_b6_s50_p50_r1.** It's 0.8 points behind v1's best on a perfect detector, and 2.9
points ahead under drift. It uses 15% of the chip's DSPs and 68% of its LUTs, which leaves room for a real build,
and it's the fastest of the five: a new jet every 5 ns. It isn't a lucky outlier: the same design is among the
best in every version.

### The answer to each question
1. **Can it fit?** Yes, with under a point of accuracy lost, by using fewer bits, a smaller network and pruning.
2. **Can training fix drift?** Partly: drift-aware training (v2) cuts the loss from 5.5 to 3.0 points.
3. **Can the trigger keep its own rate?** Yes: the self-correcting trigger stays near budget and matches a
   perfect recalibration.
4. **Can physics undo the drift?** For energy-scale slips and pileup, completely. For blurring, no.

---

## 12. Limitations

- **The drift is simulated**, and simply: a clean percentage, Gaussian noise, a fixed number of extra particles.
  Real detectors drift in messier ways.
- **The inputs are 16 summary numbers**, not the individual particles. Real pileup affects particles, not
  summaries, and newer taggers work on particles.
- **Chip numbers are tool estimates.** We didn't do a full chip build (place-and-route) or run on a real board.
- **The v5.2 correction wasn't synthesised**; its cost is estimated, not measured.
- **One training run per design.** Differences under about 0.1 points are noise.
- **Smearing is still unsolved.** It's the drift that hurts most (−14 points at 20% for v5.2) and none of our
  methods fixes it fully.
- **The chip was chosen for its free tools**, not because it's what a real trigger would use.

---

## 13. Future work

- **v5.3:** v5.2 with 50% drifted training jets again, which should recover v2's robustness to smearing.
- **Lighter pruning at reuse 1** (20–35%), between the 0% that doesn't fit and the 50% we use.
- **A full chip build** (Vivado place-and-route), including the v5.2 correction, to confirm the numbers.
- **Particle-level inputs**, with standard pileup removal (PUPPI), for a more realistic pileup test.
- **More jet types** with JetClass (10 classes, including Higgs decays). That needs tracking and vertex inputs
  for b and c jets, not just new labels.
- **Merging W and Z** into one trigger: their masses are close, and the network often confuses them.

---

## 14. How to reproduce

Setup (WSL Ubuntu 22.04, Vitis HLS 2024.1, Python environment) is in [SETUP.md](SETUP.md).

```bash
git checkout main-dev
python prepare_data.py                      # download the dataset once
VERSIONS=v1,v2,v4,v5 N_JOBS=4 python sweep.py   # train, synthesise and test every design
N_JOBS=4 python selfcal.py                  # v5.2: v5's networks with the input self-calibration
ABLATE=v5 python ablation.py                # which fix does what
python plots.py --open                      # interactive results page
```

Each version's code is also tagged (`v1`, `v2`, `v3-v4`, `v5-v5.2`), so `git checkout v2` shows the code as it was
for v2. All results are in `results/`, one file per design in `results/points/`.
