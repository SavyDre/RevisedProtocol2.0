# ADED analysis workflow

This project loads controlled releases and detection reports, applies spatial and
temporal exclusions, classifies paired observations, calculates performance
metrics, and writes tables and figures. Start with `main.py`; configure a run in
a copy of `InputVariables.py` rather than editing the analysis functions.

## Run the supplied example
`InputVariables.py` configures the files in `InputData/Sample`. The example
below processes one UTC day using the combined WellPad4and5 spatial extent:
```powershell
.\.venv\Scripts\python.exe main.py @sample.args
```
`sample.args` contains one argument or value per line. Run from this project
folder because the example uses relative paths. A path containing spaces stays
on one line, without surrounding quotes in the argument file. On the command
line, quote paths containing spaces in the usual way.


## Configure your own dataset
1. Copy `InputVariables.py` to a meaningful run-specific filename. Its
   `inputVariables()` function must return the documented dictionaries.
2. Fill in the controlled-release and detection column names, rates, baselines,
   threshold method, aggregation lists, and spatial extents. Put each rate in its
   sum aggregation list; put receipt time in an ID-based first/last list.
   Put the report ID column in an aggregation list so it survives expansion.
3. Set plot columns and units. Each plot can be enabled or disabled independently.
4. Copy `sample.args`, change its paths, time window, and ASE names, and run it.

Required CLI settings are the start/end times, controlled-release file,
detection file, ASE list, and configuration file. Maintenance, offline,
passive-observation, and operator logs are optional; omit their CLI options when
unused. An omitted log means there are no exclusions from that source. A supplied
file that cannot be read is an error, not an empty log. A header-only detection
file is supported; the controlled-release grid remains the analysis reference.

Supported files are `.csv`, `.xlsx`, and `.xls`. Ordinary timestamp columns use
`--format` (default `%Y%m%d %H:%M:%S`). Excel datetime cells are accepted.
ISO intervals in `StartEnd` are split into the configured start/end columns;
receipt timestamps are parsed separately using `--format`.

## Time, units, and spatial conventions
- CLI start/end values have format `YYYYMMDDHHMM` and use `--timezone` (default
  `US/Mountain`). Ambiguous or nonexistent daylight-saving times are rejected;
  use `UTC` when you need an unambiguous interval.
- Input timestamps without a timezone mean UTC. Explicit offsets are converted
  to UTC while preserving their instant and precision.
- Records overlapping the requested window are kept and clipped. Zero-duration
  records are dropped with a count warning; reversed intervals are rejected.
  When an endpoint is clipped, `<end column>_Original` preserves the original end
  for reporting-delay calculations.
- `--freq` defaults to `30s` and must be positive and at most 30 seconds. Endpoints
  are rounded to the nearest grid point (ties to even). Samples represent
  `(start, end]` intervals. Finer steps use more memory; events shorter than a
  step can disappear when their endpoints round to the same grid point.
- `--bufferTime "[before, after]"` is always in **minutes**, independent of `freq`.
- With **Above Threshold Across Duration**, `ThresholdDuration` is in **seconds**.
  The existing cumulative-duration definition counts consecutive nonzero-rate
  samples, then checks the current rate against baseline plus threshold.
  With **Rolling Average**, `ThresholdDuration` is the number of sampling
  intervals: 2 at `30s` means a 60-second window. Summary rows use their end
  timestamps for rolling calculations. Time-series and summary metrics therefore
  describe different representations of the observations.
- The main workflow supports those two threshold methods. The legacy Random
  Sample Average helpers are not connected to `main.py` and cannot be selected.
- Rates must share units. `Plotting['AggQuant']['timeFactor']` converts the rate
  time basis to seconds (e.g. `1/3600` for rates per hour). Plot scaling factors
  change displayed units; they do not repair incompatible input rate units.
- The existing spatial input convention is **latitude, longitude**, including
  coordinates inside the supplied WKT strings. This differs from conventional
  longitude/latitude WKT: follow the supplied configuration consistently.
- ASE definitions live in `ControlledReleaseData['ASE']`. `BoundingName` values
  require `--emissionSourceMapper`. Its JSON values contain `BoundingXY`,
  `BoundingZ`, and optionally `ReferenceCoord` as shown in the supplied files.
  Explicit polygon/height columns can be used instead. If neither spatial
  representation is configured, the existing behavior includes all input rows.
- `--bufferDistance "[analysis, source]"` buffers polygon edges in meters after
  local metric projection. The same pair is applied to both datasets. Negative
  distances shrink polygons. Inclusion uses horizontal intersection/union
  **strictly greater than** `--overlappingFraction`, with the source height range
  contained within the ASE height range. A threshold of 1 therefore includes none.
- Choose non-overlapping ASEs for an aggregate intended to count each emission
  once. Results from overlapping ASEs are concatenated, not spatially deduplicated.

## Outputs and reproducibility
Each run creates a unique timestamped directory under `--outputFilePath`
(default `Results`):
- `data/ClassifiedData.csv`: summarized overlapping CR/DR pairs and unmatched rows.
  `Classification` is the authoritative pair label; exclusions take precedence.
- `data/TestCenterData.csv` and `data/DetectionData.csv`: resolved time series,
  durations, classifications, and integrated emissions.
- `data/PairedQuantificationData.csv`: TP pairs with relative quantification errors.
- `figures/`: enabled plots, including distinct survey-time and reporting-delay
  filenames. A plot with no applicable observations may be omitted.
- `InputVariables_used.py` and `run_settings.json`: configuration source and
  invocation settings. Keep the original input files and any configuration imports
  with them to reproduce a run.
- `PaperVarsTemplate.tex`: classification counts and operational fractions.

Combined data includes an `ASE` column. Tables save before plotting so a plotting
problem does not discard completed numerical results. Read the console messages:
individual unavailable metrics/plots are reported and other outputs continue.
POD fits need suitable TP and FN observations; an out-of-range detection limit
does not establish performance beyond the tested rates. TTA/TTD retain the
existing sample-end convention; TTD does not require a valid receipt timestamp.

`--seed` defaults to 0 and controls sampling and plot bootstraps. Reproduce a
publication run using the same seed, inputs, configuration, and package versions.
The configuration is executable Python and should be a file you trust.

## Code map
`main.py` validates/configures the run and orchestrates loading, analysis, export,
and plotting. `LoadData.py` parses and restricts inputs. `Analysis.py` loops over
ASEs and combines results. `Classification.py` filters geometry, expands and
aggregates intervals, applies exclusions and thresholds, and pairs observations.
`Metrics.py` calculates timing, operational, and quantification metrics.
`Plotting.py` dispatches figures from `FiguresBuilder.py`. `writeFiles.py` writes
tables and LaTeX variables. Existing public function names are retained.
